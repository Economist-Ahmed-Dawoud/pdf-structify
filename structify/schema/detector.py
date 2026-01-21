"""Automatic schema detection from documents."""

import json
import random
from typing import Any

from structify.schema.types import Field, Schema, FieldType, ExtractionRules
from structify.schema.builder import SchemaBuilder
from structify.providers.gemini import GeminiProvider
from structify.preprocessing.loader import PDFLoader, PDFChunk
from structify.core.base import BaseTransformer
from structify.core.exceptions import SchemaError
from structify.utils.logging import get_logger, Logger
from structify.utils.json_repair import repair_json
from structify.progress.tracker import ProgressTracker

logger = get_logger("schema.detector")


SCHEMA_DETECTION_PROMPT = """You are an expert data analyst designing a schema to extract SUBSTANTIVE FINDINGS from documents.

## YOUR GOAL
Identify the CORE DATA that makes this document valuable - the actual findings, results, and measurements that someone would want to analyze in a spreadsheet or database.

## CRITICAL: What to Extract vs What to IGNORE

### EXTRACT (Substantive Data):
- **Quantitative results**: Statistics, coefficients, percentages, measurements, effect sizes
- **Research findings**: Treatment effects, outcomes, impacts, estimates with confidence intervals
- **Comparative data**: Before/after comparisons, treatment vs control, regional differences
- **Categorical classifications**: Methodologies used, sectors studied, policy types
- **Key identifiers**: Study IDs, time periods covered, geographic scope

### IGNORE (Metadata/Noise - DO NOT create fields for these):
- Bibliography/references (author names from citations, publication years of cited works)
- Table of contents, page numbers, headers/footers
- Acknowledgments, author affiliations, contact information
- Abstract summaries (extract the actual data, not the summary)
- Boilerplate text, copyright notices
- Raw literature review content (unless it contains comparative data)

## FEW-SHOT EXAMPLES

### EXAMPLE 1: Research Paper on Economic Zones
**BAD Schema** (extracts noise):
```
- author_name: "Names of authors" ❌ (metadata, not findings)
- reference_year: "Year from bibliography" ❌ (citation metadata)
- journal_title: "Where published" ❌ (not substantive data)
```

**GOOD Schema** (extracts findings):
```
- study_id: "Unique identifier like AuthorYear_N" ✓
- zone_type: "Type of economic zone (SEZ, FTZ, EPZ)" ✓
- outcome_variable: "What was measured (exports, employment, FDI)" ✓
- estimate_value: "The coefficient or effect size" ✓
- standard_error: "Standard error of estimate" ✓
- methodology: "DID, IV, RDD, OLS" ✓
- country_studied: "Country where zone is located" ✓
- time_period: "Years covered by study" ✓
- significance_level: "p<0.01, p<0.05, p<0.10, NS" ✓
```

### EXAMPLE 2: Financial Report
**BAD Schema**:
```
- report_author: "Who wrote the report" ❌
- page_number: "Page reference" ❌
- section_title: "Section headers" ❌
```

**GOOD Schema**:
```
- company_name: "Company being analyzed" ✓
- fiscal_year: "Year of the data" ✓
- revenue: "Total revenue in USD" ✓
- net_income: "Net income in USD" ✓
- growth_rate: "Year-over-year growth percentage" ✓
- profit_margin: "Profit margin percentage" ✓
```

### EXAMPLE 3: Survey Results
**BAD Schema**:
```
- surveyor_name: "Who conducted survey" ❌
- survey_date: "When survey was done" ❌
```

**GOOD Schema**:
```
- respondent_category: "Type of respondent (business, household)" ✓
- sample_size: "Number of respondents" ✓
- satisfaction_score: "Rating 1-5" ✓
- response_rate: "Percentage who responded" ✓
- key_finding: "Main result from the question" ✓
```

## FOCUS YOUR ANALYSIS ON:
1. **Tables** - These almost always contain the substantive data
2. **Results sections** - Where findings are reported
3. **Data appendices** - Raw numbers and statistics
4. **Executive summaries** - Only for the actual findings mentioned

## YOUR TASK
Analyze this document and identify fields for the SUBSTANTIVE DATA only.

For each field:
1. Give it a short, snake_case name
2. Type: string, integer, float, boolean, or categorical
3. Brief description of what data it captures
4. Required = true only if this is core data present in every record
5. For categorical: list the options you observe

RESPOND WITH ONLY VALID JSON:
{{
  "detected_fields": [
    {{
      "name": "field_name",
      "type": "string|integer|float|boolean|categorical",
      "description": "What substantive data this captures",
      "required": true,
      "options": [],
      "frequency": "always|often|sometimes|rarely"
    }}
  ],
  "document_type": "research paper|financial report|survey|policy document|other",
  "suggested_focus": ["tables", "results section", "specific sections with data"],
  "suggested_skip": ["bibliography", "references", "acknowledgments", "sections without data"]
}}

JSON only. No markdown code blocks. No explanations."""


class SchemaDetector(BaseTransformer[str, Schema]):
    """
    Automatically detect schema from sample documents.

    Uses LLM to analyze a sample of documents and identify
    common fields that can be extracted.
    """

    def __init__(
        self,
        provider: GeminiProvider | None = None,
        sample_ratio: float = 0.1,
        max_samples: int = 30,
        min_samples: int = 3,
        min_field_frequency: float = 0.3,
    ):
        """
        Initialize the schema detector.

        Args:
            provider: LLM provider for analysis
            sample_ratio: Fraction of documents to sample (0.0-1.0)
            max_samples: Maximum number of samples
            min_samples: Minimum number of samples
            min_field_frequency: Minimum frequency for a field to be included
        """
        super().__init__(
            sample_ratio=sample_ratio,
            max_samples=max_samples,
            min_samples=min_samples,
            min_field_frequency=min_field_frequency,
        )

        self.provider = provider
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.min_field_frequency = min_field_frequency

        self._detected_schema: Schema | None = None

    def fit(
        self,
        data: str,
        tracker: ProgressTracker | None = None,
        **kwargs,
    ) -> "SchemaDetector":
        """
        Detect schema from documents.

        Args:
            data: Path to directory containing documents
            tracker: Optional progress tracker

        Returns:
            self
        """
        # Initialize provider if needed
        if self.provider is None:
            self.provider = GeminiProvider()
            self.provider.initialize()

        # Load documents
        loader = PDFLoader()
        all_chunks = loader.get_all_chunks(data)

        if not all_chunks:
            raise SchemaError(f"No documents found in {data}")

        # Sample documents
        samples = self._select_samples(all_chunks)
        logger.info(f"Sampling {len(samples)} documents for schema detection")

        # Set up progress tracking
        if tracker:
            tracker.add_stage("detect_schema", len(samples))
            tracker.start_stage("detect_schema")

        # Analyze each sample
        all_fields: dict[str, list[dict[str, Any]]] = {}
        document_types: list[str] = []
        focus_suggestions: list[str] = []
        skip_suggestions: list[str] = []

        for i, chunk in enumerate(samples):
            try:
                logger.debug(f"Analyzing sample {i+1}/{len(samples)}: {chunk.name}")

                if tracker:
                    tracker.update(current_item=chunk.name)

                # Upload and analyze
                file_ref = self.provider.upload_file(str(chunk.path))
                response = self.provider.generate(SCHEMA_DETECTION_PROMPT, file_ref)

                # Parse response
                result = self._parse_detection_response(response)

                if result:
                    # Aggregate fields
                    for field_data in result.get("detected_fields", []):
                        name = field_data.get("name", "")
                        if name:
                            if name not in all_fields:
                                all_fields[name] = []
                            all_fields[name].append(field_data)

                    # Collect suggestions
                    if result.get("document_type"):
                        document_types.append(result["document_type"])
                    focus_suggestions.extend(result.get("suggested_focus", []))
                    skip_suggestions.extend(result.get("suggested_skip", []))

                if tracker:
                    tracker.increment()

            except Exception as e:
                logger.warning(f"Error analyzing sample {chunk.name}: {e}")
                if tracker:
                    tracker.log_substep(f"Error: {e}", style="warning")

        # Build schema from aggregated fields
        self._detected_schema = self._build_schema_from_fields(
            all_fields,
            document_types,
            focus_suggestions,
            skip_suggestions,
            len(samples),
        )

        if tracker:
            tracker.complete_stage("detect_schema")
            Logger.log_success(f"Detected {len(self._detected_schema.fields)} fields")

        self._is_fitted = True
        return self

    def transform(self, data: str, **kwargs) -> Schema:
        """
        Return the detected schema.

        Args:
            data: Input path (ignored, schema already detected)

        Returns:
            Detected schema
        """
        if not self._is_fitted or self._detected_schema is None:
            raise SchemaError("SchemaDetector must be fit before transform")
        return self._detected_schema

    def _select_samples(self, chunks: list[PDFChunk]) -> list[PDFChunk]:
        """Select a representative sample of chunks."""
        total = len(chunks)
        sample_size = int(total * self.sample_ratio)
        sample_size = max(self.min_samples, min(sample_size, self.max_samples))
        sample_size = min(sample_size, total)

        # Random sample
        return random.sample(chunks, sample_size)

    def _parse_detection_response(self, response: str) -> dict[str, Any] | None:
        """Parse LLM response for schema detection."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to repair
            repaired = repair_json(response)
            if repaired and len(repaired) == 1:
                return repaired[0]
            return None

    def _build_schema_from_fields(
        self,
        all_fields: dict[str, list[dict[str, Any]]],
        document_types: list[str],
        focus_suggestions: list[str],
        skip_suggestions: list[str],
        total_samples: int,
    ) -> Schema:
        """Build schema from aggregated field data."""
        fields = []

        for name, field_instances in all_fields.items():
            # Calculate frequency
            frequency = len(field_instances) / total_samples

            # Skip infrequent fields
            if frequency < self.min_field_frequency:
                continue

            # Aggregate field properties
            types = [f.get("type", "string") for f in field_instances]
            descriptions = [f.get("description", "") for f in field_instances]
            required_votes = [f.get("required", False) for f in field_instances]
            all_options = []
            for f in field_instances:
                all_options.extend(f.get("options", []))

            # Determine most common type
            type_counts: dict[str, int] = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            most_common_type = max(type_counts, key=type_counts.get)

            try:
                field_type = FieldType(most_common_type)
            except ValueError:
                field_type = FieldType.STRING

            # Use most detailed description
            description = max(descriptions, key=len) if descriptions else ""

            # Required if majority say so
            required = sum(required_votes) > len(required_votes) / 2

            # Unique options
            options = list(set(all_options))

            fields.append(
                Field(
                    name=name,
                    type=field_type,
                    description=description,
                    required=required,
                    options=options if field_type == FieldType.CATEGORICAL else [],
                )
            )

        # Sort by required first, then by name
        fields.sort(key=lambda f: (not f.required, f.name))

        # Determine document type
        doc_type = max(set(document_types), key=document_types.count) if document_types else "document"

        # Aggregate suggestions
        focus_on = list(set(focus_suggestions))[:5]
        skip = list(set(skip_suggestions))[:5]

        rules = ExtractionRules(
            focus_on=focus_on,
            skip=skip,
            context=f"Extracting data from {doc_type}s",
        )

        return Schema(
            name=f"detected_{doc_type.lower().replace(' ', '_')}_schema",
            description=f"Automatically detected schema for {doc_type}",
            fields=fields,
            extraction_rules=rules,
        )

    @property
    def schema(self) -> Schema | None:
        """Get the detected schema."""
        return self._detected_schema
