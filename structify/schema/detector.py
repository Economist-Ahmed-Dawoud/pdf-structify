"""Automatic schema detection from documents."""

import json
import random
from typing import Any, Literal

from structify.schema.types import Field, Schema, FieldType, ExtractionRules
from structify.providers.gemini import GeminiProvider
from structify.preprocessing.loader import PDFLoader, PDFChunk
from structify.core.base import BaseTransformer
from structify.core.exceptions import SchemaError
from structify.utils.logging import get_logger, Logger
from structify.utils.json_repair import repair_json
from structify.progress.tracker import ProgressTracker

logger = get_logger("schema.detector")

# Detection mode type
DetectionMode = Literal["strict", "moderate", "extended"]

# Extraction purpose type
ExtractionPurpose = Literal["findings", "policies"]

# Mode-specific instructions for the LLM (field count limits)
MODE_INSTRUCTIONS = {
    "strict": """
## STRICT MODE - CORE DATA ONLY (UP TO 5-7 fields maximum)

You are in STRICT mode. Identify UP TO 5-7 of the MOST IMPORTANT fields.
This is a MAXIMUM limit - return FEWER fields if the documents don't contain enough meaningful data.

Quality over quantity. Never pad with weak fields.
""",
    "moderate": """
## MODERATE MODE - IMPORTANT DATA (UP TO 7-12 fields maximum)

You are in MODERATE mode. Identify UP TO 7-12 of the MOST IMPORTANT fields.
This is a MAXIMUM limit - return FEWER fields if the documents don't contain enough meaningful data.

Quality over quantity. Never pad with weak fields.
""",
    "extended": """
## EXTENDED MODE - COMPREHENSIVE (UP TO 12-20 fields maximum)

You are in EXTENDED mode. Identify UP TO 12-20 of the MOST IMPORTANT fields.
This is a MAXIMUM limit - return FEWER fields if the documents don't contain enough meaningful data.

Quality over quantity. Never pad with weak fields.
"""
}

# Purpose-specific instructions for the LLM
PURPOSE_INSTRUCTIONS = {
    "findings": """
## EXTRACTION PURPOSE: RESEARCH FINDINGS

Extract QUANTITATIVE RESEARCH FINDINGS - the measurable results and estimates from studies.

Focus on extracting:
- **Estimates & Coefficients**: Treatment effects, regression coefficients, effect sizes, elasticities
- **Statistical Measures**: Standard errors, confidence intervals, p-values, t-statistics
- **Outcome Variables**: What was measured (employment, exports, GDP, health outcomes, test scores)
- **Methodology**: How the result was obtained (DID, IV, RDD, RCT, OLS, fixed effects)
- **Context**: Sample size, time period, geographic scope, unit of analysis
- **Causal vs Correlational**: Whether the estimate represents causal identification

### CONCISE CATEGORY NAMES (CRITICAL)

For ALL categorical fields, use SHORT, ABBREVIATED names:
- "Difference-in-Differences with region controls" → "DID"
- "Fixed effects with clustered standard errors" → "FE"
- "Instrumental variables two-stage least squares" → "IV"
- "Regression discontinuity design" → "RDD"
- "Randomized controlled trial" → "RCT"
- "Ordinary least squares" → "OLS"
- "Propensity score matching" → "PSM"

Details and qualifiers belong in the **notes** field, NOT in category names!

### Example Fields for Research Findings:
- estimate_value (float): The coefficient or effect size
- standard_error (float): Standard error of the estimate
- outcome_variable (string): What was measured
- methodology (categorical): DID, IV, RDD, RCT, OLS, FE, PSM, etc.
- is_causal (boolean): Whether causal identification was attempted
- sample_size (integer): Number of observations
- time_period (string): Years covered by the analysis
- country (string): Geographic location studied
- significance_level (categorical): p<0.01, p<0.05, p<0.10, NS

### MANDATORY FIELDS FOR RESEARCH FINDINGS:
You MUST always include these two fields in every schema:

1. **unit** (string, REQUIRED): Unit of measurement for the estimate. Examples:
   - "%" for percentage change
   - "log" for log points
   - "coefficient" for raw regression coefficients
   - "pp" or "percentage points" for percentage point changes
   - "USD" or currency for monetary values
   - "elasticity" for elasticity estimates
   - "odds ratio" for odds ratios
   - "hazard ratio" for hazard ratios

2. **notes** (string, REQUIRED): One sentence explaining this data point and context
""",
    "policies": """
## EXTRACTION PURPOSE: POLICIES & INTERVENTIONS

Extract POLICIES, INCENTIVES, DECISIONS, and INTERVENTIONS from documents.
This is domain-agnostic - works for economics, health, education, environment, or any policy domain.

### CRITICAL: FIELD TYPE PREFERENCES

You MUST follow this hierarchy when choosing field types:

1. **NUMERIC (float/integer)** - PREFERRED for any quantifiable data:
   - Rates, percentages, durations, amounts, counts, thresholds
   - Example: `rate_value` (float), `duration_years` (integer), `threshold_amount` (float)

2. **CATEGORICAL** - PREFERRED when you observe REPEATING PATTERNS:
   - If similar values appear across documents, make it categorical
   - DISCOVER the categories from the documents - list ALL distinct options you observe
   - Aim for 3-10 options per categorical field (merge if too granular)
   - Example: `policy_type` with options discovered from documents

3. **BOOLEAN** - For yes/no or true/false distinctions:
   - Example: `is_conditional`, `is_time_limited`, `requires_approval`

4. **STRING** - ONLY for these specific cases:
   - `notes` field (MANDATORY) - one sentence context explanation
   - `value_unit` field (MANDATORY) - unit of measurement for numeric values
   - Truly free-form text that cannot be categorized (AVOID if possible)

### CONCISE CATEGORY NAMES (CRITICAL)

For ALL categorical fields, use SHORT, SIMPLE names (2-3 words max):
- "Corporate income tax exemption for 10 years" → "Tax holiday"
- "Reduced corporate tax rate of 15%" → "Reduced tax"
- "Subsidized land lease at 50% market rate" → "Land subsidy"
- "One-stop service center for permits" → "Admin support"
- "National government ministry" → "National govt"
- "Small and medium enterprises" → "SMEs"
- "Manufacturing and processing sector" → "Manufacturing"

Details, conditions, and specifics belong in the **notes** field, NOT in category names!

### CATEGORY DISCOVERY REQUIREMENTS

For EVERY categorical field, you MUST:
1. Analyze all sample documents thoroughly
2. List ALL distinct category values you actually observe
3. Use consistent naming (pick one canonical term, avoid synonyms)
4. Merge similar categories (e.g., "Tax break" + "Tax holiday" → "Tax incentive")

### WHAT TO EXTRACT (use categorical where patterns exist)

- **WHAT** type of policy/intervention (categorize types you observe)
- **WHO** provides it (categorize: government levels, agencies, organizations)
- **WHO** benefits (categorize: firm types, sectors, demographics)
- **HOW MUCH** - numeric values (ALWAYS pair with value_unit)
- **WHEN** - durations, years (use integer/float where possible)
- **WHERE** - geographic scope (categorize if patterns exist)

### MANDATORY FIELDS

1. **value_unit** (string, REQUIRED): Unit for numeric values (%, years, USD, etc.)
2. **notes** (string, REQUIRED): One sentence explaining context and key details

### AVOID - DO NOT CREATE THESE

- Multiple string fields (e.g., "policy_details", "conditions", "specifics") - put details in notes
- String fields for things with repeating patterns - use categorical instead
- Fields without discovered options for categorical type
- More than 2 string fields total (notes + value_unit only)
- Long verbose category names - keep them short!
"""
}

# Common instructions for all extractions
COMMON_INSTRUCTIONS = """
## CRITICAL DATA QUALITY REQUIREMENTS

### 1. LONG FORMAT OUTPUT
The resulting dataset MUST be in LONG FORMAT (one observation per row).
If a document contains multiple findings/policies, each becomes a SEPARATE ROW.
Do NOT create wide format with columns like finding_1, finding_2, etc.

### 2. FIELD COVERAGE & MISSING VALUES
- Prefer fields that are likely to be available in 70-80% of cases
- ALLOW missing values (null/empty) when data is not present
- Do NOT drop rows just because some fields are missing
- AVOID highly sparse fields (present in <30% of cases) - these create mostly empty columns

### 3. MANDATORY 'notes' FIELD
ALWAYS include a field called "notes" (type: string, required: true).
This field must contain ONE SENTENCE per row explaining:
- What this data point represents
- Key context for interpretation
- Any caveats or qualifications

Example notes:
- "DID estimate of SEZ impact on exports, using neighboring counties as control, 2000-2010."
- "Tax holiday of 10 years for manufacturing, announced in 2015 SEZ policy reform."
- "RCT effect of training program on test scores, 6-month follow-up, urban schools only."

### 4. AVOID METADATA
Do NOT create fields for:
- Author names, journal titles, publication dates
- Page numbers, section headers, table numbers
- Bibliography entries, citation counts
- File names, document IDs (unless meaningful like study_id)
"""

# Frequency thresholds for each mode (field must appear in X% of samples)
MODE_FREQUENCY_THRESHOLDS = {
    "strict": 0.6,    # Field must appear in 60%+ of samples
    "moderate": 0.4,  # Field must appear in 40%+ of samples
    "extended": 0.3,  # Field must appear in 30%+ of samples
}

# Maximum fields to return per mode
MODE_MAX_FIELDS = {
    "strict": 7,
    "moderate": 12,
    "extended": 20,
}


SCHEMA_DETECTION_PROMPT_TEMPLATE = """You are an expert data analyst designing a schema to extract structured data from documents.

This extraction will be used for systematic reviews and data analysis.
Quality is MUCH more important than quantity. Every field must be highly relevant and reasonably present.

{mode_instructions}

{purpose_instructions}

{common_instructions}

## YOUR TASK

Analyze these documents and identify fields for SUBSTANTIVE DATA extraction.

For each field:
1. Give it a short, snake_case name
2. Type: Prefer numeric (integer/float) > categorical > boolean > string
3. Brief description of what data it captures
4. Required = true only for core fields present in most records
5. **CRITICAL FOR CATEGORICAL**: You MUST list ALL options you observe in the documents

### CATEGORICAL FIELD REQUIREMENTS
When you identify a categorical field:
- Scan ALL sample documents for distinct values
- List EVERY option you find in the "options" array
- **USE SHORT, CONCISE NAMES** (2-3 words max, abbreviate where standard)
- Use consistent canonical names (merge synonyms)
- Aim for 3-10 options (merge if too granular)
- If you cannot identify distinct options, use STRING type instead
- Put details/qualifiers in the notes field, NOT in category names

IMPORTANT: Always include the mandatory "notes" field (string, required: true).

RESPOND WITH ONLY VALID JSON:
{{
  "detected_fields": [
    {{
      "name": "field_name",
      "type": "string|integer|float|boolean|categorical",
      "description": "What substantive data this captures",
      "required": true,
      "options": ["option1", "option2", "option3"],
      "frequency": "always|often|sometimes"
    }},
    {{
      "name": "notes",
      "type": "string",
      "description": "One sentence explaining this data point and providing context for interpretation",
      "required": true,
      "options": [],
      "frequency": "always"
    }}
  ],
  "document_type": "research paper|financial report|survey|policy document|other",
  "suggested_focus": ["tables", "results section", "specific sections with data"],
  "suggested_skip": ["bibliography", "references", "acknowledgments", "sections without data"]
}}

CRITICAL: For categorical fields, the "options" array MUST contain the actual values found in documents.
Empty options for categorical fields will be converted to string type.

JSON only. No markdown code blocks. No explanations."""


class SchemaReviewer:
    """
    Convert schema to dict format and filter by field names.

    Workflow:
        1. Use to_dict() to get {field_name: description} for review
        2. User selects which fields to keep (list of names)
        3. Use select_fields() to create filtered schema
    """

    @staticmethod
    def to_dict(schema: Schema) -> dict[str, str]:
        """
        Convert schema to {field_name: description} dict.

        This is the primary method for reviewing detected schema.
        Returns a simple dict that can be printed, saved as JSON,
        or used for manual field selection.

        Args:
            schema: Schema to convert

        Returns:
            Dict mapping field names to their descriptions

        Example:
            >>> schema_dict = SchemaReviewer.to_dict(schema)
            >>> print(schema_dict)
            {'estimate_value': 'The coefficient...', 'methodology': 'DID, IV...'}
        """
        if not schema or not schema.fields:
            return {}

        return {
            field.name: field.description or f"({field.type.value})"
            for field in schema.fields
        }

    @staticmethod
    def select_fields(schema: Schema, field_names: list[str]) -> Schema:
        """
        Create new schema with only the specified fields.

        Args:
            schema: Original schema
            field_names: List of field names to keep

        Returns:
            New schema containing only selected fields

        Raises:
            SchemaError: If no valid fields are selected

        Example:
            >>> filtered = SchemaReviewer.select_fields(schema, ["estimate_value", "methodology"])
        """
        if not schema or not schema.fields:
            raise SchemaError("Cannot select fields from empty schema")

        # Convert to set for O(1) lookup
        names_to_keep = set(field_names)

        selected_fields = [
            f for f in schema.fields
            if f.name in names_to_keep
        ]

        if not selected_fields:
            available = [f.name for f in schema.fields]
            raise SchemaError(
                f"No valid fields selected. "
                f"Requested: {field_names}. "
                f"Available: {available}"
            )

        # Log which fields were kept
        kept_names = [f.name for f in selected_fields]
        logger.info(f"Selected {len(selected_fields)} fields: {kept_names}")

        return Schema(
            name=schema.name,
            description=schema.description,
            fields=selected_fields,
            extraction_rules=schema.extraction_rules,
        )

    @staticmethod
    def display(schema: Schema) -> None:
        """
        Print schema as dict to console for easy review.

        Args:
            schema: Schema to display
        """
        if not schema or not schema.fields:
            print("\nNo fields in schema.")
            return

        schema_dict = SchemaReviewer.to_dict(schema)

        print("\n" + "=" * 60)
        print(f"  DETECTED SCHEMA: {schema.name}")
        print("=" * 60)
        print("\n  Fields (copy field names to select):\n")

        for name, desc in schema_dict.items():
            # Truncate long descriptions
            if len(desc) > 50:
                desc = desc[:47] + "..."
            print(f"    \"{name}\": \"{desc}\"")

        print("\n" + "=" * 60)
        print(f"  Total: {len(schema_dict)} fields")
        print("=" * 60)


class SchemaDetector(BaseTransformer[str, Schema]):
    """
    Automatically detect schema from sample documents.

    Uses LLM to analyze a sample of documents and identify
    common fields that can be extracted.

    Detection Modes (field count limits):
        - "strict": UP TO 5-7 core fields only
        - "moderate": UP TO 7-12 important fields (default)
        - "extended": UP TO 12-20 comprehensive fields

    Extraction Purpose:
        - "findings": Research findings, estimates, coefficients, effects (default)
        - "policies": Policies, incentives, decisions, interventions

    Quality Principle:
        The detector will NEVER pad with weak fields to reach target counts.
        If only 5 quality fields exist, it returns 5 even in "extended" mode.
    """

    def __init__(
        self,
        provider: GeminiProvider | None = None,
        detection_mode: DetectionMode = "moderate",
        purpose: ExtractionPurpose = "findings",
        sample_ratio: float = 0.1,
        max_samples: int = 10,
        min_samples: int = 3,
        min_field_frequency: float | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the schema detector.

        Args:
            provider: LLM provider for analysis
            detection_mode: One of "strict", "moderate", or "extended"
                - strict: UP TO 5-7 core fields
                - moderate: UP TO 7-12 fields (default)
                - extended: UP TO 12-20 fields
            purpose: What type of information to extract
                - "findings": Research findings, estimates, coefficients (default)
                - "policies": Policies, incentives, decisions, interventions
            sample_ratio: Fraction of documents to sample (0.0-1.0)
            max_samples: Maximum number of samples to upload (default: 10)
            min_samples: Minimum number of samples
            min_field_frequency: Override frequency threshold (auto-set by mode if None)
            seed: Random seed for reproducible sampling. If provided, the same
                documents will be sampled each run for comparability.
        """
        # Set frequency threshold based on mode if not provided
        if min_field_frequency is None:
            min_field_frequency = MODE_FREQUENCY_THRESHOLDS.get(detection_mode, 0.4)

        super().__init__(
            detection_mode=detection_mode,
            purpose=purpose,
            sample_ratio=sample_ratio,
            max_samples=max_samples,
            min_samples=min_samples,
            min_field_frequency=min_field_frequency,
            seed=seed,
        )

        self.provider = provider
        self.detection_mode = detection_mode
        self.purpose = purpose
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.min_field_frequency = min_field_frequency
        self.seed = seed
        self._max_fields = MODE_MAX_FIELDS.get(detection_mode, 12)

        self._detected_schema: Schema | None = None

    def _get_detection_prompt(self) -> str:
        """Build the detection prompt with mode and purpose-specific instructions."""
        mode_instructions = MODE_INSTRUCTIONS.get(self.detection_mode, MODE_INSTRUCTIONS["moderate"])
        purpose_instructions = PURPOSE_INSTRUCTIONS.get(self.purpose, PURPOSE_INSTRUCTIONS["findings"])
        return SCHEMA_DETECTION_PROMPT_TEMPLATE.format(
            mode_instructions=mode_instructions,
            purpose_instructions=purpose_instructions,
            common_instructions=COMMON_INSTRUCTIONS,
        )

    def fit(
        self,
        data: str,
        tracker: ProgressTracker | None = None,
        **kwargs,
    ) -> "SchemaDetector":
        """
        Detect schema from documents.

        Uploads all sample documents and makes ONE LLM call to detect
        the schema across all samples together. This ensures we get
        exactly the target number of fields for the detection mode:
        - strict: 5-7 fields
        - moderate: 7-12 fields
        - extended: 12-20 fields

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

        # Sample chunks (10% of total, max 10)
        samples = self._select_samples(all_chunks)
        logger.info(f"Sampling {len(samples)} chunks for schema detection (mode: {self.detection_mode}, purpose: {self.purpose})")

        # Set up progress tracking
        if tracker:
            tracker.add_stage("detect_schema", len(samples) + 1)
            tracker.start_stage("detect_schema")

        # Step 1: Upload all sample files
        file_refs = []
        for i, chunk in enumerate(samples):
            try:
                logger.debug(f"Uploading sample {i+1}/{len(samples)}: {chunk.name}")
                if tracker:
                    tracker.update(current_item=f"Uploading {chunk.name}")
                file_ref = self.provider.upload_file(str(chunk.path))
                file_refs.append(file_ref)
                if tracker:
                    tracker.increment()
            except Exception as e:
                logger.warning(f"Error uploading sample {chunk.name}: {e}")

        if not file_refs:
            raise SchemaError("Failed to upload any sample chunks")

        # Step 2: ONE LLM call with all files to get exact field count
        if tracker:
            tracker.update(current_item="Analyzing all samples together")

        detection_prompt = self._get_detection_prompt()
        response = self.provider.generate_with_files(detection_prompt, file_refs)

        if tracker:
            tracker.increment()

        # Parse the single response
        result = self._parse_detection_response(response)

        if not result or not result.get("detected_fields"):
            raise SchemaError("Failed to detect any fields from samples")

        # Build schema directly from the response
        self._detected_schema = self._build_schema_from_single_response(result)

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
        """Select a representative sample of chunks.

        If seed is set, uses a seeded RNG for reproducible sampling.
        """
        total = len(chunks)
        sample_size = int(total * self.sample_ratio)
        sample_size = max(self.min_samples, min(sample_size, self.max_samples))
        sample_size = min(sample_size, total)

        # Use seeded RNG for reproducibility if seed is provided
        if self.seed is not None:
            rng = random.Random(self.seed)
            logger.debug(f"Using seed {self.seed} for reproducible sampling")
            return rng.sample(chunks, sample_size)

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
        field_frequencies: dict[str, float] = {}

        for name, field_instances in all_fields.items():
            # Calculate frequency (used for sorting, not filtering)
            frequency = len(field_instances) / total_samples
            field_frequencies[name] = frequency

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

        # Sort by: required first, then by frequency (most common), then name
        fields.sort(
            key=lambda f: (
                not f.required,
                -field_frequencies.get(f.name, 0),
                f.name
            )
        )

        # Limit to max fields for the detection mode
        if len(fields) > self._max_fields:
            logger.info(
                f"Limiting fields from {len(fields)} to {self._max_fields} "
                f"(mode: {self.detection_mode})"
            )
            fields = fields[:self._max_fields]

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

    def _build_schema_from_single_response(self, result: dict[str, Any]) -> Schema:
        """Build schema from a single LLM response (used with combined file upload)."""
        fields = []
        has_notes_field = False
        has_unit_field = False
        has_value_unit_field = False

        for field_data in result.get("detected_fields", []):
            name = field_data.get("name", "")
            if not name:
                continue

            if name == "notes":
                has_notes_field = True
            if name == "unit":
                has_unit_field = True
            if name == "value_unit":
                has_value_unit_field = True

            # Parse type
            type_str = field_data.get("type", "string")
            try:
                field_type = FieldType(type_str)
            except ValueError:
                field_type = FieldType.STRING

            # Get options for categorical fields
            options = field_data.get("options", [])

            # Validate categorical fields must have options
            if field_type == FieldType.CATEGORICAL and not options:
                logger.warning(
                    f"Categorical field '{name}' has no options - converting to string"
                )
                field_type = FieldType.STRING

            fields.append(
                Field(
                    name=name,
                    type=field_type,
                    description=field_data.get("description", ""),
                    required=field_data.get("required", False),
                    options=options if field_type == FieldType.CATEGORICAL else [],
                )
            )

        # Ensure mandatory 'notes' field is always present
        if not has_notes_field:
            logger.debug("Adding mandatory 'notes' field")
            fields.append(
                Field(
                    name="notes",
                    type=FieldType.STRING,
                    description="One sentence explaining this data point and providing context for interpretation",
                    required=True,
                    options=[],
                )
            )

        # For findings purpose, ensure mandatory 'unit' field is present
        if self.purpose == "findings" and not has_unit_field:
            logger.debug("Adding mandatory 'unit' field for findings")
            fields.append(
                Field(
                    name="unit",
                    type=FieldType.STRING,
                    description="Unit of measurement for the estimate (%, log, coefficient, percentage points, USD, elasticity, etc.)",
                    required=True,
                    options=[],
                )
            )

        # For policies purpose, ensure mandatory 'value_unit' field is present
        if self.purpose == "policies" and not has_value_unit_field:
            logger.debug("Adding mandatory 'value_unit' field for policies")
            fields.append(
                Field(
                    name="value_unit",
                    type=FieldType.STRING,
                    description="Unit of measurement for numeric values (%, years, USD, etc.)",
                    required=True,
                    options=[],
                )
            )

        # Limit to max fields as safety cap (mandatory fields don't count against limit)
        mandatory_names = {"notes", "unit", "value_unit"}
        mandatory_fields = [f for f in fields if f.name in mandatory_names]
        non_mandatory_fields = [f for f in fields if f.name not in mandatory_names]

        if len(non_mandatory_fields) > self._max_fields:
            logger.info(f"Limiting fields from {len(non_mandatory_fields)} to {self._max_fields} (mode: {self.detection_mode})")
            non_mandatory_fields = non_mandatory_fields[:self._max_fields]

        fields = non_mandatory_fields + mandatory_fields

        # Get document type and suggestions
        doc_type = result.get("document_type", "document")
        focus_on = result.get("suggested_focus", [])[:5]
        skip = result.get("suggested_skip", [])[:5]

        rules = ExtractionRules(
            focus_on=focus_on,
            skip=skip,
            context=f"Extracting {self.purpose} from {doc_type}s",
        )

        return Schema(
            name=f"detected_{doc_type.lower().replace(' ', '_')}_schema",
            description=f"Automatically detected schema for {doc_type} ({self.purpose})",
            fields=fields,
            extraction_rules=rules,
        )

    @property
    def schema(self) -> Schema | None:
        """Get the detected schema."""
        return self._detected_schema
