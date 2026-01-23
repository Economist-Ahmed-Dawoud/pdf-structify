# pdf-structify

[![PyPI version](https://badge.fury.io/py/pdf-structify.svg)](https://badge.fury.io/py/pdf-structify)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Extract structured data from PDFs using LLMs with a scikit-learn-like API.**

pdf-structify makes it easy to extract structured, tabular data from PDF documents using Large Language Models. It handles PDF splitting, schema detection, and data extraction with progress tracking, checkpoint/resume support, and intelligent sampling.

## Features

- **Scikit-learn-like API**: Familiar `fit()`, `transform()`, `fit_transform()` interface
- **Automatic Schema Detection**: LLM analyzes documents to detect extractable fields
- **Purpose-Driven Extraction**: Optimized for "findings" (research data) or "policies" (policy documents)
- **Detection Modes**: Strict, moderate, or extended field discovery
- **Schema Save/Load**: Save detected schemas and resume from any point
- **Model Selection**: Use different models for detection vs extraction
- **Extraction Sampling**: Process a random sample of files for quick testing
- **Checkpoint/Resume**: Never lose progress - automatically resume from interruptions
- **Progress Bars**: Beautiful, informative progress tracking with `rich`
- **Automatic Retry**: Built-in retry logic for API errors

## Installation

```bash
pip install pdf-structify
```

## Quick Start

### 3-Line Extraction

```python
from structify import Pipeline

pipeline = Pipeline.quick_start()
results = pipeline.fit_transform("my_pdfs/")
results.to_csv("output.csv")
```

### Research Findings Extraction

```python
from structify import Pipeline

# Optimized for academic papers and research documents
pipeline = Pipeline(purpose="findings")
results = pipeline.fit_transform("research_papers/")
```

### Policy Document Extraction

```python
from structify import Pipeline

# Optimized for policy documents, regulations, and official reports
pipeline = Pipeline(purpose="policies")
results = pipeline.fit_transform("policy_documents/")
```

### From Natural Language Description

```python
from structify import Pipeline

pipeline = Pipeline.from_description("""
    Extract research findings from academic papers:
    - Author names and publication year
    - The country being studied
    - Main numerical finding (coefficient or percentage)
    - Statistical significance (p-value)
    - Methodology used (regression, RCT, etc.)
""")

results = pipeline.fit_transform("research_papers/")
```

## Advanced Features

### Schema Save/Load (Resume Capability)

Save your detected schema and reuse it later - no need to re-run detection:

```python
from structify import Pipeline

# First run: detect schema and save it
pipeline = Pipeline(purpose="findings")
pipeline.fit("documents/")
pipeline.save_schema("my_schema.json")  # or .yaml
results = pipeline.transform("documents/")

# Later: load schema and skip detection entirely
pipeline = Pipeline(schema="my_schema.json")
pipeline.fit("documents/")  # Skips detection - instant!
results = pipeline.transform("documents/")
```

You can also load and modify schemas programmatically:

```python
from structify import Pipeline, Schema

# Load, inspect, and use
schema = Schema.load("my_schema.json")
print(schema.fields)

pipeline = Pipeline(schema=schema)
```

### Model Selection (Detection vs Extraction)

Use a fast model for schema detection and a powerful model for extraction:

```python
from structify import Pipeline

pipeline = Pipeline(
    purpose="findings",
    detection_model="gemini-2.0-flash",   # Fast for detection
    extraction_model="gemini-2.5-pro",    # Powerful for extraction
)
results = pipeline.fit_transform("documents/")
```

### Extraction Sampling

Process only a subset of files for quick testing or cost control:

```python
from structify import Pipeline

pipeline = Pipeline(
    purpose="findings",
    extraction_sample_ratio=0.2,    # Extract from 20% of files
    extraction_max_samples=50,      # But no more than 50 files
    seed=42,                        # Reproducible sampling
)
results = pipeline.fit_transform("documents/")
```

### Detection Modes

Control how aggressively the schema detector discovers fields:

```python
from structify import Pipeline

# Strict: Only essential, high-confidence fields
pipeline = Pipeline(purpose="findings", detection_mode="strict")

# Moderate (default): Balanced field discovery
pipeline = Pipeline(purpose="findings", detection_mode="moderate")

# Extended: Discover more fields, including less common ones
pipeline = Pipeline(purpose="findings", detection_mode="extended")
```

### Complete Configuration Example

```python
from structify import Pipeline

pipeline = Pipeline(
    # Purpose and detection
    purpose="findings",
    detection_mode="moderate",

    # Model selection
    detection_model="gemini-2.0-flash",
    extraction_model="gemini-2.5-pro",

    # Sampling for detection
    sample_ratio=0.1,
    max_samples=30,

    # Sampling for extraction
    extraction_sample_ratio=0.5,
    extraction_max_samples=100,

    # Reproducibility
    seed=42,

    # Checkpointing
    checkpoint=True,
)

# Fit (detect schema)
pipeline.fit("documents/")
pipeline.save_schema("schema.json")

# Transform (extract data)
results = pipeline.transform("documents/")
results.to_csv("output.csv")
```

## Schema Detection

### Purpose Modes

**"findings"** - Optimized for research papers and academic documents:
- Extracts: estimates, coefficients, p-values, methodologies, country/region, time periods
- Mandatory fields: unit, value_unit, notes

**"policies"** - Optimized for policy documents and official reports:
- Extracts: policy names, types, sectors, implementing agencies, dates, targets
- Mandatory fields: unit, value_unit, notes

### Automatic Category Discovery

For categorical fields, pdf-structify automatically:
1. Discovers valid categories from your documents
2. Uses concise, abbreviated names (e.g., "DID" not "Difference-in-Differences with controls")
3. Enforces categories strictly during extraction

## With Custom Schema

```python
from structify import Pipeline, SchemaBuilder

schema = SchemaBuilder.create(
    name="financial_metrics",
    fields=[
        {"name": "company", "type": "string", "required": True},
        {"name": "year", "type": "integer", "required": True},
        {"name": "revenue", "type": "float"},
        {"name": "profit_margin", "type": "float"},
        {"name": "sector", "type": "categorical",
         "options": ["Tech", "Finance", "Healthcare", "Energy"]}
    ],
    focus_on=["financial statements", "annual reports"],
    skip=["legal disclaimers", "boilerplate text"]
)

pipeline = Pipeline.from_schema(schema)
results = pipeline.fit_transform("annual_reports/")
```

## Configuration

### Environment Variables

```bash
export GEMINI_API_KEY="your-api-key"
```

### In Code

```python
from structify import Config

Config.set(
    gemini_api_key="your-api-key",
    pages_per_chunk=10,
    temperature=0.1,
    max_retries=5
)
```

### From .env File

```python
from structify import Config
Config.from_env()  # Loads from .env file
```

## Components

### PDFSplitter

Split large PDFs into smaller chunks:

```python
from structify import PDFSplitter

splitter = PDFSplitter(pages_per_chunk=10)
splitter.transform("large_documents/", output_path="chunks/")
```

### SchemaDetector

Automatically detect extractable fields with sampling:

```python
from structify import SchemaDetector

detector = SchemaDetector(
    purpose="findings",
    detection_mode="moderate",
    sample_ratio=0.1,
    max_samples=30,
    seed=42,
)
schema = detector.fit_transform("documents/")
print(schema.fields)
schema.save("detected_schema.json")
```

### LLMExtractor

Extract data using a schema with sampling:

```python
from structify import LLMExtractor, Schema

schema = Schema.load("my_schema.json")

extractor = LLMExtractor(
    schema=schema,
    deduplicate=True,
    sample_ratio=0.5,      # Process 50% of files
    max_samples=100,       # But no more than 100
    seed=42,
)
results = extractor.fit_transform("documents/")
```

## Progress Tracking

pdf-structify provides beautiful progress bars:

```
╭─────────────────── Structify Pipeline ───────────────────╮
│ Stage 2/3: Data Extraction                               │
╰──────────────────────────────────────────────────────────╯
Processing papers ━━━━━━━━━━━━━━━━━ 45% 12/25 papers
  Current: "Economic_Study.pdf" part 3/8
  → Found 24 records
```

## Resume After Interruption

```python
from structify import Pipeline

# If interrupted, just run again - automatically resumes!
pipeline = Pipeline.resume("my_pdfs/")
results = pipeline.transform("my_pdfs/")
```

## Output Formats

```python
# CSV
results.to_csv("output.csv")

# JSON
results.to_json("output.json")

# Parquet
results.to_parquet("output.parquet")

# Excel
results.to_excel("output.xlsx")
```

## API Retry

pdf-structify includes automatic retry logic:
- **API errors**: Automatic 1 retry with 2-second delay
- **Rate limits**: Automatic backoff and retry
- **Timeouts**: Automatic retry with increasing delays

No configuration needed - it just works.

## Tutorials

Complete end-to-end tutorials are available in the `notebooks/` directory:

| Tutorial | Description |
|----------|-------------|
| [01_quick_start.ipynb](notebooks/01_quick_start.ipynb) | Basic 3-line extraction, configuration, and saving results |
| [02_research_findings.ipynb](notebooks/02_research_findings.ipynb) | Extracting from academic papers with `purpose="findings"` |
| [03_policy_documents.ipynb](notebooks/03_policy_documents.ipynb) | Extracting from policy docs with `purpose="policies"` |
| [04_advanced_configuration.ipynb](notebooks/04_advanced_configuration.ipynb) | Schema save/load, model selection, sampling, checkpoints |
| [05_custom_schemas.ipynb](notebooks/05_custom_schemas.ipynb) | Building custom schemas with SchemaBuilder |

## Requirements

- Python 3.10+
- Google Gemini API key

## Dependencies

- google-genai
- pypdf
- rich
- pydantic
- pandas
- python-dotenv
- pyyaml

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
