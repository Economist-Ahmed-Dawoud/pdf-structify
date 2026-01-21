# pdf-structify

[![PyPI version](https://badge.fury.io/py/pdf-structify.svg)](https://badge.fury.io/py/pdf-structify)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Extract structured data from PDFs using LLMs with a scikit-learn-like API.**

pdf-structify makes it easy to extract structured, tabular data from PDF documents using Large Language Models. It handles PDF splitting, schema detection, and data extraction with progress tracking and checkpoint/resume support.

## Features

- **Scikit-learn-like API**: Familiar `fit()`, `transform()`, `fit_transform()` interface
- **Automatic Schema Detection**: Let the LLM analyze your documents and detect extractable fields
- **Natural Language Schema Definition**: Describe what you want to extract in plain English
- **Progress Bars**: Beautiful, informative progress tracking with `rich`
- **Checkpoint/Resume**: Never lose progress - automatically resume from interruptions
- **Two-Layer Prompt System**: Strict JSON enforcement for reliable extraction
- **PDF Splitting**: Automatically split large PDFs into manageable chunks

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

### With Custom Schema

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

### Resume After Interruption

```python
from structify import Pipeline

# If interrupted, just run again - automatically resumes!
pipeline = Pipeline.resume("my_pdfs/")
results = pipeline.transform("my_pdfs/")
```

## Configuration

### Environment Variables

```bash
export GEMINI_API_KEY="your-api-key"
```

### Or in Code

```python
from structify import Config

Config.set(
    gemini_api_key="your-api-key",
    pages_per_chunk=10,
    temperature=0.1,
    max_retries=5
)
```

### Or from .env File

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
Automatically detect extractable fields:

```python
from structify import SchemaDetector

detector = SchemaDetector(sample_ratio=0.1, max_samples=30)
schema = detector.fit_transform("documents/")
print(schema.fields)
```

### LLMExtractor
Extract data using a schema:

```python
from structify import LLMExtractor, Schema

extractor = LLMExtractor(schema=my_schema, deduplicate=True)
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

## Requirements

- Python 3.10+
- Google Gemini API key

## Dependencies

- google-generativeai
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
