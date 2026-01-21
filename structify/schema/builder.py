"""Schema builder for manual and LLM-assisted schema definition."""

import json
from typing import Any

from structify.schema.types import Field, Schema, FieldType, ExtractionRules
from structify.providers.gemini import GeminiProvider
from structify.core.exceptions import SchemaError
from structify.utils.logging import get_logger
from structify.utils.json_repair import repair_json

logger = get_logger("schema.builder")


SCHEMA_FROM_DESCRIPTION_PROMPT = """You are a schema designer. Based on the user's description, create a JSON schema for data extraction.

USER DESCRIPTION:
{description}

Create a schema with fields that would capture the information the user wants to extract.

For each field, specify:
- name: A short, snake_case field name
- type: One of "string", "integer", "float", "boolean", "categorical"
- description: Brief description of what this field captures
- required: true if this is essential information, false otherwise
- options: For categorical fields, list the valid options

Also provide:
- schema_name: A short name for this schema
- schema_description: A one-line description
- focus_on: List of document sections to focus on
- skip: List of document sections to skip

RESPOND WITH ONLY VALID JSON in this exact format:
{{
  "schema_name": "...",
  "schema_description": "...",
  "fields": [
    {{
      "name": "field_name",
      "type": "string",
      "description": "...",
      "required": true,
      "options": []
    }}
  ],
  "focus_on": ["..."],
  "skip": ["..."]
}}

Your response must be valid JSON only. No markdown, no explanation."""


class SchemaBuilder:
    """
    Build schemas programmatically or from natural language descriptions.

    Supports three modes:
    1. Programmatic: Define fields directly in code
    2. Natural language: Describe what to extract, LLM builds schema
    3. File-based: Load from YAML/JSON files
    """

    def __init__(self, provider: GeminiProvider | None = None):
        """
        Initialize the schema builder.

        Args:
            provider: LLM provider for natural language schema building
        """
        self.provider = provider

    @staticmethod
    def create(
        name: str,
        fields: list[dict[str, Any] | Field],
        description: str = "",
        version: str = "1.0",
        focus_on: list[str] | None = None,
        skip: list[str] | None = None,
        context: str = "",
    ) -> Schema:
        """
        Create a schema programmatically.

        Args:
            name: Schema name
            fields: List of field definitions (dicts or Field objects)
            description: Schema description
            version: Schema version
            focus_on: Document sections to focus on
            skip: Document sections to skip
            context: Domain context for extraction

        Returns:
            Schema object
        """
        # Convert dicts to Field objects
        processed_fields = []
        for f in fields:
            if isinstance(f, Field):
                processed_fields.append(f)
            elif isinstance(f, dict):
                processed_fields.append(Field.from_dict(f))
            else:
                raise SchemaError(f"Invalid field type: {type(f)}")

        rules = ExtractionRules(
            focus_on=focus_on or [],
            skip=skip or [],
            context=context,
        )

        return Schema(
            name=name,
            description=description,
            version=version,
            fields=processed_fields,
            extraction_rules=rules,
        )

    def from_description(
        self,
        description: str,
        context: str = "",
    ) -> Schema:
        """
        Build a schema from a natural language description.

        Uses an LLM to interpret the description and generate
        appropriate field definitions.

        Args:
            description: Natural language description of what to extract
            context: Additional domain context

        Returns:
            Schema object
        """
        if self.provider is None:
            self.provider = GeminiProvider()
            self.provider.initialize()

        logger.info("Building schema from natural language description...")

        # Build prompt
        prompt = SCHEMA_FROM_DESCRIPTION_PROMPT.format(
            description=description + (f"\n\nContext: {context}" if context else ""),
        )

        # Generate schema
        response = self.provider.generate(prompt)

        # Parse response
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to repair
            repaired = repair_json(response)
            if repaired and len(repaired) == 1:
                data = repaired[0]
            else:
                raise SchemaError(f"Failed to parse schema from LLM response: {response[:500]}")

        # Build schema from response
        fields = []
        for field_data in data.get("fields", []):
            # Handle type conversion
            field_type = field_data.get("type", "string")
            try:
                field_type = FieldType(field_type)
            except ValueError:
                field_type = FieldType.STRING

            fields.append(
                Field(
                    name=field_data["name"],
                    type=field_type,
                    description=field_data.get("description", ""),
                    required=field_data.get("required", False),
                    options=field_data.get("options", []),
                )
            )

        rules = ExtractionRules(
            focus_on=data.get("focus_on", []),
            skip=data.get("skip", []),
            context=context,
        )

        schema = Schema(
            name=data.get("schema_name", "generated_schema"),
            description=data.get("schema_description", description[:100]),
            fields=fields,
            extraction_rules=rules,
        )

        logger.info(f"Generated schema with {len(fields)} fields")
        return schema

    @staticmethod
    def from_file(path: str) -> Schema:
        """
        Load a schema from a YAML or JSON file.

        Args:
            path: Path to the schema file

        Returns:
            Schema object
        """
        return Schema.load(path)

    @staticmethod
    def from_yaml(yaml_str: str) -> Schema:
        """
        Create a schema from a YAML string.

        Args:
            yaml_str: YAML string

        Returns:
            Schema object
        """
        return Schema.from_yaml(yaml_str)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Schema:
        """
        Create a schema from a dictionary.

        Args:
            data: Schema dictionary

        Returns:
            Schema object
        """
        return Schema.from_dict(data)

    @staticmethod
    def merge_schemas(*schemas: Schema, name: str | None = None) -> Schema:
        """
        Merge multiple schemas into one.

        Args:
            *schemas: Schemas to merge
            name: Name for the merged schema

        Returns:
            Merged schema
        """
        if not schemas:
            raise SchemaError("At least one schema is required")

        merged_fields = []
        seen_names = set()

        for schema in schemas:
            for field in schema.fields:
                if field.name not in seen_names:
                    merged_fields.append(field)
                    seen_names.add(field.name)

        # Merge extraction rules
        focus_on = []
        skip = []
        for schema in schemas:
            focus_on.extend(schema.extraction_rules.focus_on)
            skip.extend(schema.extraction_rules.skip)

        rules = ExtractionRules(
            focus_on=list(set(focus_on)),
            skip=list(set(skip)),
        )

        return Schema(
            name=name or f"merged_{'_'.join(s.name for s in schemas)}",
            description=f"Merged schema from {len(schemas)} schemas",
            fields=merged_fields,
            extraction_rules=rules,
        )
