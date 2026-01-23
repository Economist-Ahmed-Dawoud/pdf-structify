"""Two-layer prompt generation system for structify."""

from typing import Any

from structify.schema.types import Schema, Field, FieldType


# Layer 1: Base prompt (immutable) - enforces strict JSON output
# Note: Using double braces {{ }} to escape them from .format()
BASE_PROMPT = """You are a data extraction assistant. Your task is to extract structured data from documents.

## CRITICAL OUTPUT REQUIREMENTS - READ CAREFULLY

1. Your ENTIRE response must be ONLY valid JSON - absolutely no exceptions
2. Do NOT include ANY text before or after the JSON
3. Do NOT wrap the JSON in markdown code blocks (no ```)
4. Do NOT include explanations, comments, apologies, or notes
5. Do NOT say "Here is the data" or similar phrases
6. If no data found, return exactly: []
7. Your response MUST start with [ and end with ] - NOTHING ELSE

## INVALID RESPONSES (will cause failure):
- "Here are the results: [...]"
- "```json\\n[...]\\n```"
- "[...] I hope this helps!"
- Any text that is not pure JSON

## VALID RESPONSE FORMAT:
[{{"field1": "value1", "field2": "value2"}}, ...]
[]

---

{extension_prompt}
"""


class PromptGenerator:
    """
    Generate extraction prompts from schemas.

    Implements a two-layer prompt system:
    - Layer 1 (base): Immutable JSON enforcement rules
    - Layer 2 (extension): Schema-specific extraction instructions
    """

    def __init__(
        self,
        schema: Schema,
        context: str | None = None,
        focus_on: list[str] | None = None,
        skip: list[str] | None = None,
        examples: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize the prompt generator.

        Args:
            schema: Schema defining fields to extract
            context: Domain context for extraction
            focus_on: Document sections to focus on
            skip: Document sections to skip
            examples: Example extractions to include in prompt
        """
        self.schema = schema
        self.context = context or schema.extraction_rules.context
        self.focus_on = focus_on or schema.extraction_rules.focus_on
        self.skip = skip or schema.extraction_rules.skip
        self.examples = examples or schema.extraction_rules.examples

    def build(self) -> str:
        """
        Build the complete prompt (base + extension).

        Returns:
            Complete prompt string
        """
        extension = self._build_extension()
        return BASE_PROMPT.format(extension_prompt=extension)

    def _build_extension(self) -> str:
        """Build the extension prompt from schema."""
        parts = []

        # Extraction task header
        parts.append("## EXTRACTION TASK")
        parts.append("")
        parts.append("Extract the following fields from this document:")
        parts.append("")

        # Field table
        parts.append("| Field | Type | Required | Description |")
        parts.append("|-------|------|----------|-------------|")
        for field in self.schema.fields:
            parts.append(field.to_prompt_line())
        parts.append("")

        # Categorical constraints section - enforce exact options
        categorical_fields = [
            f for f in self.schema.fields
            if f.type == FieldType.CATEGORICAL and f.options
        ]
        if categorical_fields:
            parts.append("## CATEGORICAL FIELD CONSTRAINTS")
            parts.append("")
            parts.append("For the following fields, you MUST use EXACTLY one of the listed values.")
            parts.append("Do NOT paraphrase, abbreviate differently, or use synonyms.")
            parts.append("")
            for field in categorical_fields:
                options_str = ", ".join(f'"{opt}"' for opt in field.options)
                parts.append(f"- **{field.name}**: Must be one of [{options_str}]")
            parts.append("")

        # Domain context
        if self.context:
            parts.append("## DOMAIN CONTEXT")
            parts.append(self.context)
            parts.append("")

        # Focus sections
        if self.focus_on:
            parts.append("## FOCUS ON")
            for item in self.focus_on:
                parts.append(f"- {item}")
            parts.append("")

        # Skip sections
        if self.skip:
            parts.append("## SKIP")
            for item in self.skip:
                parts.append(f"- {item}")
            parts.append("")

        # Examples
        if self.examples:
            parts.append("## EXAMPLE OUTPUT")
            import json
            parts.append(json.dumps(self.examples, indent=2))
            parts.append("")

        return "\n".join(parts)

    @property
    def base_prompt(self) -> str:
        """Get the base prompt (without extension)."""
        return BASE_PROMPT.format(extension_prompt="")

    @property
    def extension_prompt(self) -> str:
        """Get the extension prompt only."""
        return self._build_extension()

    def with_document_info(
        self,
        document_name: str,
        part_num: int | None = None,
        total_parts: int | None = None,
    ) -> str:
        """
        Build prompt with document-specific information.

        Args:
            document_name: Name of the document being processed
            part_num: Current part number (for split documents)
            total_parts: Total number of parts

        Returns:
            Complete prompt with document info
        """
        extension = self._build_extension()

        # Add document context
        doc_info = ["\n## DOCUMENT CONTEXT"]
        doc_info.append(f"- Document: {document_name}")
        if part_num is not None and total_parts is not None:
            doc_info.append(f"- Part {part_num} of {total_parts}")
        doc_info.append("")

        extension += "\n".join(doc_info)

        return BASE_PROMPT.format(extension_prompt=extension)


def create_prompt_for_schema(
    schema: Schema,
    document_name: str | None = None,
    part_num: int | None = None,
    total_parts: int | None = None,
) -> str:
    """
    Convenience function to create a prompt from a schema.

    Args:
        schema: Schema defining fields to extract
        document_name: Optional document name
        part_num: Optional part number
        total_parts: Optional total parts

    Returns:
        Complete prompt string
    """
    generator = PromptGenerator(schema)

    if document_name:
        return generator.with_document_info(document_name, part_num, total_parts)

    return generator.build()
