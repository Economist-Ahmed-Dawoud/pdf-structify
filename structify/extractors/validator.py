"""Response validation and JSON repair for extraction results."""

import json
from typing import Any

from structify.schema.types import Schema, Field, FieldType
from structify.core.exceptions import InvalidResponseError, ValidationError
from structify.utils.json_repair import repair_json, strip_markdown
from structify.utils.logging import get_logger

logger = get_logger("validator")


class ResponseValidator:
    """
    Multi-layer validation for LLM responses.

    Implements defense-in-depth validation:
    1. Format check (starts with [)
    2. JSON parsing
    3. JSON repair (if parsing fails)
    4. Schema validation
    """

    def __init__(self, schema: Schema | None = None, strict: bool = False):
        """
        Initialize the validator.

        Args:
            schema: Schema for validation (optional)
            strict: If True, raise errors on validation failures
        """
        self.schema = schema
        self.strict = strict

    def validate(self, response_text: str) -> list[dict[str, Any]]:
        """
        Validate and parse LLM response.

        Args:
            response_text: Raw response text from LLM

        Returns:
            List of validated records

        Raises:
            InvalidResponseError: If response cannot be parsed
            ValidationError: If strict mode and records fail validation
        """
        if not response_text or not response_text.strip():
            return []

        # Layer 1: Strip markdown and whitespace
        text = strip_markdown(response_text.strip())

        # Layer 2: Check format
        if not text.startswith("["):
            logger.debug("Response doesn't start with [, attempting repair")

        # Layer 3: Try JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]
            else:
                records = []
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse failed: {e}, attempting repair")

            # Layer 4: Attempt repair
            records = repair_json(text)

            if not records:
                if self.strict:
                    raise InvalidResponseError(
                        f"Failed to parse response as JSON: {text[:200]}..."
                    )
                logger.warning("Could not parse or repair JSON response")
                return []

            logger.info(f"Repaired JSON: salvaged {len(records)} records")

        # Layer 5: Schema validation
        if self.schema:
            records = self._validate_records(records)

        return records

    def _validate_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Validate records against schema.

        Args:
            records: List of records to validate

        Returns:
            List of valid records
        """
        valid_records = []

        for i, record in enumerate(records):
            try:
                validated = self._validate_record(record)
                valid_records.append(validated)
            except ValidationError as e:
                if self.strict:
                    raise
                logger.debug(f"Record {i} failed validation: {e}")

        return valid_records

    def _validate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Validate a single record against schema.

        Args:
            record: Record to validate

        Returns:
            Validated record with type coercion

        Raises:
            ValidationError: If required field is missing
        """
        validated = {}

        for field in self.schema.fields:
            value = record.get(field.name)

            # Check required fields
            if field.required and value is None:
                raise ValidationError(f"Required field '{field.name}' is missing")

            # Apply default if missing
            if value is None:
                if field.default is not None:
                    value = field.default
                else:
                    continue

            # Type coercion and validation
            try:
                value = self._coerce_type(value, field)
            except (ValueError, TypeError) as e:
                if self.strict:
                    raise ValidationError(
                        f"Field '{field.name}' type error: {e}"
                    )
                logger.debug(f"Type coercion failed for {field.name}: {e}")
                continue

            # Validate categorical options
            if field.type == FieldType.CATEGORICAL and field.options:
                if value not in field.options:
                    if self.strict:
                        raise ValidationError(
                            f"Field '{field.name}' value '{value}' not in options: {field.options}"
                        )
                    logger.debug(f"Invalid categorical value for {field.name}: {value}")

            validated[field.name] = value

        # Include any extra fields not in schema
        for key, value in record.items():
            if key not in validated and self.schema.get_field(key) is None:
                validated[key] = value

        return validated

    def _coerce_type(self, value: Any, field: Field) -> Any:
        """
        Coerce value to expected type.

        Args:
            value: Value to coerce
            field: Field definition

        Returns:
            Coerced value
        """
        if value is None:
            return None

        target_type = field.type

        if target_type == FieldType.STRING:
            return str(value)

        elif target_type == FieldType.INTEGER:
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                # Handle common formats
                value = value.replace(",", "").strip()
                return int(float(value))

        elif target_type == FieldType.FLOAT:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.replace(",", "").strip()
                return float(value)

        elif target_type == FieldType.BOOLEAN:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1", "y")
            return bool(value)

        elif target_type == FieldType.CATEGORICAL:
            return str(value)

        elif target_type == FieldType.LIST:
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                # Try to parse as JSON list
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
                # Split by comma
                return [v.strip() for v in value.split(",")]
            return [value]

        return value

    def validate_batch(
        self,
        responses: list[str],
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Validate multiple responses.

        Args:
            responses: List of response texts

        Returns:
            Tuple of (all valid records, number of failed responses)
        """
        all_records = []
        failures = 0

        for response in responses:
            try:
                records = self.validate(response)
                all_records.extend(records)
            except (InvalidResponseError, ValidationError):
                failures += 1

        return all_records, failures
