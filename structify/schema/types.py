"""Schema types for structify."""

import hashlib
import json
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml
from pydantic import BaseModel, Field as PydanticField


class FieldType(str, Enum):
    """Supported field types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    LIST = "list"
    DATE = "date"


@dataclass
class Field:
    """
    Definition of a single field to extract.

    Attributes:
        name: Field name (will be used as column header)
        type: Field type (string, integer, float, etc.)
        description: Human-readable description for the LLM
        required: Whether this field is required
        default: Default value if not found
        options: List of valid options (for categorical fields)
        pattern: Regex pattern for validation (optional)
        examples: Example values to show the LLM
    """

    name: str
    type: FieldType = FieldType.STRING
    description: str = ""
    required: bool = False
    default: Any = None
    options: list[str] = field(default_factory=list)
    pattern: str | None = None
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert field to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value if isinstance(self.type, FieldType) else self.type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "options": self.options,
            "pattern": self.pattern,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Field":
        """Create field from dictionary."""
        # Handle type conversion
        field_type = data.get("type", "string")
        if isinstance(field_type, str):
            try:
                field_type = FieldType(field_type)
            except ValueError:
                field_type = FieldType.STRING

        return cls(
            name=data["name"],
            type=field_type,
            description=data.get("description", ""),
            required=data.get("required", False),
            default=data.get("default"),
            options=data.get("options", []),
            pattern=data.get("pattern"),
            examples=data.get("examples", []),
        )

    def to_prompt_line(self) -> str:
        """Generate a prompt line for this field."""
        type_str = self.type.value if isinstance(self.type, FieldType) else self.type
        required_str = "Yes" if self.required else "No"

        line = f"| {self.name} | {type_str} | {required_str} | {self.description}"

        if self.options:
            line += f" (one of: {', '.join(self.options)})"

        return line + " |"


@dataclass
class ExtractionRules:
    """Rules for guiding the extraction process."""

    focus_on: list[str] = field(default_factory=list)
    skip: list[str] = field(default_factory=list)
    context: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Schema:
    """
    Complete schema definition for data extraction.

    Attributes:
        name: Schema name
        description: Human-readable description
        version: Schema version
        fields: List of field definitions
        extraction_rules: Rules for guiding extraction
    """

    name: str
    description: str = ""
    version: str = "1.0"
    fields: list[Field] = field(default_factory=list)
    extraction_rules: ExtractionRules = field(default_factory=ExtractionRules)

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "fields": [f.to_dict() for f in self.fields],
            "extraction_rules": {
                "focus_on": self.extraction_rules.focus_on,
                "skip": self.extraction_rules.skip,
                "context": self.extraction_rules.context,
                "examples": self.extraction_rules.examples,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Schema":
        """Create schema from dictionary."""
        fields = [Field.from_dict(f) for f in data.get("fields", [])]

        rules_data = data.get("extraction_rules", {})
        rules = ExtractionRules(
            focus_on=rules_data.get("focus_on", []),
            skip=rules_data.get("skip", []),
            context=rules_data.get("context", ""),
            examples=rules_data.get("examples", []),
        )

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            fields=fields,
            extraction_rules=rules,
        )

    def to_yaml(self) -> str:
        """Convert schema to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Schema":
        """Create schema from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Save schema to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    @classmethod
    def load(cls, path: str | Path) -> "Schema":
        """Load schema from a YAML or JSON file."""
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if path.suffix.lower() == ".json":
            data = json.loads(content)
        else:
            data = yaml.safe_load(content)

        return cls.from_dict(data)

    def get_field(self, name: str) -> Field | None:
        """Get a field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def get_required_fields(self) -> list[Field]:
        """Get all required fields."""
        return [f for f in self.fields if f.required]

    def get_field_names(self) -> list[str]:
        """Get all field names."""
        return [f.name for f in self.fields]

    def compute_hash(self) -> str:
        """Compute a hash of the schema for caching purposes."""
        schema_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()

    def __len__(self) -> int:
        """Get number of fields."""
        return len(self.fields)

    def __iter__(self):
        """Iterate over fields."""
        return iter(self.fields)
