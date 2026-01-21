"""JSON repair utilities for handling malformed LLM responses."""

import json
import re
from typing import Any


def strip_markdown(text: str) -> str:
    """
    Remove markdown code blocks from text.

    Args:
        text: Text potentially containing markdown

    Returns:
        Text with markdown removed
    """
    text = text.strip()

    # Remove ```json ... ``` blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    return text.strip()


def extract_json_array(text: str) -> str | None:
    """
    Extract a JSON array from text that may contain extra content.

    Args:
        text: Text containing JSON

    Returns:
        Extracted JSON string or None if not found
    """
    # Find the first [ and last ]
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


def find_complete_objects(json_str: str) -> list[dict[str, Any]]:
    """
    Find all complete JSON objects in a potentially truncated array.

    Args:
        json_str: JSON string that may be truncated

    Returns:
        List of successfully parsed objects
    """
    objects = []
    depth = 0
    current_obj_start = None
    in_string = False
    escape_next = False

    for i, char in enumerate(json_str):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == "{":
            if depth == 0:
                current_obj_start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and current_obj_start is not None:
                obj_str = json_str[current_obj_start : i + 1]
                try:
                    obj = json.loads(obj_str)
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                current_obj_start = None

    return objects


def repair_json(json_str: str) -> list[dict[str, Any]]:
    """
    Attempt to repair truncated or malformed JSON arrays.

    This function tries multiple strategies to salvage data from
    malformed JSON responses.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        List of successfully parsed objects, or empty list if repair fails
    """
    if not json_str or not json_str.strip():
        return []

    json_str = json_str.strip()

    # Strategy 0: Strip markdown formatting
    json_str = strip_markdown(json_str)

    # Strategy 1: Try direct parse
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        return [result] if result else []
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON array from surrounding text
    extracted = extract_json_array(json_str)
    if extracted:
        try:
            result = json.loads(extracted)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find complete objects in truncated array
    if json_str.startswith("[") or "[" in json_str:
        objects = find_complete_objects(json_str)
        if objects:
            return objects

    # Strategy 4: Try to find and parse just the first object
    try:
        start = json_str.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(json_str)):
                if json_str[i] == "{":
                    depth += 1
                elif json_str[i] == "}":
                    depth -= 1
                    if depth == 0:
                        obj = json.loads(json_str[start : i + 1])
                        return [obj]
    except json.JSONDecodeError:
        pass

    # Strategy 5: Handle "Extra data" error by truncating
    try:
        for end_char in ["]", "}"]:
            last_valid = json_str.rfind(end_char)
            while last_valid > 0:
                try:
                    test_str = json_str[: last_valid + 1]
                    if not test_str.endswith("]"):
                        test_str += "]"
                    result = json.loads(test_str)
                    if isinstance(result, list) and result:
                        return result
                except json.JSONDecodeError:
                    pass
                last_valid = json_str.rfind(end_char, 0, last_valid)
    except Exception:
        pass

    return []


def validate_json_structure(data: list[dict[str, Any]], required_fields: list[str]) -> list[dict[str, Any]]:
    """
    Validate that JSON objects have required fields.

    Args:
        data: List of JSON objects
        required_fields: List of required field names

    Returns:
        List of valid objects
    """
    valid = []
    for obj in data:
        if isinstance(obj, dict) and all(field in obj for field in required_fields):
            valid.append(obj)
    return valid
