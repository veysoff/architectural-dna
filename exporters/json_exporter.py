"""JSON export format implementation."""

import json
import time
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .base import BaseExporter, ExportResult


class JsonExporter(BaseExporter):
    """JSON format exporter with comprehensive serialization handling.

    CRITICAL FEATURES:
    - Proper Enum serialization (recursive field iteration, not asdict())
    - Set serialization with deterministic output (sorted)
    - Pydantic model support (model_dump)
    - Custom to_metadata/to_dict method support
    - Atomic file writes (temp → rename)
    - Per-item error handling with graceful degradation
    """

    def export(self, data: Any, output_path: str, **options) -> ExportResult:
        """Export data to JSON file.

        Args:
            data: Data to export (dataclass, Pydantic model, list, dict)
            output_path: Output file path
            **options: Format options:
                - indent: JSON indent (default: 2)
                - include_metadata: Add metadata section (default: True)
                - skip_on_error: Skip failed items instead of failing (default: False)

        Returns:
            ExportResult with export metrics and error details
        """
        start_time = time.time()
        indent = options.get("indent", 2)
        include_metadata = options.get("include_metadata", True)
        skip_on_error = options.get("skip_on_error", False)

        errors = []
        warnings = []
        serialized_items = []

        try:
            # 1. Pre-flight validation: Check output path writability
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Test writability
            test_file = output_file.with_suffix(".tmp_test")
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError as e:
                duration = time.time() - start_time
                return ExportResult(
                    success=False,
                    output_path=output_path,
                    records_exported=0,
                    records_failed=0,
                    errors=[{"type": "PermissionError", "error": str(e)}],
                    warnings=[],
                    duration_seconds=duration,
                )

            # 2. Serialize data with per-item error handling
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    try:
                        serialized_items.append(self._serialize(item))
                    except Exception as e:
                        errors.append(
                            {"index": idx, "type": type(item).__name__, "error": str(e)}
                        )
                        if skip_on_error:
                            warnings.append(f"Skipped item {idx}: {str(e)}")
                        else:
                            raise
            else:
                try:
                    serialized_items = self._serialize(data)
                except Exception as e:
                    duration = time.time() - start_time
                    return ExportResult(
                        success=False,
                        output_path=output_path,
                        records_exported=0,
                        records_failed=1,
                        errors=[{"type": type(data).__name__, "error": str(e)}],
                        warnings=[],
                        duration_seconds=duration,
                    )

            # Count records: list → length, single item → 1
            records_count = (
                len(serialized_items) if isinstance(serialized_items, list) else 1
            )

            # 3. Prepare output structure
            if include_metadata:
                output_data = {
                    "metadata": {
                        "exported_at": datetime.now().isoformat(),
                        "format": "json",
                        "version": "1.0",
                    },
                    "data": serialized_items,
                    "export_stats": {
                        "total": len(data) if isinstance(data, list) else 1,
                        "exported": len(serialized_items),
                        "failed": len(errors),
                    },
                }
            else:
                output_data = serialized_items

            # 4. Atomic write: write to temp, then rename
            temp_file = output_file.with_suffix(".tmp")
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(
                        output_data, f, indent=indent, default=self._json_serializer
                    )

                # Atomic rename (prevents partial file corruption)
                temp_file.replace(output_file)

            except Exception as e:
                if temp_file.exists():
                    temp_file.unlink()
                raise e

            duration = time.time() - start_time

            return ExportResult(
                success=len(errors) == 0,
                output_path=str(output_file),
                records_exported=records_count,
                records_failed=len(errors),
                errors=errors,
                warnings=warnings,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ExportResult(
                success=False,
                output_path=output_path,
                records_exported=len(serialized_items),
                records_failed=len(errors) + 1,
                errors=errors + [{"type": "WriteError", "error": str(e)}],
                warnings=warnings,
                duration_seconds=duration,
            )

    def _serialize(self, obj: Any) -> Any:
        """Recursive serialization with proper Enum/set handling.

        CRITICAL: dataclasses.asdict() does NOT recursively serialize Enums.
        This method manually iterates dataclass fields and recursively serializes
        to handle nested Enums and other special types.

        Args:
            obj: Object to serialize

        Returns:
            Serializable representation
        """
        # Handle dataclasses FIRST (before checking for attributes)
        if is_dataclass(obj) and not isinstance(obj, type):
            # Check for custom serialization methods FIRST
            if hasattr(obj, "to_metadata"):
                return obj.to_metadata()  # Pattern has this!
            elif hasattr(obj, "to_dict"):
                return obj.to_dict()  # BatchProgress has this!

            # Manual field iteration to recursively handle nested Enums
            result = {}
            for field in fields(obj):
                value = getattr(obj, field.name)
                result[field.name] = self._serialize(value)
            return result

        # Pydantic models
        elif hasattr(obj, "model_dump"):
            return obj.model_dump(mode="python")

        # Collections
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize(item) for item in obj]
        elif isinstance(obj, set):
            # Sort sets for deterministic output; key=str handles heterogeneous types
            return sorted([self._serialize(item) for item in obj], key=str)

        # Primitives and special types
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)

        # Fallback: convert to string
        else:
            return str(obj)

    def _json_serializer(self, obj: Any) -> Any:
        """Handle special types in JSON default handler.

        Used by json.dump() for objects that aren't JSON serializable.
        This is a fallback for objects missed in _serialize().
        """
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return sorted((self._serialize(item) for item in obj), key=str)
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def supports_format(self, format_name: str) -> bool:
        """Check if this exporter handles JSON format."""
        return format_name.lower() == "json"

    def get_file_extension(self) -> str:
        """Get JSON file extension."""
        return ".json"

    def verify(self, output_path: str) -> tuple[bool, str]:
        """Verify exported JSON file is valid.

        Args:
            output_path: Path to JSON file

        Returns:
            (success, message): True if valid JSON
        """
        try:
            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)

            # Check for expected structure
            if isinstance(data, dict):
                if "data" in data and "metadata" in data:
                    record_count = len(data.get("data", []))
                    return True, f"Valid JSON with {record_count} records"
                else:
                    return True, f"Valid JSON ({type(data).__name__})"
            else:
                return True, f"Valid JSON ({type(data).__name__})"

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except FileNotFoundError:
            return False, "File not found"
        except Exception as e:
            return False, f"Verification failed: {e}"
