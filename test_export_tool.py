"""Integration tests for ExportTool with mock Qdrant client."""

import json
from unittest.mock import MagicMock

import pytest

from exporters import ExportResult
from tools.export_tool import ExportTool


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = MagicMock()

    # Mock scroll results: return tuple (records, next_offset)
    # Use MagicMock for PointStruct to avoid Pydantic validation issues with vector field
    point1 = MagicMock()
    point1.id = "1"
    point1.payload = {
        "title": "Test Pattern 1",
        "description": "Description 1",
        "language": "python",
        "category": "utilities",
        "content": "def foo(): pass",
        "quality_score": 8,
        "source_repo": "test/repo1",
        "source_path": "foo.py",
    }
    point1.vector = None

    point2 = MagicMock()
    point2.id = "2"
    point2.payload = {
        "title": "Test Pattern 2",
        "description": "Description 2",
        "language": "java",
        "category": "testing",
        "content": "public class Test {}",
        "quality_score": 7,
        "source_repo": "test/repo2",
        "source_path": "Test.java",
    }
    point2.vector = None

    client.scroll.return_value = ([point1, point2], None)

    return client


@pytest.fixture
def export_tool(mock_qdrant_client):
    """Initialize ExportTool with mock Qdrant client."""
    config = {}  # Minimal config for testing
    return ExportTool(mock_qdrant_client, "test_collection", config)


# ============================================================================
# ExportTool Integration Tests
# ============================================================================


class TestExportToolIntegration:
    """Integration tests for ExportTool with mock Qdrant."""

    def test_export_patterns_json(self, export_tool, tmp_path):
        """Test JSON export through ExportTool."""
        output_file = tmp_path / "patterns.json"

        result = export_tool.export_patterns(
            output_path=str(output_file), export_format="json", limit=10
        )

        assert isinstance(result, ExportResult)
        assert result.success
        assert output_file.exists()
        assert result.records_exported == 2

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "data" in data
        assert len(data["data"]) == 2

    def test_export_patterns_csv(self, export_tool, tmp_path):
        """Test CSV export through ExportTool."""
        output_file = tmp_path / "patterns.csv"

        result = export_tool.export_patterns(
            output_path=str(output_file), export_format="csv", limit=10
        )

        assert result.success
        assert output_file.exists()
        assert result.records_exported == 2

        # Verify CSV is readable
        import csv

        with open(output_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

    def test_export_patterns_markdown(self, export_tool, tmp_path):
        """Test Markdown export through ExportTool."""
        output_file = tmp_path / "patterns.md"

        result = export_tool.export_patterns(
            output_path=str(output_file), export_format="md", limit=10
        )

        assert result.success
        assert output_file.exists()
        assert result.records_exported == 2

        # Verify Markdown content
        content = output_file.read_text()
        assert "# DNA Export Report" in content
        assert "Test Pattern 1" in content
        assert "```python" in content

    def test_export_patterns_with_language_filter(self, export_tool, tmp_path):
        """Test exporting with language filter."""
        output_file = tmp_path / "python_patterns.json"

        result = export_tool.export_patterns(
            output_path=str(output_file),
            export_format="json",
            language="python",
            limit=10,
        )

        assert result.success
        # Verify filter was applied to Qdrant call
        export_tool.client.scroll.assert_called()
        call_kwargs = export_tool.client.scroll.call_args.kwargs
        assert call_kwargs["scroll_filter"] is not None

    def test_export_patterns_with_quality_filter(self, export_tool, tmp_path):
        """Test exporting with quality score filter."""
        output_file = tmp_path / "high_quality_patterns.json"

        result = export_tool.export_patterns(
            output_path=str(output_file), export_format="json", min_quality=7, limit=10
        )

        assert result.success
        call_kwargs = export_tool.client.scroll.call_args.kwargs
        assert call_kwargs["scroll_filter"] is not None

    def test_export_patterns_with_multiple_filters(self, export_tool, tmp_path):
        """Test exporting with multiple filters."""
        output_file = tmp_path / "filtered_patterns.json"

        result = export_tool.export_patterns(
            output_path=str(output_file),
            export_format="json",
            language="python",
            category="utilities",
            min_quality=7,
            limit=10,
        )

        assert result.success
        call_kwargs = export_tool.client.scroll.call_args.kwargs
        # Multiple conditions should be in filter
        assert call_kwargs["scroll_filter"] is not None

    def test_export_unsupported_format_raises_error(self, export_tool, tmp_path):
        """Test that unsupported format is handled gracefully."""
        result = export_tool.export_patterns(
            output_path=str(tmp_path / "test.xyz"), export_format="xyz"
        )

        # Should handle error gracefully and return failed result
        assert result.success is False
        assert len(result.errors) > 0
        assert "Unknown export format" in str(result.errors[0])

    def test_export_result_contains_metrics(self, export_tool, tmp_path):
        """Test that ExportResult contains all expected metrics."""
        output_file = tmp_path / "patterns.json"

        result = export_tool.export_patterns(
            output_path=str(output_file), export_format="json"
        )

        # Check all ExportResult fields are populated
        assert hasattr(result, "success")
        assert hasattr(result, "output_path")
        assert hasattr(result, "records_exported")
        assert hasattr(result, "records_failed")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "duration_seconds")

        # Verify values
        assert result.success is True
        assert result.output_path == str(output_file)
        assert result.records_exported == 2
        assert result.records_failed == 0
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert result.duration_seconds > 0

    def test_export_pagination(self, export_tool, tmp_path):
        """Test that ExportTool handles pagination correctly."""

        # Set up mock to return multiple batches
        def mock_scroll(*args, **kwargs):
            offset = kwargs.get("offset")
            if offset is None:
                # First call
                point = MagicMock()
                point.id = "1"
                point.payload = {
                    "title": "Pattern 1",
                    "language": "python",
                    "category": "utilities",
                }
                point.vector = None
                return ([point], "cursor_1")  # Return cursor for next page
            elif offset == "cursor_1":
                # Second call
                point = MagicMock()
                point.id = "2"
                point.payload = {
                    "title": "Pattern 2",
                    "language": "java",
                    "category": "testing",
                }
                point.vector = None
                return ([point], None)  # No more pages
            return ([], None)

        export_tool.client.scroll.side_effect = mock_scroll

        output_file = tmp_path / "paginated.json"

        export_tool.export_patterns(
            output_path=str(output_file), export_format="json", limit=100
        )

        # Should have called scroll twice (once for each page)
        assert export_tool.client.scroll.call_count == 2

    def test_export_with_metadata_option(self, export_tool, tmp_path):
        """Test exporting with metadata option."""
        output_file = tmp_path / "with_metadata.json"

        result = export_tool.export_patterns(
            output_path=str(output_file), export_format="json", include_metadata=True
        )

        assert result.success

        with open(output_file) as f:
            data = json.load(f)

        # Should have metadata section
        assert "metadata" in data
        assert "data" in data

    def test_export_without_metadata_option(self, export_tool, tmp_path):
        """Test exporting without metadata option."""
        # Re-mock client for this test
        point = MagicMock()
        point.id = "1"
        point.payload = {
            "title": "Pattern",
            "language": "python",
            "category": "utilities",
        }
        point.vector = None
        export_tool.client.scroll.return_value = ([point], None)

        output_file = tmp_path / "no_metadata.json"

        result = export_tool.export_patterns(
            output_path=str(output_file), export_format="json", include_metadata=False
        )

        assert result.success

        with open(output_file) as f:
            data = json.load(f)

        # Should be a list, not a dict with metadata
        assert isinstance(data, list)
        assert len(data) == 1

    def test_export_error_handling(self, export_tool, tmp_path):
        """Test error handling in export operation."""
        # Make client.scroll raise an exception
        export_tool.client.scroll.side_effect = RuntimeError("Connection failed")

        output_file = tmp_path / "error_test.json"

        result = export_tool.export_patterns(
            output_path=str(output_file), export_format="json"
        )

        # Should handle error gracefully
        assert result.success is False
        assert len(result.errors) > 0
        assert "RuntimeError" in str(result.errors[0])


class TestExportToolQdrantScrolling:
    """Test Qdrant scroll cursor handling (critical)."""

    def test_scroll_uses_correct_cursor_sequence(self, export_tool):
        """Test that scroll uses correct cursor progression.

        CRITICAL: First call uses offset=None, subsequent calls use next_offset from previous.
        """

        call_count = 0

        def mock_scroll(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            offset = kwargs.get("offset")

            if call_count == 1:
                # First call should have offset=None
                assert offset is None
                point = MagicMock()
                point.id = "1"
                point.payload = {
                    "title": "P1",
                    "language": "python",
                    "category": "utilities",
                }
                point.vector = None
                return ([point], "cursor_abc")
            elif call_count == 2:
                # Second call should use returned cursor
                assert offset == "cursor_abc"
                point = MagicMock()
                point.id = "2"
                point.payload = {
                    "title": "P2",
                    "language": "java",
                    "category": "testing",
                }
                point.vector = None
                return ([point], None)
            return ([], None)

        export_tool.client.scroll.side_effect = mock_scroll

        result = export_tool.export_patterns(
            output_path="/tmp/test.json", export_format="json", limit=100
        )

        assert result.success
        assert result.records_exported == 2
        assert call_count == 2


class TestPathTraversalSecurity:
    """Security tests for path traversal vulnerability prevention."""

    def test_path_traversal_with_double_dot_rejected(self, export_tool, tmp_path):
        """Test that path traversal with .. is rejected."""
        result = export_tool.export_patterns(
            output_path="../../../etc/passwd", export_format="json"
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "cannot contain" in str(result.errors[0]).lower() or ".." in str(
            result.errors[0]
        )

    def test_path_with_null_byte_rejected(self, export_tool):
        """Test that paths with null bytes are rejected."""
        result = export_tool.export_patterns(
            output_path="test\0malicious.json", export_format="json"
        )

        assert result.success is False
        assert len(result.errors) > 0

    def test_relative_path_allowed(self, export_tool, tmp_path):
        """Test that relative paths without traversal are allowed."""
        # Change to tmp_path so relative paths work
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = export_tool.export_patterns(
                output_path="safe_export.json", export_format="json"
            )
            # Should succeed or at least not fail for path security reasons
            if result.success is False:
                # Check it's not a path security issue
                assert "Output path" not in str(result.errors[0])
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
