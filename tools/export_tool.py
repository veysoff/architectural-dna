"""Export tool for DNA bank data export functionality."""

import time

from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from export_utils import sanitize_export_path
from exporters import ExporterFactory, ExportResult

from .base import BaseTool
from .pattern_tool import PatternTool


class ExportTool(BaseTool):
    """Tool for exporting DNA bank data in various formats.

    Coordinates export operations with pattern filtering, Qdrant scrolling,
    and format-specific handlers through the ExporterFactory.

    CRITICAL IMPLEMENTATION DETAILS:
    - Qdrant scroll() returns (records, next_offset) where next_offset is an opaque cursor
    - Must use None for first page, then use cursor returned from previous call
    - Uses Range filter for quality_score threshold
    """

    def export_patterns(
        self,
        output_path: str,
        export_format: str = "json",
        language: str | None = None,
        category: str | None = None,
        min_quality: int = 5,
        limit: int = 1000,
        **export_options,
    ) -> ExportResult:
        """Export patterns from Qdrant with filtering and pagination.

        CRITICAL: Qdrant scroll() returns (records, next_offset) where
        next_offset is an opaque cursor, not a page number.

        Args:
            output_path: Output file path
            format: Export format (json, csv, md)
            language: Filter by language (e.g., 'python', 'java')
            category: Filter by category
            min_quality: Minimum quality score threshold
            limit: Maximum patterns to export
            **export_options: Format-specific options

        Returns:
            ExportResult with export metrics and error details
        """
        start_time = time.time()

        # SECURITY: Validate path to prevent traversal attacks
        # Use strict=False to allow absolute paths (backward compatible with C# audit reporter)
        valid, _, error_msg = sanitize_export_path(output_path, strict=False)
        if not valid:
            return ExportResult(
                success=False,
                output_path=output_path,
                records_exported=0,
                records_failed=0,
                errors=[
                    {
                        "type": "SecurityError",
                        "error": f"Path validation failed: {error_msg}",
                    }
                ],
                warnings=[],
                duration_seconds=time.time() - start_time,
            )

        try:
            # 1. Build Qdrant filter
            conditions = []

            if language:
                conditions.append(
                    FieldCondition(key="language", match=MatchValue(value=language))
                )

            if category:
                conditions.append(
                    FieldCondition(key="category", match=MatchValue(value=category))
                )

            if min_quality > 0:
                conditions.append(
                    FieldCondition(key="quality_score", range=Range(gte=min_quality))
                )

            query_filter = Filter(must=conditions) if conditions else None

            # 2. Fetch patterns from Qdrant with proper scrolling
            patterns = []
            offset = None  # Start with None for first page

            while len(patterns) < limit:
                batch_limit = min(100, limit - len(patterns))

                # scroll() returns tuple: (records, next_offset)
                records, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_limit,
                    offset=offset,  # None for first page, then use cursor
                    scroll_filter=query_filter,
                    with_payload=True,
                    with_vectors=False,
                )

                # No more results
                if not records:
                    break

                # Convert Qdrant points to dicts
                for point in records:
                    pattern_data = {"id": str(point.id), **point.payload}
                    patterns.append(pattern_data)

                # Update offset for next iteration
                offset = next_offset

                # If next_offset is None, we've reached the end
                if offset is None:
                    break

            # 3. Export using factory
            exporter = ExporterFactory.get_exporter(export_format)
            result = exporter.export(patterns, output_path, **export_options)

            return result

        except Exception as e:
            duration = time.time() - start_time
            return ExportResult(
                success=False,
                output_path=output_path,
                records_exported=0,
                records_failed=0,
                errors=[{"type": type(e).__name__, "error": str(e)}],
                warnings=[],
                duration_seconds=duration,
            )

    def export_search_results(
        self,
        query: str,
        output_path: str,
        export_format: str = "json",
        limit: int = 10,
        **export_options,
    ) -> ExportResult:
        """Export search results to file.

        Args:
            query: Search query string
            output_path: Output file path
            format: Export format
            limit: Maximum results to export
            **export_options: Format-specific options

        Returns:
            ExportResult with export metrics
        """
        start_time = time.time()

        # SECURITY: Validate path to prevent traversal attacks
        # Use strict=False to allow absolute paths (backward compatible with C# audit reporter)
        valid, _, error_msg = sanitize_export_path(output_path, strict=False)
        if not valid:
            return ExportResult(
                success=False,
                output_path=output_path,
                records_exported=0,
                records_failed=0,
                errors=[
                    {
                        "type": "SecurityError",
                        "error": f"Path validation failed: {error_msg}",
                    }
                ],
                warnings=[],
                duration_seconds=time.time() - start_time,
            )

        try:
            # Use existing search logic
            pattern_tool = PatternTool(self.client, self.collection_name, self.config)
            results = pattern_tool.search_dna(query=query, limit=limit)

            # Export results
            exporter = ExporterFactory.get_exporter(export_format)
            result = exporter.export(results, output_path, **export_options)

            return result

        except Exception as e:
            duration = time.time() - start_time
            return ExportResult(
                success=False,
                output_path=output_path,
                records_exported=0,
                records_failed=0,
                errors=[{"type": type(e).__name__, "error": str(e)}],
                warnings=[],
                duration_seconds=duration,
            )
