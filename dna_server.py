"""
Architectural DNA MCP Server

An MCP server that:
1. Connects to your GitHub repositories
2. Extracts code patterns using LLM analysis
3. Stores patterns in a Qdrant vector database
4. Enables RAG-powered project scaffolding

Usage:
    python dna_server.py         # stdio mode (local)
    python dna_server.py --sse   # SSE mode (Docker/remote)

Environment variables can be set via:
    - .env file
    - Docker environment
    - MCP client headers (X-GITHUB-TOKEN, X-GEMINI-API-KEY, X-QDRANT-URL)
"""

import logging
import os

# Disable FastMCP banner and logging to prevent stdout pollution
os.environ["FASTMCP_SHOW_CLI_BANNER"] = "false"
os.environ["FASTMCP_LOG_ENABLED"] = "false"

from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastmcp import FastMCP
from qdrant_client import QdrantClient

from embedding_manager import EmbeddingManager
from tools import (
    ExportTool,
    MaintenanceTool,
    PatternTool,
    RepositoryTool,
    ScaffoldTool,
    StatsTool,
)
from tools.batch_processor import BatchProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dna_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (can be overridden by headers)
load_dotenv()


def get_env_or_header(
    key: str, header_key: str, default: str | None = None
) -> str | None:
    """Get value from environment variable, with header override support.

    Priority: Environment variable > Default
    Headers are handled at request time via middleware.
    """
    return os.getenv(key, default)


def load_configuration(config_path: Path) -> dict:
    """Load and validate configuration file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Validated configuration dictionary

    Raises:
        SystemExit: If configuration is missing, invalid, or incomplete
    """
    try:
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create config.yaml in {config_path.parent}\n"
                f"See .env.example and config.yaml.example for reference"
            )

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not cfg:
            raise ValueError("Configuration file is empty")

        # Validate required sections
        required_sections = ["qdrant", "embeddings"]
        for section in required_sections:
            if section not in cfg:
                raise ValueError(f"Missing required config section: [{section}]")

        # Validate required keys in qdrant section
        if "url" not in cfg["qdrant"]:
            raise ValueError(
                "Missing required config: qdrant.url\n"
                "Set via config.yaml or QDRANT_URL environment variable"
            )
        if "collection_name" not in cfg["qdrant"]:
            raise ValueError("Missing required config: qdrant.collection_name")

        logger.info(f"Configuration loaded successfully from {config_path}")
        return cfg

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit(f"FATAL: {e}") from e
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config.yaml: {e}")
        raise SystemExit(
            f"FATAL: Configuration file syntax error: {e}\n"
            f"Please check {config_path} for YAML syntax errors"
        ) from e
    except (KeyError, ValueError) as e:
        logger.error(f"Configuration validation failed: {e}")
        raise SystemExit(f"FATAL: {e}") from e


# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = load_configuration(CONFIG_PATH)

# Initialize MCP server
mcp = FastMCP("Architectural DNA")

# Initialize embedding manager
embedding_manager = EmbeddingManager(config)
logger.info(
    f"Embedding model: {embedding_manager.model} ({embedding_manager.get_vector_size()} dimensions)"
)

# Initialize Qdrant client with configured embeddings
qdrant_url = os.getenv("QDRANT_URL", config["qdrant"]["url"])
client = QdrantClient(url=qdrant_url)
COLLECTION_NAME = config["qdrant"]["collection_name"]

# Set up the embedding model and ensure collection exists
embedding_manager.setup_qdrant_client(client)
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=client.get_fastembed_vector_params(),
    )

# Initialize tool classes
pattern_tool = PatternTool(client, COLLECTION_NAME, config)
scaffold_tool = ScaffoldTool(client, COLLECTION_NAME, config)
stats_tool = StatsTool(client, COLLECTION_NAME, config)
batch_processor = BatchProcessor(client, COLLECTION_NAME, config)

# Initialize repository tool with batch processor for large repos
repository_tool = RepositoryTool(client, COLLECTION_NAME, config, batch_processor)
maintenance_tool = MaintenanceTool(client, COLLECTION_NAME, config)

# Initialize export tool for data export functionality
export_tool = ExportTool(client, COLLECTION_NAME, config)


# ==============================================================================
# MCP Tool Registrations
# ==============================================================================


@mcp.tool()
def store_pattern(
    content: str,
    title: str,
    description: str,
    category: str,
    language: str = "python",
    quality_score: int = 5,
    source_repo: str = "manual",
    source_path: str = "",
    use_cases: list[str] | None = None,
) -> str:
    """
    Stores a high-quality code snippet or architectural pattern in the DNA bank.

    Args:
        content: The code content to store (min 10 chars)
        title: Pattern title (3-200 chars)
        description: A detailed description (min 10 chars)
        category: Pattern category (architecture, error_handling, configuration, etc.)
        language: Programming language (python, java, typescript, go)
        quality_score: Quality rating 1-10 (default: 5)
        source_repo: Source repository name (default: "manual")
        source_path: Source file path (default: "")
        use_cases: List of use case descriptions (default: [])

    Returns:
        Confirmation message
    """
    return pattern_tool.store_pattern(
        content=content,
        title=title,
        description=description,
        category=category,
        language=language,
        quality_score=quality_score,
        source_repo=source_repo,
        source_path=source_path,
        use_cases=use_cases or [],
    )


@mcp.tool()
def search_dna(
    query: str,
    language: str | None = None,
    category: str | None = None,
    min_quality: int = 5,
    limit: int = 10,
) -> str:
    """
    Searches the DNA bank for best practices matching the query.

    Args:
        query: Natural language description (min 3 chars)
        language: Filter by programming language (python, java, typescript, go)
        category: Filter by pattern category (architecture, error_handling, etc.)
        min_quality: Minimum quality score 1-10 (default: 5)
        limit: Maximum results to return 1-100 (default: 10)

    Returns:
        Formatted list of matching patterns with their code
    """
    return pattern_tool.search_dna(
        query=query,
        language=language,
        category=category,
        min_quality=min_quality,
        limit=limit,
    )


@mcp.tool()
def list_my_repos(include_private: bool = True, include_orgs: bool = True) -> str:
    """
    Lists all your GitHub repositories available for DNA extraction.

    Args:
        include_private: Include private repositories
        include_orgs: Include repositories from organizations you belong to

    Returns:
        Formatted list of repositories with their details
    """
    return repository_tool.list_my_repos(
        include_private=include_private, include_orgs=include_orgs
    )


@mcp.tool()
def sync_github_repo(
    repo_name: str, analyze_patterns: bool = True, min_quality: int = 5
) -> str:
    """
    Syncs a GitHub repository into the DNA bank.

    Fetches code from the repository, extracts patterns, optionally analyzes
    them with LLM, and stores high-quality patterns in the vector database.

    Args:
        repo_name: Full repository name (e.g., "username/repo-name")
        analyze_patterns: If True, use LLM to identify and rate patterns
        min_quality: Minimum quality score (1-10) for patterns to store

    Returns:
        Summary of the sync operation
    """
    return repository_tool.sync_github_repo(
        repo_name=repo_name, analyze_patterns=analyze_patterns, min_quality=min_quality
    )


@mcp.tool()
def scaffold_project(
    project_name: str, project_type: str, tech_stack: str, output_dir: str | None = None
) -> str:
    """
    Scaffolds a new project using best practices from the DNA bank.

    Searches for relevant patterns based on the project type and tech stack,
    then generates a complete project structure with files.

    Args:
        project_name: Name for the new project (will be the directory name)
        project_type: Type of project - "api", "cli", "library", or "web-app"
        tech_stack: Comma-separated technologies (e.g., "python, fastapi, postgresql")
        output_dir: Directory to create the project in (defaults to ./generated_projects)

    Returns:
        Path to the created project and summary of what was generated
    """
    return scaffold_tool.scaffold_project(
        project_name=project_name,
        project_type=project_type,
        tech_stack=tech_stack,
        output_dir=output_dir,
    )


@mcp.tool()
def get_dna_stats() -> str:
    """
    Get statistics about the DNA bank.

    Returns:
        Statistics including total patterns, languages, categories, and top sources
    """
    return stats_tool.get_dna_stats()


@mcp.resource("dna://stats")
def get_stats_resource() -> str:
    """
    DNA Bank statistics as a readable resource.

    Access via: dna://stats
    """
    return stats_tool.get_dna_stats()


@mcp.tool()
def get_embedding_info() -> str:
    """
    Get information about the current embedding configuration.

    Returns:
        Embedding model details and configuration
    """
    info = embedding_manager.get_model_info()

    output = "[*] **Embedding Configuration**\n\n"
    output += f"**Provider:** {info['provider']}\n"
    output += f"**Model:** {info['model']}\n"
    output += f"**Vector Size:** {info['vector_size']} dimensions\n"
    output += (
        f"**Chunking:** {'Enabled' if info['chunking_enabled'] else 'Disabled'}\n\n"
    )

    output += "**Preprocessing:**\n"
    for key, value in info["preprocessing"].items():
        output += f"  - {key}: {value}\n"

    output += "\n**Supported Models:**\n"
    for model, dims in EmbeddingManager.SUPPORTED_MODELS.items():
        current = " (current)" if model == info["model"] else ""
        output += f"  - {model} ({dims}d){current}\n"

    return output


@mcp.tool()
def batch_sync_repo(
    repo_name: str,
    batch_size: int | None = None,
    analyze_patterns: bool | None = None,
    min_quality: int | None = None,
    resume: bool = True,
) -> str:
    """
    Sync a large GitHub repository in batches with progress tracking.

    Designed for large repositories that may timeout with regular sync.
    Processes files in configurable batches, tracks progress, and supports
    resuming after interruption.

    Args:
        repo_name: Full repository name (e.g., "username/repo-name")
        batch_size: Files per batch (default: from config.yaml)
        analyze_patterns: Use LLM analysis (default: from config.yaml llm.provider)
        min_quality: Min quality score 1-10 (default: from config.yaml llm.min_quality_score)
        resume: If True, resume from previous progress if available

    Returns:
        Summary of the sync operation with progress details
    """
    # Start with defaults from config, then apply any user overrides
    batch_config = batch_processor._get_default_batch_config()

    if batch_size is not None:
        batch_config.batch_size = batch_size
    if analyze_patterns is not None:
        batch_config.analyze_patterns = analyze_patterns
    if min_quality is not None:
        batch_config.min_quality = min_quality

    return batch_processor.batch_sync_repo(
        repo_name=repo_name, batch_config=batch_config, resume=resume
    )


@mcp.tool()
def get_sync_progress(repo_name: str) -> str:
    """
    Get the current progress for a repository batch sync.

    Use this to check on long-running sync operations or to see
    if there's a resumable sync available.

    Args:
        repo_name: Full repository name (e.g., "username/repo-name")

    Returns:
        Progress details or message if no progress exists
    """
    progress = batch_processor.get_sync_progress(repo_name)
    if not progress:
        return f"No sync progress found for {repo_name}"

    output = f"[*] **Sync Progress for {repo_name}**\n\n"
    output += f"**Files:** {progress['processed_files']}/{progress['total_files']}"
    output += f" ({progress['progress_percent']}%)\n"
    output += f"**Chunks extracted:** {progress['total_chunks']}\n"
    output += f"**Patterns stored:** {progress['stored_patterns']}\n"
    output += f"**Failed files:** {len(progress['failed_files'])}\n"
    output += f"**Current file:** {progress['current_file']}\n"
    output += f"**Elapsed:** {progress['elapsed_seconds']:.1f}s\n"
    output += f"**Est. remaining:** {progress['estimated_remaining_seconds']:.1f}s\n"

    return output


@mcp.tool()
def clear_sync_progress(repo_name: str) -> str:
    """
    Clear saved progress for a repository sync.

    Use this to start a fresh sync instead of resuming.

    Args:
        repo_name: Full repository name (e.g., "username/repo-name")

    Returns:
        Confirmation message
    """
    return batch_processor.clear_sync_progress(repo_name)


@mcp.tool()
def recategorize_patterns(
    from_category: str = "other",
    batch_size: int = 10,
    delay_between_batches: float = 1.0,
    dry_run: bool = False,
) -> str:
    """
    Re-categorize patterns using LLM analysis.

    Use this to fix patterns that were stored without proper categorization
    or to re-analyze patterns with a different/better LLM.

    Args:
        from_category: Which patterns to re-analyze:
            - "other" (default): Only patterns with category 'other'
            - "all": All patterns regardless of current category
            - Any category name: Only patterns with that specific category
            Valid categories: architecture, error_handling, configuration,
            testing, api_design, data_access, security, logging, utilities, other
        batch_size: Number of patterns to process per batch (default: 10)
        delay_between_batches: Seconds between batches for rate limiting (default: 1.0)
        dry_run: If True, only show what would be changed without updating

    Returns:
        Summary of recategorization results
    """
    return maintenance_tool.recategorize_patterns(
        from_category=from_category,
        batch_size=batch_size,
        delay_between_batches=delay_between_batches,
        dry_run=dry_run,
    )


@mcp.tool()
def get_category_stats() -> str:
    """
    Get statistics about pattern categories in the DNA bank.

    Shows distribution of patterns across categories with visual bars.

    Returns:
        Category distribution statistics
    """
    return maintenance_tool.get_category_stats()


@mcp.tool()
def analyze_csharp_project(
    project_path: str,
    repo_name: str = "unknown-repo",
    output_dir: str = "csharp_audit_reports",
) -> str:
    """
    Analyze a C# project for architectural violations and patterns.

    Uses advanced semantic analysis to detect architectural issues, generate
    audit reports in multiple formats (JSON, Markdown, SARIF), and extract
    architectural patterns for the DNA bank.

    Args:
        project_path: Path to C# project root or .csproj file
        repo_name: Repository name for report naming (e.g., "mycompany/myproject")
        output_dir: Directory for generated audit reports (default: csharp_audit_reports)

    Returns:
        Summary of analysis including violation counts and report locations
    """
    try:
        from pathlib import Path

        from csharp_audit_integration import CSharpArchitecturalAuditor
        from csharp_audit_reporter import CSharpAuditReporter

        logger.info(f"Starting C# project analysis: {project_path}")

        auditor = CSharpArchitecturalAuditor()

        # Analyze project
        result = auditor.analyze_csharp_project(project_path)

        # Validate analysis result
        if not result:
            raise ValueError("Project analysis returned no results")

        audit_result = result.get("audit_result")
        types = result.get("types", {})

        if audit_result is None:
            logger.warning(f"No audit result for {project_path}, creating empty result")
            from csharp_audit_engine import AuditResult

            audit_result = AuditResult(
                total_types=0,
                total_violations=0,
                violations_by_severity={},
                violations_by_rule={},
                violations=[],
            )

        if not isinstance(types, dict):
            logger.error(f"Invalid types in result: {type(types)}, expected dict")
            types = {}

        # Generate reports
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = Path(output_dir) / f"{repo_name}_audit.json"
        CSharpAuditReporter.generate_json_report(audit_result, str(json_path), types)

        # Markdown report
        md_path = Path(output_dir) / f"{repo_name}_audit.md"
        CSharpAuditReporter.generate_markdown_report(
            audit_result, types, str(md_path), auditor.audit_engine.config
        )

        # SARIF report (for IDE integration)
        sarif_path = Path(output_dir) / f"{repo_name}_audit.sarif"
        CSharpAuditReporter.generate_sarif_report(audit_result, str(sarif_path))

        # Build summary
        summary = f"""C# Project Analysis Complete

Summary:
  • Total Types Analyzed: {audit_result.total_types}
  • Total Violations: {audit_result.total_violations}
  • Violations by Severity:"""

        for severity, count in sorted(audit_result.violations_by_severity.items()):
            summary += f"\n    {severity.upper()}: {count}"

        summary += """

Reports Generated:"""
        summary += f"\n  • JSON: `{json_path}`"
        summary += f"\n  • Markdown: `{md_path}`"
        summary += f"\n  • SARIF (IDE): `{sarif_path}`"

        summary += "\n\nTop 5 Rules Violated:"
        for i, (rule_id, count) in enumerate(
            sorted(audit_result.violations_by_rule.items(), key=lambda x: -x[1])[:5], 1
        ):
            summary += f"\n  {i}. {rule_id}: {count} violations"

        logger.info(f"C# analysis completed successfully for {repo_name}")

        return summary

    except ImportError as e:
        logger.warning(f"C# audit module import failed: {e}", exc_info=True)
        return (
            f"Error: C# audit module not available: {e}\n"
            f"This feature requires: pip install -r requirements.txt"
        )
    except Exception as e:
        logger.error(f"C# project analysis failed: {e}", exc_info=True)
        return f"Error analyzing C# project: {str(e)}"


@mcp.tool()
def export_dna(
    output_path: str,
    format: str = "json",
    language: str | None = None,
    category: str | None = None,
    min_quality: int = 5,
    limit: int = 1000,
    include_metadata: bool = True,
) -> str:
    """Export DNA bank patterns to file in specified format.

    Exports patterns from the Qdrant vector database to a file in JSON, CSV, or Markdown format.
    Supports filtering by language, category, and quality score with pagination for large exports.

    Args:
        output_path: Path where to save the exported file
        format: Export format - 'json', 'csv', or 'md'/'markdown'
        language: Filter by programming language (e.g., 'python', 'java', 'csharp')
        category: Filter by pattern category (e.g., 'architecture', 'testing')
        min_quality: Minimum quality score threshold (0-10)
        limit: Maximum number of patterns to export (default: 1000)
        include_metadata: Include metadata section in export (default: True)

    Returns:
        Status message with export details and file location

    Example:
        export_dna(
            output_path="exports/patterns.json",
            format="json",
            language="python",
            min_quality=7,
            limit=100
        )
    """
    try:
        logger.info(
            f"Exporting patterns to {output_path} "
            f"(format={format}, language={language}, category={category}, min_quality={min_quality})"
        )

        result = export_tool.export_patterns(
            output_path=output_path,
            export_format=format,
            language=language,
            category=category,
            min_quality=min_quality,
            limit=limit,
            include_metadata=include_metadata,
        )

        if result.success:
            logger.info(
                f"Export successful: {result.records_exported} patterns "
                f"exported to {result.output_path} "
                f"({result.duration_seconds:.2f}s)"
            )
            return str(result)
        else:
            logger.error(f"Export failed: {result}")
            return str(result)

    except Exception as e:
        logger.error(f"Export operation failed: {e}", exc_info=True)
        return f"Error exporting patterns: {str(e)}"


def apply_header_overrides(headers: dict) -> dict:
    """Apply environment overrides from request headers.

    Supported headers:
        X-GITHUB-TOKEN: Override GITHUB_TOKEN
        X-GEMINI-API-KEY: Override GEMINI_API_KEY
        X-QDRANT-URL: Override QDRANT_URL

    Returns dict of applied overrides for logging.
    """
    overrides = {}

    header_mapping = {
        "x-github-token": "GITHUB_TOKEN",
        "x-gemini-api-key": "GEMINI_API_KEY",
        "x-qdrant-url": "QDRANT_URL",
    }

    for header_name, env_name in header_mapping.items():
        value = headers.get(header_name)
        if value:
            os.environ[env_name] = value
            overrides[env_name] = (
                "***" if "token" in header_name or "key" in header_name else value
            )

    return overrides


if __name__ == "__main__":
    import sys

    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8080"))

    logger.info(
        f"Starting Architectural DNA MCP Server "
        f"(Qdrant: {qdrant_url}, Collection: {COLLECTION_NAME})"
    )

    if transport == "sse" or "--sse" in sys.argv:
        logger.info(f"Running in SSE mode on http://{host}:{port}")
        logger.info("Headers supported: X-GITHUB-TOKEN, X-GEMINI-API-KEY, X-QDRANT-URL")

        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request

        class HeaderAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Extract headers and apply overrides
                headers = {k.lower(): v for k, v in request.headers.items()}
                overrides = apply_header_overrides(headers)
                if overrides:
                    logger.info(f"Applied header overrides: {list(overrides.keys())}")
                return await call_next(request)

        mcp.run(transport="sse", host=host, port=port)
    else:
        logger.info("Running in stdio mode")
        mcp.run()
