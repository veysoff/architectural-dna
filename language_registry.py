"""Language registry for dynamic tree-sitter parser management.

Provides lazy-loading and caching of tree-sitter parsers for multiple languages.
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import Language

logger = logging.getLogger(__name__)


class ExtractionMethod(StrEnum):
    """Method used for code extraction."""

    AST = "ast"  # Tree-sitter AST parsing
    REGEX = "regex"  # Enhanced regex patterns
    PSEUDO_AST = "pseudo_ast"  # Advanced structural regex
    SEMANTIC = "semantic"  # Fallback line-based chunking


@dataclass
class LanguageConfig:
    """Configuration for a supported language.

    Attributes:
        language: Language enum value
        parser_module: Tree-sitter module name (e.g., "tree_sitter_python")
        type_declarations: Map of AST node types to chunk types
        query_strategy: Strategy for AST traversal ("recursive", "s-expression", "query")
        node_type_docs: Documentation of AST node types (optional)
    """

    language: Language  # Resolved via __future__ annotations
    parser_module: str
    type_declarations: dict[str, str] = field(default_factory=dict)
    query_strategy: str = "recursive"
    node_type_docs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if not self.language:
            raise ValueError("language is required")
        if not self.parser_module:
            raise ValueError("parser_module is required")
        if self.query_strategy not in ("recursive", "s-expression", "query"):
            raise ValueError(
                f"Invalid query_strategy: {self.query_strategy}. "
                "Must be one of: recursive, s-expression, query"
            )


class LanguageRegistry:
    """Central registry for language parsers and extraction strategies.

    Features:
    - Lazy loading: Parsers loaded on first use
    - Caching: Loaded parsers cached for reuse
    - Graceful degradation: Fallback to regex if AST unavailable
    - Extensible: Add languages via registration or config
    """

    def __init__(self):
        """Initialize the registry with empty caches."""
        self._parsers: dict = {}  # Cache for loaded parsers
        self._languages: dict = {}  # Cache for language objects
        self._configs: dict = {}  # Cache for language configs
        self._initialized: set = set()  # Track initialization attempts
        self._lock = threading.Lock()  # Guard lazy-load critical section

        # Check tree-sitter availability globally
        self._tree_sitter_available = self._check_tree_sitter()

        # Register default languages
        self._register_defaults()

    def _check_tree_sitter(self) -> bool:
        """Check if tree-sitter is available.

        Returns:
            True if tree-sitter core is available
        """
        try:
            from tree_sitter import Parser  # noqa: F401

            return True
        except ImportError:
            logger.warning("tree-sitter not available, using regex extraction")
            return False

    def _register_defaults(self) -> None:
        """Register default language configurations."""
        # Import here to avoid circular dependency
        from models import Language

        # Python
        self.register(
            LanguageConfig(
                language=Language.PYTHON,
                parser_module="tree_sitter_python",
                type_declarations={
                    "class_definition": "class",
                    "function_definition": "function",
                    "decorated_definition": "function",
                },
                query_strategy="recursive",
                node_type_docs={
                    "class_definition": "Top-level class: class Name:",
                    "function_definition": "Function or method: def name():",
                    "decorated_definition": "@decorator wrapper around function",
                },
            )
        )

        # Java
        self.register(
            LanguageConfig(
                language=Language.JAVA,
                parser_module="tree_sitter_java",
                type_declarations={
                    "class_declaration": "class",
                    "interface_declaration": "interface",
                    "enum_declaration": "enum",
                    "record_declaration": "record",
                    "annotation_type_declaration": "annotation",
                },
                query_strategy="recursive",
                node_type_docs={
                    "class_declaration": "class Foo { }",
                    "interface_declaration": "interface Bar { }",
                    "enum_declaration": "enum Baz { }",
                },
            )
        )

        # JavaScript
        self.register(
            LanguageConfig(
                language=Language.JAVASCRIPT,
                parser_module="tree_sitter_javascript",
                type_declarations={
                    "class_declaration": "class",
                    "function_declaration": "function",
                    "method_definition": "method",
                    "arrow_function": "function",
                    "lexical_declaration": "variable",
                },
                query_strategy="recursive",
                node_type_docs={
                    "class_declaration": "class Foo { }",
                    "function_declaration": "function bar() { }",
                    "arrow_function": "const fn = () => { }",
                },
            )
        )

        # TypeScript (shares parser with JavaScript)
        self.register(
            LanguageConfig(
                language=Language.TYPESCRIPT,
                parser_module="tree_sitter_javascript",
                type_declarations={
                    "class_declaration": "class",
                    "interface_declaration": "interface",
                    "type_alias_declaration": "type",
                    "function_declaration": "function",
                    "method_definition": "method",
                },
                query_strategy="recursive",
                node_type_docs={
                    "class_declaration": "class Foo { }",
                    "interface_declaration": "interface Bar { }",
                    "type_alias_declaration": "type T = string",
                },
            )
        )

        # C#
        self.register(
            LanguageConfig(
                language=Language.CSHARP,
                parser_module="tree_sitter_c_sharp",
                type_declarations={
                    "class_declaration": "class",
                    "interface_declaration": "interface",
                    "struct_declaration": "struct",
                    "record_declaration": "record",
                    "enum_declaration": "enum",
                },
                query_strategy="recursive",
                node_type_docs={
                    "class_declaration": "class Foo { }",
                    "interface_declaration": "interface IBar { }",
                    "struct_declaration": "struct Baz { }",
                },
            )
        )

        # Go (uses pseudo-AST extraction, future AST enhancement)
        self.register(
            LanguageConfig(
                language=Language.GO,
                parser_module="tree_sitter_go",  # Not currently used, reserved for future
                type_declarations={
                    "type_spec": "type",
                    "function_declaration": "function",
                    "struct_type": "struct",
                    "interface_type": "interface",
                },
                query_strategy="recursive",
                node_type_docs={
                    "type_spec": "type Reader struct { }",
                    "function_declaration": "func main() { }",
                    "struct_type": "type Person struct { Name string }",
                    "interface_type": "type Writer interface { Write() }",
                },
            )
        )

    def register(self, config: LanguageConfig) -> None:
        """Register a language configuration.

        Args:
            config: Language configuration

        Raises:
            ValueError: If config is invalid
        """
        if not config:
            raise ValueError("config cannot be None")

        self._configs[config.language] = config
        logger.debug(f"Registered language: {config.language.value}")

    def get_parser(self, language):
        """Get or load parser for a language.

        Implements lazy loading with caching. Returns None if tree-sitter
        unavailable or language not registered.

        IMPORTANT: This method does NOT retry failed parser initialization.
        If tree-sitter fails to load on first call (ImportError, missing module),
        subsequent calls will return None without attempting to reload. This is
        intentional to avoid repeated failures in long-lived processes.

        For dynamic environments where tree-sitter may become available later,
        consider creating a fresh LanguageRegistry instance.

        Args:
            language: Target language (Language enum)

        Returns:
            Parser instance or None if unavailable (or failed to initialize)
        """
        # Fast path: return cached result without lock (safe after initialization)
        if language in self._initialized:
            return self._parsers.get(language)

        # Slow path: acquire lock and double-check to prevent concurrent initialization
        with self._lock:
            if language in self._initialized:
                return self._parsers.get(language)

            # Mark as initialized (even if fails, don't retry to avoid repeated errors)
            self._initialized.add(language)

        # Check if tree-sitter available
        if not self._tree_sitter_available:
            self._parsers[language] = None
            return None

        # Check if language registered
        if language not in self._configs:
            logger.debug(f"Language {language.value} not registered")
            self._parsers[language] = None
            return None

        # Load parser
        config = self._configs[language]
        try:
            from tree_sitter import Parser

            # Dynamically import language module
            module = importlib.import_module(config.parser_module)
            language_obj = module.language()

            # Create parser
            parser = Parser()
            parser.set_language(language_obj)

            # Cache
            self._parsers[language] = parser
            self._languages[language] = language_obj

            logger.info(f"Loaded tree-sitter parser for {language.value}")
            return parser

        except ImportError as e:
            logger.warning(
                f"Tree-sitter {language.value} not available: {e}. "
                "Falling back to regex extraction."
            )
            self._parsers[language] = None
            return None
        except Exception as e:
            logger.error(f"Failed to load parser for {language.value}: {e}")
            self._parsers[language] = None
            return None

    def get_config(self, language) -> LanguageConfig | None:
        """Get language configuration.

        Args:
            language: Target language (Language enum)

        Returns:
            Language config or None if not found
        """
        return self._configs.get(language)

    def supports_ast(self, language) -> bool:
        """Check if AST extraction is available for a language.

        Args:
            language: Target language (Language enum)

        Returns:
            True if AST parsing available
        """
        parser = self.get_parser(language)
        return parser is not None

    def get_supported_languages(self) -> list:
        """Get list of languages with AST support.

        Returns:
            List of supported languages (Language enums)
        """
        supported = []
        for lang in self._configs:
            if self.supports_ast(lang):
                supported.append(lang)
        return supported
