"""Advanced C# Semantic Analysis for Architectural Intelligence.

This module provides deep architectural analysis including:
- Dependency Injection mapping
- Attribute-based categorization
- Complexity metrics (LCOM, Instability)
- Architectural rule enforcement
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from csharp_constants import CSHARP_CONSTANTS, SQL_LIBRARIES
from csharp_pattern_detector import CSharpPatternDetector

logger = logging.getLogger(__name__)


class ArchitecturalRole(StrEnum):
    """Architectural roles derived from attributes and patterns."""

    CONTROLLER = "controller"
    SERVICE = "service"
    REPOSITORY = "repository"
    DOMAIN_ENTITY = "domain_entity"
    VALUE_OBJECT = "value_object"
    HANDLER = "handler"
    VALIDATOR = "validator"
    MIDDLEWARE = "middleware"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class CSharpAttribute:
    """Represents a C# attribute."""

    name: str
    arguments: list[str] = field(default_factory=list)
    location: str | None = None


@dataclass
class CSharpMember:
    """Represents a class member (field, property, method)."""

    name: str
    member_type: str  # "field", "property", "method"
    return_type: str | None = None
    accessed_members: set[str] = field(default_factory=set)
    is_static: bool = False


@dataclass
class DIRegistration:
    """Represents a dependency injection registration."""

    interface_type: str
    implementation_type: str
    lifetime: str  # "Transient", "Scoped", "Singleton"
    location: str


@dataclass
class CSharpTypeInfo:
    """Extended type information for architectural analysis."""

    name: str
    namespace: str
    file_path: str
    type_kind: str  # "class", "interface", "struct", "record", "enum"

    # Architectural metadata
    attributes: list[CSharpAttribute] = field(default_factory=list)
    architectural_role: ArchitecturalRole = ArchitecturalRole.UNKNOWN

    # Dependencies
    dependencies: set[str] = field(default_factory=set)  # Types this depends on
    dependents: set[str] = field(default_factory=set)  # Types that depend on this

    # Members for cohesion analysis
    members: list[CSharpMember] = field(default_factory=list)

    # Metrics
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    lcom_score: float = 0.0

    # Partial class tracking
    is_partial: bool = False
    partial_locations: list[str] = field(default_factory=list)

    # Async safety violations (line_number, message)
    async_violations: list[tuple[int, str]] = field(default_factory=list)

    # Detected design patterns
    design_patterns: list[dict] = field(
        default_factory=list
    )  # List of {pattern, confidence, indicators}


@dataclass
class ArchitecturalViolation:
    """Represents an architectural rule violation."""

    rule_id: str
    severity: str  # "error", "warning", "info"
    message: str
    type_name: str
    file_path: str
    line_number: int | None = None
    suggestion: str | None = None


class MetricsConfig(BaseModel):
    """Validated configuration for code metrics thresholds."""

    lcom_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="LCOM threshold for God Object detection (0.0-1.0)",
    )
    loc_threshold: int = Field(
        default=500,
        gt=0,
        le=10000,
        description="Lines of code threshold for large classes",
    )
    cyclomatic_complexity_limit: int = Field(
        default=15, gt=0, le=100, description="Maximum cyclomatic complexity per method"
    )


class DependenciesConfig(BaseModel):
    """Validated configuration for dependency limits."""

    max_per_class: int = Field(
        default=7, gt=0, le=50, description="Maximum dependencies per class"
    )
    max_per_namespace: int = Field(
        default=50,
        gt=0,
        le=500,
        description="Maximum external dependencies per namespace",
    )


class PatternsConfig(BaseModel):
    """Validated configuration for pattern detection."""

    include_partial_classes: bool = Field(
        default=True, description="Aggregate partial class declarations"
    )
    extract_di_registrations: bool = Field(
        default=True, description="Extract DI registrations from Program.cs/Startup.cs"
    )
    detect_async_patterns: bool = Field(
        default=True, description="Detect async-over-sync anti-patterns"
    )
    detect_design_patterns: bool = Field(
        default=True, description="Detect design patterns (Singleton, Factory, etc.)"
    )


class CSharpAuditConfig(BaseModel):
    """Validated configuration for C# architectural audit."""

    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    dependencies: DependenciesConfig = Field(default_factory=DependenciesConfig)
    patterns: PatternsConfig = Field(default_factory=PatternsConfig)


class CSharpSemanticAnalyzer:
    """Advanced semantic analyzer for C# architectural patterns."""

    # Attribute patterns for role detection
    ROLE_ATTRIBUTES = {
        ArchitecturalRole.CONTROLLER: [
            r"ApiController",
            r"Controller",
            r"RouteAttribute",
        ],
        ArchitecturalRole.SERVICE: [r"Service", r"Injectable", r"Transient", r"Scoped"],
        ArchitecturalRole.REPOSITORY: [r"Repository", r"DataAccess"],
        ArchitecturalRole.DOMAIN_ENTITY: [r"Entity", r"DomainEntity", r"Aggregate"],
        ArchitecturalRole.VALUE_OBJECT: [r"ValueObject", r"Immutable"],
        ArchitecturalRole.HANDLER: [
            r"Handler",
            r"RequestHandler",
            r"CommandHandler",
            r"QueryHandler",
        ],
        ArchitecturalRole.VALIDATOR: [r"Validator", r"FluentValidation"],
        ArchitecturalRole.MIDDLEWARE: [r"Middleware"],
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.types: dict[str, CSharpTypeInfo] = {}
        self.di_registrations: list[DIRegistration] = []
        self.config = self._load_config(config_path)
        self.pattern_detector = CSharpPatternDetector()

    def _load_config(self, config_path: str) -> dict:
        """Load and validate C# configuration from YAML file.

        Uses Pydantic for validation to ensure all values are within safe ranges.
        Falls back to defaults if config is invalid or missing.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Validated configuration dictionary
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.debug(f"Config file not found: {config_path}. Using defaults.")
                return CSharpAuditConfig().model_dump()

            with open(config_file) as f:
                raw_config = yaml.safe_load(f)
                if raw_config is None:
                    raw_config = {}
                csharp_config = raw_config.get("csharp_audit", {})

                # Validate configuration using Pydantic
                validated_config = CSharpAuditConfig(**csharp_config)
                logger.info(f"Loaded config from {config_path}")
                return validated_config.model_dump()

        except PermissionError:
            logger.error(
                f"Permission denied reading config at {config_path}. "
                f"Check file permissions. Using defaults."
            )
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {config_path}: {e}. Using defaults.")
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}", exc_info=True)

        # Return validated defaults if config loading fails
        return CSharpAuditConfig().model_dump()

    def extract_attributes(
        self, content: str, start_line: int
    ) -> list[CSharpAttribute]:
        """Extract C# attributes from code."""
        attributes = []
        lines = content.split("\n")
        # Match attributes like [Name] or [Name(...)] - note parentheses can contain nested brackets
        attr_pattern = re.compile(r"^\s*\[(\w+)(?:\(([^)]*)\))?\]")

        search_start = max(
            0, start_line - CSHARP_CONSTANTS.ATTRIBUTE_SEARCH_LINES_BEFORE
        )
        search_end = min(
            len(lines), start_line + CSHARP_CONSTANTS.ATTRIBUTE_SEARCH_LINES_AFTER
        )

        for i in range(search_start, search_end):
            match = attr_pattern.match(lines[i])
            if match:
                attr_name = match.group(1)
                attr_args = match.group(2).split(",") if match.group(2) else []
                attributes.append(
                    CSharpAttribute(
                        name=attr_name,
                        arguments=[arg.strip() for arg in attr_args],
                        location=f"line {i + 1}",
                    )
                )

        return attributes

    def determine_architectural_role(
        self, attributes: list[CSharpAttribute], type_name: str, base_types: list[str]
    ) -> ArchitecturalRole:
        """Determine architectural role based on attributes and conventions."""
        # Check attributes first
        for attr in attributes:
            for role, patterns in self.ROLE_ATTRIBUTES.items():
                if any(
                    re.search(pattern, attr.name, re.IGNORECASE) for pattern in patterns
                ):
                    return role

        # Check naming conventions using mapping
        naming_patterns = {
            ArchitecturalRole.CONTROLLER: "Controller",
            ArchitecturalRole.SERVICE: "Service",
            ArchitecturalRole.REPOSITORY: "Repository",
            ArchitecturalRole.HANDLER: "Handler",
            ArchitecturalRole.VALIDATOR: "Validator",
        }

        for role, suffix in naming_patterns.items():
            if type_name.endswith(suffix):
                return role

        # Check base types/interfaces
        base_type_patterns = {
            ArchitecturalRole.CONTROLLER: "Controller",
            ArchitecturalRole.REPOSITORY: "Repository",
        }

        for base_type in base_types:
            for role, pattern in base_type_patterns.items():
                if pattern in base_type:
                    return role

        return ArchitecturalRole.UNKNOWN

    def extract_di_registrations(
        self, program_cs_content: str, file_path: str
    ) -> list[DIRegistration]:
        """Extract dependency injection registrations from Program.cs or Startup.cs."""
        registrations = []
        patterns = [
            r"Add(Transient|Scoped|Singleton)<(\w+),\s*(\w+)>\(",
            r"Add(Transient|Scoped|Singleton)<(\w+)>\([^)]*new\s+(\w+)",
            r"Add(Transient|Scoped|Singleton)\(typeof\((\w+)\),\s*typeof\((\w+)\)",
        ]

        for i, line in enumerate(program_cs_content.split("\n"), 1):
            for pattern in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    lifetime = match.group(1)
                    interface_type = match.group(2)
                    impl_type = match.group(3)

                    registrations.append(
                        DIRegistration(
                            interface_type=interface_type,
                            implementation_type=impl_type,
                            lifetime=lifetime,
                            location=f"{file_path}:{i}",
                        )
                    )

        self.di_registrations.extend(registrations)
        return registrations

    def extract_dependencies(self, content: str) -> set[str]:
        """Extract type dependencies from C# code."""
        dependencies = set()

        field_pattern = re.compile(
            r"(?:private|public|protected|internal)\s+(?:readonly\s+)?(\w+(?:<\w+>)?)\s+\w+"
        )
        for match in field_pattern.finditer(content):
            dep_type = re.sub(r"<.*?>", "", match.group(1))
            if dep_type and dep_type[0].isupper():
                dependencies.add(dep_type)

        method_pattern = re.compile(
            r"(?:public|private|protected|internal)\s+(?:async\s+)?(?:Task<)?(\w+)>?\s+\w+\([^)]*\)"
        )
        for match in method_pattern.finditer(content):
            return_type = match.group(1)
            if (
                return_type
                and return_type[0].isupper()
                and return_type not in ["Task", "void"]
            ):
                dependencies.add(return_type)

        # Detect SQL and data access library usage
        using_pattern = re.compile(r"using\s+([\w.]+);")
        for match in using_pattern.finditer(content):
            namespace = match.group(1)
            # Check against comprehensive list of SQL libraries
            if any(sql_lib in namespace for sql_lib in SQL_LIBRARIES):
                dependencies.add(f"__SQL__{namespace}")

        return dependencies

    def extract_members(self, content: str) -> list[CSharpMember]:
        """Extract class members for cohesion analysis."""
        members = []

        field_pattern = re.compile(
            r"(?:private|public|protected|internal)\s+(?:readonly\s+)?(?:static\s+)?(\w+)\s+(\w+)\s*[;=]",
            re.MULTILINE,
        )
        for match in field_pattern.finditer(content):
            members.append(
                CSharpMember(
                    name=match.group(2),
                    member_type="field",
                    return_type=match.group(1),
                    is_static="static" in match.group(0),
                )
            )

        property_pattern = re.compile(
            r"(?:public|private|protected|internal)\s+(?:static\s+)?(\w+)\s+(\w+)\s*\{",
            re.MULTILINE,
        )
        for match in property_pattern.finditer(content):
            members.append(
                CSharpMember(
                    name=match.group(2),
                    member_type="property",
                    return_type=match.group(1),
                    is_static="static" in match.group(0),
                )
            )

        method_pattern = re.compile(
            r"(?:public|private|protected|internal)\s+(?:static\s+)?(?:async\s+)?(?:Task<)?(\w+)>?\s+(\w+)\s*\(",
            re.MULTILINE,
        )
        for match in method_pattern.finditer(content):
            members.append(
                CSharpMember(
                    name=match.group(2),
                    member_type="method",
                    return_type=match.group(1),
                    is_static="static" in match.group(0),
                )
            )

        return members

    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize field name by removing common prefixes."""
        normalized = field_name.lstrip("_")
        return normalized

    def _field_accessed_in_region(self, field_name: str, region: str) -> bool:
        """Check if field is accessed in code region with normalization."""
        normalized_field = self._normalize_field_name(field_name)

        patterns = [
            rf"\b{re.escape(field_name)}\b",
            rf"\bthis\.{re.escape(field_name)}\b",
            rf"\b{re.escape(normalized_field)}\b",
            rf"\bthis\.{re.escape(normalized_field)}\b",
        ]

        return any(re.search(pattern, region) for pattern in patterns)

    def calculate_lcom(self, members: list[CSharpMember], content: str) -> float:
        """Calculate Lack of Cohesion in Methods (LCOM4)."""
        if not content or not content.strip():
            logger.warning("Empty content provided to calculate_lcom")
            return 0.0

        if not members:
            return 0.0

        methods = [m for m in members if m.member_type == "method" and not m.is_static]
        fields = [
            m
            for m in members
            if m.member_type in ("field", "property") and not m.is_static
        ]

        if not methods or not fields:
            return 0.0

        total_accesses = 0
        for method in methods:
            method_pattern = rf"(?:public|private|protected|internal)?\s*(?:async\s+)?(?:Task<?)?\w+>?\s+{re.escape(method.name)}\s*\("
            method_match = re.search(method_pattern, content)

            if method_match:
                method_start = method_match.start()
                method_end = self._find_method_end(content, method_start)
                method_region = content[method_start:method_end]

                for field in fields:
                    if self._field_accessed_in_region(field.name, method_region):
                        total_accesses += 1
                        method.accessed_members.add(field.name)

        max_accesses = len(methods) * len(fields)
        if max_accesses == 0:
            return 0.0

        lcom = 1.0 - (total_accesses / max_accesses)
        return round(lcom, 3)

    def _find_method_end(
        self, content: str, start: int, max_chars: int = 500_000
    ) -> int:
        """Find the end of a C# method by counting braces."""
        from csharp_code_parser import BraceFindMode, CSharpCodeParser

        result = CSharpCodeParser.find_block_end(
            content=content,
            start=start,
            mode=BraceFindMode.WAIT_FOR_OPENING,
            max_iterations=max_chars,
        )

        if not result.success:
            logger.warning(f"Method end search failed: {result.reason}")

        return result.end_position

    def _remove_comments_and_strings_safe(self, content: str) -> str:
        """Remove comments and string literals using state machine."""
        result = []
        state = "CODE"
        i = 0
        length = len(content)

        while i < length:
            char = content[i]
            next_char = content[i + 1] if i + 1 < length else ""

            if state == "CODE":
                if char == "/" and next_char == "/":
                    state = "LINE_COMMENT"
                    i += 2
                    continue
                elif char == "/" and next_char == "*":
                    state = "BLOCK_COMMENT"
                    i += 2
                    continue
                elif char == "$" and next_char == '"':
                    # Interpolated string $"..." — treat as regular string
                    state = "STRING"
                    i += 2
                    continue
                elif (
                    char == "$"
                    and next_char == "@"
                    and i + 2 < length
                    and content[i + 2] == '"'
                ):
                    # Interpolated verbatim string $@"..."
                    state = "VERBATIM_STRING"
                    i += 3
                    continue
                elif (
                    char == "@"
                    and next_char == "$"
                    and i + 2 < length
                    and content[i + 2] == '"'
                ):
                    # Interpolated verbatim string @$"..."
                    state = "VERBATIM_STRING"
                    i += 3
                    continue
                elif char == "@" and next_char == '"':
                    state = "VERBATIM_STRING"
                    i += 2
                    continue
                elif char == '"':
                    state = "STRING"
                    i += 1
                    continue
                elif char == "'":
                    state = "CHAR"
                    i += 1
                    continue
                else:
                    result.append(char)
                    i += 1

            elif state == "LINE_COMMENT":
                if char == "\n":
                    state = "CODE"
                    result.append("\n")
                i += 1

            elif state == "BLOCK_COMMENT":
                if char == "*" and next_char == "/":
                    state = "CODE"
                    i += 2
                else:
                    i += 1

            elif state == "STRING":
                if char == "\\" and next_char:
                    i += 2
                elif char == '"':
                    state = "CODE"
                    i += 1
                else:
                    i += 1

            elif state == "VERBATIM_STRING":
                if char == '"':
                    if next_char == '"':
                        i += 2
                    else:
                        state = "CODE"
                        i += 1
                else:
                    i += 1

            elif state == "CHAR":
                if char == "\\" and next_char:
                    i += 2
                elif char == "'":
                    state = "CODE"
                    i += 1
                else:
                    i += 1

        return "".join(result)

    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity by counting decision points."""
        content_cleaned = self._remove_comments_and_strings_safe(content)

        # Count decision points with word boundaries
        decision_patterns = [
            r"\bif\b",
            r"\belse\s+if\b",  # Count else-if as separate decision
            r"\bwhile\b",
            r"\bfor\b",
            r"\bforeach\b",
            r"\bcase\b",  # Each case is a decision point
            r"\bcatch\b",  # Exception handlers add complexity
            r"&&",  # Logical AND
            r"\|\|",  # Logical OR
            r"\?(?!=)",  # Ternary operator (but not null-coalescing ??)
        ]

        complexity = 1  # Base complexity
        for pattern in decision_patterns:
            complexity += len(re.findall(pattern, content_cleaned))

        return complexity

    def detect_async_over_sync(self, content: str) -> list[tuple[int, str]]:
        """Detect async-over-sync anti-patterns and async best practice violations."""
        violations = []

        # Anti-patterns that block async code
        blocking_patterns = [
            (r"\.Result\b", "Using .Result blocks the thread (async-over-sync)"),
            (r"\.Wait\(\)", "Using .Wait() blocks the thread (async-over-sync)"),
            (r"\.GetAwaiter\(\)\.GetResult\(\)", "Using GetResult() blocks the thread"),
            (
                r"Task\.Run\([^)]*\)\.Wait\(\)",
                "Task.Run().Wait() is async-over-sync anti-pattern",
            ),
            (
                r"Task\.WaitAll\(",
                "Task.WaitAll() blocks the thread, prefer await Task.WhenAll()",
            ),
            (
                r"Task\.WaitAny\(",
                "Task.WaitAny() blocks the thread, prefer await Task.WhenAny()",
            ),
        ]

        # Best practice violations (warnings, not errors)
        best_practice_patterns = [
            (
                r"async\s+void\s+\w+\s*\([^)]*\)",
                "async void should only be used for event handlers",
            ),
        ]

        # CancellationToken check (separate logic for better accuracy)
        cancellation_token_pattern = r"async\s+Task.*\((?![^)]*CancellationToken)"

        for i, line in enumerate(content.split("\n"), 1):
            # Check blocking patterns
            for pattern, message in blocking_patterns:
                if re.search(pattern, line):
                    violations.append((i, message))

            # Check best practices (only for public/internal methods)
            if re.search(r"^\s*(?:public|internal)\s+async", line):
                for pattern, message in best_practice_patterns:
                    if re.search(pattern, line):
                        violations.append((i, f"Best practice: {message}"))

                # Check CancellationToken for async Task methods
                if re.search(cancellation_token_pattern, line):
                    violations.append(
                        (
                            i,
                            "Best practice: Async method should accept CancellationToken parameter",
                        )
                    )

        return violations

    def analyze_type(self, type_info: CSharpTypeInfo, content: str) -> CSharpTypeInfo:
        """Perform deep analysis on a C# type."""
        # Extract attributes from content - find class definition line first
        lines = content.split("\n")
        class_line = 0
        for i, line in enumerate(lines):
            if (
                f"class {type_info.name}" in line
                or f"interface {type_info.name}" in line
            ):
                class_line = i
                break
        type_info.attributes = self.extract_attributes(content, class_line)

        # Extract members
        type_info.members = self.extract_members(content)

        # Calculate LCOM
        type_info.lcom_score = self.calculate_lcom(type_info.members, content)

        # Extract dependencies
        type_info.dependencies = self.extract_dependencies(content)

        # Count lines of code (excluding blanks and comments)
        lines = [
            line.strip()
            for line in content.split("\n")
            if line.strip() and not line.strip().startswith("//")
        ]
        type_info.lines_of_code = len(lines)

        # Calculate cyclomatic complexity with proper regex to avoid false positives
        type_info.cyclomatic_complexity = self._calculate_cyclomatic_complexity(content)

        # Determine architectural role if not already set
        if type_info.architectural_role == ArchitecturalRole.UNKNOWN:
            base_types = [m.return_type for m in type_info.members if m.return_type]
            type_info.architectural_role = self.determine_architectural_role(
                type_info.attributes, type_info.name, base_types
            )

        # Detect design patterns
        if self.config.get("patterns", {}).get("detect_design_patterns", True):
            try:
                patterns = self.pattern_detector.detect_patterns(
                    content, type_info.name
                )

                # Validate and convert patterns
                validated_patterns = []
                for p in patterns:
                    try:
                        # Validate pattern structure
                        if not all(
                            hasattr(p, attr)
                            for attr in [
                                "pattern",
                                "confidence",
                                "indicators",
                                "description",
                            ]
                        ):
                            logger.warning(
                                f"Invalid pattern structure from detector: {p}"
                            )
                            continue

                        # Validate confidence is in valid range
                        confidence = float(p.confidence)
                        if not (0.0 <= confidence <= 1.0):
                            logger.warning(
                                f"Pattern confidence out of range [0-1]: {confidence}"
                            )
                            continue

                        validated_patterns.append(
                            {
                                "pattern": str(p.pattern.value),
                                "confidence": confidence,
                                "indicators": list(p.indicators)
                                if p.indicators
                                else [],
                                "description": str(p.description),
                            }
                        )
                    except (AttributeError, ValueError, TypeError) as pattern_error:
                        logger.warning(
                            f"Failed to process pattern {p}: {pattern_error}"
                        )
                        continue

                type_info.design_patterns = validated_patterns

            except RecursionError as e:
                logger.error(
                    f"CRITICAL: Pattern detection recursion limit in {type_info.name}: {e}"
                )
                raise RuntimeError(
                    "Pattern detection failed: recursion limit exceeded"
                ) from e
            except MemoryError as e:
                logger.error(
                    f"CRITICAL: Pattern detection out of memory for {type_info.name}"
                )
                raise RuntimeError(
                    "Pattern detection failed: insufficient memory"
                ) from e
            except Exception as e:
                logger.error(
                    f"Unexpected error in pattern detection for {type_info.name}: {e}. "
                    f"Design patterns will not be included in analysis.",
                    exc_info=True,
                )
                type_info.design_patterns = []

        # Store in analyzer
        self.types[type_info.name] = type_info

        return type_info

    def calculate_instability(self, namespace: str) -> float:
        """Calculate Instability Index for a namespace (0=stable, 1=unstable)."""
        types_in_namespace = [
            t for t in self.types.values() if t.namespace == namespace
        ]

        if not types_in_namespace:
            return 0.0

        efferent = set()
        for type_info in types_in_namespace:
            efferent.update(type_info.dependencies)

        afferent = set()
        for type_info in types_in_namespace:
            afferent.update(type_info.dependents)

        ce = len(efferent)
        ca = len(afferent)

        if ce + ca == 0:
            return 0.0

        return round(ce / (ce + ca), 3)

    def link_di_registrations(self):
        """Link interfaces to implementations via DI registrations."""
        for reg in self.di_registrations:
            if reg.interface_type in self.types:
                interface = self.types[reg.interface_type]
                if reg.implementation_type in self.types:
                    implementation = self.types[reg.implementation_type]
                    # Create dependency link
                    interface.dependents.add(reg.implementation_type)
                    implementation.dependencies.add(reg.interface_type)

    def aggregate_partial_classes(self):
        """Aggregate data from partial class declarations."""
        partial_groups = defaultdict(list)

        for _type_name, type_info in self.types.items():
            if type_info.is_partial:
                key = f"{type_info.namespace}.{type_info.name}"
                partial_groups[key].append(type_info)

        for _key, partials in partial_groups.items():
            if len(partials) <= 1:
                continue

            base = partials[0]
            base.partial_locations = [p.file_path for p in partials]

            for other in partials[1:]:
                base.members.extend(other.members)
                base.attributes.extend(other.attributes)
                base.dependencies.update(other.dependencies)
                base.lines_of_code += other.lines_of_code
                base.cyclomatic_complexity += other.cyclomatic_complexity
