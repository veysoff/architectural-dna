"""C# Design Pattern Detection and Classification.

Detects common C# design patterns like:
- Singleton
- Factory
- Observer/Pub-Sub
- Strategy
- Decorator
- Repository
- Unit of Work
- CQRS
- Event-Driven Architecture
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from csharp_constants import CSHARP_CONSTANTS


class DesignPattern(StrEnum):
    """Common C# design patterns."""

    SINGLETON = "singleton"
    FACTORY = "factory"
    ABSTRACT_FACTORY = "abstract_factory"
    BUILDER = "builder"
    PROTOTYPE = "prototype"
    ADAPTER = "adapter"
    BRIDGE = "bridge"
    COMPOSITE = "composite"
    DECORATOR = "decorator"
    FACADE = "facade"
    FLYWEIGHT = "flyweight"
    PROXY = "proxy"
    CHAIN_OF_RESPONSIBILITY = "chain_of_responsibility"
    COMMAND = "command"
    INTERPRETER = "interpreter"
    ITERATOR = "iterator"
    MEDIATOR = "mediator"
    MEMENTO = "memento"
    OBSERVER = "observer"
    STATE = "state"
    STRATEGY = "strategy"
    TEMPLATE_METHOD = "template_method"
    VISITOR = "visitor"
    REPOSITORY = "repository"
    UNIT_OF_WORK = "unit_of_work"
    CQRS = "cqrs"
    EVENT_SOURCING = "event_sourcing"
    PUBSUB = "pubsub"


@dataclass
class PatternMatch:
    """Result of pattern detection."""

    pattern: DesignPattern
    confidence: float  # 0-1, higher is more confident
    indicators: list[str]  # What indicators were found
    description: str


class CSharpPatternDetector:
    """Detects design patterns in C# code."""

    def __init__(self):
        """Initialize pattern detector."""
        self.patterns_found: dict[str, list[PatternMatch]] = {}

    def _create_pattern_match(
        self,
        pattern: DesignPattern,
        indicators: list[str],
        max_indicators: int,
        confidence_threshold: float,
        type_name: str,
    ) -> PatternMatch | None:
        """
        Helper method to create a PatternMatch if confidence threshold is met.

        Args:
            pattern: The design pattern detected
            indicators: List of indicator strings found
            max_indicators: Maximum possible indicators for this pattern
            confidence_threshold: Minimum confidence required (0.0-1.0)
            type_name: Name of the type being analyzed

        Returns:
            PatternMatch if threshold met, None otherwise
        """
        if not indicators:
            return None

        confidence = len(indicators) / float(max_indicators)

        if confidence >= confidence_threshold:
            return PatternMatch(
                pattern=pattern,
                confidence=confidence,
                indicators=indicators,
                description=f"{pattern.value.replace('_', ' ').title()} pattern detected in {type_name}",
            )

        return None

    def detect_patterns(self, code: str, type_name: str) -> list[PatternMatch]:
        """
        Detect design patterns in C# code.

        Args:
            code: C# source code
            type_name: Class/interface name

        Returns:
            List of detected patterns with confidence scores
        """
        patterns = []

        # Creational patterns
        patterns.extend(self._detect_singleton(code, type_name))
        patterns.extend(self._detect_factory(code, type_name))
        patterns.extend(self._detect_builder(code, type_name))

        # Structural patterns
        patterns.extend(self._detect_decorator(code, type_name))
        patterns.extend(self._detect_adapter(code, type_name))
        patterns.extend(self._detect_facade(code, type_name))
        patterns.extend(self._detect_proxy(code, type_name))

        # Behavioral patterns
        patterns.extend(self._detect_observer(code, type_name))
        patterns.extend(self._detect_strategy(code, type_name))
        patterns.extend(self._detect_command(code, type_name))
        patterns.extend(self._detect_chain_of_responsibility(code, type_name))
        patterns.extend(self._detect_state(code, type_name))

        # Architectural patterns
        patterns.extend(self._detect_repository(code, type_name))
        patterns.extend(self._detect_unit_of_work(code, type_name))
        patterns.extend(self._detect_cqrs(code, type_name))
        patterns.extend(self._detect_event_sourcing(code, type_name))
        patterns.extend(self._detect_pubsub(code, type_name))

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    # Creational Patterns

    def _detect_singleton(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Singleton pattern."""
        indicators = []

        # Check for static instance
        if re.search(r"static\s+(?:readonly\s+)?.*\s+Instance", code):
            indicators.append("Static Instance property")

        # Check for private constructor
        if re.search(r"private\s+" + type_name + r"\s*\(", code):
            indicators.append("Private constructor")

        # Check for public static property
        if re.search(
            r"public\s+static\s+" + type_name + r"\s+(?:Instance|Current|Default)", code
        ):
            indicators.append("Public static property")

        pattern_match = self._create_pattern_match(
            DesignPattern.SINGLETON,
            indicators,
            CSHARP_CONSTANTS.SINGLETON_INDICATORS,
            CSHARP_CONSTANTS.PATTERN_CONFIDENCE_HIGH,
            type_name,
        )

        return [pattern_match] if pattern_match else []

    def _detect_factory(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Factory pattern."""
        indicators = []

        # Check for static Create method
        if re.search(
            r"public\s+static\s+(?:abstract\s+)?(?:new\s+)?(\w+)\s+Create", code
        ):
            indicators.append("Static Create method")

        # Check for factory method returning interface
        if re.search(r"I\w+\s+Create|I\w+\s+Make|I\w+\s+Build", code):
            indicators.append("Returns interface type")

        # Check for switch on enum/type
        if re.search(
            r"switch\s*\(.*type.*\)|switch\s*\(.*kind.*\)", code, re.IGNORECASE
        ):
            indicators.append("Switch on type/kind")

        pattern_match = self._create_pattern_match(
            DesignPattern.FACTORY,
            indicators,
            CSHARP_CONSTANTS.FACTORY_INDICATORS,
            CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM,
            type_name,
        )

        return [pattern_match] if pattern_match else []

    def _detect_builder(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Builder pattern."""
        indicators = []

        # Check for With* methods
        if re.search(r"public\s+\w+\s+With\w+\s*\(", code):
            indicators.append("With* fluent methods")

        # Check for Build method
        if re.search(r"public\s+\w+\s+Build\s*\(\s*\)", code):
            indicators.append("Build method")

        # Check for method chaining (returns this)
        if re.search(r"return\s+this;", code):
            indicators.append("Returns this for chaining")

        pattern_match = self._create_pattern_match(
            DesignPattern.BUILDER,
            indicators,
            CSHARP_CONSTANTS.BUILDER_INDICATORS,
            CSHARP_CONSTANTS.PATTERN_CONFIDENCE_HIGH,
            type_name,
        )

        return [pattern_match] if pattern_match else []

    # Structural Patterns

    def _detect_decorator(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Decorator pattern."""
        matches = []
        indicators = []

        # Check for composition of interface type (field holding interface)
        if re.search(r"private\s+(?:readonly\s+)?I\w+\s+\w+\s*;", code):
            indicators.append("Wraps interface type")

        # Check for delegating methods (method calling wrapped object's method)
        if re.search(
            r"(?:public|protected)\s+(?:async\s+)?(?:Task\s*<\w+>|\w+)\s+\w+\s*\([^)]*\)\s*\{[^}]*\w+\.\w+\(",
            code,
            re.DOTALL,
        ):
            indicators.append("Delegates to wrapped object")

        # Check for decorator-like constructor (takes interface parameter)
        if re.search(r"public\s+" + type_name + r"\s*\(\s*I\w+\s+", code):
            indicators.append("Takes interface in constructor")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.DECORATOR_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.DECORATOR,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Decorator pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_adapter(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Adapter pattern."""
        matches = []
        indicators = []

        # Check for implementing multiple interfaces
        if re.search(r":\s*\w+\s*,\s*\w+", code):
            indicators.append("Implements multiple interfaces")

        # Check for wrapping incompatible class
        if re.search(r"private\s+(?:readonly\s+)?\w+\s+\w+;", code):
            indicators.append("Wraps incompatible type")

        # Check for adapter-like name
        if "Adapter" in type_name or "Wrapper" in type_name:
            indicators.append("Adapter/Wrapper in name")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.ADAPTER_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_LOW:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.ADAPTER,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Adapter pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_facade(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Facade pattern."""
        matches = []
        indicators = []

        # Check for many private dependencies
        private_deps = len(re.findall(r"private\s+(?:readonly\s+)?\w+\s+\w+;", code))
        if private_deps >= 3:
            indicators.append(f"Multiple dependencies ({private_deps})")

        # Check for simple public methods
        simple_methods = len(
            re.findall(
                r"public\s+(?:async\s+)?(?:Task<)?[\w\[\]]+\s+\w+\s*\([^)]*\)\s*{[^}]{0,100}?}",
                code,
            )
        )
        if simple_methods >= 2:
            indicators.append("Simple public interface")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.FACADE_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_HIGH:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.FACADE,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Facade pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_proxy(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Proxy pattern."""
        matches = []
        indicators = []

        # Check for same interface wrapping
        if re.search(
            r"private\s+(?:readonly\s+)?I\w+\s+\w+;.*public\s+class\s+\w+\s*:\s*I\w+",
            code,
            re.DOTALL,
        ):
            indicators.append("Implements same interface as wrapped object")

        # Check for access control or lazy loading
        if re.search(
            r"if\s*\([^)]*\w+\s*==\s*null\)|lock\s*\(|IsAuthorized|Permission",
            code,
            re.IGNORECASE,
        ):
            indicators.append("Access control or lazy loading")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.PROXY_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.PROXY,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Proxy pattern detected in {type_name}",
                )
            )

        return matches

    # Behavioral Patterns

    def _detect_observer(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Observer pattern."""
        matches = []
        indicators = []

        # Check for events or delegates (including generic EventHandler)
        if re.search(r"event\s+[\w<>,\s]+\s+\w+\s*;|EventHandler\s*<", code):
            indicators.append("Event definition")

        # Check for event raising (?.Invoke pattern or OnChanged/RaiseEvent calls)
        if re.search(
            r"\w+\?\s*\.Invoke\(|OnChanged\s*\(|RaiseEvent\s*\(|PropertyChanged\?\s*\.Invoke\(",
            code,
        ):
            indicators.append("Event raising")

        # Check for IObserver or IObservable
        if re.search(r"IObserver|IObservable|INotifyPropertyChanged", code):
            indicators.append("Standard observer interface")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.OBSERVER_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.OBSERVER,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Observer pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_strategy(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Strategy pattern."""
        matches = []
        indicators = []

        # Check for interface dependency
        if re.search(r"private\s+(?:readonly\s+)?I\w+Strategy\s+", code):
            indicators.append("Strategy interface")

        # Check for executing strategy
        if re.search(r"\w+Strategy\.\w+\(|\w+\.Execute\(", code):
            indicators.append("Executes strategy")

        # Check for strategy switching
        if re.search(r"Strategy\s*=|SetStrategy|ChangeStrategy", code):
            indicators.append("Strategy assignment")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.STRATEGY_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.STRATEGY,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Strategy pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_command(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Command pattern."""
        matches = []
        indicators = []

        # Check for Execute method
        if re.search(r"public\s+(?:async\s+)?(?:Task<)?[\w\[\]]*\s+Execute\s*\(", code):
            indicators.append("Execute method")

        # Check for Undo/Redo
        if re.search(
            r"(?:public|protected)\s+(?:async\s+)?(?:Task<)?[\w\[\]]*\s+Undo\s*\(|Redo\s*\(",
            code,
        ):
            indicators.append("Undo/Redo support")

        # Check for command queue or history
        if re.search(
            r"Queue.*Command|List.*Command|command.*history", code, re.IGNORECASE
        ):
            indicators.append("Command queue/history")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.COMMAND_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.COMMAND,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Command pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_chain_of_responsibility(
        self, code: str, type_name: str
    ) -> list[PatternMatch]:
        """Detect Chain of Responsibility pattern."""
        matches = []
        indicators = []

        # Check for next handler
        if re.search(
            r"private\s+(?:readonly\s+)?\w+\s+\w*[Nn]ext|_successor|_next", code
        ):
            indicators.append("Next handler reference")

        # Check for Handle method
        if re.search(
            r"public\s+(?:abstract\s+)?(?:async\s+)?(?:Task<)?[\w\[\]]*\s+Handle\s*\(",
            code,
        ):
            indicators.append("Handle method")

        # Check for passing to next
        if re.search(r"\w+[Nn]ext\.\w+\(|\w+_successor\.\w+\(", code):
            indicators.append("Delegates to next handler")

        confidence = len(indicators) / float(
            CSHARP_CONSTANTS.CHAIN_OF_RESPONSIBILITY_INDICATORS
        )

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.CHAIN_OF_RESPONSIBILITY,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Chain of Responsibility pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_state(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect State pattern."""
        matches = []
        indicators = []

        # Check for state interface
        if re.search(r"IState|I\w+State\s+", code):
            indicators.append("State interface")

        # Check for state assignment
        if re.search(r"_state\s*=|CurrentState\s*=|SetState\(", code):
            indicators.append("State assignment")

        # Check for behavior delegation to state
        if re.search(r"_state\.\w+\(|CurrentState\.\w+\(", code):
            indicators.append("Delegates to state")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.STATE_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.STATE,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"State pattern detected in {type_name}",
                )
            )

        return matches

    # Architectural Patterns

    def _detect_repository(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Repository pattern."""
        matches = []
        indicators = []

        # Check for repository methods
        if re.search(
            r"public\s+.*Get\w+\s*\(|FindBy\w+\s*\(|Add\s*\(|Remove\s*\(|Update\s*\(",
            code,
        ):
            indicators.append("CRUD methods")

        # Check for data access interface
        if re.search(r"IRepository|IDataAccess|DbContext|DbSet", code):
            indicators.append("Data access pattern")

        # Repository in name
        if "Repository" in type_name or "DAO" in type_name:
            indicators.append("Repository/DAO in name")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.REPOSITORY_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_HIGH:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.REPOSITORY,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Repository pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_unit_of_work(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Unit of Work pattern."""
        matches = []
        indicators = []

        # Check for multiple repositories (properties or fields)
        if re.search(r"I\w+Repository\s+[\w\s{};]+I\w+Repository", code, re.DOTALL):
            indicators.append("Multiple repositories")

        # Check for SaveChanges/Commit (including async variants)
        if re.search(
            r"(?:SaveChanges|Commit|Complete)\s*(?:Async)?\s*\(",
            code,
        ):
            indicators.append("SaveChanges/Commit method")

        # Check for transaction handling (BeginTransaction, CommitTransaction, etc)
        if re.search(
            r"BeginTransaction|CommitTransaction|RollbackTransaction|RollbackAsync",
            code,
        ):
            indicators.append("Transaction handling")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.UNIT_OF_WORK_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.UNIT_OF_WORK,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Unit of Work pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_cqrs(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect CQRS pattern."""
        matches = []
        indicators = []

        # Check for Command/Query separation
        if re.search(r"ICommand|IQuery|Command\s+class|Query\s+class", code):
            indicators.append("Command/Query separation")

        # Check for handler
        if re.search(r"ICommandHandler|IQueryHandler|Handle\s*\(", code):
            indicators.append("Handler interface")

        # CQRS in name
        if "Command" in type_name and "Query" in type_name:
            indicators.append("CQRS in name")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.CQRS_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_LOW:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.CQRS,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"CQRS pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_event_sourcing(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Event Sourcing pattern."""
        matches = []
        indicators = []

        # Check for event store
        if re.search(r"EventStore|AppendEvent|GetEvents|EventStream", code):
            indicators.append("Event store")

        # Check for event classes
        if re.search(r"class\s+\w+Event|: Event|DomainEvent", code):
            indicators.append("Event classes")

        # Check for event replay
        if re.search(r"Replay|Rebuild|Reconstruct", code):
            indicators.append("Event replay")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.EVENT_SOURCING_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_MEDIUM:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.EVENT_SOURCING,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Event Sourcing pattern detected in {type_name}",
                )
            )

        return matches

    def _detect_pubsub(self, code: str, type_name: str) -> list[PatternMatch]:
        """Detect Pub/Sub pattern."""
        matches = []
        indicators = []

        # Check for publisher/subscriber
        if re.search(r"IPublisher|ISubscriber|Subscribe|Publish", code):
            indicators.append("Pub/Sub interfaces")

        # Check for event broker
        if re.search(r"EventBroker|MessageBroker|EventBus", code):
            indicators.append("Event broker")

        # Check for async events
        if re.search(r"async\s+Task.*Event|await.*Event", code):
            indicators.append("Async event handling")

        confidence = len(indicators) / float(CSHARP_CONSTANTS.PUBSUB_INDICATORS)

        if confidence >= CSHARP_CONSTANTS.PATTERN_CONFIDENCE_LOW:
            matches.append(
                PatternMatch(
                    pattern=DesignPattern.PUBSUB,
                    confidence=confidence,
                    indicators=indicators,
                    description=f"Pub/Sub pattern detected in {type_name}",
                )
            )

        return matches
