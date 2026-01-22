"""ModelOps contracts - stable interface between infrastructure and science."""

from .version import CONTRACTS_VERSION
from .types import (
    Scalar,
    TrialStatus,
    UniqueParameterSet,
    SeedInfo,
    TrialResult,
    MAX_DIAG_BYTES,
)
from .param_hashing import make_param_id, digest_bytes
from .adaptive import AdaptiveAlgorithm
from .simulation import (
    SimTask,
    TableIPC,
    ReplicateSet,
    AggregationTask,
    AggregationReturn,
)
from .artifacts import TableArtifact, SimReturn, ErrorInfo, INLINE_CAP
from .entrypoint import (
    EntryPointId,
    ENTRYPOINT_GRAMMAR_VERSION,
    EntrypointFormatError,
    format_entrypoint,
    parse_entrypoint,
)
from .errors import ContractViolationError
from .ports import (
    Future,
    SimulationService,
    ExecutionEnvironment,
    BundleRepository,
    WireFunction,
)
from .manifest import BundleManifest
from .registry import (
    ModelEntry,
    TargetEntry,
    TargetSet,
    BundleRegistry,
    discover_model_classes,
    discover_model_outputs,
    discover_target_functions,
    BUNDLE_STORAGE_DIR,
    REGISTRY_FILE,
    REGISTRY_PATH,
)
from .jobs import Job, SimJob, CalibrationJob, TargetSpec
from .study import ParameterSetEntry, SimulationStudy, CalibrationSpec
from .bundle_environment import (
    BundleEnvironment,
    RegistryConfig,
    StorageConfig,
    DEFAULT_ENVIRONMENT,
    ENVIRONMENTS_DIR,
)
from .auth import (
    Credential,
    AuthProvider,
)
from .environment import EnvironmentDigest
from .utils import canonical_task_id, bundle12, aggregation_id

__version__ = CONTRACTS_VERSION

__all__ = [
    # Version
    "CONTRACTS_VERSION",
    # Core task specification (ESSENTIAL - not internal!)
    "SimTask",
    "ReplicateSet",
    "AggregationTask",
    "AggregationReturn",
    "UniqueParameterSet",
    "SeedInfo",
    # Type aliases
    "Scalar",
    "TableIPC",
    # Protocols
    "AdaptiveAlgorithm",
    # Results and status
    "TrialStatus",
    "TrialResult",
    "SimReturn",
    "ErrorInfo",
    "TableArtifact",
    "INLINE_CAP",
    "MAX_DIAG_BYTES",
    # Entrypoint utilities
    "EntryPointId",
    "ENTRYPOINT_GRAMMAR_VERSION",
    "EntrypointFormatError",
    "format_entrypoint",
    "parse_entrypoint",
    # Parameter utilities
    "make_param_id",
    "digest_bytes",
    # Errors
    "ContractViolationError",
    # Ports (for hexagonal architecture)
    "Future",
    "SimulationService",
    "ExecutionEnvironment",
    "BundleRepository",
    "WireFunction",
    # Manifest and registry types
    "BundleManifest",
    "ModelEntry",
    "TargetEntry",
    "TargetSet",
    "BundleRegistry",
    "discover_model_classes",
    "discover_model_outputs",
    "discover_target_functions",
    "BUNDLE_STORAGE_DIR",
    "REGISTRY_FILE",
    "REGISTRY_PATH",
    # Job types (discriminated union)
    "Job",
    "SimJob",
    "CalibrationJob",
    "TargetSpec",
    # Study types (parameter-space exploration)
    "ParameterSetEntry",
    "SimulationStudy",
    "CalibrationSpec",
    # Bundle environment configuration
    "BundleEnvironment",
    "RegistryConfig",
    "StorageConfig",
    "DEFAULT_ENVIRONMENT",
    "ENVIRONMENTS_DIR",
    # Authentication protocol
    "Credential",
    "AuthProvider",
    # Environment tracking
    "EnvironmentDigest",
    # Utility functions
    "canonical_task_id",
    "bundle12",
    "aggregation_id",
]
