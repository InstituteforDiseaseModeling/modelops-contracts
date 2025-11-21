"""Model and target registry for provenance tracking.

This module provides the registry system for tracking models and their
dependencies. The registry is the foundation of the provenance system,
allowing explicit declaration of what files affect model behavior.

This is the contract interface - implementations may vary between
modelops-bundle (for authoring) and modelops (for consumption).
"""

import ast
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Protocol, runtime_checkable
import yaml
from pydantic import BaseModel as PydanticBaseModel, Field, model_validator


# Standard bundle storage location constants
BUNDLE_STORAGE_DIR = ".modelops-bundle"
REGISTRY_FILE = "registry.yaml"
REGISTRY_PATH = f"{BUNDLE_STORAGE_DIR}/{REGISTRY_FILE}"


class ModelEntry(PydanticBaseModel):
    """Unified registry entry for a model - discovery, dependencies, and tracking.

    Combines model capabilities (what it can do) with dependencies (what it needs)
    and digest tracking for invalidation detection.

    Attributes:
        entrypoint: Full entrypoint like "models.sir:StochasticSIR"
        path: Path to the Python file containing the model
        class_name: Name of the model class
        scenarios: Available scenarios/configurations for this model
        parameters: Parameter names this model accepts
        outputs: List of output names this model produces
        data: List of data file dependencies
        data_digests: Mapping of data file paths to their digests
        code: List of code file dependencies
        code_digests: Mapping of code file paths to their digests
        model_digest: Hash of the model file itself
    """
    # Identification
    entrypoint: str  # Primary identifier like "models.sir:StochasticSIR"
    path: Path
    class_name: str

    # Capabilities (from old manifest.ModelEntry)
    scenarios: List[str] = Field(default_factory=list)
    parameters: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)

    # Dependencies with digest tracking
    data: List[Path] = Field(default_factory=list)
    data_digests: Dict[str, str] = Field(default_factory=dict)  # path -> digest

    code: List[Path] = Field(default_factory=list)
    code_digests: Dict[str, str] = Field(default_factory=dict)  # path -> digest

    # Model's own digest
    model_digest: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def compute_digest(self, base_path: Optional[Path] = None) -> Optional[str]:
        """Compute and store the digest of the model file.

        Args:
            base_path: Base directory for resolving relative paths

        Returns:
            The computed digest in format "sha256:xxxx" or None if file doesn't exist
        """
        import hashlib
        base = base_path or Path.cwd()
        model_file = base / self.path if not self.path.is_absolute() else self.path

        if not model_file.exists():
            return None

        sha256 = hashlib.sha256()
        with model_file.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        digest = f"sha256:{sha256.hexdigest()}"
        self.model_digest = digest
        return digest

    def compute_dependency_digests(self, base_path: Optional[Path] = None) -> None:
        """Compute and store digests for all dependencies.

        Updates data_digests and code_digests dictionaries with current file digests.

        Args:
            base_path: Base directory for resolving relative paths
        """
        import hashlib
        base = base_path or Path.cwd()

        def compute_file_digest(file_path: Path) -> str:
            """Compute SHA256 digest with prefix."""
            sha256 = hashlib.sha256()
            with file_path.open('rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return f"sha256:{sha256.hexdigest()}"

        # Compute data file digests
        for data_file in self.data:
            abs_path = base / data_file if not data_file.is_absolute() else data_file
            if abs_path.exists():
                # Store with relative path as key
                path_key = str(data_file)
                self.data_digests[path_key] = compute_file_digest(abs_path)

        # Compute code file digests
        for code_file in self.code:
            abs_path = base / code_file if not code_file.is_absolute() else code_file
            if abs_path.exists():
                # Store with relative path as key
                path_key = str(code_file)
                self.code_digests[path_key] = compute_file_digest(abs_path)

    def check_invalidation(self, base_path: Optional[Path] = None) -> List[str]:
        """Check what changed since digests were computed.

        This compares stored digests against current files to identify changes.

        Args:
            base_path: Base directory for resolving relative paths

        Returns:
            List of human-readable change descriptions
        """
        import hashlib
        changes = []
        base = base_path or Path.cwd()

        def compute_file_digest(file_path: Path) -> str:
            """Simple SHA256 hash of file contents."""
            sha256 = hashlib.sha256()
            with file_path.open('rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return f"sha256:{sha256.hexdigest()}"

        # Check model file itself
        if self.path and self.model_digest:
            model_file = base / self.path
            if model_file.exists():
                current = compute_file_digest(model_file)
                if current != self.model_digest:
                    changes.append(f"MODEL {self.path}: content changed")
            else:
                changes.append(f"MODEL {self.path}: file missing")

        # Check data dependencies
        for data_file in self.data:
            stored_digest = self.data_digests.get(str(data_file))
            if stored_digest:
                abs_path = base / data_file
                if abs_path.exists():
                    current = compute_file_digest(abs_path)
                    if current != stored_digest:
                        changes.append(f"DATA {data_file}: content changed")
                else:
                    changes.append(f"DATA {data_file}: file missing")
            else:
                changes.append(f"DATA {data_file}: no digest stored")

        # Check code dependencies
        for code_file in self.code:
            stored_digest = self.code_digests.get(str(code_file))
            if stored_digest:
                abs_path = base / code_file
                if abs_path.exists():
                    current = compute_file_digest(abs_path)
                    if current != stored_digest:
                        changes.append(f"CODE {code_file}: content changed")
                else:
                    changes.append(f"CODE {code_file}: file missing")
            else:
                changes.append(f"CODE {code_file}: no digest stored")

        return changes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "entrypoint": self.entrypoint,
            "path": str(self.path),
            "class_name": self.class_name,
            "scenarios": self.scenarios,
            "parameters": self.parameters,
            "outputs": self.outputs,
            "data": [str(p) for p in self.data],
            "data_digests": self.data_digests,
            "code": [str(p) for p in self.code],
            "code_digests": self.code_digests,
            "model_digest": self.model_digest
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        """Create from dictionary (YAML deserialization)."""
        return cls(
            entrypoint=data["entrypoint"],
            path=Path(data["path"]),
            class_name=data["class_name"],
            scenarios=data.get("scenarios", []),
            parameters=data.get("parameters", []),
            outputs=data.get("outputs", []),
            data=[Path(p) for p in data.get("data", [])],
            data_digests=data.get("data_digests", {}),
            code=[Path(p) for p in data.get("code", [])],
            code_digests=data.get("code_digests", {}),
            model_digest=data.get("model_digest")
        )


class TargetEntry(PydanticBaseModel):
    """Registry entry for a calibration target.

    Attributes:
        path: Path to the Python file containing the target
        entrypoint: Module path and function name (e.g., "targets.prevalence:prevalence_target")
        model_output: Name of the model output to calibrate against
        data: List of data file dependencies
        target_digest: Token-based hash of the target file
    """
    path: Path
    entrypoint: str
    model_output: str
    data: List[Path] = []
    target_digest: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def compute_digest(self, base_path: Optional[Path] = None) -> Optional[str]:
        """Compute and store the digest of the target file.

        Args:
            base_path: Base directory for resolving relative paths

        Returns:
            The computed digest in format "sha256:xxxx" or None if file doesn't exist
        """
        import hashlib
        base = base_path or Path.cwd()
        target_file = base / self.path if not self.path.is_absolute() else self.path

        if not target_file.exists():
            return None

        sha256 = hashlib.sha256()
        with target_file.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        digest = f"sha256:{sha256.hexdigest()}"
        self.target_digest = digest
        return digest

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "path": str(self.path),
            "entrypoint": self.entrypoint,
            "model_output": self.model_output,
            "data": [str(p) for p in self.data],
            "target_digest": self.target_digest
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetEntry":
        """Create from dictionary (YAML deserialization)."""
        return cls(
            path=Path(data["path"]),
            entrypoint=data["entrypoint"],
            model_output=data["model_output"],
            data=[Path(p) for p in data.get("data", [])],
            target_digest=data.get("target_digest")
        )


class TargetSet(PydanticBaseModel):
    """Logical grouping of targets with optional weights."""

    targets: List[str] = Field(default_factory=list)
    weights: Dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "targets": list(self.targets),
            "weights": dict(self.weights),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetSet":
        return cls(
            targets=list(data.get("targets", [])),
            weights=dict(data.get("weights", {})),
        )


@runtime_checkable
class RegistryReader(Protocol):
    """Protocol for reading model registry information.

    This is the minimal interface that modelops needs to read
    registry information from bundles.
    """

    @property
    def models(self) -> Dict[str, ModelEntry]:
        """Get all registered models."""
        ...

    @property
    def targets(self) -> Dict[str, TargetEntry]:
        """Get all registered targets."""
        ...

    def get_all_dependencies(self) -> List[Path]:
        """Get all files referenced in the registry."""
        ...


class BundleRegistry(PydanticBaseModel):
    """Registry of models and targets for provenance tracking.

    This is the base implementation that both modelops-bundle
    and modelops can use. Extended functionality (like compute_digest)
    should be added in the implementing package.
    """
    version: str = "1.0"
    models: Dict[str, ModelEntry] = Field(default_factory=dict)
    targets: Dict[str, TargetEntry] = Field(default_factory=dict)
    target_sets: Dict[str, TargetSet] = Field(default_factory=dict)

    def add_model(
        self,
        model_id: str,
        path: Path,
        class_name: str,
        outputs: List[str] = None,
        data: List[Path] = None,
        code: List[Path] = None
    ) -> ModelEntry:
        """Add a model to the registry.

        Automatically generates entrypoint from path and class_name.
        """
        # Generate entrypoint from path
        # Convert path like "src/models/sir.py" to "models.sir:ClassName"
        if path.suffix == '.py':
            # Remove .py extension and convert path to module notation
            module_path = str(path.with_suffix(''))
            # Remove common prefixes like 'src/' if present
            if module_path.startswith('src/'):
                module_path = module_path[4:]
            # Convert / to .
            module_path = module_path.replace('/', '.')
            entrypoint = f"{module_path}:{class_name}"
        else:
            # Fallback for non-Python files
            entrypoint = f"{path.stem}:{class_name}"

        entry = ModelEntry(
            entrypoint=entrypoint,
            path=path,
            class_name=class_name,
            outputs=outputs or [],
            data=data or [],
            code=code or []
        )
        self.models[model_id] = entry
        return entry

    def add_target(
        self,
        target_id: str,
        path: Path,
        entrypoint: str,
        model_output: str,
        data: List[Path] = None
    ) -> TargetEntry:
        """Add a target to the registry."""
        entry = TargetEntry(
            path=path,
            entrypoint=entrypoint,
            model_output=model_output,
            data=data or []
        )
        self.targets[target_id] = entry
        return entry

    def set_target_set(
        self,
        name: str,
        targets: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> TargetSet:
        """Create or replace a named target set."""
        unknown = [tid for tid in targets if tid not in self.targets]
        if unknown:
            raise ValueError(f"Target set '{name}' references unknown targets: {', '.join(unknown)}")

        normalized_weights: Dict[str, float] = {}
        if weights:
            for tid, weight in weights.items():
                if tid not in targets:
                    continue
                normalized_weights[tid] = float(weight)

        target_set = TargetSet(targets=list(dict.fromkeys(targets)), weights=normalized_weights)
        self.target_sets[name] = target_set
        return target_set

    def delete_target_set(self, name: str) -> bool:
        """Delete a target set if it exists."""
        return self.target_sets.pop(name, None) is not None

    def remove_target_from_sets(self, target_id: str) -> None:
        """Remove target references from all sets."""
        for target_set in self.target_sets.values():
            if target_id in target_set.targets:
                target_set.targets = [tid for tid in target_set.targets if tid != target_id]
            if target_id in target_set.weights:
                target_set.weights.pop(target_id, None)

    def validate(self) -> List[str]:
        """Validate registry entries."""
        errors = []

        for model_id, model in self.models.items():
            if not model.path.exists():
                errors.append(f"Model {model_id}: file not found at {model.path}")
            for data_file in model.data:
                if not data_file.exists():
                    errors.append(f"Model {model_id}: data dependency not found at {data_file}")
            for code_file in model.code:
                if not code_file.exists():
                    errors.append(f"Model {model_id}: code dependency not found at {code_file}")

        for target_id, target in self.targets.items():
            if not target.path.exists():
                errors.append(f"Target {target_id}: file not found at {target.path}")
            for data_file in target.data:
                if not data_file.exists():
                    errors.append(f"Target {target_id}: data dependency not found at {data_file}")

        for set_name, target_set in self.target_sets.items():
            missing = [tid for tid in target_set.targets if tid not in self.targets]
            if missing:
                errors.append(
                    f"Target set {set_name}: references unknown targets {', '.join(sorted(missing))}"
                )
            unused_weights = [tid for tid in target_set.weights if tid not in target_set.targets]
            if unused_weights:
                errors.append(
                    f"Target set {set_name}: weights configured for non-member targets "
                    f"{', '.join(sorted(unused_weights))}"
                )

        return errors

    def get_all_dependencies(self) -> List[Path]:
        """Get all files referenced in the registry."""
        dependencies = set()

        # Add model files and their dependencies
        for model in self.models.values():
            dependencies.add(model.path)
            dependencies.update(model.data)
            dependencies.update(model.code)

        # Add target files and their data dependencies
        for target in self.targets.values():
            dependencies.add(target.path)
            dependencies.update(target.data)

        return sorted(list(dependencies))

    def save(self, path: Path) -> None:
        """Save registry to YAML file.

        Args:
            path: Path to YAML file to write
        """
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "BundleRegistry":
        """Load registry from YAML file.

        Args:
            path: Path to YAML file to read

        Returns:
            Loaded BundleRegistry instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "version": self.version,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "targets": {k: v.to_dict() for k, v in self.targets.items()},
            "target_sets": {k: v.to_dict() for k, v in self.target_sets.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleRegistry":
        """Create from dictionary (YAML deserialization)."""
        registry = cls(version=data.get("version", "1.0"))

        for model_id, model_data in data.get("models", {}).items():
            registry.models[model_id] = ModelEntry.from_dict(model_data)

        for target_id, target_data in data.get("targets", {}).items():
            registry.targets[target_id] = TargetEntry.from_dict(target_data)

        for set_name, set_data in data.get("target_sets", {}).items():
            registry.target_sets[set_name] = TargetSet.from_dict(set_data)

        return registry


def discover_model_classes(file_path: Path) -> List[Tuple[str, List[str]]]:
    """Discover classes that inherit from BaseModel in a Python file.

    This function uses AST parsing to find classes without executing code,
    making it safe to use on untrusted files. It looks for classes that:
    - Directly inherit from BaseModel
    - Inherit from calabaria.BaseModel or modelops_calabaria.BaseModel
    - Inherit from other classes in the same file that inherit from BaseModel

    Args:
        file_path: Path to Python file to analyze

    Returns:
        List of (class_name, base_classes) tuples where base_classes
        is a list of base class names as strings

    Example:
        >>> discover_model_classes(Path("models.py"))
        [("StochasticSEIR", ["BaseModel"]),
         ("DeterministicSEIR", ["calabaria.BaseModel"]),
         ("NetworkSEIR", ["StochasticSEIR"])]
    """
    with open(file_path) as f:
        tree = ast.parse(f.read())

    # First pass: find all classes and their bases
    all_classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            base_names = []
            for base in node.bases:
                # Handle direct names (e.g., BaseModel)
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                # Handle attribute access (e.g., calabaria.BaseModel)
                elif isinstance(base, ast.Attribute):
                    parts = []
                    current = base
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                    base_names.append('.'.join(reversed(parts)))
            all_classes[node.name] = base_names

    # Second pass: find all BaseModel descendants
    model_classes = []

    def is_basemodel_descendant(class_name: str, visited: set = None) -> bool:
        """Recursively check if a class descends from BaseModel."""
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        if class_name in visited:
            return False
        visited.add(class_name)

        # Check if this class exists in our file
        if class_name not in all_classes:
            # External class - check if it's BaseModel
            return 'BaseModel' in class_name

        # Check direct bases
        for base in all_classes[class_name]:
            if 'BaseModel' in base:
                return True
            # Recursively check parent classes
            if is_basemodel_descendant(base, visited):
                return True

        return False

    # Collect all BaseModel descendants
    for class_name, base_names in all_classes.items():
        if is_basemodel_descendant(class_name):
            model_classes.append((class_name, base_names))

    return model_classes


def discover_model_outputs(file_path: Path) -> Dict[str, List[str]]:
    """Discover @model_output decorators grouped by class name.

    This is a lightweight contract with Calabaria's decorator API so that
    tooling (e.g. modelops-bundle) can infer which outputs a model exposes
    without executing user code. If Calabaria changes the decorator name or
    signature, this helper will need to be updated accordingly.
    """
    outputs: Dict[str, List[str]] = {}
    with open(file_path) as f:
        tree = ast.parse(f.read())

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_outputs: List[str] = []
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    for decorator in method.decorator_list:
                        decorator_name = None
                        output_name = None

                        if isinstance(decorator, ast.Name):
                            decorator_name = decorator.id
                        elif isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Name):
                                decorator_name = decorator.func.id
                            elif isinstance(decorator.func, ast.Attribute):
                                decorator_name = decorator.func.attr

                            if decorator.args:
                                try:
                                    output_name = ast.literal_eval(decorator.args[0])
                                except Exception:
                                    output_name = ast.unparse(decorator.args[0])
                            elif decorator.keywords:
                                for kw in decorator.keywords:
                                    if kw.arg in ("name", "output"):
                                        try:
                                            output_name = ast.literal_eval(kw.value)
                                        except Exception:
                                            output_name = ast.unparse(kw.value)
                                        break

                        if decorator_name == "model_output":
                            class_outputs.append(str(output_name or method.name))
            if class_outputs:
                outputs[node.name] = class_outputs

    return outputs


def discover_target_functions(file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Discover functions decorated with @calibration_target in a Python file.

    This function uses AST parsing to find decorated functions without executing code,
    making it safe to use on untrusted files.

    Args:
        file_path: Path to Python file to analyze

    Returns:
        List of (function_name, metadata) tuples where metadata is a dict
        extracted from the decorator arguments

    Example:
        >>> discover_target_functions(Path("targets.py"))
        [("prevalence_target", {
            "model_output": "prevalence",
            "data": {
                "observed": "data/observed_prevalence.csv",
                "population": "data/population.csv"
            }
        })]
    """
    with open(file_path) as f:
        tree = ast.parse(f.read())

    target_functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Look for @calibration_target decorator
            for decorator in node.decorator_list:
                decorator_name = None
                decorator_kwargs = {}

                # Handle direct name (e.g., @calibration_target)
                if isinstance(decorator, ast.Name):
                    decorator_name = decorator.id
                # Handle call with arguments (e.g., @calibration_target(...))
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name):
                        decorator_name = decorator.func.id
                    elif isinstance(decorator.func, ast.Attribute):
                        # Handle module.calibration_target
                        decorator_name = decorator.func.attr

                    # Extract keyword arguments
                    for keyword in decorator.keywords:
                        # Try to evaluate the argument as a literal
                        try:
                            decorator_kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                        except (ValueError, TypeError):
                            # If it's not a literal, store as string representation
                            decorator_kwargs[keyword.arg] = ast.unparse(keyword.value)

                if decorator_name == "calibration_target":
                    target_functions.append((node.name, decorator_kwargs))

    return target_functions


__all__ = [
    "ModelEntry",
    "TargetEntry",
    "BundleRegistry",
    "RegistryReader",
    "discover_model_classes",
    "discover_target_functions",
]
