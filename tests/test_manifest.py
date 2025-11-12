"""Tests for manifest types."""

import pytest
from pathlib import Path
from modelops_contracts import ModelEntry, BundleManifest


class TestModelEntry:
    """Tests for ModelEntry validation and functionality."""

    def test_model_entry_creation(self):
        """Test basic ModelEntry creation with required fields."""
        entry = ModelEntry(
            entrypoint="models.seir:StochasticSEIR",
            path=Path("models/seir.py"),
            class_name="StochasticSEIR",
            scenarios=["baseline"],
            outputs=["infections"],
            parameters=["beta", "gamma"],
            model_digest="sha256:" + "a" * 64
        )
        assert entry.entrypoint == "models.seir:StochasticSEIR"
        assert entry.class_name == "StochasticSEIR"
        assert entry.model_digest == "sha256:" + "a" * 64

    def test_model_entry_optional_fields(self):
        """Test ModelEntry with minimal required fields."""
        entry = ModelEntry(
            entrypoint="models.test:Model",
            path=Path("models/test.py"),
            class_name="Model"
        )
        assert entry.scenarios == []
        assert entry.parameters == []
        assert entry.outputs == []
        assert entry.model_digest is None

    def test_model_entry_with_scenarios(self):
        """Test ModelEntry with multiple scenarios."""
        entry = ModelEntry(
            entrypoint="models.seir:StochasticSEIR",
            path=Path("models/seir.py"),
            class_name="StochasticSEIR",
            scenarios=["baseline", "lockdown", "vaccination"],
            outputs=["infections"],
            parameters=["beta"]
        )
        assert len(entry.scenarios) == 3
        assert "baseline" in entry.scenarios
        assert "lockdown" in entry.scenarios

    def test_model_entry_with_parameters(self):
        """Test ModelEntry with multiple parameters."""
        entry = ModelEntry(
            entrypoint="models.test:Model",
            path=Path("models/test.py"),
            class_name="Model",
            parameters=["x", "y", "z"]
        )
        assert len(entry.parameters) == 3
        assert "x" in entry.parameters


class TestBundleManifest:
    """Tests for BundleManifest validation and functionality."""

    def test_bundle_manifest_creation(self):
        """Test basic BundleManifest creation."""
        model = ModelEntry(
            entrypoint="models.test:Model",
            path=Path("models/test.py"),
            class_name="Model",
            scenarios=["baseline"],
            outputs=["results"],
            parameters=["x"],
            model_digest="sha256:" + "1" * 64
        )

        # Valid manifest
        manifest = BundleManifest(
            bundle_ref="oci://registry/bundle:latest",
            bundle_digest="2" * 64,  # No sha256: prefix
            models={"models.test:Model": model},
            version=1
        )
        assert manifest.bundle_digest == "2" * 64
        assert len(manifest.models) == 1

    def test_bundle_manifest_with_multiple_models(self):
        """Test BundleManifest with multiple models."""
        model1 = ModelEntry(
            entrypoint="models.seir:SEIR",
            path=Path("models/seir.py"),
            class_name="SEIR",
            scenarios=["baseline"],
            outputs=["infections"],
            parameters=["beta"],
            model_digest="sha256:" + "1" * 64
        )
        model2 = ModelEntry(
            entrypoint="models.sir:SIR",
            path=Path("models/sir.py"),
            class_name="SIR",
            scenarios=["baseline"],
            outputs=["infections"],
            parameters=["beta"],
            model_digest="sha256:" + "2" * 64
        )

        manifest = BundleManifest(
            bundle_ref="local://dev",
            bundle_digest="3" * 64,  # No sha256: prefix
            models={
                "models.seir:SEIR": model1,
                "models.sir:SIR": model2
            }
        )

        assert len(manifest.models) == 2
        assert "models.seir:SEIR" in manifest.models
        assert "models.sir:SIR" in manifest.models