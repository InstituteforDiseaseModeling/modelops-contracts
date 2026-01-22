"""Tests for SimulationStudy and ParameterSetEntry."""

import pytest
from modelops_contracts import (
    ParameterSetEntry,
    SimulationStudy,
)

# Valid test bundle reference (SHA256 with 64 hex chars)
TEST_BUNDLE = "sha256:" + "a" * 64


class TestParameterSetEntry:
    """Tests for ParameterSetEntry dataclass."""

    def test_creation_without_seed(self):
        """Test creating entry without explicit seed."""
        entry = ParameterSetEntry(params={"x": 1.0, "y": 2.0})
        assert entry.params == {"x": 1.0, "y": 2.0}
        assert entry.seed is None

    def test_creation_with_seed(self):
        """Test creating entry with explicit seed."""
        entry = ParameterSetEntry(params={"x": 1.0}, seed=42)
        assert entry.params == {"x": 1.0}
        assert entry.seed == 42

    def test_seed_validation_negative(self):
        """Test that negative seeds are rejected."""
        with pytest.raises(ValueError, match="seed must be int in uint64 range"):
            ParameterSetEntry(params={"x": 1.0}, seed=-1)

    def test_seed_validation_too_large(self):
        """Test that seeds larger than uint64 are rejected."""
        with pytest.raises(ValueError, match="seed must be int in uint64 range"):
            ParameterSetEntry(params={"x": 1.0}, seed=2**64)

    def test_seed_validation_uint64_max(self):
        """Test that uint64 max is valid."""
        entry = ParameterSetEntry(params={"x": 1.0}, seed=2**64 - 1)
        assert entry.seed == 2**64 - 1

    def test_seed_validation_zero(self):
        """Test that zero is valid."""
        entry = ParameterSetEntry(params={"x": 1.0}, seed=0)
        assert entry.seed == 0

    def test_immutability(self):
        """Test that ParameterSetEntry is frozen."""
        entry = ParameterSetEntry(params={"x": 1.0}, seed=42)
        with pytest.raises(AttributeError):
            entry.seed = 100


class TestSimulationStudyExplicitSeeds:
    """Tests for SimulationStudy with explicit seeds."""

    def test_explicit_seed(self):
        """Test that explicit seeds are used directly."""
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[
                ParameterSetEntry(params={"x": 1.0}, seed=42),
                ParameterSetEntry(params={"x": 2.0}, seed=123),
            ],
            sampling_method="manual",
        )
        job = study.to_simjob(TEST_BUNDLE)
        assert len(job.tasks) == 2
        assert job.tasks[0].seed == 42
        assert job.tasks[1].seed == 123

    def test_auto_seed(self):
        """Test that None seeds generate n_replicates tasks."""
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[
                ParameterSetEntry(params={"x": 1.0}, seed=None),
            ],
            sampling_method="manual",
            n_replicates=5,
        )
        job = study.to_simjob(TEST_BUNDLE)
        assert len(job.tasks) == 5
        # Seeds should be deterministic and all different
        seeds = [t.seed for t in job.tasks]
        assert len(set(seeds)) == 5

    def test_mixed_explicit_and_auto_seeds(self):
        """Test mixing explicit and auto-generated seeds."""
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[
                ParameterSetEntry(params={"x": 1.0}, seed=42),  # 1 task
                ParameterSetEntry(params={"x": 2.0}, seed=None),  # 3 tasks
                ParameterSetEntry(params={"x": 3.0}, seed=999),  # 1 task
            ],
            sampling_method="manual",
            n_replicates=3,
        )
        job = study.to_simjob(TEST_BUNDLE)
        assert len(job.tasks) == 5  # 1 + 3 + 1

        # First task should have explicit seed 42
        assert job.tasks[0].seed == 42
        # Last task should have explicit seed 999
        assert job.tasks[4].seed == 999
        # Middle 3 tasks should have auto-generated seeds (all different)
        middle_seeds = [job.tasks[i].seed for i in range(1, 4)]
        assert len(set(middle_seeds)) == 3

    def test_total_simulation_count_explicit(self):
        """Test total_simulation_count with explicit seeds."""
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[
                ParameterSetEntry(params={"x": 1.0}, seed=42),
                ParameterSetEntry(params={"x": 2.0}, seed=123),
            ],
            sampling_method="manual",
            n_replicates=10,  # Should be ignored for explicit seeds
        )
        assert study.total_simulation_count() == 2

    def test_total_simulation_count_auto(self):
        """Test total_simulation_count with auto seeds."""
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[
                ParameterSetEntry(params={"x": 1.0}, seed=None),
                ParameterSetEntry(params={"x": 2.0}, seed=None),
            ],
            sampling_method="manual",
            n_replicates=5,
        )
        assert study.total_simulation_count() == 10  # 2 * 5

    def test_total_simulation_count_mixed(self):
        """Test total_simulation_count with mixed seeds."""
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[
                ParameterSetEntry(params={"x": 1.0}, seed=42),  # 1
                ParameterSetEntry(params={"x": 2.0}, seed=None),  # 3
                ParameterSetEntry(params={"x": 3.0}, seed=None),  # 3
            ],
            sampling_method="manual",
            n_replicates=3,
        )
        assert study.total_simulation_count() == 7  # 1 + 3 + 3


class TestSimulationStudyFromDict:
    """Tests for SimulationStudy.from_dict() backwards compatibility."""

    def test_from_dict_old_format(self):
        """Test backwards compatibility with old JSON format."""
        data = {
            "model": "test.model",
            "scenario": "baseline",
            "parameter_sets": [{"x": 1.0}, {"x": 2.0}],
            "sampling_method": "sobol",
            "n_replicates": 3,
        }
        study = SimulationStudy.from_dict(data)
        assert len(study.parameter_sets) == 2
        assert all(e.seed is None for e in study.parameter_sets)
        assert study.parameter_sets[0].params == {"x": 1.0}
        assert study.parameter_sets[1].params == {"x": 2.0}
        assert study.n_replicates == 3

    def test_from_dict_new_format(self):
        """Test parsing new format with explicit seeds."""
        data = {
            "model": "test.model",
            "scenario": "baseline",
            "parameter_sets": [
                {"params": {"x": 1.0}, "seed": 42},
                {"params": {"x": 2.0}, "seed": 123},
            ],
            "sampling_method": "manual",
        }
        study = SimulationStudy.from_dict(data)
        assert study.parameter_sets[0].params == {"x": 1.0}
        assert study.parameter_sets[0].seed == 42
        assert study.parameter_sets[1].params == {"x": 2.0}
        assert study.parameter_sets[1].seed == 123

    def test_from_dict_new_format_no_seed(self):
        """Test parsing new format without explicit seeds."""
        data = {
            "model": "test.model",
            "scenario": "baseline",
            "parameter_sets": [
                {"params": {"x": 1.0}},
                {"params": {"x": 2.0}, "seed": None},
            ],
            "sampling_method": "manual",
            "n_replicates": 5,
        }
        study = SimulationStudy.from_dict(data)
        assert study.parameter_sets[0].seed is None
        assert study.parameter_sets[1].seed is None

    def test_from_dict_mixed_formats(self):
        """Test that old and new format can be mixed in parameter_sets."""
        data = {
            "model": "test.model",
            "scenario": "baseline",
            "parameter_sets": [
                {"x": 1.0},  # Old format
                {"params": {"x": 2.0}, "seed": 42},  # New format with seed
            ],
            "sampling_method": "manual",
        }
        study = SimulationStudy.from_dict(data)
        assert study.parameter_sets[0].params == {"x": 1.0}
        assert study.parameter_sets[0].seed is None
        assert study.parameter_sets[1].params == {"x": 2.0}
        assert study.parameter_sets[1].seed == 42

    def test_from_dict_defaults(self):
        """Test default values when fields are omitted."""
        data = {
            "model": "test.model",
            "scenario": "baseline",
            "parameter_sets": [],
        }
        study = SimulationStudy.from_dict(data)
        assert study.sampling_method == "manual"
        assert study.n_replicates == 1
        assert study.outputs is None
        assert study.targets is None
        assert study.metadata == {}

    def test_from_dict_with_all_fields(self):
        """Test parsing with all optional fields."""
        data = {
            "model": "test.model",
            "scenario": "baseline",
            "parameter_sets": [{"params": {"x": 1.0}, "seed": 42}],
            "sampling_method": "manual",
            "n_replicates": 5,
            "outputs": ["metric1", "metric2"],
            "targets": ["target1"],
            "metadata": {"author": "test"},
        }
        study = SimulationStudy.from_dict(data)
        assert study.outputs == ["metric1", "metric2"]
        assert study.targets == ["target1"]
        assert study.metadata == {"author": "test"}


class TestSimulationStudyJobCreation:
    """Tests for SimulationStudy.to_simjob() integration."""

    def test_to_simjob_with_custom_job_id(self):
        """Test creating job with custom job ID."""
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[ParameterSetEntry(params={"x": 1.0}, seed=42)],
            sampling_method="manual",
        )
        job = study.to_simjob(TEST_BUNDLE, job_id="my-custom-job")
        assert job.job_id == "my-custom-job"

    def test_to_simjob_params_preserved(self):
        """Test that parameters are correctly passed to tasks."""
        params = {"alpha": 0.5, "beta": 1.0, "gamma": 2.0}
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[ParameterSetEntry(params=params, seed=42)],
            sampling_method="manual",
        )
        job = study.to_simjob(TEST_BUNDLE)
        assert job.tasks[0].params.params == params

    def test_to_simjob_deterministic_auto_seeds(self):
        """Test that auto-generated seeds are deterministic."""
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[ParameterSetEntry(params={"x": 1.0}, seed=None)],
            sampling_method="manual",
            n_replicates=3,
        )
        job1 = study.to_simjob(TEST_BUNDLE)
        job2 = study.to_simjob(TEST_BUNDLE)

        seeds1 = [t.seed for t in job1.tasks]
        seeds2 = [t.seed for t in job2.tasks]
        assert seeds1 == seeds2

    def test_to_simjob_invalid_bundle_ref(self):
        """Test that invalid bundle_ref raises ValueError."""
        study = SimulationStudy(
            model="test.model",
            scenario="baseline",
            parameter_sets=[ParameterSetEntry(params={"x": 1.0}, seed=42)],
            sampling_method="manual",
        )
        with pytest.raises(ValueError, match="bundle_ref must be"):
            study.to_simjob("invalid-bundle-ref")
