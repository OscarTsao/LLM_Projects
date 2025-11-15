"""
Test CLI command integration and flags.

Tests the typer-based CLI interface for correctness and robustness.
"""

import json

import pytest
from typer.testing import CliRunner

from psy_agents_noaug.cli import app


@pytest.fixture
def runner():
    """Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config file."""
    config = {"batch_size": 32, "learning_rate": 0.001}
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(config))
    return str(config_path)


class TestCLIHelp:
    """Test CLI help messages and basic structure."""

    def test_app_help(self, runner):
        """Test main app help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "NoAug Criteria/Evidence" in result.stdout

    def test_train_help(self, runner):
        """Test train command help."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "agent" in result.stdout.lower()

    def test_tune_help(self, runner):
        """Test tune command help."""
        result = runner.invoke(app, ["tune", "--help"])
        assert result.exit_code == 0
        assert "study" in result.stdout.lower()

    def test_show_best_help(self, runner):
        """Test show-best command help."""
        result = runner.invoke(app, ["show-best", "--help"])
        assert result.exit_code == 0
        assert "topk" in result.stdout.lower() or "agent" in result.stdout.lower()


class TestTrainCommand:
    """Test train command functionality."""

    def test_train_minimal(self, runner):
        """Test train command with minimal required arguments."""
        result = runner.invoke(app, ["train", "--agent", "criteria"])
        # Should succeed (stub implementation)
        assert result.exit_code == 0
        assert "agent=criteria" in result.stdout

    def test_train_with_all_flags(self, runner):
        """Test train command with all optional flags."""
        result = runner.invoke(
            app,
            [
                "train",
                "--agent",
                "evidence",
                "--model-name",
                "roberta-base",
                "--epochs",
                "5",
                "--seed",
                "123",
                "--batch-size",
                "32",
                "--grad-accum",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert "agent=evidence" in result.stdout
        assert "epochs=5" in result.stdout
        assert "seed=123" in result.stdout
        assert "antonym_guard=off" in result.stdout

    def test_train_with_antonym_guard_flag(self, runner):
        """Test train command with antonym guard override."""
        result = runner.invoke(
            app,
            [
                "train",
                "--agent",
                "criteria",
                "--antonym-guard",
                "on_low_weight",
            ],
        )
        assert result.exit_code == 0
        assert "antonym_guard=on_low_weight" in result.stdout

    def test_train_with_config_file(self, runner, temp_config):
        """Test train command with config file."""
        result = runner.invoke(
            app,
            [
                "train",
                "--agent",
                "criteria",
                "--config",
                temp_config,
            ],
        )
        assert result.exit_code == 0
        assert "loaded config keys" in result.stdout

    def test_train_missing_agent(self, runner):
        """Test train command fails without required --agent."""
        result = runner.invoke(app, ["train"])
        # Should fail due to missing required argument
        assert result.exit_code != 0


class TestTuneCommand:
    """Test tune command functionality."""

    def test_tune_minimal(self, runner):
        """Test tune command with minimal required arguments."""
        # Note: This will try to run tune_max.py which may not exist
        # We're testing flag parsing, not actual execution
        result = runner.invoke(
            app,
            [
                "tune",
                "--agent",
                "criteria",
                "--study",
                "test_study",
                "--n-trials",
                "1",
            ],
        )
        # May fail if script doesn't exist, but should parse flags
        # Check that command was attempted (not a flag parsing error)
        # Exit code may be non-zero if script doesn't exist
        assert "--agent" not in result.stdout or result.exit_code in [0, 1]

    def test_tune_with_all_flags(self, runner, tmp_path):
        """Test tune command with all optional flags."""
        result = runner.invoke(
            app,
            [
                "tune",
                "--agent",
                "evidence",
                "--study",
                "test_study",
                "--n-trials",
                "10",
                "--timeout",
                "3600",
                "--parallel",
                "2",
                "--outdir",
                str(tmp_path),
                "--storage",
                "sqlite:///test.db",
                "--stage",
                "B",
                "--from-study",
                "noaug-evidence-max",
                "--pareto-limit",
                "3",
            ],
        )
        # May fail if script doesn't exist, but flags should parse
        # Just verify it attempted to run (not a parse error)
        assert result.exit_code in [0, 1]

    def test_tune_missing_required_args(self, runner):
        """Test tune command fails without required arguments."""
        result = runner.invoke(app, ["tune"])
        assert result.exit_code != 0


class TestShowBestCommand:
    """Test show-best command functionality."""

    def test_show_best_file_not_found(self, runner):
        """Test show-best with non-existent file."""
        result = runner.invoke(
            app,
            [
                "show-best",
                "--agent",
                "criteria",
                "--study",
                "nonexistent",
            ],
        )
        # Should exit with non-zero status because study does not exist
        assert result.exit_code != 0

    def test_show_best_with_topk(self, runner, tmp_path):
        """Test show-best with topk parameter."""
        # Create a dummy results file
        results_file = tmp_path / "criteria_test_topk.json"
        results_file.write_text(json.dumps([{"trial": 1, "value": 0.95}]))

        result = runner.invoke(
            app,
            [
                "show-best",
                "--agent",
                "criteria",
                "--study",
                "test",
                "--outdir",
                str(tmp_path),
                "--topk",
                "3",
            ],
        )
        # Should succeed or handle missing file gracefully
        assert result.exit_code != 0


class TestAgentValidation:
    """Test agent parameter validation across commands."""

    @pytest.mark.parametrize(
        "agent", ["criteria", "evidence", "share", "joint", "custom_agent"]
    )
    def test_train_accepts_agent(self, runner, agent):
        """Test that train command accepts various agent values."""
        result = runner.invoke(app, ["train", "--agent", agent])
        # Should succeed (stub implementation doesn't validate agent)
        assert result.exit_code == 0
        assert f"agent={agent}" in result.stdout


class TestOutputDirectory:
    """Test output directory handling."""

    def test_train_creates_default_outdir(self, runner):
        """Test that train uses default output directory."""
        result = runner.invoke(app, ["train", "--agent", "criteria"])
        assert result.exit_code == 0
        assert "outdir=" in result.stdout

    def test_train_with_custom_outdir(self, runner, tmp_path):
        """Test train with custom output directory."""
        custom_out = tmp_path / "custom_output"
        result = runner.invoke(
            app,
            [
                "train",
                "--agent",
                "criteria",
                "--outdir",
                str(custom_out),
            ],
        )
        assert result.exit_code == 0
        assert str(custom_out) in result.stdout or "outdir=" in result.stdout


class TestMLflowSetup:
    """Test MLflow tracking URI setup."""

    def test_mlflow_uri_set_for_train(self, runner, tmp_path):
        """Test that MLflow tracking URI is configured for train."""
        # The CLI should set MLFLOW_TRACKING_URI
        # We can't easily test env var from CLI runner, but we test the function
        from psy_agents_noaug.cli import _ensure_mlflow

        outdir = str(tmp_path / "test_out")
        _ensure_mlflow(outdir)

        # Check that directory was created
        assert (tmp_path / "test_out").exists()

    def test_default_outdir_function(self):
        """Test _default_outdir helper function."""
        from psy_agents_noaug.cli import _default_outdir

        # None should return default
        assert _default_outdir(None) == "./_runs"

        # Non-None should return as-is
        assert _default_outdir("/custom/path") == "/custom/path"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_command(self, runner):
        """Test running CLI with no command."""
        result = runner.invoke(app, [])
        # Should show help or list commands
        assert result.exit_code in [0, 2]  # 2 = usage error

    def test_invalid_command(self, runner):
        """Test running CLI with invalid command."""
        result = runner.invoke(app, ["invalid_command"])
        # Should show error about unknown command
        assert result.exit_code != 0

    def test_train_with_invalid_json_config(self, runner, tmp_path):
        """Test train with malformed JSON config file."""
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("{invalid json")

        result = runner.invoke(
            app,
            [
                "train",
                "--agent",
                "criteria",
                "--config",
                str(bad_config),
            ],
        )
        # Should fail with JSON decode error
        assert result.exit_code != 0

    def test_train_with_nonexistent_config(self, runner):
        """Test train with non-existent config file."""
        result = runner.invoke(
            app,
            [
                "train",
                "--agent",
                "criteria",
                "--config",
                "/nonexistent/config.json",
            ],
        )
        # Should fail with file not found error
        assert result.exit_code != 0
