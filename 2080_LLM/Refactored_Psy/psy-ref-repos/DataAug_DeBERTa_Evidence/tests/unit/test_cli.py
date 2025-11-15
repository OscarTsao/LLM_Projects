"""Unit tests for CLI entry point."""

import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
from src.dataaug_multi_both.cli.train import cli, train, hpo, resume, status


class TestCLI:
    """Test suite for CLI commands."""
    
    def test_cli_help(self):
        """Test that CLI help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        
        assert result.exit_code == 0
        assert "DataAug Multi Both" in result.output
    
    def test_train_help(self):
        """Test that train command help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        
        assert result.exit_code == 0
        assert "trial-id" in result.output
        assert "model-name" in result.output
    
    def test_hpo_help(self):
        """Test that HPO command help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["hpo", "--help"])
        
        assert result.exit_code == 0
        assert "study-name" in result.output
        assert "n-trials" in result.output
    
    def test_resume_help(self):
        """Test that resume command help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--help"])
        
        assert result.exit_code == 0
        assert "trial-dir" in result.output
    
    def test_status_help(self):
        """Test that status command help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--help"])
        
        assert result.exit_code == 0
        assert "experiments-dir" in result.output


class TestTrainCommand:
    """Test suite for train command."""
    
    def test_train_with_required_args(self):
        """Test train command with required arguments."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "train",
                "--trial-id", "test_trial",
                "--experiments-dir", tmpdir
            ])
            
            assert result.exit_code == 0
            assert "Starting training: test_trial" in result.output
    
    def test_train_creates_trial_directory(self):
        """Test that train command creates trial directory."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "train",
                "--trial-id", "test_trial",
                "--experiments-dir", tmpdir
            ])
            
            assert result.exit_code == 0
            
            # Verify trial directory was created
            trial_dir = Path(tmpdir) / "test_trial"
            assert trial_dir.exists()
    
    def test_train_with_custom_params(self):
        """Test train command with custom parameters."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "train",
                "--trial-id", "test_trial",
                "--model-name", "bert-base",
                "--learning-rate", "1e-4",
                "--batch-size", "32",
                "--max-epochs", "5",
                "--seed", "42",
                "--experiments-dir", tmpdir
            ])
            
            assert result.exit_code == 0
            assert "test_trial" in result.output
    
    def test_train_with_resume_disabled(self):
        """Test train command with resume disabled."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "train",
                "--trial-id", "test_trial",
                "--no-resume",
                "--experiments-dir", tmpdir
            ])
            
            assert result.exit_code == 0


class TestHPOCommand:
    """Test suite for HPO command."""
    
    def test_hpo_with_required_args(self):
        """Test HPO command with required arguments."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "hpo",
                "--study-name", "test_study",
                "--experiments-dir", tmpdir
            ])
            
            assert result.exit_code == 0
            assert "Starting HPO: test_study" in result.output
    
    def test_hpo_creates_experiments_directory(self):
        """Test that HPO command creates experiments directory."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "experiments"
            
            result = runner.invoke(cli, [
                "hpo",
                "--study-name", "test_study",
                "--experiments-dir", str(exp_dir)
            ])
            
            assert result.exit_code == 0
            assert exp_dir.exists()
    
    def test_hpo_with_custom_n_trials(self):
        """Test HPO command with custom number of trials."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "hpo",
                "--study-name", "test_study",
                "--n-trials", "20",
                "--experiments-dir", tmpdir
            ])
            
            assert result.exit_code == 0
            assert "Number of trials: 20" in result.output


class TestResumeCommand:
    """Test suite for resume command."""
    
    def test_resume_requires_trial_dir(self):
        """Test that resume command requires trial directory."""
        runner = CliRunner()
        result = runner.invoke(cli, ["resume"])
        
        assert result.exit_code != 0
        assert "trial-dir" in result.output.lower() or "missing" in result.output.lower()
    
    def test_resume_with_existing_dir(self):
        """Test resume command with existing directory."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "resume",
                "--trial-dir", tmpdir
            ])
            
            assert result.exit_code == 0
            assert "Resuming from" in result.output


class TestStatusCommand:
    """Test suite for status command."""
    
    def test_status_with_no_experiments(self):
        """Test status command with no experiments."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "status",
                "--experiments-dir", tmpdir
            ])
            
            assert result.exit_code == 0
            assert "Number of trials: 0" in result.output
    
    def test_status_with_experiments(self):
        """Test status command with existing experiments."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some trial directories
            (Path(tmpdir) / "trial_001").mkdir()
            (Path(tmpdir) / "trial_002").mkdir()
            
            result = runner.invoke(cli, [
                "status",
                "--experiments-dir", tmpdir
            ])
            
            assert result.exit_code == 0
            assert "Number of trials: 2" in result.output
            assert "trial_001" in result.output
            assert "trial_002" in result.output
    
    def test_status_with_nonexistent_dir(self):
        """Test status command with nonexistent directory."""
        runner = CliRunner()

        result = runner.invoke(cli, [
            "status",
            "--experiments-dir", "/nonexistent/path"
        ])

        # Should fail with error
        assert result.exit_code != 0

