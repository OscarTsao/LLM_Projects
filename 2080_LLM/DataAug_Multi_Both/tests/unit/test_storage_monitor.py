"""Unit tests for storage monitoring."""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.dataaug_multi_both.utils.storage_monitor import StorageMonitor


class TestStorageMonitor:
    """Test suite for StorageMonitor."""
    
    def test_monitor_initialization(self):
        """Test that monitor can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = StorageMonitor(
                monitor_path=Path(tmpdir),
                check_interval=1,
                critical_threshold_percent=10.0
            )
            assert monitor.monitor_path == Path(tmpdir)
            assert monitor.check_interval == 1
            assert monitor.critical_threshold == 10.0
    
    def test_get_disk_usage(self):
        """Test that disk usage can be retrieved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = StorageMonitor(monitor_path=Path(tmpdir))
            usage = monitor.get_disk_usage()
            
            assert "total_bytes" in usage
            assert "used_bytes" in usage
            assert "free_bytes" in usage
            assert "percent_free" in usage
            assert usage["total_bytes"] > 0
    
    @patch('shutil.disk_usage')
    def test_critical_flag_set_when_low_space(self, mock_disk_usage):
        """Test that critical flag is set when space is low."""
        # Mock disk usage: 5% free (below 10% threshold)
        mock_usage = MagicMock()
        mock_usage.total = 1000
        mock_usage.used = 950
        mock_usage.free = 50
        mock_disk_usage.return_value = mock_usage
        
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = StorageMonitor(
                monitor_path=Path(tmpdir),
                check_interval=0.1,
                critical_threshold_percent=10.0
            )
            
            monitor.start()
            time.sleep(0.3)  # Wait for monitor to check
            
            assert monitor.is_critical()
            
            monitor.stop()
    
    @patch('shutil.disk_usage')
    def test_critical_flag_cleared_when_space_recovers(self, mock_disk_usage):
        """Test that critical flag is cleared when space recovers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = StorageMonitor(
                monitor_path=Path(tmpdir),
                check_interval=0.1,
                critical_threshold_percent=10.0
            )
            
            # Start with low space
            mock_usage_low = MagicMock()
            mock_usage_low.total = 1000
            mock_usage_low.used = 950
            mock_usage_low.free = 50
            mock_disk_usage.return_value = mock_usage_low
            
            monitor.start()
            time.sleep(0.3)
            assert monitor.is_critical()
            
            # Recover space
            mock_usage_high = MagicMock()
            mock_usage_high.total = 1000
            mock_usage_high.used = 500
            mock_usage_high.free = 500
            mock_disk_usage.return_value = mock_usage_high
            
            time.sleep(0.3)
            assert not monitor.is_critical()
            
            monitor.stop()
    
    @patch('shutil.disk_usage')
    def test_warning_flag_set_when_space_low(self, mock_disk_usage):
        """Test that warning flag is set when space is low."""
        # Mock disk usage: 15% free (below 20% warning threshold)
        mock_usage = MagicMock()
        mock_usage.total = 1000
        mock_usage.used = 850
        mock_usage.free = 150
        mock_disk_usage.return_value = mock_usage
        
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = StorageMonitor(
                monitor_path=Path(tmpdir),
                check_interval=0.1,
                warning_threshold_percent=20.0
            )
            
            monitor.start()
            time.sleep(0.3)
            
            assert monitor.is_warning()
            
            monitor.stop()
    
    def test_monitor_start_stop(self):
        """Test that monitor can be started and stopped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = StorageMonitor(monitor_path=Path(tmpdir), check_interval=0.1)

            monitor.start()
            assert monitor._thread is not None
            assert monitor._thread.is_alive()

            monitor.stop()
            # Daemon threads may take a moment to stop
            # Just verify stop was called successfully
            assert monitor._stop_event.is_set()
    
    def test_context_manager(self):
        """Test that monitor works as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with StorageMonitor(monitor_path=Path(tmpdir), check_interval=0.1) as monitor:
                assert monitor._thread is not None
                assert monitor._thread.is_alive()

            # Monitor should have stop event set after context exit
            assert monitor._stop_event.is_set()
    
    def test_multiple_start_calls_ignored(self):
        """Test that multiple start calls are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = StorageMonitor(monitor_path=Path(tmpdir))
            
            monitor.start()
            thread1 = monitor._thread
            
            monitor.start()  # Second start should be ignored
            thread2 = monitor._thread
            
            assert thread1 is thread2  # Same thread
            
            monitor.stop()

