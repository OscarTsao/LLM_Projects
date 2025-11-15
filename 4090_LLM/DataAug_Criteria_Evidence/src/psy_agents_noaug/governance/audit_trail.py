#!/usr/bin/env python
"""Audit trail and lineage tracking (Phase 25).

This module provides tools for logging model operations, tracking lineage,
and maintaining audit trails for governance and compliance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of audit events."""

    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_INFERENCE = "model_inference"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    COMPLIANCE_CHECK = "compliance_check"
    BIAS_ASSESSMENT = "bias_assessment"
    MODEL_UPDATE = "model_update"
    MODEL_RETIREMENT = "model_retirement"


@dataclass
class AuditEvent:
    """Single audit event."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    user: str
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    model_name: str | None = None
    model_version: str | None = None
    data_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary.

        Returns:
            Dictionary representation
        """
        event_dict = asdict(self)
        # Convert datetime and enum to strings
        event_dict["timestamp"] = self.timestamp.isoformat()
        event_dict["event_type"] = self.event_type.value
        return event_dict

    def to_json(self, indent: int = 2) -> str:
        """Convert event to JSON.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)


class AuditLogger:
    """Logger for audit events."""

    def __init__(self, log_file: Path | str | None = None):
        """Initialize audit logger.

        Args:
            log_file: Path to audit log file (optional)
        """
        self.events: list[AuditEvent] = []
        self.log_file = Path(log_file) if log_file else None
        self._event_counter = 0

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Initialized AuditLogger")

    def _generate_event_id(self) -> str:
        """Generate unique event ID.

        Returns:
            Event ID
        """
        self._event_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"event_{timestamp}_{self._event_counter:06d}"

    def log_event(
        self,
        event_type: EventType,
        user: str,
        description: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            event_type: Type of event
            user: User who triggered the event
            description: Event description
            **kwargs: Additional event properties

        Returns:
            Created audit event
        """
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.now(),
            user=user,
            description=description,
            **kwargs,
        )

        # Store event
        self.events.append(event)

        # Write to file if configured
        if self.log_file:
            self._write_to_file(event)

        LOGGER.info(f"Logged audit event: {event.event_id} ({event_type.value})")
        return event

    def _write_to_file(self, event: AuditEvent) -> None:
        """Write event to log file.

        Args:
            event: Audit event
        """
        with open(self.log_file, "a") as f:
            f.write(event.to_json(indent=None) + "\n")

    def get_events(
        self,
        event_type: EventType | None = None,
        user: str | None = None,
        model_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[AuditEvent]:
        """Get filtered audit events.

        Args:
            event_type: Filter by event type
            user: Filter by user
            model_name: Filter by model name
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Filtered events
        """
        events = self.events

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if user:
            events = [e for e in events if e.user == user]

        if model_name:
            events = [e for e in events if e.model_name == model_name]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events

    def get_event_by_id(self, event_id: str) -> AuditEvent | None:
        """Get event by ID.

        Args:
            event_id: Event ID

        Returns:
            Event or None if not found
        """
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None

    def get_lineage(
        self,
        model_name: str,
        model_version: str | None = None,
    ) -> list[AuditEvent]:
        """Get model lineage (all related events).

        Args:
            model_name: Model name
            model_version: Model version (optional)

        Returns:
            List of related events
        """
        events = [e for e in self.events if e.model_name == model_name]

        if model_version:
            events = [e for e in events if e.model_version == model_version]

        # Sort by timestamp
        return sorted(events, key=lambda e: e.timestamp)

    def generate_audit_report(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate audit report.

        Args:
            start_time: Report start time
            end_time: Report end time

        Returns:
            Audit report
        """
        events = self.get_events(start_time=start_time, end_time=end_time)

        # Count by event type
        event_counts: dict[str, int] = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Count by user
        user_counts: dict[str, int] = {}
        for event in events:
            user_counts[event.user] = user_counts.get(event.user, 0) + 1

        # Count by model
        model_counts: dict[str, int] = {}
        for event in events:
            if event.model_name:
                model_counts[event.model_name] = (
                    model_counts.get(event.model_name, 0) + 1
                )

        return {
            "total_events": len(events),
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "events_by_type": event_counts,
            "events_by_user": user_counts,
            "events_by_model": model_counts,
            "generated_at": datetime.now().isoformat(),
        }

    def export_to_json(self, filepath: Path | str) -> None:
        """Export all events to JSON file.

        Args:
            filepath: Path to export file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        events_data = [event.to_dict() for event in self.events]

        with open(filepath, "w") as f:
            json.dumps(events_data, f, indent=2)

        LOGGER.info(f"Exported {len(self.events)} events to {filepath}")

    def load_from_json(self, filepath: Path | str) -> None:
        """Load events from JSON file.

        Args:
            filepath: Path to import file
        """
        filepath = Path(filepath)

        with open(filepath) as f:
            events_data = json.load(f)

        for event_dict in events_data:
            # Convert timestamp back to datetime
            event_dict["timestamp"] = datetime.fromisoformat(event_dict["timestamp"])
            # Convert event type back to enum
            event_dict["event_type"] = EventType(event_dict["event_type"])

            event = AuditEvent(**event_dict)
            self.events.append(event)

        LOGGER.info(f"Loaded {len(events_data)} events from {filepath}")

    def clear_events(self, before: datetime | None = None) -> int:
        """Clear events from memory.

        Args:
            before: Clear events before this time (None for all)

        Returns:
            Number of events cleared
        """
        if before:
            original_count = len(self.events)
            self.events = [e for e in self.events if e.timestamp >= before]
            cleared = original_count - len(self.events)
        else:
            cleared = len(self.events)
            self.events.clear()

        LOGGER.info(f"Cleared {cleared} events")
        return cleared


def log_event(
    event_type: EventType,
    user: str,
    description: str,
    **kwargs: Any,
) -> AuditEvent:
    """Log audit event (convenience function).

    Args:
        event_type: Type of event
        user: User who triggered the event
        description: Event description
        **kwargs: Additional event properties

    Returns:
        Created audit event
    """
    logger = AuditLogger()
    return logger.log_event(event_type, user, description, **kwargs)
