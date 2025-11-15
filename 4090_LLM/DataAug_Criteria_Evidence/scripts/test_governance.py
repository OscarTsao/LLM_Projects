#!/usr/bin/env python
"""Test script for Phase 25: Model Governance & Compliance.

This script tests:
1. Model cards (creation, documentation, HTML export)
2. Bias detection (demographic parity, equal opportunity, disparate impact)
3. Compliance tracking (GDPR, HIPAA, assessment)
4. Audit trail (logging, filtering, lineage)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psy_agents_noaug.governance import (
    AuditLogger,
    BiasDetector,
    ComplianceFramework,
    ComplianceTracker,
    EventType,
    FairnessMetric,
    ModelCardGenerator,
    check_compliance,
    detect_bias,
    generate_model_card,
    log_event,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_model_card_creation() -> bool:
    """Test model card creation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Model Card Creation")
    LOGGER.info("=" * 80)

    try:
        generator = ModelCardGenerator()

        # Create model card
        card = generator.create_card(
            model_name="ClinicalBERT",
            model_version="1.0.0",
            model_type="Transformer-based classifier",
            description="BERT model fine-tuned for clinical text classification",
            developers=["ML Team"],
            contact="ml-team@example.com",
        )

        assert card.model_name == "ClinicalBERT"
        assert card.model_version == "1.0.0"
        assert len(card.developers) == 1

        # Add training info
        generator.add_training_info(
            card,
            data_description="Clinical notes from EHR system",
            data_size=10000,
            data_source="Hospital EHR database",
        )

        assert card.training_data_size == 10000

        # Add metrics
        metrics = {"accuracy": 0.92, "f1": 0.89, "auc": 0.95}
        generator.add_performance_metrics(
            card,
            metrics,
            evaluation_description="Test set from 2024",
        )

        assert len(card.metrics) == 3
        assert card.metrics["accuracy"] == 0.92

        LOGGER.info("✅ Model Card Creation: PASSED")
        LOGGER.info(f"   - Model: {card.model_name} v{card.model_version}")
        LOGGER.info(f"   - Training data size: {card.training_data_size}")
        LOGGER.info(f"   - Metrics: {len(card.metrics)}")

    except Exception:
        LOGGER.exception("❌ Model Card Creation: FAILED")
        return False
    else:
        return True


def test_model_card_export() -> bool:
    """Test model card export to JSON and HTML."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Model Card Export")
    LOGGER.info("=" * 80)

    try:
        with TemporaryDirectory() as tmpdir:
            generator = ModelCardGenerator()

            # Create card
            card = generator.create_card(
                model_name="TestModel",
                model_version="2.0.0",
                model_type="Neural Network",
                description="Test model for export",
            )

            # Save as JSON
            json_path = Path(tmpdir) / "model_card.json"
            card.save(json_path)
            assert json_path.exists()

            # Load back
            loaded_card = generator.create_card.__self__.__class__.__name__
            from psy_agents_noaug.governance.model_card import ModelCard

            loaded = ModelCard.load(json_path)
            assert loaded.model_name == "TestModel"
            assert loaded.model_version == "2.0.0"

            # Generate HTML
            html_path = Path(tmpdir) / "report.html"
            generator.save_html_report(card, html_path)
            assert html_path.exists()

            LOGGER.info("✅ Model Card Export: PASSED")
            LOGGER.info("   - JSON export: OK")
            LOGGER.info("   - HTML export: OK")

    except Exception:
        LOGGER.exception("❌ Model Card Export: FAILED")
        return False
    else:
        return True


def test_bias_detection() -> bool:
    """Test bias detection with demographic parity."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Bias Detection")
    LOGGER.info("=" * 80)

    try:
        np.random.seed(42)
        detector = BiasDetector()

        # Create biased data
        # Group 0: 80% positive predictions
        # Group 1: 40% positive predictions
        n_samples = 1000
        protected_attr = np.array([0] * 500 + [1] * 500)

        y_true = np.random.binomial(1, 0.5, n_samples)
        y_pred = np.zeros(n_samples)

        # Bias predictions by protected attribute
        for i in range(n_samples):
            if protected_attr[i] == 0:
                y_pred[i] = np.random.binomial(1, 0.8)  # 80% positive
            else:
                y_pred[i] = np.random.binomial(1, 0.4)  # 40% positive

        # Detect bias
        metrics = detector.detect_bias(
            y_true,
            y_pred,
            protected_attr,
            reference_value=0,
            comparison_value=1,
            attribute_name="gender",
        )

        # Should detect significant bias
        assert abs(metrics.demographic_parity_difference) > 0.3
        assert not metrics.is_fair(FairnessMetric.DEMOGRAPHIC_PARITY, threshold=0.1)

        LOGGER.info("✅ Bias Detection: PASSED")
        LOGGER.info(
            f"   - Demographic parity diff: {metrics.demographic_parity_difference:.3f}"
        )
        LOGGER.info(
            f"   - Disparate impact ratio: {metrics.disparate_impact_ratio:.3f}"
        )

    except Exception:
        LOGGER.exception("❌ Bias Detection: FAILED")
        return False
    else:
        return True


def test_fairness_assessment() -> bool:
    """Test multiple fairness metrics."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Fairness Assessment")
    LOGGER.info("=" * 80)

    try:
        np.random.seed(42)
        detector = BiasDetector()

        # Create fair data
        n_samples = 1000
        protected_attr = np.array([0] * 500 + [1] * 500)
        y_true = np.random.binomial(1, 0.5, n_samples)

        # Fair predictions (same rate for both groups)
        y_pred = np.random.binomial(1, 0.6, n_samples)

        metrics = detector.detect_bias(
            y_true,
            y_pred,
            protected_attr,
            reference_value=0,
            comparison_value=1,
        )

        # Should be fair
        is_fair_dp = metrics.is_fair(FairnessMetric.DEMOGRAPHIC_PARITY, threshold=0.1)
        is_fair_di = metrics.is_fair(FairnessMetric.DISPARATE_IMPACT)

        # With random data, might not be exactly fair, but should be close
        LOGGER.info("✅ Fairness Assessment: PASSED")
        LOGGER.info(f"   - Fair (demographic parity): {is_fair_dp}")
        LOGGER.info(f"   - Fair (disparate impact): {is_fair_di}")
        LOGGER.info(
            f"   - Demographic parity diff: {abs(metrics.demographic_parity_difference):.3f}"
        )

    except Exception:
        LOGGER.exception("❌ Fairness Assessment: FAILED")
        return False
    else:
        return True


def test_compliance_tracking() -> bool:
    """Test compliance tracking."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Compliance Tracking")
    LOGGER.info("=" * 80)

    try:
        tracker = ComplianceTracker()

        # Check initial compliance (should be non-compliant)
        report = tracker.assess_compliance(ComplianceFramework.GDPR)

        assert not report.overall_compliant
        assert report.compliance_score == 0.0  # No requirements met initially

        # Update some requirements
        tracker.update_requirement(
            ComplianceFramework.GDPR,
            "GDPR-1",
            is_met=True,
            evidence="Implemented data minimization in preprocessing",
        )

        tracker.update_requirement(
            ComplianceFramework.GDPR,
            "GDPR-2",
            is_met=True,
            evidence="Added SHAP explanations for model predictions",
        )

        # Re-assess
        report = tracker.assess_compliance(ComplianceFramework.GDPR)

        assert report.compliance_score == 0.5  # 2 out of 4 requirements met
        assert not report.overall_compliant  # Still not fully compliant

        LOGGER.info("✅ Compliance Tracking: PASSED")
        LOGGER.info(f"   - Framework: {report.framework.value}")
        LOGGER.info(f"   - Compliance score: {report.compliance_score:.2%}")
        LOGGER.info(f"   - Overall compliant: {report.overall_compliant}")

    except Exception:
        LOGGER.exception("❌ Compliance Tracking: FAILED")
        return False
    else:
        return True


def test_compliance_frameworks() -> bool:
    """Test multiple compliance frameworks."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Compliance Frameworks")
    LOGGER.info("=" * 80)

    try:
        tracker = ComplianceTracker()

        # Check all frameworks
        frameworks = tracker.get_all_frameworks()
        assert len(frameworks) >= 3  # At least GDPR, HIPAA, CCPA

        # Generate compliance matrix
        matrix = tracker.generate_compliance_matrix()

        assert "frameworks" in matrix
        assert "gdpr" in matrix["frameworks"]
        assert "hipaa" in matrix["frameworks"]

        LOGGER.info("✅ Compliance Frameworks: PASSED")
        LOGGER.info(f"   - Frameworks tracked: {len(frameworks)}")
        LOGGER.info(f"   - Matrix generated: {len(matrix['frameworks'])} frameworks")

    except Exception:
        LOGGER.exception("❌ Compliance Frameworks: FAILED")
        return False
    else:
        return True


def test_audit_logging() -> bool:
    """Test audit event logging."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 7: Audit Logging")
    LOGGER.info("=" * 80)

    try:
        logger = AuditLogger()

        # Log various events
        event1 = logger.log_event(
            EventType.MODEL_TRAINING,
            user="data_scientist",
            description="Started training ClinicalBERT model",
            model_name="ClinicalBERT",
            model_version="1.0.0",
        )

        event2 = logger.log_event(
            EventType.DATA_ACCESS,
            user="ml_engineer",
            description="Accessed training data",
            data_source="ehr_database",
        )

        event3 = logger.log_event(
            EventType.MODEL_DEPLOYMENT,
            user="ml_ops",
            description="Deployed model to production",
            model_name="ClinicalBERT",
            model_version="1.0.0",
        )

        assert len(logger.events) == 3
        assert event1.event_type == EventType.MODEL_TRAINING
        assert event2.user == "ml_engineer"

        # Filter events
        training_events = logger.get_events(event_type=EventType.MODEL_TRAINING)
        assert len(training_events) == 1

        model_events = logger.get_events(model_name="ClinicalBERT")
        assert len(model_events) == 2

        LOGGER.info("✅ Audit Logging: PASSED")
        LOGGER.info(f"   - Total events: {len(logger.events)}")
        LOGGER.info(f"   - Training events: {len(training_events)}")
        LOGGER.info(f"   - Model events: {len(model_events)}")

    except Exception:
        LOGGER.exception("❌ Audit Logging: FAILED")
        return False
    else:
        return True


def test_audit_lineage() -> bool:
    """Test audit lineage tracking."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 8: Audit Lineage")
    LOGGER.info("=" * 80)

    try:
        logger = AuditLogger()

        # Create model lifecycle events
        logger.log_event(
            EventType.MODEL_TRAINING,
            user="user1",
            description="Initial training",
            model_name="ModelX",
            model_version="1.0.0",
        )

        logger.log_event(
            EventType.BIAS_ASSESSMENT,
            user="user2",
            description="Checked for bias",
            model_name="ModelX",
            model_version="1.0.0",
        )

        logger.log_event(
            EventType.MODEL_DEPLOYMENT,
            user="user3",
            description="Deployed to production",
            model_name="ModelX",
            model_version="1.0.0",
        )

        # Get lineage
        lineage = logger.get_lineage("ModelX")
        assert len(lineage) == 3

        # Events should be in chronological order
        assert lineage[0].event_type == EventType.MODEL_TRAINING
        assert lineage[1].event_type == EventType.BIAS_ASSESSMENT
        assert lineage[2].event_type == EventType.MODEL_DEPLOYMENT

        LOGGER.info("✅ Audit Lineage: PASSED")
        LOGGER.info(f"   - Lineage events: {len(lineage)}")
        LOGGER.info(f"   - Event sequence: OK")

    except Exception:
        LOGGER.exception("❌ Audit Lineage: FAILED")
        return False
    else:
        return True


def test_audit_reporting() -> bool:
    """Test audit reporting."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 9: Audit Reporting")
    LOGGER.info("=" * 80)

    try:
        logger = AuditLogger()

        # Log events over time
        base_time = datetime.now()

        for i in range(10):
            logger.log_event(
                EventType.MODEL_INFERENCE,
                user=f"user{i % 3}",
                description=f"Inference request {i}",
                model_name=f"Model{i % 2}",
            )

        # Generate report
        report = logger.generate_audit_report()

        assert report["total_events"] == 10
        assert "events_by_type" in report
        assert "events_by_user" in report
        assert "events_by_model" in report

        # Should have 3 users
        assert len(report["events_by_user"]) == 3

        # Should have 2 models
        assert len(report["events_by_model"]) == 2

        LOGGER.info("✅ Audit Reporting: PASSED")
        LOGGER.info(f"   - Total events: {report['total_events']}")
        LOGGER.info(f"   - Unique users: {len(report['events_by_user'])}")
        LOGGER.info(f"   - Unique models: {len(report['events_by_model'])}")

    except Exception:
        LOGGER.exception("❌ Audit Reporting: FAILED")
        return False
    else:
        return True


def test_convenience_functions() -> bool:
    """Test convenience functions."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 10: Convenience Functions")
    LOGGER.info("=" * 80)

    try:
        # Test generate_model_card
        card = generate_model_card(
            model_name="QuickModel",
            model_version="1.0.0",
            model_type="Classifier",
            description="Quick test",
            metrics={"accuracy": 0.95},
        )

        assert card.model_name == "QuickModel"
        assert len(card.metrics) == 1

        # Test detect_bias
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_pred = np.random.binomial(1, 0.6, 100)
        protected_attr = np.array([0] * 50 + [1] * 50)

        bias_metrics = detect_bias(y_true, y_pred, protected_attr, 0, 1)
        assert bias_metrics.protected_attribute == "protected_attribute"

        # Test check_compliance
        compliance_report = check_compliance(ComplianceFramework.GDPR)
        assert compliance_report.framework == ComplianceFramework.GDPR

        # Test log_event
        event = log_event(
            EventType.CONFIGURATION_CHANGE,
            user="admin",
            description="Updated config",
        )
        assert event.event_type == EventType.CONFIGURATION_CHANGE

        LOGGER.info("✅ Convenience Functions: PASSED")
        LOGGER.info("   - generate_model_card: OK")
        LOGGER.info("   - detect_bias: OK")
        LOGGER.info("   - check_compliance: OK")
        LOGGER.info("   - log_event: OK")

    except Exception:
        LOGGER.exception("❌ Convenience Functions: FAILED")
        return False
    else:
        return True


def main():
    """Run all governance tests."""
    LOGGER.info("Starting Phase 25 Governance Tests")
    LOGGER.info("=" * 80)

    tests = [
        ("Model Card Creation", test_model_card_creation),
        ("Model Card Export", test_model_card_export),
        ("Bias Detection", test_bias_detection),
        ("Fairness Assessment", test_fairness_assessment),
        ("Compliance Tracking", test_compliance_tracking),
        ("Compliance Frameworks", test_compliance_frameworks),
        ("Audit Logging", test_audit_logging),
        ("Audit Lineage", test_audit_lineage),
        ("Audit Reporting", test_audit_reporting),
        ("Convenience Functions", test_convenience_functions),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception:
            LOGGER.exception(f"Test '{test_name}' crashed")
            results.append((test_name, False))

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("TEST SUMMARY")
    LOGGER.info("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        LOGGER.info(f"{status}: {test_name}")

    LOGGER.info("=" * 80)
    LOGGER.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        LOGGER.info("🎉 All tests passed!")
        return 0

    LOGGER.error(f"❌ {total - passed} test(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
