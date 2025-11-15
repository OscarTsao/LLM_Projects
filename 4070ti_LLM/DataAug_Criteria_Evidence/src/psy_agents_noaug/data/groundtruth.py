"""Ground truth generation with STRICT validation rules.

CRITICAL REQUIREMENTS (HIGHEST PRIORITY):
1. Criteria labels come ONLY from the 'status' field
2. Evidence comes ONLY from the 'cases' field
3. Assertions FAIL if any other fields are used
"""

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _assert_field_usage(field_name: str, expected_field: str, operation: str):
    """Assert that the correct field is being used for an operation.

    Args:
        field_name: The field being used
        expected_field: The expected field name
        operation: Description of the operation (for error message)

    Raises:
        AssertionError: If wrong field is used
    """
    assert field_name == expected_field, (
        f"STRICT VALIDATION FAILURE: {operation} must use '{expected_field}' field, "
        f"but '{field_name}' was used. This violates the core data pipeline rules."
    )


def load_field_map(config_path: Path) -> dict[str, Any]:
    """Load field mapping configuration.

    Args:
        config_path: Path to field_map.yaml

    Returns:
        Dictionary with field mappings
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def normalize_status_value(value: Any, status_map: dict[str, list]) -> int | None:
    """Normalize status value to binary (0 or 1).

    Args:
        value: Raw status value
        status_map: Mapping from status_values in field_map.yaml

    Returns:
        0 for negative, 1 for positive, None for invalid
    """
    if pd.isna(value):
        return None

    # Check positive values
    if value in status_map.get("positive", []):
        return 1

    # Check negative values
    if value in status_map.get("negative", []):
        return 0

    # Check string representation
    value_str = str(value).strip().lower()
    if value_str in [str(v).strip().lower() for v in status_map.get("positive", [])]:
        return 1
    if value_str in [str(v).strip().lower() for v in status_map.get("negative", [])]:
        return 0

    return None


def parse_cases_field(cases_value: Any) -> list[dict[str, Any]]:
    """Parse cases field to extract evidence spans.

    Args:
        cases_value: Raw cases value (can be list, stringified JSON, etc.)

    Returns:
        List of evidence dictionaries with start_char, end_char, sentence_id, text
    """
    # Handle None explicitly
    if cases_value is None:
        return []

    # Handle pd.NA, np.nan, float('nan') - use try/except to avoid ValueError on arrays
    try:
        if pd.isna(cases_value):
            return []
    except (ValueError, TypeError):
        # pd.isna() raises ValueError for arrays/lists
        pass

    # If already a list
    if isinstance(cases_value, list):
        # Empty list check
        if len(cases_value) == 0:
            return []
        return cases_value

    # If stringified JSON
    if isinstance(cases_value, str):
        try:
            parsed = json.loads(cases_value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            # Try ast.literal_eval
            try:
                parsed = ast.literal_eval(cases_value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass

    # Single value - wrap in list
    if isinstance(cases_value, dict):
        return [cases_value]

    return []


def create_criteria_groundtruth(
    annotations: pd.DataFrame,
    posts: pd.DataFrame,
    field_map: dict[str, Any],
    valid_criterion_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Create criteria groundtruth from annotations.

    STRICT RULE: Uses ONLY the status field for labels.

    Args:
        annotations: Annotations DataFrame
        posts: Posts DataFrame
        field_map: Field mapping configuration
        valid_criterion_ids: Set of valid criterion IDs (optional)

    Returns:
        DataFrame with columns: post_id, criterion_id, status, label

    Raises:
        AssertionError: If status field is not used
    """
    # STRICT VALIDATION: Assert we're using the correct field
    status_field = field_map["annotations"]["status"]
    _assert_field_usage(status_field, "status", "Criteria labels")

    # Extract required columns
    post_id_field = field_map["annotations"]["post_id"]
    criterion_id_field = field_map["annotations"]["criterion_id"]

    # Validate columns exist
    required_cols = {post_id_field, criterion_id_field, status_field}
    missing = required_cols - set(annotations.columns)
    if missing:
        raise ValueError(f"Missing required columns in annotations: {missing}")

    # Create working DataFrame
    df = annotations[[post_id_field, criterion_id_field, status_field]].copy()
    df.columns = ["post_id", "criterion_id", "status"]  # type: ignore[assignment]

    # Normalize status values to binary labels
    status_map = field_map.get("status_values", {})
    df["label"] = df["status"].apply(lambda x: normalize_status_value(x, status_map))

    # Drop rows with invalid status values
    invalid_mask = df["label"].isna()
    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        print(f"WARNING: Dropping {invalid_count} rows with invalid status values")
        df = df[~invalid_mask].copy()

    # Validate criterion IDs
    if valid_criterion_ids:
        invalid_ids = set(df["criterion_id"].unique()) - valid_criterion_ids
        if invalid_ids:
            if field_map.get("validation", {}).get(
                "fail_on_invalid_criterion_id", True
            ):
                raise ValueError(f"Invalid criterion IDs found: {invalid_ids}")
            print(f"WARNING: Invalid criterion IDs found: {invalid_ids}")
            df = df[df["criterion_id"].isin(valid_criterion_ids)].copy()

    # Validate post IDs exist in posts
    post_ids_set = set(posts[field_map["posts"]["post_id"]])
    missing_posts = set(df["post_id"].unique()) - post_ids_set
    if missing_posts:
        if field_map.get("validation", {}).get("fail_on_missing_post_id", True):
            raise ValueError(
                f"Post IDs in annotations not found in posts: {missing_posts}"
            )
        print(
            f"WARNING: Dropping {len(missing_posts)} annotations with missing post_ids"
        )
        df = df[df["post_id"].isin(post_ids_set)].copy()

    # Handle duplicates
    if field_map.get("validation", {}).get("drop_duplicates", True):
        dup_mask = df.duplicated(subset=["post_id", "criterion_id"], keep="first")
        if dup_mask.any():
            print(f"WARNING: Dropping {dup_mask.sum()} duplicate annotations")
            df = df[~dup_mask].copy()

    # Reset index
    df = df.reset_index(drop=True)

    return df[["post_id", "criterion_id", "status", "label"]]


def create_evidence_groundtruth(
    annotations: pd.DataFrame,
    posts: pd.DataFrame,
    field_map: dict[str, Any],
    valid_criterion_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Create evidence groundtruth from annotations.

    STRICT RULE: Uses ONLY the cases field for evidence.

    Args:
        annotations: Annotations DataFrame
        posts: Posts DataFrame
        field_map: Field mapping configuration
        valid_criterion_ids: Set of valid criterion IDs (optional)

    Returns:
        DataFrame with columns: post_id, criterion_id, case_id, evidence_text,
                                start_char, end_char, sentence_id

    Raises:
        AssertionError: If cases field is not used
    """
    # STRICT VALIDATION: Assert we're using the correct field
    cases_field = field_map["annotations"]["cases"]
    _assert_field_usage(cases_field, "cases", "Evidence extraction")

    # Extract required columns
    post_id_field = field_map["annotations"]["post_id"]
    criterion_id_field = field_map["annotations"]["criterion_id"]

    # Validate columns exist
    required_cols = {post_id_field, criterion_id_field, cases_field}
    missing = required_cols - set(annotations.columns)
    if missing:
        raise ValueError(f"Missing required columns in annotations: {missing}")

    # Create working DataFrame
    df = annotations[[post_id_field, criterion_id_field, cases_field]].copy()
    df.columns = ["post_id", "criterion_id", "cases"]  # type: ignore[assignment]

    # Parse cases field and explode
    df["cases_parsed"] = df["cases"].apply(parse_cases_field)

    # Explode per‑annotation cases into distinct rows (one row per span)
    df = df.explode("cases_parsed", ignore_index=True)

    # Drop rows with no cases
    df = df[df["cases_parsed"].notna()].copy()
    df = df[df["cases_parsed"].apply(lambda x: bool(x))].copy()

    # Extract evidence fields
    cases_config = field_map.get("cases_structure", {}).get("fields", {})

    df["evidence_text"] = df["cases_parsed"].apply(
        lambda x: (
            x.get(cases_config.get("text", "text"), "") if isinstance(x, dict) else ""
        )
    )
    df["start_char"] = df["cases_parsed"].apply(
        lambda x: (
            x.get(cases_config.get("start_char", "start_char"), None)
            if isinstance(x, dict)
            else None
        )
    )
    df["end_char"] = df["cases_parsed"].apply(
        lambda x: (
            x.get(cases_config.get("end_char", "end_char"), None)
            if isinstance(x, dict)
            else None
        )
    )
    df["sentence_id"] = df["cases_parsed"].apply(
        lambda x: (
            x.get(cases_config.get("sentence_id", "sentence_id"), None)
            if isinstance(x, dict)
            else None
        )
    )

    # Add case_id (0..N-1) within each (post_id, criterion_id) to track spans
    df["case_id"] = df.groupby(["post_id", "criterion_id"]).cumcount()

    # Validate criterion IDs
    if valid_criterion_ids:
        invalid_ids = set(df["criterion_id"].unique()) - valid_criterion_ids
        if invalid_ids:
            if field_map.get("validation", {}).get(
                "fail_on_invalid_criterion_id", True
            ):
                raise ValueError(f"Invalid criterion IDs found: {invalid_ids}")
            print(f"WARNING: Invalid criterion IDs found: {invalid_ids}")
            df = df[df["criterion_id"].isin(valid_criterion_ids)].copy()

    # Validate post IDs exist in posts
    post_ids_set = set(posts[field_map["posts"]["post_id"]])
    missing_posts = set(df["post_id"].unique()) - post_ids_set
    if missing_posts:
        if field_map.get("validation", {}).get("fail_on_missing_post_id", True):
            raise ValueError(
                f"Post IDs in annotations not found in posts: {missing_posts}"
            )
        print("WARNING: Dropping annotations with missing post_ids")
        df = df[df["post_id"].isin(post_ids_set)].copy()

    # Drop cases_parsed column
    df = df.drop("cases_parsed", axis=1)

    # Reset index
    df = df.reset_index(drop=True)

    return df[
        [
            "post_id",
            "criterion_id",
            "case_id",
            "evidence_text",
            "start_char",
            "end_char",
            "sentence_id",
        ]
    ]


def validate_strict_separation(
    criteria_df: pd.DataFrame, evidence_df: pd.DataFrame, field_map: dict[str, Any]
):
    """Validate that criteria and evidence use separate fields.

    Args:
        criteria_df: Criteria groundtruth DataFrame
        evidence_df: Evidence groundtruth DataFrame
        field_map: Field mapping configuration

    Raises:
        AssertionError: If field separation is violated
    """
    # Check that criteria has 'status' column
    assert "status" in criteria_df.columns, "Criteria must have 'status' column"

    # Check that evidence does NOT have 'status' column
    assert (
        "status" not in evidence_df.columns
    ), "STRICT VIOLATION: Evidence groundtruth must NOT contain 'status' field"

    # Check that evidence has evidence-specific columns
    assert (
        "evidence_text" in evidence_df.columns
    ), "Evidence must have 'evidence_text' column"

    # Check that criteria does NOT have evidence-specific columns
    evidence_specific = {
        "evidence_text",
        "case_id",
        "start_char",
        "end_char",
        "sentence_id",
    }
    criteria_specific = set(criteria_df.columns) & evidence_specific
    assert (
        not criteria_specific
    ), f"STRICT VIOLATION: Criteria groundtruth must NOT contain evidence fields: {criteria_specific}"

    print("VALIDATION PASSED: Strict field separation maintained")
    print(f"  Criteria columns: {list(criteria_df.columns)}")
    print(f"  Evidence columns: {list(evidence_df.columns)}")


class GroundTruthValidator:
    """Validates ground truth data against STRICT rules."""

    def __init__(self, field_map: dict[str, Any], valid_criterion_ids: set[str]):
        """Initialize validator.

        Args:
            field_map: Field mapping configuration
            valid_criterion_ids: Set of valid DSM criterion IDs
        """
        self.field_map = field_map
        self.valid_criterion_ids = valid_criterion_ids

    def validate_criteria_groundtruth(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Validate criteria groundtruth.

        Args:
            df: Criteria groundtruth DataFrame

        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []

        # Check required columns
        required = {"post_id", "criterion_id", "status", "label"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")

        # Check for evidence contamination
        evidence_fields = {"evidence_text", "case_id", "cases"}
        contamination = evidence_fields & set(df.columns)
        if contamination:
            errors.append(
                f"STRICT VIOLATION: Evidence fields found in criteria groundtruth: {contamination}"
            )

        # Validate criterion IDs
        invalid_ids = set(df["criterion_id"].unique()) - self.valid_criterion_ids
        if invalid_ids:
            errors.append(f"Invalid criterion IDs: {invalid_ids}")

        # Check for nulls in critical columns
        for col in ["post_id", "criterion_id", "label"]:
            if col in df.columns and df[col].isna().any():
                null_count = df[col].isna().sum()
                warnings.append(f"{null_count} null values in {col}")

        return {"errors": errors, "warnings": warnings}

    def validate_evidence_groundtruth(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Validate evidence groundtruth.

        Args:
            df: Evidence groundtruth DataFrame

        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []

        # Check required columns
        required = {"post_id", "criterion_id", "case_id", "evidence_text"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")

        # Check for criteria contamination
        if "status" in df.columns or "label" in df.columns:
            errors.append(
                "STRICT VIOLATION: Criteria fields (status/label) found in evidence groundtruth"
            )

        # Validate criterion IDs
        invalid_ids = set(df["criterion_id"].unique()) - self.valid_criterion_ids
        if invalid_ids:
            errors.append(f"Invalid criterion IDs: {invalid_ids}")

        # Check for empty evidence text
        if "evidence_text" in df.columns:
            empty_count = (
                df["evidence_text"].isna() | (df["evidence_text"] == "")
            ).sum()
            if empty_count > 0:
                warnings.append(f"{empty_count} rows with empty evidence_text")

        return {"errors": errors, "warnings": warnings}
