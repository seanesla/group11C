from __future__ import annotations

import logging
from typing import List

from models.model_utils import load_latest_models
from services.result_types import ModelLoadResult, ServiceResult


logger = logging.getLogger(__name__)


def load_and_validate_models(
    expected_feature_names: List[str],
) -> ServiceResult[ModelLoadResult]:
    """Load latest classifier/regressor models and validate feature schema.

    The returned :class:`ModelLoadResult` may contain ``None`` values when
    models are missing or incompatible with the deployed feature pipeline.
    """
    try:
        classifier, regressor = load_latest_models()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load ML models")
        return ServiceResult.fail(f"Failed to load ML models: {exc}")

    # No models at all – degrade gracefully and let callers decide how to handle.
    if classifier is None and regressor is None:
        return ServiceResult.ok(ModelLoadResult(classifier=None, regressor=None))

    warnings: List[str] = []

    # Require both models for predictions
    if classifier is None or regressor is None:
        warnings.append(
            "Only one of classifier/regressor loaded; ML predictions are disabled."
        )
        return ServiceResult.ok(
            ModelLoadResult(classifier=None, regressor=None),
            warnings=warnings,
        )

    clf_features = getattr(classifier, "feature_names", None)
    reg_features = getattr(regressor, "feature_names", None)

    if clf_features != expected_feature_names or reg_features != expected_feature_names:
        warnings.append(
            "ML models were trained with a feature schema that does not match the "
            "deployed US prediction pipeline. To enable ML predictions in the app, "
            "retrain models with the core-parameter feature set "
            "(`train_models.py --core-params-only`) and redeploy."
        )
        # Expose a clear warning and disable models for the UI
        return ServiceResult.ok(
            ModelLoadResult(classifier=None, regressor=None),
            warnings=warnings,
        )

    return ServiceResult.ok(
        ModelLoadResult(classifier=classifier, regressor=regressor),
        warnings=warnings,
    )

