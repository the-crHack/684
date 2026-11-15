"""Trigger-blind defenses for the small tabular classification experiment.

Neither entry point accepts feature names, attack targets, poison identities,
evaluation examples, or trigger metadata. The returned audit objects contain
only JSON-compatible values.
"""

from copy import deepcopy

import numpy as np
from scipy import sparse
from scipy.stats import fisher_exact


def _matrix(x, *, name):
    """Validate and materialize an experiment-sized numeric matrix."""
    a = x.toarray() if sparse.issparse(x) else np.asarray(x)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains nonfinite values")
    return a


def prune_unsupported_weights(model, trusted_x):
    """Zero input coefficients unsupported by the trusted clean distribution.

    For a linear classifier, a feature that is identically zero on trusted
    examples contributes nothing to their logits. Zeroing its coefficient
    therefore leaves trusted logits unchanged while removing any dependence
    on that feature elsewhere. This deliberately narrow defense cannot remove
    a backdoor expressed through features active on clean inputs.
    """
    trusted = _matrix(trusted_x, name="trusted_x")
    coefficients = np.asarray(model.coef_)
    if coefficients.ndim != 2 or coefficients.shape[1] != trusted.shape[1]:
        raise ValueError("model coefficient dimension does not match trusted_x")
    tolerance = 1e-12
    unsupported = np.max(np.abs(trusted), axis=0) <= tolerance
    defended = deepcopy(model)
    defended.coef_ = coefficients.copy()
    defended.coef_[:, unsupported] = 0.0
    audit = {
        "method": "trusted_clean_unsupported_input_weight_pruning",
        "zero_tolerance": tolerance,
        "trusted_examples": int(trusted.shape[0]),
        "total_features": int(trusted.shape[1]),
        "pruned_feature_count": int(unsupported.sum()),
        "pruned_feature_indices": np.flatnonzero(unsupported).tolist(),
        "changed_coefficient_count": int(np.count_nonzero(coefficients[:, unsupported])),
        "max_trusted_logit_change": float(
            np.max(np.abs(trusted @ (defended.coef_ - coefficients).T))
        ),
    }
    return defended, audit


def detect_poisoned_samples(x, y):
    """Flag rare binary features with nearly pure, significant label association.

    The settings are fixed before evaluation: support in [5, floor(0.10*n)],
    label purity >= .98, and one-sided Fisher enrichment p <= .01 after
    Bonferroni correction over ALL binary features times observed classes.
    Rows activating any flagged feature are removed. Poison identities may
    only be used by a caller afterward to score precision and recall.
    """
    features = _matrix(x, name="x")
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("y must contain one label per row")
    classes, inverse = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("detection requires at least two observed classes")
    n, d = features.shape
    binary = np.all((features == 0) | (features == 1), axis=0)
    binary_indices = np.flatnonzero(binary)
    number_tests = int(len(binary_indices) * len(classes))
    min_support, max_support, min_purity, alpha = 5, int(np.floor(0.10 * n)), 0.98, 0.01
    removed = np.zeros(n, dtype=bool)
    flagged = []
    tested = []
    for j in binary_indices:
        active = features[:, j] == 1
        support = int(active.sum())
        if not min_support <= support <= max_support:
            continue
        for k, label in enumerate(classes):
            is_class = inverse == k
            a = int(np.count_nonzero(active & is_class))
            b = support - a
            c = int(is_class.sum()) - a
            remaining = n - a - b - c
            p = float(fisher_exact([[a, b], [c, remaining]], alternative="greater").pvalue)
            adjusted = min(1.0, p * number_tests)
            purity = a / support
            row = {
                "feature_index": int(j),
                "enriched_label": label.item() if isinstance(label, np.generic) else label,
                "support": support,
                "class_support": a,
                "purity": float(purity),
                "p_value": p,
                "bonferroni_p_value": adjusted,
            }
            tested.append(row)
            if purity >= min_purity and adjusted <= alpha:
                flagged.append(row.copy())
                removed |= active
    audit = {
        "method": "rare_binary_feature_label_enrichment",
        "training_examples": int(n),
        "total_features": int(d),
        "binary_feature_count": int(len(binary_indices)),
        "bonferroni_number_tests": number_tests,
        "min_support": min_support,
        "max_support_fraction": 0.10,
        "max_support": max_support,
        "min_purity": min_purity,
        "familywise_alpha": alpha,
        "flagged_features": flagged,
        "tested_associations": tested,
        "removed_count": int(removed.sum()),
    }
    return removed, audit
