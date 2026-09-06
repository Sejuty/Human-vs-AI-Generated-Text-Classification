"""Criterion importance, recorded as a linguistic judgement."""

import numpy as np

# Five-level importance scale, as TFNs on a 1-9 range.
IMPORTANCE_SCALE = {
    "very low": (1.0, 1.0, 3.0),
    "low": (1.0, 3.0, 5.0),
    "medium": (3.0, 5.0, 7.0),
    "high": (5.0, 7.0, 9.0),
    "very high": (7.0, 9.0, 9.0),
}

# The judgement itself. Order matches CRITERIA in benchmark.py.
IMPORTANCE = {
    "accuracy": (
        "very high",
        "the headline measure of the task",
    ),
    "f1": (
        "high",
        "balances the two error directions, which accuracy alone hides",
    ),
    "latency_ms": (
        "medium",
        "experienced directly every time the classifier is used",
    ),
    "size_mb": (
        "low",
        "nothing here targets a memory-constrained device, so size is "
        "an inconvenience rather than a constraint",
    ),
    "lime_seconds": (
        "medium",
        "in a system built to explain itself, waiting for the explanation "
        "costs the user as much as waiting for the prediction",
    ),
    "content_share": (
        "high",
        "an explanation made of function words is produced as quickly as any "
        "other and tells the reader nothing, so speed alone cannot stand in "
        "for explanation quality",
    ),
}


def fuzzy_weights(criteria):
    """TFN weights for the given criteria, as an (n, 3) array."""
    return np.array([IMPORTANCE_SCALE[IMPORTANCE[c][0]] for c in criteria],
                    dtype=float)


def crisp_weights(criteria):
    """Scalar weights for the given criteria, summing to 1."""
    centroids = fuzzy_weights(criteria).mean(axis=1)
    return centroids / centroids.sum()


def describe(criteria):
    """The judgement as a table, for the console and the report."""
    fuzzy = fuzzy_weights(criteria)
    crisp = crisp_weights(criteria)
    width = max(len(c) for c in criteria)

    lines = [
        f"{'criterion'.ljust(width)}  {'importance':<11}  {'TFN':<16}  weight",
        f"{'-' * width}  {'-' * 11}  {'-' * 16}  ------",
    ]
    for i, c in enumerate(criteria):
        level = IMPORTANCE[c][0]
        tfn = f"({fuzzy[i][0]:.0f}, {fuzzy[i][1]:.0f}, {fuzzy[i][2]:.0f})"
        lines.append(f"{c.ljust(width)}  {level:<11}  {tfn:<16}  {crisp[i]:.3f}")

    lines.append("")
    lines.append("Rationale:")
    for c in criteria:
        lines.append(f"  {c}: {IMPORTANCE[c][1]}")
    return "\n".join(lines)


def _self_test():
    criteria = ["accuracy", "f1", "latency_ms", "size_mb", "lime_seconds"]

    fuzzy = fuzzy_weights(criteria)
    assert fuzzy.shape == (5, 3), fuzzy.shape
    for l, m, u in fuzzy:
        assert l <= m <= u, f"invalid TFN weight: {(l, m, u)}"
    print("TFN weights          shape (5, 3), all valid triangles          ok")

    crisp = crisp_weights(criteria)
    assert np.isclose(crisp.sum(), 1.0), crisp.sum()
    print(f"crisp weights        sum to 1.0                                ok")

    expected = np.array([0.294, 0.247, 0.176, 0.106, 0.176])
    assert np.allclose(crisp, expected, atol=5e-4), np.round(crisp, 4)
    print("centroids            " + " / ".join(f"{w:.3f}" for w in crisp) + "   ok")

    # Importance ordering must survive defuzzification: a criterion rated
    # higher must not end up with a smaller crisp weight.
    order = ["very low", "low", "medium", "high", "very high"]
    for c1 in criteria:
        for c2 in criteria:
            r1, r2 = IMPORTANCE[c1][0], IMPORTANCE[c2][0]
            if order.index(r1) > order.index(r2):
                i, j = criteria.index(c1), criteria.index(c2)
                assert crisp[i] > crisp[j], f"{c1} rated above {c2} but weighs less"
    print("consistency          ratings and crisp weights agree in order   ok")

    print("\nAll weights.py self-tests passed.\n")
    print(describe(criteria))


if __name__ == "__main__":
    _self_test()
