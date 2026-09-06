"""TOPSIS and fuzzy TOPSIS."""

import numpy as np

BENEFIT = "benefit"
COST = "cost"


def _check_directions(directions, n_criteria):
    if len(directions) != n_criteria:
        raise ValueError(f"got {len(directions)} directions for {n_criteria} criteria")
    bad = [d for d in directions if d not in (BENEFIT, COST)]
    if bad:
        raise ValueError(f"directions must be {BENEFIT!r} or {COST!r}, got {bad!r}")


def topsis(matrix, weights, directions):
    """Crisp TOPSIS."""
    x = np.asarray(matrix, dtype=float)
    m, n = x.shape
    _check_directions(directions, n)

    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("criterion weights must sum to a positive number")
    w = w / w.sum()

    # Step 1 — vector normalisation: divide each column by its Euclidean norm,
    # giving unitless values that preserve ratios between alternatives.
    norms = np.sqrt((x ** 2).sum(axis=0))
    norms[norms == 0] = 1.0  # a constant column carries no information
    r = x / norms

    # Step 2 — weight.
    v = r * w

    # Step 3 — ideal and anti-ideal, respecting each criterion's direction.
    is_benefit = np.array(directions) == BENEFIT
    ideal_best = np.where(is_benefit, v.max(axis=0), v.min(axis=0))
    ideal_worst = np.where(is_benefit, v.min(axis=0), v.max(axis=0))

    # Step 4 — Euclidean distance to each.
    d_plus = np.sqrt(((v - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((v - ideal_worst) ** 2).sum(axis=1))

    # Step 5 — closeness. The guard covers the degenerate case where
    # every alternative is identical and both distances vanish.
    denom = d_plus + d_minus
    cc = np.divide(d_minus, denom, out=np.full(m, 0.5), where=denom > 0)

    return {
        "normalised": r,
        "weighted": v,
        "ideal_best": ideal_best,
        "ideal_worst": ideal_worst,
        "d_plus": d_plus,
        "d_minus": d_minus,
        "closeness": cc,
        "ranking": np.argsort(-cc),
    }


def fuzzy_topsis(matrix, weights, directions):
    """Fuzzy TOPSIS over triangular fuzzy numbers."""
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("fuzzy matrix must have shape (alternatives, criteria, 3)")
    m, n, _ = x.shape
    _check_directions(directions, n)

    if np.any(x[..., 0] > x[..., 1]) or np.any(x[..., 1] > x[..., 2]):
        raise ValueError("every TFN must satisfy l <= m <= u")

    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        # A crisp weight is the degenerate triangle (w, w, w).
        w = np.repeat(w[:, None], 3, axis=1)
    if w.shape != (n, 3):
        raise ValueError(f"weights must be (n, 3) or length n, got {w.shape}")

    # Step 1 — linear normalisation, defined piecewise by direction.
    r = np.zeros_like(x)
    for j, direction in enumerate(directions):
        col = x[:, j, :]
        if direction == BENEFIT:
            c_star = col[:, 2].max()
            if c_star == 0:
                c_star = 1.0
            r[:, j, :] = col / c_star
        else:
            # Smallest lower bound over each of the alternative's own three
            # numbers.
            a_minus = col[:, 0].min()
            safe = np.where(col == 0, np.finfo(float).eps, col)
            r[:, j, 0] = a_minus / safe[:, 2]
            r[:, j, 1] = a_minus / safe[:, 1]
            r[:, j, 2] = a_minus / safe[:, 0]

    # Step 2 — weight, componentwise triangle by triangle.
    v = r * w[None, :, :]

    # Step 3 — FPIS and FNIS, per criterion, componentwise across alternatives.
    fpis = v.max(axis=0)
    fnis = v.min(axis=0)

    # Step 4 — vertex distance, summed across criteria.
    def vertex(a, b):
        return np.sqrt(((a - b) ** 2).mean(axis=-1))

    d_plus = vertex(v, fpis[None, :, :]).sum(axis=1)
    d_minus = vertex(v, fnis[None, :, :]).sum(axis=1)

    # Step 5 — closeness.
    denom = d_plus + d_minus
    cc = np.divide(d_minus, denom, out=np.full(m, 0.5), where=denom > 0)

    return {
        "normalised": r,
        "weighted": v,
        "ideal_best": fpis,
        "ideal_worst": fnis,
        "d_plus": d_plus,
        "d_minus": d_minus,
        "closeness": cc,
        "ranking": np.argsort(-cc),
    }


def find_ties(closeness, tol=1e-9):
    """Groups of alternatives whose closeness coefficients are equal."""
    cc = np.asarray(closeness, dtype=float)
    groups, used = [], set()
    for i in range(len(cc)):
        if i in used:
            continue
        tied = [j for j in range(len(cc)) if abs(cc[j] - cc[i]) <= tol]
        if len(tied) > 1:
            groups.append(tied)
            used.update(tied)
    return groups


def sensitivity_analysis(matrix, directions, group_a_idx, group_b_idx,
                         base_weights, steps=21, fuzzy_matrix=None,
                         fuzzy_weights=None, fuzzy_directions=None):
    """Sweep the balance of weight between two groups of criteria."""
    base = np.asarray(base_weights, dtype=float)
    if fuzzy_directions is None:
        fuzzy_directions = directions
    a_share = base[group_a_idx].sum()
    b_share = base[group_b_idx].sum()

    rows = []
    for t in np.linspace(0.0, 1.0, steps):
        w = np.zeros_like(base)
        w[group_a_idx] = base[group_a_idx] / a_share * t if a_share else 0.0
        w[group_b_idx] = base[group_b_idx] / b_share * (1 - t) if b_share else 0.0
        if w.sum() <= 0:
            continue

        crisp = topsis(matrix, w, directions)["closeness"]
        entry = {
            "group_a_weight": float(t),
            "crisp_closeness": crisp,
            "crisp_winner": int(np.argmax(crisp)),
        }

        if fuzzy_matrix is not None:
            # Scale the TFN weights by the same group proportions so the
            # fuzzy sweep tracks the crisp one.
            fw = np.asarray(fuzzy_weights, dtype=float).copy()
            if fw.ndim == 1:
                fw = np.repeat(fw[:, None], 3, axis=1)
            scale = np.ones(len(base))
            scale[group_a_idx] = t / a_share * len(group_a_idx) if a_share else 0.0
            scale[group_b_idx] = (1 - t) / b_share * len(group_b_idx) if b_share else 0.0
            scaled = fw * scale[:, None]
            if scaled.sum() > 0:
                fz = fuzzy_topsis(fuzzy_matrix, scaled, fuzzy_directions)["closeness"]
                entry["fuzzy_closeness"] = fz
                entry["fuzzy_winner"] = int(np.argmax(fz))
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Checks against published worked examples

def _check_crisp_car_example():
    """The standard car-selection illustration: 4 alternatives, 4 criteria,
    three benefit and one cost, weights 0.1 / 0.4 / 0.3 / 0.2."""
    matrix = np.array([
        [7.0, 9.0, 9.0, 8.0],   # Civic
        [8.0, 7.0, 8.0, 7.0],   # Saturn
        [9.0, 6.0, 8.0, 9.0],   # Ford
        [6.0, 7.0, 8.0, 6.0],   # Mazda
    ])
    directions = [BENEFIT, BENEFIT, BENEFIT, COST]
    weights = [0.1, 0.4, 0.3, 0.2]
    names = ["Civic", "Saturn", "Ford", "Mazda"]

    res = topsis(matrix, weights, directions)
    cc = res["closeness"]
    expected = np.array([0.74, 0.41, 0.17, 0.45])

    assert np.allclose(cc, expected, atol=1e-2), (
        f"expected {expected}, got {np.round(cc, 4)}")
    assert [names[i] for i in res["ranking"]] == ["Civic", "Mazda", "Saturn", "Ford"]

    print("crisp / car example  " + "  ".join(
        f"{n} {c:.2f}" for n, c in zip(names, cc)) + "   ok")


def _check_fuzzy_hiring_example():
    """The standard fuzzy hiring illustration: 4 candidates, 3 criteria (two
    benefit, one cost), rated as TFNs and weighted with TFNs."""
    matrix = np.array([
        [[3, 5.667, 9], [5, 8.333, 9], [5, 7, 9]],          # C1
        [[5, 7, 9], [3, 7, 9], [3, 5, 7]],                  # C2
        [[5, 8.333, 9], [3, 5, 7], [1, 2.333, 5]],          # C3
        [[1, 2.333, 5], [1, 4.333, 7], [1, 1, 3]],          # C4
    ], dtype=float)
    directions = [BENEFIT, BENEFIT, COST]
    weights = np.array([[5, 7, 9], [7, 9, 9], [3, 5, 7]], dtype=float)
    names = ["C1", "C2", "C3", "C4"]

    res = fuzzy_topsis(matrix, weights, directions)

    # Intermediate table: normalisation, benefit and cost columns.
    assert np.allclose(res["normalised"][0, 0], [0.333, 0.630, 1.000], atol=5e-3), \
        res["normalised"][0, 0]
    assert np.allclose(res["normalised"][3, 2], [0.333, 1.000, 1.000], atol=5e-3), \
        res["normalised"][3, 2]

    # Intermediate table: weighted values.
    assert np.allclose(res["weighted"][0, 0], [1.667, 4.407, 9.000], atol=5e-3), \
        res["weighted"][0, 0]

    # The computed fuzzy ideal — the shortcut (1,1,1) would fail here.
    assert np.allclose(res["ideal_best"][0], [2.778, 6.481, 9.000], atol=5e-3), \
        res["ideal_best"][0]
    assert np.allclose(res["ideal_worst"][2], [0.333, 0.714, 1.400], atol=5e-3), \
        res["ideal_worst"][2]

    # Distances and the final answer.
    assert np.allclose(res["d_plus"], [5.448, 5.346, 4.083, 6.919], atol=5e-3), \
        np.round(res["d_plus"], 3)
    assert np.allclose(res["d_minus"], [5.971, 6.062, 8.091, 4.089], atol=5e-3), \
        np.round(res["d_minus"], 3)

    cc = res["closeness"]
    expected = np.array([0.523, 0.531, 0.665, 0.371])
    assert np.allclose(cc, expected, atol=5e-3), (
        f"expected {expected}, got {np.round(cc, 4)}")
    assert [names[i] for i in res["ranking"]] == ["C3", "C2", "C1", "C4"]

    print("fuzzy / hiring       " + "  ".join(
        f"{n} {c:.3f}" for n, c in zip(names, cc)) + "   ok")


def _check_properties():
    """Structural properties that must hold regardless of the data."""
    # An alternative best on every criterion is the ideal, so CC = 1
    # exactly; one worst on every criterion is the anti-ideal, CC = 0.
    matrix = np.array([[10.0, 1.0], [5.0, 5.0], [1.0, 10.0]])
    directions = [BENEFIT, COST]
    cc = topsis(matrix, [0.5, 0.5], directions)["closeness"]
    assert np.isclose(cc[0], 1.0) and np.isclose(cc[2], 0.0), cc
    assert cc[0] > cc[1] > cc[2]
    print(f"dominance            {cc[0]:.3f} > {cc[1]:.3f} > {cc[2]:.3f}"
          "                        ok")

    # Reading the cost criterion as a benefit must change the answer,
    # otherwise direction is not being applied.
    wrong = topsis(matrix, [0.5, 0.5], [BENEFIT, BENEFIT])["closeness"]
    assert not np.isclose(wrong[0], 1.0), "direction has no effect"
    print("direction            cost read as benefit changes the ranking   ok")

    # Two alternatives trading off symmetrically must tie at 0.5.
    sym = topsis(np.array([[10.0, 10.0], [5.0, 5.0]]), [0.5, 0.5], directions)
    assert np.allclose(sym["closeness"], 0.5), sym["closeness"]
    assert len(find_ties(sym["closeness"])) == 1
    print("symmetry / ties      symmetric trade-off ties at 0.500,"
          " detected     ok")

    # Scaling every weight must leave the closeness coefficients alone.
    a = topsis(matrix, [0.5, 0.5], directions)["closeness"]
    b = topsis(matrix, [5.0, 5.0], directions)["closeness"]
    assert np.allclose(a, b)
    print("weight scaling       invariant to a common factor               ok")


def _self_test():
    _check_crisp_car_example()
    _check_fuzzy_hiring_example()
    _check_properties()
    print("\nAll topsis.py checks passed.")


if __name__ == "__main__":
    _self_test()
