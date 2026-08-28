"""
fuzzify.py
----------
Converts raw measurements into triangular fuzzy numbers via a linguistic
scale, which is the input fuzzy TOPSIS operates on.

A triangular fuzzy number (TFN) is a triple (l, m, u): a lower bound, a
most-likely value at which membership equals 1, and an upper bound
(Zadeh, 1965). Fuzzy TOPSIS replaces every crisp score in the decision
matrix with such a triple, which lets a criterion be rated as a band of
plausible values rather than one falsely precise number.

Ratings are assigned the standard way: a raw value is compared against
threshold bands, the band names a linguistic category, and that
category's fixed TFN is looked up. The raw number plays no further part
in the arithmetic — only the category's TFN does.

Constructing the scale
----------------------
Rather than hardcoding a lookup table, the scale is built so its shape is
explicit:

  1. Count the categories.                       n = 6
  2. Choose a numeric range to spread them over. 0-10
  3. Space the peaks evenly in (n-1) steps.      step = 10/5 = 2
                                                 peaks at 0,2,4,6,8,10
  4. Give each triangle one step either side.    (peak-step, peak, peak+step)
  5. Clip the end categories at the bounds.      lowest  -> (0, 0, peak+step)
                                                 highest -> (peak-step, max, max)

The triangles deliberately overlap. A value sitting on a band boundary
should shade smoothly between two categories instead of jumping, and
overlapping triangles guarantee every point on the scale has nonzero
membership in at least one category — the property that makes the
representation fuzzy rather than merely discretised.

Threshold bands
---------------
Accuracy and F1 are percentages and use the conventional performance
bands in QUALITY_BANDS.

Latency, model size and explanation time are not percentages, so no
conventional band table applies. The construction above still holds, but
the thresholds themselves are a judgement, and a judgement of that kind
has to be recorded rather than left implicit — COST_BANDS states each one
with its reasoning. They are deliberately *absolute*, chosen from what
the numbers mean for an interactively driven classifier, and are not
derived from the observed measurements; the bands therefore stay fixed
if the set of candidate models changes.

Direction
---------
Every band table here is phrased in terms of goodness rather than
magnitude: a fast model rates "outstanding", exactly as a high accuracy
does. The benefit/cost distinction is therefore resolved once, here, and
every criterion reaches fuzzy TOPSIS as a benefit criterion where a
larger TFN is better.

The alternative convention rates the attribute's magnitude ("time to
solve = High") and inverts cost criteria later, at the normalisation
step. Both are sound; combining them is not, which is why the choice is
stated here and applied without exception.

Crisp TOPSIS is untouched by any of this. It works from raw values, where
latency and size remain genuine cost criteria and invert at the
ideal-solution step.

Reference
---------
Zadeh, L. A. (1965). Fuzzy sets. Information and Control, 8(3), 338-353.
"""

# ---------------------------------------------------------------------------
# Scale construction
# ---------------------------------------------------------------------------

CATEGORIES = ["poor", "fair", "good", "very good", "excellent", "outstanding"]
SCALE_MAX = 10.0


def build_tfn_scale(categories=CATEGORIES, scale_max=SCALE_MAX):
    """
    Build the TFN for each category by the construction described above.

    Returns {category: (l, m, u)} ordered worst to best. For six
    categories over 0-10 this yields:

        poor        (0,  0,  2)
        fair        (0,  2,  4)
        good        (2,  4,  6)
        very good   (4,  6,  8)
        excellent   (6,  8, 10)
        outstanding (8, 10, 10)
    """
    n = len(categories)
    step = scale_max / (n - 1)

    scale = {}
    for i, name in enumerate(categories):
        peak = i * step
        lower, upper = peak - step, peak + step
        # The end categories cannot extend past the ends of the scale.
        if i == 0:
            lower = peak = 0.0
        if i == n - 1:
            peak = upper = scale_max
        scale[name] = (max(0.0, lower), peak, min(scale_max, upper))
    return scale


TFN_SCALE = build_tfn_scale()


# ---------------------------------------------------------------------------
# Threshold bands
# ---------------------------------------------------------------------------

# Percentages. Read as: strictly greater than the threshold earns the
# category.
QUALITY_BANDS = [
    (95.0, "outstanding"),
    (90.0, "excellent"),
    (80.0, "very good"),
    (70.0, "good"),
    (60.0, "fair"),
    # at or below 60 falls through to "poor"
]

QUALITY_CRITERIA = ("accuracy", "f1", "sensitivity", "specificity")

# Cost criteria. Read as: strictly less than the threshold earns the
# category. Each band is an absolute judgement about what the quantity
# means in use, not a rescaling of what was observed.
COST_BANDS = {
    # Under a millisecond is imperceptible. By ~100ms a single
    # classification has become something the user waits for.
    "latency_ms": [
        (1.0, "outstanding"),
        (5.0, "excellent"),
        (20.0, "very good"),
        (100.0, "good"),
        (500.0, "fair"),
    ],
    # Under a megabyte is trivial to move and version. Past ~500MB the
    # model is no longer something to casually ship alongside the code.
    "size_mb": [
        (1.0, "outstanding"),
        (10.0, "excellent"),
        (100.0, "very good"),
        (500.0, "good"),
        (2000.0, "fair"),
    ],
    # A tenth of a second reads as instant. Past ~30s an explanation
    # cannot be produced during a live demonstration.
    "lime_seconds": [
        (0.1, "outstanding"),
        (1.0, "excellent"),
        (5.0, "very good"),
        (30.0, "good"),
        (120.0, "fair"),
    ],
}


def classify(criterion, value):
    """
    Return the linguistic category a raw measurement falls into.

    Thresholds are strict on both sides: a band stated as ">70%" excludes
    exactly 70.00, which drops to the band below, and a cost band stated
    as "<1.0" excludes exactly 1.0.
    """
    if criterion in QUALITY_CRITERIA:
        # Accept either a 0-1 proportion or an already-scaled percentage.
        pct = value * 100.0 if value <= 1.0 else value
        for threshold, name in QUALITY_BANDS:
            if pct > threshold:
                return name
        return "poor"

    if criterion in COST_BANDS:
        for threshold, name in COST_BANDS[criterion]:
            if value < threshold:
                return name
        return "poor"

    raise KeyError(f"no threshold bands defined for criterion {criterion!r}")


def fuzzify(criterion, value):
    """Raw measurement -> linguistic category -> that category's TFN."""
    return TFN_SCALE[classify(criterion, value)]


def describe_scale():
    """The constructed scale as text, for the console and the report."""
    lines = [f"Linguistic scale ({len(CATEGORIES)} categories, "
             f"0-{SCALE_MAX:g}, step {SCALE_MAX / (len(CATEGORIES) - 1):g}):"]
    for name, tfn in TFN_SCALE.items():
        lines.append(f"  {name:12s} ({tfn[0]:4.1f}, {tfn[1]:4.1f}, {tfn[2]:4.1f})")
    return "\n".join(lines)


def describe_bands():
    """The threshold tables, including the stated cost assumptions."""
    lines = ["Quality bands (percentages):"]
    last = None
    for threshold, name in QUALITY_BANDS:
        lines.append(f"  > {threshold:g}%".ljust(18) + name)
        last = threshold
    lines.append(f"  <= {last:g}%".ljust(18) + "poor")

    lines.append("")
    lines.append("Cost bands (stated assumptions):")
    for criterion, bands in COST_BANDS.items():
        lines.append(f"  {criterion}:")
        last = None
        for threshold, name in bands:
            lines.append(f"    < {threshold:g}".ljust(18) + name)
            last = threshold
        lines.append(f"    >= {last:g}".ljust(18) + "poor")
    return "\n".join(lines)


def _self_test():
    expected = {
        "poor": (0.0, 0.0, 2.0),
        "fair": (0.0, 2.0, 4.0),
        "good": (2.0, 4.0, 6.0),
        "very good": (4.0, 6.0, 8.0),
        "excellent": (6.0, 8.0, 10.0),
        "outstanding": (8.0, 10.0, 10.0),
    }
    for name, tfn in expected.items():
        assert TFN_SCALE[name] == tfn, f"{name}: expected {tfn}, built {TFN_SCALE[name]}"
    print("scale construction   reproduces the six-category table          ok")

    for name, (l, m, u) in TFN_SCALE.items():
        assert l <= m <= u, f"{name} is not a valid TFN: {(l, m, u)}"
    print("TFN validity         l <= m <= u for all six                    ok")

    # Consecutive categories must overlap, or the scale is merely a set
    # of bins and the representation is not fuzzy at all.
    ordered = [TFN_SCALE[c] for c in CATEGORIES]
    for lo, hi in zip(ordered, ordered[1:]):
        assert lo[2] > hi[0], f"no overlap between {lo} and {hi}"
    print("overlap              consecutive triangles overlap              ok")

    # Strict thresholds.
    assert classify("accuracy", 0.70) == "fair", "70.00 must not count as > 70"
    assert classify("accuracy", 0.7001) == "good"
    assert classify("accuracy", 0.95) == "excellent", "95.00 must not count as > 95"
    assert classify("accuracy", 0.9501) == "outstanding"
    print("boundary rule        70.00 -> fair, 70.01 -> good               ok")

    # Percentages accepted in either form.
    assert classify("accuracy", 0.9117) == classify("accuracy", 91.17)
    print("input form           0-1 and 0-100 agree                        ok")

    # The project's measured values.
    for criterion, value, want in [
        ("accuracy", 0.9117, "excellent"), ("accuracy", 0.9783, "outstanding"),
        ("latency_ms", 0.285, "outstanding"), ("latency_ms", 17.27, "very good"),
        ("size_mb", 0.219, "outstanding"), ("size_mb", 256.33, "good"),
        ("lime_seconds", 0.051, "outstanding"), ("lime_seconds", 10.64, "good"),
    ]:
        got = classify(criterion, value)
        assert got == want, f"{criterion}={value}: expected {want}, got {got}"
    print("measured values      land in the expected categories            ok")

    print("\nAll fuzzify.py self-tests passed.\n")
    print(describe_scale())
    print()
    print(describe_bands())


if __name__ == "__main__":
    _self_test()
