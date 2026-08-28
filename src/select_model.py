"""
select_model.py
---------------
Ranks the benchmarked models with TOPSIS and fuzzy TOPSIS and records
which one to deploy.

Reads results/decision_matrix.json, written by benchmark.py, so the
ranking can be re-run under different weights without repeating the
measurements.

The two methods answer slightly different questions and are both
reported. Crisp TOPSIS works from the raw measurements and can resolve
arbitrarily small differences between models. Fuzzy TOPSIS works from
linguistic ratings, which deliberately discard resolution in exchange
for saying something robust: two models in the same category are being
called equivalent on that criterion, and that is a claim worth being
able to make.

Where they agree the conclusion is well supported. Where they differ —
in particular where the fuzzy method ties two models the crisp method
separates — the difference is itself informative, and is reported rather
than resolved silently.

What this script decides, and what it does not
----------------------------------------------
It selects one model to deploy and explain. It does not build a weighted
ensemble: closeness coefficients rank alternatives, they are not
calibrated probabilities, and using them as vote weights would mean
treating a model's cheapness as a reason to trust its verdict.

Direction handling
------------------
Crisp TOPSIS receives the raw values, where latency, size and
explanation time are cost criteria and invert at the ideal-solution
step. Fuzzy TOPSIS receives linguistic ratings that are already phrased
as goodness — a fast model rates "outstanding" — so every criterion
reaches it as a benefit criterion. The direction is applied exactly once
in each path.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # no display in a terminal session
import matplotlib.pyplot as plt
import numpy as np

import weights as weighting
from fuzzify import describe_bands, describe_scale
from metrics import format_table
from topsis import BENEFIT, find_ties, fuzzy_topsis, sensitivity_analysis, topsis

from paths import DECISION_MATRIX, FIGURES_DIR, SELECTED_JSON, ensure_dirs

QUALITY_CRITERIA = ["accuracy", "f1"]


def load_matrix(path=DECISION_MATRIX):
    """Raw values, linguistic TFNs, and the criterion metadata."""
    with open(path) as fh:
        payload = json.load(fh)

    criteria = payload["criteria"]
    directions = payload["directions"]
    names = list(payload["models"])

    crisp = np.array([[payload["models"][n]["crisp"][c] for c in criteria]
                      for n in names], dtype=float)
    fuzzy = np.array([[payload["models"][n]["linguistic"][c]["tfn"] for c in criteria]
                      for n in names], dtype=float)
    categories = [[payload["models"][n]["linguistic"][c]["category"] for c in criteria]
                  for n in names]
    return names, criteria, directions, crisp, fuzzy, categories


def print_matrix(title, names, criteria, matrix, fmt="{:.4f}"):
    print(f"\n{title}")
    rows = [[n] + [fmt.format(v) for v in row] for n, row in zip(names, matrix)]
    print(format_table(rows, ["model"] + list(criteria)))


def print_fuzzy_matrix(title, names, criteria, matrix):
    print(f"\n{title}")
    rows = [[n] + [f"({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})" for t in row]
            for n, row in zip(names, matrix)]
    print(format_table(rows, ["model"] + list(criteria)))


def print_ranking(title, names, result):
    print(f"\n{title}")
    rows = []
    for rank, idx in enumerate(result["ranking"], start=1):
        rows.append([rank, names[idx],
                     f"{result['d_plus'][idx]:.4f}",
                     f"{result['d_minus'][idx]:.4f}",
                     f"{result['closeness'][idx]:.4f}"])
    print(format_table(rows, ["rank", "model", "d+", "d-", "closeness"]))

    ties = find_ties(result["closeness"])
    for group in ties:
        tied = ", ".join(names[i] for i in group)
        print(f"\n  Tie: {tied} share a closeness of "
              f"{result['closeness'][group[0]]:.4f}.")
    return ties


def plot_sensitivity(sweep, names, path):
    """Closeness against the quality/cost weight balance, both methods."""
    xs = [row["group_a_weight"] for row in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for ax, key, label in ((axes[0], "crisp_closeness", "TOPSIS"),
                           (axes[1], "fuzzy_closeness", "Fuzzy TOPSIS")):
        if key not in sweep[0]:
            continue
        for i, name in enumerate(names):
            ax.plot(xs, [row[key][i] for row in sweep], label=name, linewidth=2)

        winner_key = "crisp_winner" if key.startswith("crisp") else "fuzzy_winner"
        winners = [row[winner_key] for row in sweep]
        for j in range(1, len(winners)):
            if winners[j] != winners[j - 1]:
                flip = (xs[j] + xs[j - 1]) / 2
                ax.axvline(flip, color="grey", linestyle="--", linewidth=1)
                ax.text(flip, ax.get_ylim()[1], f" {flip:.2f}", rotation=90,
                        va="top", fontsize=8, color="grey")

        ax.set_title(label)
        ax.set_xlabel("weight on quality criteria  (cost takes the remainder)")
        ax.set_ylabel("closeness coefficient")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Sensitivity of the ranking to the quality/cost weight balance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def weight_invariance(crisp, fuzzy, directions, fuzzy_directions, criteria, names):
    """
    Re-rank under several plausible weightings.

    The standard objection to any multi-criteria result is that the
    weights were chosen to produce it. Rather than argue for one set,
    show what happens under others.
    """
    schemes = {
        "uniform": (np.full(len(criteria), 1.0),
                    np.full((len(criteria), 3), 1.0)),
        "linguistic": (weighting.crisp_weights(criteria),
                       weighting.fuzzy_weights(criteria)),
        "quality-led": (np.array([0.35, 0.25, 0.15, 0.10, 0.15]),
                        np.repeat(np.array([0.35, 0.25, 0.15, 0.10, 0.15])[:, None],
                                  3, axis=1)),
    }

    rows = []
    for label, (cw, fw) in schemes.items():
        c_rank = topsis(crisp, cw, directions)["ranking"]
        f_rank = fuzzy_topsis(fuzzy, fw, fuzzy_directions)["ranking"]
        rows.append([label,
                     " > ".join(names[i] for i in c_rank),
                     " > ".join(names[i] for i in f_rank)])
    return rows


def main():
    ensure_dirs()
    names, criteria, directions, crisp, fuzzy, categories = load_matrix()

    # Linguistic ratings are phrased as goodness, so every criterion is a
    # benefit criterion by the time fuzzy TOPSIS sees it.
    fuzzy_directions = [BENEFIT] * len(criteria)

    crisp_w = weighting.crisp_weights(criteria)
    fuzzy_w = weighting.fuzzy_weights(criteria)

    print("=" * 74)
    print("MULTI-CRITERIA MODEL SELECTION")
    print("=" * 74)

    print("\nCriteria and directions (as measured):")
    print(format_table([[c, d] for c, d in zip(criteria, directions)],
                       ["criterion", "direction"]))

    print("\nCriterion importance:")
    print(weighting.describe(criteria))

    print_matrix("\nDecision matrix (raw measurements):", names, criteria, crisp)

    # ---- Crisp TOPSIS -------------------------------------------------
    print("\n" + "=" * 74)
    print("TOPSIS")
    print("=" * 74)
    cr = topsis(crisp, crisp_w, directions)
    print_matrix("\nVector-normalised matrix:", names, criteria, cr["normalised"])
    print_matrix("Weighted normalised matrix:", names, criteria, cr["weighted"])
    print("\nIdeal      A+: " + "  ".join(f"{v:.4f}" for v in cr["ideal_best"]))
    print("Anti-ideal A-: " + "  ".join(f"{v:.4f}" for v in cr["ideal_worst"]))
    crisp_ties = print_ranking("Ranking:", names, cr)

    # ---- Fuzzy TOPSIS -------------------------------------------------
    print("\n" + "=" * 74)
    print("FUZZY TOPSIS")
    print("=" * 74)
    print()
    print(describe_scale())
    print()
    print(describe_bands())

    print("\nLinguistic ratings:")
    print(format_table([[n] + row for n, row in zip(names, categories)],
                       ["model"] + list(criteria)))

    # There is one set of measurements, so one rater: the aggregation
    # step that would combine several decision-makers' ratings is the
    # identity here and is stated rather than skipped.
    print("\nAggregation across raters: identity (a single set of measurements).")

    print_fuzzy_matrix("Fuzzy decision matrix, TFN (l, m, u):",
                       names, criteria, fuzzy)
    fz = fuzzy_topsis(fuzzy, fuzzy_w, fuzzy_directions)
    print_fuzzy_matrix("Normalised:", names, criteria, fz["normalised"])
    print_fuzzy_matrix("Weighted normalised:", names, criteria, fz["weighted"])
    print("\nFPIS: " + "  ".join(f"({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"
                                 for t in fz["ideal_best"]))
    print("FNIS: " + "  ".join(f"({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"
                               for t in fz["ideal_worst"]))
    fuzzy_ties = print_ranking("Ranking:", names, fz)

    # ---- Agreement ----------------------------------------------------
    crisp_order = [names[i] for i in cr["ranking"]]
    fuzzy_order = [names[i] for i in fz["ranking"]]

    print("\n" + "=" * 74)
    print("COMPARISON")
    print("=" * 74)
    print(f"\nTOPSIS:        {' > '.join(crisp_order)}")
    print(f"Fuzzy TOPSIS:  {' > '.join(fuzzy_order)}")

    if fuzzy_ties:
        tied_names = [", ".join(names[i] for i in g) for g in fuzzy_ties]
        print(f"\nThe fuzzy ranking cannot separate {'; '.join(tied_names)}: they")
        print("hold identical categories on every criterion. The linguistic scale")
        print("is too coarse to distinguish them, which is a real statement about")
        print("how similar they are rather than a defect. TOPSIS, working from the")
        print("raw values, does separate them.")

    if crisp_order[-1] == fuzzy_order[-1]:
        print(f"\nBoth methods place {crisp_order[-1]} last.")

    # ---- Robustness to the weighting ----------------------------------
    print("\n" + "=" * 74)
    print("ROBUSTNESS TO THE WEIGHTING")
    print("=" * 74)
    rows = weight_invariance(crisp, fuzzy, directions, fuzzy_directions,
                             criteria, names)
    print()
    print(format_table(rows, ["weighting", "TOPSIS", "Fuzzy TOPSIS"]))

    first = {r[1].split(" > ")[0] for r in rows}
    last = {r[1].split(" > ")[-1] for r in rows}
    if len(first) == 1 and len(last) == 1:
        print(f"\n{first.pop()} ranks first and {last.pop()} last under every")
        print("weighting tested, so the conclusion does not rest on the particular")
        print("importance ratings chosen.")

    # ---- Sensitivity sweep --------------------------------------------
    print("\n" + "=" * 74)
    print("SENSITIVITY TO THE QUALITY/COST BALANCE")
    print("=" * 74)
    quality_idx = [criteria.index(c) for c in QUALITY_CRITERIA]
    cost_idx = [i for i in range(len(criteria)) if i not in quality_idx]
    sweep = sensitivity_analysis(crisp, directions, quality_idx, cost_idx,
                                 crisp_w, steps=21, fuzzy_matrix=fuzzy,
                                 fuzzy_weights=fuzzy_w,
                                 fuzzy_directions=fuzzy_directions)

    print("\nWinner as the balance shifts:")
    print(format_table(
        [[f"{r['group_a_weight']:.2f}", names[r["crisp_winner"]],
          names[r.get("fuzzy_winner", r["crisp_winner"])]] for r in sweep],
        ["quality weight", "TOPSIS", "Fuzzy TOPSIS"]))

    flips = [(sweep[j]["group_a_weight"], names[sweep[j - 1]["crisp_winner"]],
              names[sweep[j]["crisp_winner"]])
             for j in range(1, len(sweep))
             if sweep[j]["crisp_winner"] != sweep[j - 1]["crisp_winner"]]
    for at, before, after in flips:
        print(f"\nWinner changes from {before} to {after} near a quality weight "
              f"of {at:.2f}.")

    fig_path = FIGURES_DIR / "sensitivity.png"
    plot_sensitivity(sweep, names, fig_path)
    print(f"\nSensitivity plot written to {fig_path}")

    # ---- Record the decision ------------------------------------------
    selected = crisp_order[0]
    selection = {
        "selected": selected,
        "selected_by": "TOPSIS closeness coefficient",
        "selection_note": (
            "TOPSIS is used to make the final choice because it works from the "
            "raw measurements and can separate models the linguistic scale rates "
            "identically."
        ),
        "importance": {c: weighting.IMPORTANCE[c][0] for c in criteria},
        "crisp_weights": {c: float(w) for c, w in zip(criteria, crisp_w)},
        "topsis_ranking": crisp_order,
        "fuzzy_ranking": fuzzy_order,
        "topsis_closeness": {n: float(cr["closeness"][i]) for i, n in enumerate(names)},
        "fuzzy_closeness": {n: float(fz["closeness"][i]) for i, n in enumerate(names)},
        "fuzzy_ties": [[names[i] for i in g] for g in fuzzy_ties],
        "rankings_agree": crisp_order == fuzzy_order,
    }
    with open(SELECTED_JSON, "w") as fh:
        json.dump(selection, fh, indent=2)

    print("\n" + "=" * 74)
    print(f"SELECTED MODEL: {selected}")
    print("=" * 74)
    print(f"Written to {SELECTED_JSON}")


if __name__ == "__main__":
    main()
