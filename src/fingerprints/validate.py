"""Validation analyses (Milestone 4): lung SBS4, melanoma SBS7, colorectal MMR.

The analysis plan was fixed before any real result was seen (docs/decisions.md D3, D4, D5, D7,
"Validation specification"). This module holds pure, testable functions; file reading, figures
and orchestration are in ``scripts/run_validation.py``.

Conventions
-----------
* Per-tumour fractions are NNLS-attributed exposures divided by their sum (see ``fit.py``).
* Two-group tests are two-sided Mann-Whitney U. The effect size is the AUC
  (``U1 / (n1 * n2)``, ties count half): the probability that a random tumour from group 1 has a
  higher value than a random tumour from group 2. 0.5 means no difference. The rank-biserial
  correlation is ``2 * AUC - 1`` (positive: group 1 higher). Group 1 is always the group
  expected to be higher (smokers, MSI, SKCM).
* Confidence intervals are percentile bootstrap (2000 resamples, fixed seed, each group
  resampled with replacement). The seed is fixed per call, so results do not depend on call order.
* Filters (``min_snvs``, ``min_cosine``) use only the SNV total and the reconstruction cosine.
  They never see smoking, MSI or any other group label.
* The pooled lung model is a van Elteren stratified rank test (strata = cancer type).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261122
N_BOOT = 2000

SNV_COL = "observed_snvs"
COSINE_COL = "cosine"

#: Signature groups (COSMIC v3.4 names). Exactly these; tests pin them.
UV_SIGS = ["SBS7a", "SBS7b", "SBS7c", "SBS7d"]
UV_AB_SIGS = ["SBS7a", "SBS7b"]
UV_PLUS_38_SIGS = [*UV_SIGS, "SBS38"]
MMR_SIGS = ["SBS6", "SBS14", "SBS15", "SBS20", "SBS26", "SBS44"]

BORDERLINE_LOW = 0.15
BORDERLINE_HIGH = 0.25
MATERIAL_CHANGE = 0.05  # reporting convention for the D5 check: 5 percentage points
#: Listing aids (fixed in advance, data-independent; not tests): MSI tumours below this summed
#: MMR fraction are listed in a supplementary table; "non-UV-dominated" melanomas have a summed
#: SBS7a-d fraction <= NON_UV_CUT (the complement of the secondary ">50%" measure); a
#: never-smoker is listed as having "tobacco-like" SBS4 when its SBS4 fraction exceeds
#: NEVER_SMOKER_SBS4_CUT.
MSI_LOW_MMR_CUT = 0.25
NON_UV_CUT = 0.5
NEVER_SMOKER_SBS4_CUT = 0.25

#: Row schema shared by all validation tables.
TABLE_COLUMNS = [
    "analysis",
    "cohort",
    "comparison",
    "label",
    "setting",
    "min_snvs",
    "min_cosine",
    "group1",
    "group2",
    "n1",
    "n2",
    "median1",
    "iqr1",
    "median2",
    "iqr2",
    "effect_name",
    "effect",
    "ci_low",
    "ci_high",
    "rank_biserial",
    "p",
    "min",
    "max",
    "median_cosine1",
    "median_cosine2",
    "notes",
]

LABELS = ("primary", "secondary", "sensitivity", "descriptive")


@dataclass(frozen=True)
class Setting:
    """A filter setting for one analysis run."""

    name: str
    min_snvs: int
    min_cosine: float | None = None


PRIMARY = Setting("primary", 50)
SETTINGS: list[Setting] = [
    PRIMARY,
    Setting("min_snvs_20", 20),
    Setting("min_snvs_100", 100),
    Setting("cosine_ge_0.8", 50, 0.8),
]


def label_for(setting: Setting, is_primary_cell: bool = True) -> str:
    """'primary' only for the primary cell at the primary setting; everything else sensitivity."""
    return "primary" if (is_primary_cell and setting.name == PRIMARY.name) else "sensitivity"


# --------------------------------------------------------------------------------------
# Filters and sums
# --------------------------------------------------------------------------------------


def apply_filters(
    table: pd.DataFrame,
    min_snvs: int = 50,
    min_cosine: float | None = None,
    snv_col: str = SNV_COL,
    cosine_col: str = COSINE_COL,
) -> pd.DataFrame:
    """Keep tumours with ``>= min_snvs`` SNVs and (optionally) ``cosine >= min_cosine``.

    Only the SNV and cosine columns are read, so no group label can influence which tumours
    are kept. A missing cosine fails the cosine filter.
    """
    keep = table[snv_col].astype(float) >= float(min_snvs)
    if min_cosine is not None:
        keep &= table[cosine_col].astype(float) >= float(min_cosine)
    return table.loc[keep]


def sum_signatures(table: pd.DataFrame, signatures: Sequence[str]) -> pd.Series:
    """Per-tumour sum of the fractions of ``signatures``; absent columns are an error."""
    missing = [s for s in signatures if s not in table.columns]
    if missing:
        raise KeyError(f"signature column(s) not in table: {missing}")
    return table[list(signatures)].sum(axis=1)


def is_borderline(
    score: pd.Series, low: float = BORDERLINE_LOW, high: float = BORDERLINE_HIGH
) -> pd.Series:
    """True where ``low <= msi_score <= high`` (both ends included, D4). NaN -> False."""
    s = pd.to_numeric(score, errors="coerce")
    return (s >= low) & (s <= high)


def uv_largest_component(table: pd.DataFrame, other_sigs: Sequence[str]) -> pd.Series:
    """True where summed SBS7a-d exceeds every single other fitted signature."""
    uv = sum_signatures(table, UV_SIGS)
    others = table[list(other_sigs)].max(axis=1) if other_sigs else pd.Series(0.0, index=uv.index)
    return uv > others


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def _clean(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[~np.isnan(arr)]


def describe(values: Iterable[float]) -> dict[str, float]:
    """n, median, Q1, Q3, min, max (NaN dropped)."""
    arr = _clean(values)
    if arr.size == 0:
        return dict.fromkeys(["n", "median", "q1", "q3", "min", "max"], float("nan")) | {"n": 0}
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(arr.size),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def auc_from_ranks(x: np.ndarray, y: np.ndarray) -> float:
    """AUC = P(x > y) + 0.5 P(x == y) = U1 / (n1 n2), via mid-ranks."""
    n1, n2 = len(x), len(y)
    ranks = stats.rankdata(np.concatenate([x, y]))
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u1 / (n1 * n2))


def _auc_boot(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    n1, n2 = len(x), len(y)
    xi = rng.integers(0, n1, size=(n_boot, n1))
    yi = rng.integers(0, n2, size=(n_boot, n2))
    both = np.concatenate([x[xi], y[yi]], axis=1)
    ranks = stats.rankdata(both, axis=1)
    u1 = ranks[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2
    return u1 / (n1 * n2)


def bootstrap_auc_ci(
    x: Iterable[float],
    y: Iterable[float],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the AUC (deterministic for a given seed)."""
    xa, ya = _clean(x), _clean(y)
    if len(xa) < 1 or len(ya) < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = _auc_boot(xa, ya, rng, n_boot)
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return float(lo), float(hi)


def mann_whitney(
    x: Iterable[float], y: Iterable[float], n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, float]:
    """Two-sided Mann-Whitney U of group 1 (x) vs group 2 (y), with AUC and bootstrap CI.

    Returns n1, n2, U1, p, auc, rank_biserial, ci_low, ci_high. NaN effect/p when a group
    has fewer than one observation or both groups are constant and equal.
    """
    xa, ya = _clean(x), _clean(y)
    out = {
        "n1": len(xa),
        "n2": len(ya),
        "U1": float("nan"),
        "p": float("nan"),
        "auc": float("nan"),
        "rank_biserial": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if len(xa) < 1 or len(ya) < 1:
        return out
    auc = auc_from_ranks(xa, ya)
    out["auc"] = auc
    out["rank_biserial"] = 2 * auc - 1
    out["U1"] = auc * len(xa) * len(ya)
    if np.ptp(np.concatenate([xa, ya])) > 0:
        out["p"] = float(stats.mannwhitneyu(xa, ya, alternative="two-sided").pvalue)
    else:
        out["p"] = 1.0
    out["ci_low"], out["ci_high"] = bootstrap_auc_ci(xa, ya, n_boot=n_boot, seed=seed)
    return out


def spearman(
    x: Iterable[float], y: Iterable[float], n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, float]:
    """Spearman rho with two-sided p, n and a bootstrap (pairs) percentile CI."""
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    ok = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[ok], ya[ok]
    out = {
        "n": int(len(xa)),
        "rho": float("nan"),
        "p": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if len(xa) < 3 or np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return out
    res = stats.spearmanr(xa, ya)
    out["rho"], out["p"] = float(res.statistic), float(res.pvalue)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xa), size=(n_boot, len(xa)))
    rx = stats.rankdata(xa[idx], axis=1)
    ry = stats.rankdata(ya[idx], axis=1)
    rxc = rx - rx.mean(axis=1, keepdims=True)
    ryc = ry - ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rxc**2).sum(axis=1) * (ryc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (rxc * ryc).sum(axis=1) / denom
    boots = boots[~np.isnan(boots)]
    if boots.size:
        out["ci_low"], out["ci_high"] = (float(v) for v in np.quantile(boots, [0.025, 0.975]))
    return out


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def van_elteren(
    strata: Sequence[tuple[np.ndarray, np.ndarray]],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, float]:
    """Van Elteren stratified rank-sum test of group 1 vs group 2 across strata.

    Each stratum is ``(x, y)`` (group 1, group 2). Within a stratum, observations are ranked
    together (mid-ranks) and the group-1 rank sum ``W_s`` is weighted by ``1 / (N_s + 1)``.
    Statistic ``T = sum w_s W_s`` is compared with its null mean and (tie-corrected) variance;
    the p value is two-sided from the normal approximation. Strata with an empty group carry no
    information and are skipped. The effect size is the weighted AUC
    ``sum(w_s n1_s n2_s AUC_s) / sum(w_s n1_s n2_s)`` (the same weights as the test), with a
    bootstrap CI that resamples within strata. Returns NaN effect/p when no stratum is usable.
    """
    used = []
    for x, y in strata:
        xa, ya = _clean(x), _clean(y)
        if len(xa) >= 1 and len(ya) >= 1:
            used.append((xa, ya))
    out = {
        "n1": int(sum(len(x) for x, _ in used)),
        "n2": int(sum(len(y) for _, y in used)),
        "z": float("nan"),
        "p": float("nan"),
        "auc": float("nan"),
        "rank_biserial": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n_strata": len(used),
    }
    if not used:
        return out
    num = den = t_stat = t_mean = t_var = 0.0
    for xa, ya in used:
        n1, n2 = len(xa), len(ya)
        n = n1 + n2
        w = 1.0 / (n + 1)
        allv = np.concatenate([xa, ya])
        ranks = stats.rankdata(allv)
        w1 = ranks[:n1].sum()
        t_stat += w * w1
        t_mean += w * n1 * (n + 1) / 2
        _, counts = np.unique(allv, return_counts=True)
        tie = float(((counts**3 - counts).sum()) / (n * (n - 1))) if n > 1 else 0.0
        t_var += w**2 * n1 * n2 / 12 * ((n + 1) - tie)
        auc_s = (w1 - n1 * (n1 + 1) / 2) / (n1 * n2)
        num += w * n1 * n2 * auc_s
        den += w * n1 * n2
    auc = num / den
    out["auc"] = float(auc)
    out["rank_biserial"] = float(2 * auc - 1)
    if t_var > 0:
        z = (t_stat - t_mean) / math.sqrt(t_var)
        out["z"] = float(z)
        out["p"] = float(2 * stats.norm.sf(abs(z)))
    else:
        out["p"] = 1.0
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for xa, ya in used:
        n1, n2 = len(xa), len(ya)
        w = 1.0 / (n1 + n2 + 1)
        b = _auc_boot(xa, ya, rng, n_boot)
        boots += w * n1 * n2 * b
    boots /= den
    out["ci_low"], out["ci_high"] = (float(v) for v in np.quantile(boots, [0.025, 0.975]))
    return out


# --------------------------------------------------------------------------------------
# Row builders
# --------------------------------------------------------------------------------------


def _fmt_iqr(d: dict[str, float]) -> str:
    if d["n"] == 0:
        return ""
    return f"{d['q1']:.4g}-{d['q3']:.4g}"


def _median_or_nan(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else float("nan")


def _blank_row() -> dict[str, object]:
    return dict.fromkeys(TABLE_COLUMNS, float("nan")) | {"notes": ""}


def two_group_row(
    analysis: str,
    cohort: str,
    comparison: str,
    label: str,
    setting: Setting,
    value_col: str,
    table: pd.DataFrame,
    group_col: str,
    g1: str,
    g2: str,
    notes: str = "",
) -> dict[str, object]:
    """Filter ``table`` by ``setting``, then compare ``value_col`` between two groups.

    The filter is applied first and uses only SNV total and cosine. Group labels are applied
    afterwards.
    """
    filt = apply_filters(table, setting.min_snvs, setting.min_cosine)
    a = filt[filt[group_col] == g1]
    b = filt[filt[group_col] == g2]
    mw = mann_whitney(a[value_col], b[value_col])
    d1, d2 = describe(a[value_col]), describe(b[value_col])
    row = _blank_row()
    row.update(
        analysis=analysis,
        cohort=cohort,
        comparison=comparison,
        label=label,
        setting=setting.name,
        min_snvs=setting.min_snvs,
        min_cosine=setting.min_cosine if setting.min_cosine is not None else float("nan"),
        group1=g1,
        group2=g2,
        n1=mw["n1"],
        n2=mw["n2"],
        median1=d1["median"],
        iqr1=_fmt_iqr(d1),
        median2=d2["median"],
        iqr2=_fmt_iqr(d2),
        effect_name="AUC",
        effect=mw["auc"],
        ci_low=mw["ci_low"],
        ci_high=mw["ci_high"],
        rank_biserial=mw["rank_biserial"],
        p=mw["p"],
        median_cosine1=_median_or_nan(a[COSINE_COL]),
        median_cosine2=_median_or_nan(b[COSINE_COL]),
        notes=notes,
    )
    return row


def spearman_row(
    analysis: str,
    cohort: str,
    comparison: str,
    label: str,
    setting: Setting,
    table: pd.DataFrame,
    x_col: str,
    y_col: str,
    notes: str = "",
) -> dict[str, object]:
    """Spearman correlation of two columns after filtering by ``setting``."""
    filt = apply_filters(table, setting.min_snvs, setting.min_cosine)
    sp = spearman(filt[x_col], filt[y_col])
    ok = filt[[x_col, y_col]].notna().all(axis=1)
    row = _blank_row()
    row.update(
        analysis=analysis,
        cohort=cohort,
        comparison=comparison,
        label=label,
        setting=setting.name,
        min_snvs=setting.min_snvs,
        min_cosine=setting.min_cosine if setting.min_cosine is not None else float("nan"),
        group1=x_col,
        group2=y_col,
        n1=sp["n"],
        effect_name="Spearman rho",
        effect=sp["rho"],
        ci_low=sp["ci_low"],
        ci_high=sp["ci_high"],
        p=sp["p"],
        median_cosine1=_median_or_nan(filt.loc[ok, COSINE_COL]),
        notes=notes,
    )
    return row


def describe_row(
    analysis: str,
    cohort: str,
    comparison: str,
    label: str,
    setting: Setting,
    table: pd.DataFrame,
    value_col: str,
    group_label: str = "all",
    notes: str = "",
) -> dict[str, object]:
    """Single-group description (n, median, IQR, cosine); no test."""
    filt = apply_filters(table, setting.min_snvs, setting.min_cosine)
    d = describe(filt[value_col])
    row = _blank_row()
    row.update(
        analysis=analysis,
        cohort=cohort,
        comparison=comparison,
        label=label,
        setting=setting.name,
        min_snvs=setting.min_snvs,
        min_cosine=setting.min_cosine if setting.min_cosine is not None else float("nan"),
        group1=group_label,
        n1=d["n"],
        median1=d["median"],
        iqr1=_fmt_iqr(d),
        effect_name="n/a (descriptive)",
        min=d["min"],
        max=d["max"],
        median_cosine1=_median_or_nan(filt[COSINE_COL]),
        notes=notes,
    )
    return row


def to_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Rows to a DataFrame with the standard column order."""
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


# --------------------------------------------------------------------------------------
# Cohort assembly
# --------------------------------------------------------------------------------------


class CohortMismatchError(ValueError):
    """Chosen cases and fit tables disagree in a way that is not explained by low SNV counts."""


def assemble_cohort(
    fractions: pd.DataFrame,
    qc: pd.DataFrame,
    clinical: pd.DataFrame,
    matrix_totals: pd.Series | None = None,
    min_fit_snvs: int = 20,
) -> pd.DataFrame:
    """Join fractions (tumours x signatures), QC and clinical (by ``barcode``) per tumour.

    ``fractions`` and ``qc`` are indexed by tumour barcode. The QC SNV column may be called
    ``observed_snvs`` or ``total_snvs``.

    Silent-drop guard: a chosen case missing from the fit table is allowed only when its SBS96
    matrix total (``matrix_totals``, indexed by barcode) is below ``min_fit_snvs``, the fit
    table's own inclusion threshold. Any other unmatched case (total at or above the threshold,
    absent from ``matrix_totals``, or ``matrix_totals`` not supplied) raises
    :class:`CohortMismatchError`, as does a fitted tumour with no clinical row (see
    :func:`unmatched_chosen_cases`).
    """
    if "total_snvs" in qc.columns:  # the matrix total; preferred over observed_snvs
        qc = qc.drop(columns=[SNV_COL], errors="ignore").rename(columns={"total_snvs": SNV_COL})
    for col in (SNV_COL, COSINE_COL):
        if col not in qc.columns:
            raise KeyError(f"QC table lacks column {col!r}")
    clin = clinical.drop_duplicates("barcode").set_index("barcode")
    unmatched_chosen_cases(clin.index, fractions.index, matrix_totals, min_fit_snvs)
    out = fractions.join(qc[[SNV_COL, COSINE_COL]], how="inner").join(clin, how="inner")
    out.index.name = "tumour"
    return out


def unmatched_chosen_cases(
    chosen: pd.Index,
    fitted: pd.Index,
    matrix_totals: pd.Series | None,
    min_fit_snvs: int = 20,
) -> pd.DataFrame:
    """Chosen cases absent from the fit table, with their matrix totals; raise if unexplained.

    Returns (barcode, total_snvs) for chosen cases not in ``fitted``, every one of which must
    have a total below ``min_fit_snvs``. Raises :class:`CohortMismatchError` for a fitted tumour
    with no chosen case, or a chosen case that is missing without a total below the threshold.
    """
    extra = [b for b in fitted if b not in chosen]
    if extra:
        raise CohortMismatchError(
            f"{len(extra)} fitted tumour(s) have no clinical row, e.g. {extra[:3]}"
        )
    missing = [b for b in chosen if b not in fitted]
    if not missing:
        return pd.DataFrame({"barcode": [], SNV_COL: []})
    if matrix_totals is None:
        raise CohortMismatchError(
            f"{len(missing)} chosen case(s) are not in the fit table and no matrix totals "
            f"were given to check them, e.g. {missing[:3]}"
        )
    totals = matrix_totals.reindex(missing).astype(float)
    bad = totals[totals.isna() | (totals >= min_fit_snvs)]
    if len(bad):
        raise CohortMismatchError(
            f"{len(bad)} chosen case(s) missing from the fit table but not explained by a "
            f"total below {min_fit_snvs} SNVs (absent from matrix or total >= threshold), "
            f"e.g. {dict(bad.head(3))}"
        )
    return pd.DataFrame({"barcode": totals.index, SNV_COL: totals.to_numpy()})


def cohort_flow(
    cohort: str,
    groups: pd.Series,
    totals: pd.Series,
    fit_index: pd.Index,
    cosine: pd.Series,
    min_fit_snvs: int = 20,
) -> pd.DataFrame:
    """Per-group attrition: chosen cases -> in the fit table -> >=20/50/100 SNVs -> cosine >=0.8.

    ``groups`` is indexed by chosen-case barcode (group label per case; use a label such as
    'unknown' rather than NaN). ``totals`` are SBS96 matrix totals (a missing barcode counts as
    0 and is reported in ``absent_from_matrix``). ``cosine`` is indexed by fitted tumour. The
    cosine column counts tumours with >=50 SNVs and cosine >=0.8 (the cosine sensitivity set).
    Labels are never used by the SNV or cosine criteria; they only split the counts.
    """
    tot = totals.reindex(groups.index)
    cos = cosine.reindex(groups.index)
    rows = []
    for g in sorted(groups.unique(), key=str):
        m = groups == g
        t = tot[m]
        rows.append(
            {
                "cohort": cohort,
                "group": g,
                "chosen_cases": int(m.sum()),
                "absent_from_matrix": int(t.isna().sum()),
                "in_fit_table": int(groups.index[m].isin(fit_index).sum()),
                f"snvs_ge_{min_fit_snvs}": int((t >= min_fit_snvs).sum()),
                "snvs_ge_50": int((t >= 50).sum()),
                "snvs_ge_100": int((t >= 100).sum()),
                "snvs_ge_50_and_cosine_ge_0.8": int(((t >= 50) & (cos[m] >= 0.8)).sum()),
            }
        )
    return pd.DataFrame(rows)


def borderline_counts(clinical: pd.DataFrame, fitted: pd.DataFrame, cohort: str) -> list[dict]:
    """MSI scores 0.15-0.25 (D4), counted three ways per MSI group.

    ``clinical_cases``: chosen cases with an MSI call in the clinical table;
    ``clinical_borderline``: of those, score in [0.15, 0.25];
    ``in_fit_table_borderline``: borderline among tumours in the fit table (>=20 SNVs);
    ``analysed_ge50_borderline``: borderline among tumours with >=50 SNVs (primary analysis).
    """
    rows = []
    called = clinical[clinical["msi_status"].isin(["MSI", "MSS"])]
    fit = fitted[fitted["msi_group"].notna()]
    for g in ("MSI", "MSS"):
        c = called[called["msi_status"] == g]
        f = fit[fit["msi_group"] == g]
        rows.append(
            {
                "cohort": cohort,
                "group": g,
                "clinical_cases": len(c),
                "clinical_borderline": int(is_borderline(c["msi_score"]).sum()),
                "in_fit_table": len(f),
                "in_fit_table_borderline": int(f["borderline"].sum()),
                "analysed_ge50": int((f[SNV_COL] >= 50).sum()),
                "analysed_ge50_borderline": int(f.loc[f[SNV_COL] >= 50, "borderline"].sum()),
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# A. Lung
# --------------------------------------------------------------------------------------

SBS4 = "SBS4"


def _lung_groups(table: pd.DataFrame) -> pd.DataFrame:
    """Add ``primary_group`` / ``ever_group`` / ``other_ever`` helper columns."""
    t = table.copy()
    t["primary_group"] = t["smoking_primary_group"]
    t["ever_group"] = t["smoking_ever_group"]
    return t


def lung_two_group(
    cohorts: dict[str, pd.DataFrame],
    variant: str = "with SBS18",
    include_pooled: bool = True,
    settings: Sequence[Setting] = SETTINGS,
) -> list[dict[str, object]]:
    """Never vs smoker (primary contrast) and never vs ever (sensitivity), per cohort + pooled.

    ``cohorts`` maps 'LUAD'/'LUSC' to assembled tables. Only (LUAD, primary contrast, with
    SBS18, primary setting) is labelled primary; LUSC rows are 'descriptive' (18 never-smokers);
    pooled rows are 'secondary' at the primary setting.
    """
    rows: list[dict[str, object]] = []
    default_variant = variant == "with SBS18"
    for cohort, table in cohorts.items():
        t = _lung_groups(table)
        for setting in settings:
            for contrast, col, g1 in (
                ("never vs smoker (current + reformed <=15y)", "primary_group", "smoker"),
                ("never vs ever-smoker (codes 2-5)", "ever_group", "ever"),
            ):
                is_primary_contrast = col == "primary_group"
                if cohort == "LUAD":
                    label = label_for(setting, is_primary_contrast and default_variant)
                    note = f"SBS4 fraction; fit {variant}"
                else:
                    label = "descriptive"
                    note = f"SBS4 fraction; fit {variant}; underpowered (few never-smokers)"
                rows.append(
                    two_group_row(
                        f"lung_sbs4_{'primary_contrast' if is_primary_contrast else 'ever'}",
                        cohort,
                        f"{contrast}; fit {variant}",
                        label,
                        setting,
                        SBS4,
                        t,
                        col,
                        g1,
                        "never",
                        notes=note,
                    )
                )
    if include_pooled and set(cohorts) >= {"LUAD", "LUSC"}:
        rows.extend(lung_pooled(cohorts, variant, settings))
    return rows


def lung_pooled(
    cohorts: dict[str, pd.DataFrame],
    variant: str = "with SBS18",
    settings: Sequence[Setting] = SETTINGS,
) -> list[dict[str, object]]:
    """Van Elteren test (strata LUAD/LUSC) of SBS4 fraction, smoker vs never and ever vs never."""
    rows = []
    for setting in settings:
        for contrast, col, g1 in (
            ("never vs smoker (current + reformed <=15y)", "smoking_primary_group", "smoker"),
            ("never vs ever-smoker (codes 2-5)", "smoking_ever_group", "ever"),
        ):
            strata = []
            per = []
            for cohort, table in cohorts.items():
                f = apply_filters(table, setting.min_snvs, setting.min_cosine)
                a = f.loc[f[col] == g1, SBS4].to_numpy(float)
                b = f.loc[f[col] == "never", SBS4].to_numpy(float)
                strata.append((a, b))
                per.append(f"{cohort} n={len(a)}/{len(b)}")
            ve = van_elteren(strata)
            row = _blank_row()
            row.update(
                analysis="lung_sbs4_pooled_van_elteren",
                cohort="LUAD+LUSC",
                comparison=f"{contrast}; fit {variant}",
                label="secondary" if setting.name == PRIMARY.name else "sensitivity",
                setting=setting.name,
                min_snvs=setting.min_snvs,
                min_cosine=setting.min_cosine if setting.min_cosine is not None else float("nan"),
                group1=g1,
                group2="never",
                n1=ve["n1"],
                n2=ve["n2"],
                effect_name="weighted AUC (van Elteren weights)",
                effect=ve["auc"],
                ci_low=ve["ci_low"],
                ci_high=ve["ci_high"],
                rank_biserial=ve["rank_biserial"],
                p=ve["p"],
                notes=f"strata: {'; '.join(per)} (group1/group2); z={ve['z']:.3f}",
            )
            used = []
            for table in cohorts.values():
                f = apply_filters(table, setting.min_snvs, setting.min_cosine)
                used.append(f[f[col].isin([g1, "never"])])
            row["median_cosine1"] = _median_or_nan(pd.concat(used)[COSINE_COL])
            rows.append(row)
    return rows


def lung_dose_response(
    cohorts: dict[str, pd.DataFrame],
    settings: Sequence[Setting] = (PRIMARY,),
    measure: str = "fraction",
) -> list[dict[str, object]]:
    """Spearman SBS4 vs pack-years within ever-smokers (D3), per cohort.

    ``measure`` is 'fraction' (the pre-specified version) or 'mutation count'. For the count,
    pass tables whose ``SBS4`` column holds the NNLS-attributed SBS4 exposure (a mutation count,
    from ``exposures_<P>_all20.tsv``), joined to the same QC and clinical columns, so the same
    filters apply. The count version is an added secondary test (D9).
    """
    analysis = (
        "lung_sbs4_dose_response" if measure == "fraction" else "lung_sbs4_count_dose_response"
    )
    rows = []
    for cohort, table in cohorts.items():
        ever = table[table["smoking_ever_group"] == "ever"].copy()
        ever["pack_years"] = pd.to_numeric(ever["pack_years_smoked"], errors="coerce")
        for setting in settings:
            rows.append(
                spearman_row(
                    analysis,
                    cohort,
                    f"SBS4 {measure} vs pack_years_smoked, ever-smokers",
                    "secondary" if setting.name == PRIMARY.name else "sensitivity",
                    setting,
                    ever,
                    "pack_years",
                    SBS4,
                    notes="ever-smokers with pack-years recorded",
                )
            )
    return rows


def lung_group_descriptives(
    cohorts: dict[str, pd.DataFrame], setting: Setting = PRIMARY
) -> list[dict[str, object]]:
    """Median (IQR), range of SBS4 fraction by smoking code, per cohort (descriptive)."""
    rows = []
    for cohort, table in cohorts.items():
        t = table.copy()
        t["group"] = t["smoking_label"].fillna("no usable smoking status")
        for grp in sorted(t["group"].unique()):
            sub = t[t["group"] == grp]
            rows.append(
                describe_row(
                    "lung_sbs4_by_smoking_group",
                    cohort,
                    "SBS4 fraction by smoking group",
                    "descriptive",
                    setting,
                    sub,
                    SBS4,
                    group_label=str(grp),
                )
            )
    return rows


def lung_never_smoker_sbs4(
    cohorts: dict[str, pd.DataFrame], setting: Setting = PRIMARY
) -> list[dict[str, object]]:
    """Descriptive: SBS4 fraction among lifelong never-smokers (D9).

    Rows per cohort: share with SBS4 fraction above ``NEVER_SMOKER_SBS4_CUT`` (a listing aid,
    not a test; Wilson CI), the upper quartile and the maximum.
    """
    rows = []
    for cohort, table in cohorts.items():
        f = apply_filters(table, setting.min_snvs, setting.min_cosine)
        nv = f[f["smoking_primary_group"] == "never"].copy()
        nv["above_cut"] = (nv[SBS4] > NEVER_SMOKER_SBS4_CUT).astype(float)
        d = describe(nv[SBS4])
        r = _prop_row(
            "lung_never_smoker_sbs4",
            cohort,
            f"share of lifelong never-smokers with SBS4 fraction > {NEVER_SMOKER_SBS4_CUT}",
            "descriptive",
            setting,
            nv,
            "above_cut",
            f"cut {NEVER_SMOKER_SBS4_CUT} is a descriptive listing aid, not a test",
        )
        r["group1"] = "never"
        r["median1"], r["iqr1"] = d["median"], _fmt_iqr(d)
        r["min"], r["max"] = d["min"], d["max"]
        rows.append(r)
        for name, val in (("upper quartile (Q3)", d["q3"]), ("maximum", d["max"])):
            q = _blank_row()
            q.update(
                analysis="lung_never_smoker_sbs4",
                cohort=cohort,
                comparison=f"SBS4 fraction in lifelong never-smokers: {name}",
                label="descriptive",
                setting=setting.name,
                min_snvs=setting.min_snvs,
                group1="never",
                n1=d["n"],
                effect_name=name,
                effect=val,
            )
            rows.append(q)
    return rows


# --------------------------------------------------------------------------------------
# B. Melanoma
# --------------------------------------------------------------------------------------


def add_uv_columns(table: pd.DataFrame, fitted_signatures: Sequence[str]) -> pd.DataFrame:
    """Add summed UV measures and the 'UV is the largest component' flag."""
    t = table.copy()
    t["uv_sbs7a_d"] = sum_signatures(t, UV_SIGS)
    t["uv_sbs7a_b"] = sum_signatures(t, UV_AB_SIGS)
    t["uv_sbs7a_d_plus_38"] = sum_signatures(t, UV_PLUS_38_SIGS)
    others = [s for s in fitted_signatures if s not in UV_SIGS]
    t["uv_is_largest"] = uv_largest_component(t, others).astype(float)
    t["uv_gt_50"] = (t["uv_sbs7a_d"] > 0.5).astype(float)
    return t


def _prop_row(
    analysis: str,
    cohort: str,
    comparison: str,
    label: str,
    setting: Setting,
    filt: pd.DataFrame,
    flag_col: str,
    notes: str,
) -> dict[str, object]:
    n = len(filt)
    k = int(filt[flag_col].sum())
    lo, hi = wilson_ci(k, n)
    row = _blank_row()
    row.update(
        analysis=analysis,
        cohort=cohort,
        comparison=comparison,
        label=label,
        setting=setting.name,
        min_snvs=setting.min_snvs,
        min_cosine=setting.min_cosine if setting.min_cosine is not None else float("nan"),
        group1="all",
        n1=n,
        effect_name="proportion (Wilson 95% CI)",
        effect=k / n if n else float("nan"),
        ci_low=lo,
        ci_high=hi,
        median_cosine1=_median_or_nan(filt[COSINE_COL]),
        notes=f"{k}/{n}. {notes}".strip(),
    )
    return row


def melanoma_summary(
    table: pd.DataFrame,
    variant: str = "exclude (default)",
    settings: Sequence[Setting] = SETTINGS,
    primary_variant: bool = True,
) -> list[dict[str, object]]:
    """Median (IQR) of UV measures, % largest component and % >50%, per setting.

    ``table`` must already carry the columns from :func:`add_uv_columns`. All melanoma rows are
    descriptive summaries (there is no group test); the only one labelled 'primary' is the summed
    SBS7a-d fraction at the primary setting with the default profile. The '% largest' and
    '% > 50%' proportions are secondary.
    """
    rows = []
    for setting in settings:
        filt = apply_filters(table, setting.min_snvs, setting.min_cosine)
        for col, name in (
            ("uv_sbs7a_d", "summed SBS7a-d fraction"),
            ("uv_sbs7a_b", "summed SBS7a+b fraction"),
            ("uv_sbs7a_d_plus_38", "summed SBS7a-d + SBS38 fraction"),
        ):
            primary_cell = primary_variant and col == "uv_sbs7a_d"
            row = describe_row(
                "melanoma_uv_fraction",
                "SKCM",
                f"{name}; profile adjacent_snvs {variant}",
                label_for(setting, primary_cell) if primary_cell else "sensitivity",
                setting,
                table,
                col,
            )
            if not primary_cell and primary_variant and setting.name == PRIMARY.name:
                row["label"] = "secondary"
            rows.append(row)
        for flag, name in (
            ("uv_is_largest", "SBS7a-d (summed) is the single largest component"),
            ("uv_gt_50", "summed SBS7a-d fraction > 50%"),
        ):
            rows.append(
                _prop_row(
                    "melanoma_uv_proportion",
                    "SKCM",
                    f"{name}; profile adjacent_snvs {variant}",
                    "secondary"
                    if (primary_variant and setting.name == "primary")
                    else "sensitivity",
                    setting,
                    filt,
                    flag,
                    "largest = summed SBS7a-d exceeds every other single fitted signature"
                    if flag == "uv_is_largest"
                    else "",
                )
            )
    return rows


def melanoma_sample_type(
    table: pd.DataFrame, setting: Setting = PRIMARY
) -> list[dict[str, object]]:
    """Descriptive: summed SBS7a-d by sample type code (01 primary vs 06 metastatic)."""
    rows = []
    for code in sorted(table["sample_type_code"].astype(str).unique()):
        sub = table[table["sample_type_code"].astype(str) == code]
        rows.append(
            describe_row(
                "melanoma_uv_by_sample_type",
                "SKCM",
                "summed SBS7a-d fraction by sample type",
                "descriptive",
                setting,
                sub,
                "uv_sbs7a_d",
                group_label=f"sample type {code}",
            )
        )
    return rows


def melanoma_adjacent_check(
    default_table: pd.DataFrame, count_table: pd.DataFrame, setting: Setting = PRIMARY
) -> dict[str, object]:
    """D5 check: paired change in summed SBS7a-d between adjacent_snvs exclude and count_as_sbs.

    Pairs are tumours present in both tables after filtering each by ``setting``. Reports the
    median and maximum absolute paired difference (count_as_sbs minus exclude). 'Material' uses
    a reporting convention of 0.05 (5 percentage points) on the median paired difference; it is
    stated in the notes, not a statistical test.
    """
    a = apply_filters(default_table, setting.min_snvs, setting.min_cosine)["uv_sbs7a_d"]
    b = apply_filters(count_table, setting.min_snvs, setting.min_cosine)["uv_sbs7a_d"]
    common = a.index.intersection(b.index)
    diff = (b.loc[common] - a.loc[common]).to_numpy(float)
    row = _blank_row()
    d = describe(diff)
    d_a, d_b = describe(a.loc[common]), describe(b.loc[common])
    material = bool(len(diff) and abs(d["median"]) > MATERIAL_CHANGE)
    p = float("nan")
    if len(diff) >= 2 and np.any(diff != 0):
        p = float(stats.wilcoxon(diff).pvalue)
    row.update(
        analysis="melanoma_adjacent_snvs_check",
        cohort="SKCM",
        comparison="summed SBS7a-d: count_as_sbs minus exclude (paired)",
        label="sensitivity",
        setting=setting.name,
        min_snvs=setting.min_snvs,
        group1="count_as_sbs",
        group2="exclude",
        n1=len(common),
        n2=len(common),
        median1=d_b["median"],
        iqr1=_fmt_iqr(d_b),
        median2=d_a["median"],
        iqr2=_fmt_iqr(d_a),
        effect_name="median paired difference",
        effect=d["median"],
        min=d["min"],
        max=d["max"],
        p=p,
        median_cosine1=_median_or_nan(default_table.loc[common, COSINE_COL]),
        median_cosine2=_median_or_nan(count_table.loc[common, COSINE_COL]),
        notes=(
            f"IQR of paired differences {d['q1']:.4g} to {d['q3']:.4g} (min/max columns are the "
            "extreme paired differences, not a CI); "
            f"max |diff|={np.abs(diff).max() if len(diff) else float('nan'):.4g}; "
            f"material change (|median diff| > {MATERIAL_CHANGE}): {'yes' if material else 'no'}; "
            "p is Wilcoxon signed-rank on the paired differences"
        ),
    )
    return row


#: Fields read from the assembled SKCM table (clinical_<P>.tsv), not the raw download.
SKCM_TABLE_FIELDS = ["site_of_resection_or_biopsy", "tumor_of_origin"]
SKCM_CLINICAL_EXTRA = [
    "primary_diagnosis",
    "primary_site",
    "tissue_or_organ_of_origin",
    "morphology",
]


def melanoma_non_uv_listing(
    table: pd.DataFrame,
    fitted_signatures: Sequence[str],
    extra_clinical: pd.DataFrame | None = None,
    setting: Setting = PRIMARY,
    cut: float = NON_UV_CUT,
) -> pd.DataFrame:
    """Supplementary listing of non-UV-dominated melanomas (summed SBS7a-d <= ``cut``).

    Flags tumours; nothing is excluded from any analysis. The boundary is inclusive (exactly
    0.5 is listed), the complement of the '> 50%' measure. ``extra_clinical`` (indexed by
    ``case_id``) supplies histology / site fields; fields it lacks are filled with
    'not available'.
    """
    t = apply_filters(table, setting.min_snvs, setting.min_cosine)
    t = t[t["uv_sbs7a_d"] <= cut]
    sigs = list(fitted_signatures)
    top = t[sigs].idxmax(axis=1) if len(t) else pd.Series(dtype=str)
    out = pd.DataFrame(
        {
            "barcode": t.index,
            "case_id": t["case_id"].to_numpy(),
            "sample_type_code": t["sample_type_code"].astype(str).to_numpy(),
            "summed_sbs7a_d": t["uv_sbs7a_d"].to_numpy(),
            "top_fitted_signature": top.to_numpy(),
            "top_signature_fraction": t[sigs].max(axis=1).to_numpy() if len(t) else [],
            "total_snvs": t[SNV_COL].to_numpy(),
            "cosine": t[COSINE_COL].to_numpy(),
        }
    )
    for col in SKCM_TABLE_FIELDS:
        if col in t.columns:
            out[col] = t[col].fillna("not reported").replace("", "not reported").to_numpy()
        else:
            out[col] = "not available"
    for col in SKCM_CLINICAL_EXTRA:
        if extra_clinical is not None and col in extra_clinical.columns:
            vals = extra_clinical[col].reindex(out["case_id"]).to_numpy()
            out[col] = pd.Series(vals).fillna("not reported").to_numpy()
        else:
            out[col] = "not available"
    return out.sort_values("summed_sbs7a_d").reset_index(drop=True)


def non_uv_field_counts(
    non_uv: pd.DataFrame,
    all_tumours: pd.DataFrame,
    fields: Sequence[str],
    by: str | None = None,
) -> pd.DataFrame:
    """Value counts of clinical fields among non-UV melanomas vs all analysed SKCM tumours.

    With ``by`` (e.g. 'sample_type_code', present in both frames) counts are split by that
    column.
    """
    rows = []
    for f in fields:
        if f not in all_tumours.columns:
            continue
        keys = [None] if by is None else sorted(all_tumours[by].astype(str).unique())
        for key in keys:
            a_sub = all_tumours if by is None else all_tumours[all_tumours[by].astype(str) == key]
            n_sub = non_uv
            if by is not None and by in non_uv.columns:
                n_sub = non_uv[non_uv[by].astype(str) == key]
            a = a_sub[f].fillna("not reported").value_counts()
            n = n_sub[f].value_counts() if f in n_sub.columns else pd.Series(dtype=int)
            for val in a.index:
                row = {"field": f}
                if by is not None:
                    row[by] = key
                row |= {"value": val, "non_uv_n": int(n.get(val, 0)), "all_analysed_n": int(a[val])}
                rows.append(row)
    return pd.DataFrame(rows)


# ---- model-free C>T check -------------------------------------------------------------


def _channel_parts(channels: Sequence[str]) -> pd.DataFrame:
    ch = pd.Series(list(channels), index=list(channels))
    return pd.DataFrame({"five": ch.str[0], "sub": ch.str[2:5], "three": ch.str[6]}, index=ch.index)


def ct_fractions(matrix: pd.DataFrame) -> pd.DataFrame:
    """Model-free C>T measures per tumour from a 96 x tumours count matrix.

    Columns (fractions of the tumour's total SBS count):
    ``ct_all`` all C>T (pyrimidine-strand notation); ``ct_dipyrimidine`` C>T where the mutated
    C has a pyrimidine (C or T) as 5' or 3' neighbour; ``ct_cc`` C>T where a neighbour is C
    (a CC dinucleotide on the pyrimidine strand). Tumours with no SNVs get NaN.
    """
    parts = _channel_parts(matrix.index)
    is_ct = parts["sub"] == "C>T"
    pyr = parts["five"].isin(["C", "T"]) | parts["three"].isin(["C", "T"])
    cc = (parts["five"] == "C") | (parts["three"] == "C")
    total = matrix.sum(axis=0).astype(float).replace(0, np.nan)
    out = pd.DataFrame(
        {
            "ct_all": matrix[is_ct].sum(axis=0) / total,
            "ct_dipyrimidine": matrix[is_ct & pyr].sum(axis=0) / total,
            "ct_cc": matrix[is_ct & cc].sum(axis=0) / total,
            SNV_COL: matrix.sum(axis=0).astype(float),
        }
    )
    out.index.name = "tumour"
    return out


def melanoma_ct_control(
    ct_by_cohort: dict[str, pd.DataFrame],
    settings: Sequence[Setting] = (PRIMARY,),
) -> list[dict[str, object]]:
    """SKCM vs each other cohort on model-free C>T measures (AUC: SKCM higher).

    ``ct_by_cohort`` maps cohort name to :func:`ct_fractions` output restricted to the selected
    tumours. The cosine column is not applicable (no model), so only min_snvs filtering is used.
    """
    rows = []
    for setting in settings:
        rows.extend(_ct_control_one(ct_by_cohort, setting))
    return rows


def _ct_control_one(
    ct_by_cohort: dict[str, pd.DataFrame], setting: Setting
) -> list[dict[str, object]]:
    rows = []
    skcm = apply_filters(ct_by_cohort["SKCM"], setting.min_snvs)
    for other, tab in ct_by_cohort.items():
        if other == "SKCM":
            continue
        o = apply_filters(tab, setting.min_snvs)
        for col, name in (
            ("ct_all", "C>T fraction (all contexts)"),
            ("ct_dipyrimidine", "C>T at dipyrimidine fraction"),
            ("ct_cc", "C>T at CC context fraction"),
        ):
            mw = mann_whitney(skcm[col], o[col])
            d1, d2 = describe(skcm[col]), describe(o[col])
            row = _blank_row()
            row.update(
                analysis="melanoma_ct_model_free_control",
                cohort=f"SKCM vs {other}",
                comparison=f"{name}, raw SBS96 profile (no fit)",
                label="secondary" if setting.name == PRIMARY.name else "sensitivity",
                setting=setting.name,
                min_snvs=setting.min_snvs,
                group1="SKCM",
                group2=other,
                n1=mw["n1"],
                n2=mw["n2"],
                median1=d1["median"],
                iqr1=_fmt_iqr(d1),
                median2=d2["median"],
                iqr2=_fmt_iqr(d2),
                effect_name="AUC",
                effect=mw["auc"],
                ci_low=mw["ci_low"],
                ci_high=mw["ci_high"],
                rank_biserial=mw["rank_biserial"],
                p=mw["p"],
                notes="model-free: reconstruction cosine not applicable",
            )
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# C. Colorectal
# --------------------------------------------------------------------------------------


def add_mmr_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add summed MMR fraction (SBS6+14+15+20+26+44), MSI group and borderline flag."""
    t = table.copy()
    t["mmr_sum"] = sum_signatures(t, MMR_SIGS)
    t["msi_group"] = t["msi_status"].where(t["msi_status"].isin(["MSI", "MSS"]))
    t["borderline"] = is_borderline(t["msi_score"])
    t["total_snvs"] = t[SNV_COL]
    return t


def colorectal_msi_tests(
    table: pd.DataFrame,
    settings: Sequence[Setting] = SETTINGS,
    value_col: str = "mmr_sum",
    value_name: str = "summed MMR fraction (SBS6+14+15+20+26+44)",
    include_borderline_exclusion: bool = True,
    cohorts: Sequence[str] = ("COAD+READ", "COAD", "READ"),
) -> list[dict[str, object]]:
    """MSI vs MSS on ``value_col``: pooled and per project, across settings.

    Primary = pooled COAD+READ summed MMR at the primary setting. Per-project rows are
    'secondary' at the primary setting. The borderline exclusion (scores 0.15-0.25 removed) is
    an additional sensitivity run on the primary setting filters.
    """
    t = table[table["msi_group"].notna()]
    rows = []
    for cohort in cohorts:
        sub = t if cohort == "COAD+READ" else t[t["project"] == f"TCGA-{cohort}"]
        for setting in settings:
            if cohort == "COAD+READ":
                label = label_for(setting)
            else:
                label = "secondary" if setting.name == PRIMARY.name else "sensitivity"
            rows.append(
                two_group_row(
                    "colorectal_mmr_msi_vs_mss",
                    cohort,
                    f"MSI vs MSS: {value_name}",
                    label,
                    setting,
                    value_col,
                    sub,
                    "msi_group",
                    "MSI",
                    "MSS",
                    notes="GDC msi_status (WXS call preferred); bioinformatic label",
                )
            )
        if include_borderline_exclusion:
            nb = sub[~sub["borderline"]]
            r = two_group_row(
                "colorectal_mmr_msi_vs_mss",
                cohort,
                f"MSI vs MSS: {value_name}",
                "sensitivity",
                Setting("exclude_borderline_0.15-0.25", PRIMARY.min_snvs),
                value_col,
                nb,
                "msi_group",
                "MSI",
                "MSS",
                notes=f"MSI scores 0.15-0.25 excluded ({len(sub) - len(nb)} borderline tumours in "
                "the fit table, i.e. >=20 SNVs; see validation_colorectal_borderline.tsv for the "
                "clinical-table and >=50 SNV counts)",
            )
            rows.append(r)
    return rows


def colorectal_per_signature(
    table: pd.DataFrame, setting: Setting = PRIMARY
) -> list[dict[str, object]]:
    """Each MMR signature's fraction, MSI vs MSS, pooled (descriptive; no multiplicity claim)."""
    t = table[table["msi_group"].notna()]
    rows = []
    for sig in MMR_SIGS:
        rows.append(
            two_group_row(
                "colorectal_mmr_per_signature",
                "COAD+READ",
                f"MSI vs MSS: {sig} fraction",
                "descriptive",
                setting,
                sig,
                t,
                "msi_group",
                "MSI",
                "MSS",
            )
        )
    return rows


def colorectal_hypermutation(
    table: pd.DataFrame, setting: Setting = PRIMARY
) -> list[dict[str, object]]:
    """Total SNVs, MSI vs MSS (pooled and per project)."""
    t = table[table["msi_group"].notna()]
    rows = []
    for cohort in ("COAD+READ", "COAD", "READ"):
        sub = t if cohort == "COAD+READ" else t[t["project"] == f"TCGA-{cohort}"]
        rows.append(
            two_group_row(
                "colorectal_hypermutation",
                cohort,
                "MSI vs MSS: total SBS96 SNVs per tumour (exome)",
                "secondary" if cohort == "COAD+READ" else "descriptive",
                setting,
                "total_snvs",
                sub,
                "msi_group",
                "MSI",
                "MSS",
                notes="SNV counts after profile-step drops; hypermutation expected in MSI-H",
            )
        )
    return rows


def msi_fraction_by_case(clinical_by_project: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    """Fraction of cases called MSI among cases with an MSI call, per project (D4 sanity check).

    Counts cases (one row per case, as built by ``samples.choose_msi``), not BAM files. COAD is
    expected to be roughly 10-20% (TCGA Network 2012, Nature 487:330-337).
    """
    rows = []
    for project, clin in clinical_by_project.items():
        called = clin[clin["msi_status"].isin(["MSI", "MSS"])].drop_duplicates("case_id")
        n, k = len(called), int((called["msi_status"] == "MSI").sum())
        lo, hi = wilson_ci(k, n)
        row = _blank_row()
        row.update(
            analysis="colorectal_msi_fraction_by_case",
            cohort=project.replace("TCGA-", ""),
            comparison="share of cases with an MSI call that are MSI",
            label="descriptive",
            setting="all cases with a call",
            group1="MSI",
            group2="MSS",
            n1=k,
            n2=n - k,
            effect_name="proportion MSI (Wilson 95% CI)",
            effect=k / n if n else float("nan"),
            ci_low=lo,
            ci_high=hi,
            notes=f"{k}/{n} cases; "
            + (
                "D4 sanity check expects roughly 10-20% in COAD "
                "(TCGA Network 2012, Nature 487:330-337)"
                if project == "TCGA-COAD"
                else "the 10-20% COAD expectation does not cover this project"
            ),
        )
        rows.append(row)
    return rows


def sbs1_sbs6_overlap(table: pd.DataFrame, setting: Setting = PRIMARY) -> list[dict[str, object]]:
    """SBS6 fraction distribution and SBS1-SBS6 Spearman correlation, in MSS (and MSI) tumours."""
    rows = []
    t = table[table["msi_group"].notna()]
    for group in ("MSS", "MSI"):
        sub = apply_filters(t[t["msi_group"] == group], setting.min_snvs, setting.min_cosine)
        s6 = sub["SBS6"]
        row = describe_row(
            "colorectal_sbs6_leakage",
            "COAD+READ",
            f"SBS6 fraction distribution in {group} tumours",
            "descriptive",
            setting,
            t[t["msi_group"] == group],
            "SBS6",
            group_label=group,
        )
        q90, q95 = (float(v) for v in np.percentile(s6, [90, 95])) if len(s6) else (np.nan,) * 2
        row["notes"] = f"P90={q90:.4g}, P95={q95:.4g}"
        rows.append(row)
        flagged = sub.assign(sbs6_gt0=(s6 > 0).astype(float), sbs6_gt5pct=(s6 > 0.05).astype(float))
        for flag, name in (("sbs6_gt0", "SBS6 > 0"), ("sbs6_gt5pct", "SBS6 > 5%")):
            r = _prop_row(
                "colorectal_sbs6_leakage",
                "COAD+READ",
                f"share of {group} tumours with {name}",
                "descriptive",
                setting,
                flagged,
                flag,
                "",
            )
            r["group1"] = group
            rows.append(r)
        rows.append(
            spearman_row(
                "colorectal_sbs6_leakage",
                "COAD+READ",
                f"SBS1 fraction vs SBS6 fraction in {group} tumours",
                "descriptive",
                setting,
                t[t["msi_group"] == group],
                "SBS1",
                "SBS6",
                notes="a negative rho is the expected signature of exposure trading between "
                "SBS1 and SBS6",
            )
        )
    return rows


POLE_SIGS = ["SBS10a", "SBS10b", "SBS28"]


def msi_low_mmr_listing(
    table: pd.DataFrame, setting: Setting = PRIMARY, cut: float = MSI_LOW_MMR_CUT
) -> pd.DataFrame:
    """Supplementary listing of MSI tumours with summed MMR fraction below ``cut``.

    A listing aid with a fixed, data-independent cut (not a test, not an exclusion). Uses the
    primary-setting tumours (>= 50 SNVs); a fraction exactly equal to the cut is not listed.
    No explanation is asserted for why these tumours score low.
    """
    t = apply_filters(table[table["msi_group"] == "MSI"], setting.min_snvs, setting.min_cosine)
    t = t[t["mmr_sum"] < cut]
    out = pd.DataFrame(
        {
            "project": t["project"],
            "barcode": t.index,
            "case_id": t["case_id"],
            "msi_status": t["msi_status"],
            "msi_score": t["msi_score"],
            "msi_borderline": t["borderline"],
            "total_snvs": t["total_snvs"],
            "summed_mmr_fraction": t["mmr_sum"],
            "pole_like_fraction_sbs10a_10b_28": sum_signatures(t, POLE_SIGS),
            "cosine": t[COSINE_COL],
        }
    )
    return out.sort_values("summed_mmr_fraction").reset_index(drop=True)


def pole_like_counts(table: pd.DataFrame, setting: Setting = PRIMARY) -> list[dict[str, object]]:
    """Descriptive: tumours dominated by POLE-like signatures (SBS10a+10b+28 fraction > 0.5).

    Reported per MSI group (and per project). Not an exclusion: these tumours are expected to
    score low on summed MMR even when called MSI.
    """
    t = apply_filters(table[table["msi_group"].notna()], setting.min_snvs, setting.min_cosine)
    t = t.assign(pole_like=(sum_signatures(t, POLE_SIGS) > 0.5).astype(float))
    rows = []
    for cohort in ("COAD+READ", "COAD", "READ"):
        sub = t if cohort == "COAD+READ" else t[t["project"] == f"TCGA-{cohort}"]
        for group in ("MSI", "MSS"):
            g = sub[sub["msi_group"] == group]
            r = _prop_row(
                "colorectal_pole_like",
                cohort,
                f"{group} tumours with SBS10a+SBS10b+SBS28 fraction > 0.5",
                "descriptive",
                setting,
                g,
                "pole_like",
                "descriptive count, not an exclusion",
            )
            r["group1"] = group
            rows.append(r)
    return rows


# --------------------------------------------------------------------------------------
# Group labels for the cohort-flow table
# --------------------------------------------------------------------------------------


def lung_flow_groups(clinical: pd.DataFrame) -> pd.Series:
    """never / smoker (codes 2+4) / other_ever (3+5) / unknown_or_conflict, by barcode."""
    prim = clinical["smoking_primary_group"]
    ever = clinical["smoking_ever_group"]
    g = pd.Series("unknown_or_conflict", index=clinical["barcode"].to_numpy())
    g[(prim == "never").to_numpy()] = "never"
    g[(prim == "smoker").to_numpy()] = "smoker"
    g[(prim.isna() & (ever == "ever")).to_numpy()] = "other_ever"
    return g


def colorectal_flow_groups(clinical: pd.DataFrame) -> pd.Series:
    """MSI / MSS / no_msi_call, by barcode."""
    st = clinical["msi_status"].where(clinical["msi_status"].isin(["MSI", "MSS"]), "no_msi_call")
    return pd.Series(st.to_numpy(), index=clinical["barcode"].to_numpy())


def skcm_flow_groups(clinical: pd.DataFrame) -> pd.Series:
    """'sample type 01' / 'sample type 06' etc., by barcode."""
    return pd.Series(
        ("sample type " + clinical["sample_type_code"].astype(str)).to_numpy(),
        index=clinical["barcode"].to_numpy(),
    )


# --------------------------------------------------------------------------------------
# Summary writer
# --------------------------------------------------------------------------------------


def _fmt(v: object, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NA"
    if isinstance(v, float):
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.{nd}f}"
    return str(v)


def table_to_markdown(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render selected columns as a GitHub-flavoured Markdown table."""
    cols = [c for c in columns if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def write_markdown(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------------------
# Figures (matplotlib, Okabe-Ito colourblind-safe palette)
# --------------------------------------------------------------------------------------

#: Standard SBS substitution colours (C>A blue, C>G black, C>T red, T>A grey, T>C green, T>G pink).
SUB_COLOURS = {
    "C>A": "#03BCEE",
    "C>G": "#010101",
    "C>T": "#E32926",
    "T>A": "#999999",
    "T>C": "#A1CE63",
    "T>G": "#EBC6C4",
}
OI = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#999999",
}
DPI = 200


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _box_points(ax, groups: dict[str, np.ndarray], colours: dict[str, str], seed: int = SEED):
    rng = np.random.default_rng(seed)
    names = list(groups)
    data = [groups[g] for g in names]
    ax.boxplot(
        data,
        positions=range(len(names)),
        widths=0.55,
        showfliers=False,
        medianprops={"color": "black"},
    )
    for i, g in enumerate(names):
        v = groups[g]
        ax.scatter(
            i + rng.uniform(-0.18, 0.18, len(v)),
            v,
            s=10,
            alpha=0.55,
            color=colours.get(g, OI["grey"]),
            edgecolor="none",
            zorder=3,
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{g}\n(n={len(groups[g])})" for g in names])


def plot_mean_spectra(
    fractions_by_cohort: dict[str, pd.DataFrame], path: str | Path, plot_order: Sequence[str]
) -> Path:
    """Mean per-tumour SBS96 spectrum (fractions) per cohort, standard six colours."""
    plt = _plt()
    n = len(fractions_by_cohort)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.1 * n + 0.8), sharex=True)
    axes = np.atleast_1d(axes)
    cols = [SUB_COLOURS[c[2:5]] for c in plot_order]
    for ax, (name, frac) in zip(axes, fractions_by_cohort.items(), strict=True):
        mean = frac.reindex(plot_order).mean(axis=1)
        ax.bar(range(96), mean.to_numpy(), color=cols, width=0.85)
        ax.set_ylabel("mean fraction")
        ax.set_title(f"{name} (n={frac.shape[1]} tumours)", loc="left", fontsize=10)
        ax.set_xlim(-0.5, 95.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(range(96))
    axes[-1].set_xticklabels([c[0] + c[2] + c[6] for c in plot_order], rotation=90, fontsize=5)
    for i, sub in enumerate(SUB_COLOURS):
        axes[0].text(i * 16 + 7.5, axes[0].get_ylim()[1] * 0.95, sub, ha="center", fontsize=9)
    fig.suptitle(
        "Mean SBS96 mutation spectrum per cohort (tumours with >= 50 SNVs); "
        "y-axis scales differ between panels",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_lung_groups(
    cohorts: dict[str, pd.DataFrame], stats_rows: dict[str, str], path: str | Path
) -> Path:
    """SBS4 fraction by smoking group (never / smoker 2+4 / other ever 3+5), LUAD and LUSC."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.2 * len(cohorts), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    colours = {
        "never": OI["sky"],
        "smoker (2+4)": OI["vermillion"],
        "other ever (3+5)": OI["orange"],
    }
    for ax, (name, t) in zip(axes, cohorts.items(), strict=True):
        t = apply_filters(t, PRIMARY.min_snvs)
        code = t["smoking_code"].astype(str)
        groups = {
            "never": t.loc[code == "1", SBS4].to_numpy(float),
            "smoker (2+4)": t.loc[code.isin(["2", "4"]), SBS4].to_numpy(float),
            "other ever (3+5)": t.loc[code.isin(["3", "5"]), SBS4].to_numpy(float),
        }
        _box_points(ax, groups, colours)
        ax.set_title(f"{name}\n{stats_rows.get(name, '')}", fontsize=9)
        ax.set_ylabel("SBS4 fraction (NNLS-attributed)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SBS4 fraction by smoking history (>= 50 SNVs)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(
        0.5,
        0.01,
        "The primary contrast is never vs smoker (2+4). 'other ever (3+5)' (reformed >15 y or "
        "duration not given) is shown for context only and is not part of it.",
        ha="center",
        fontsize=8,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pack_years(
    cohorts: dict[str, pd.DataFrame], stats_rows: dict[str, str], path: str | Path
) -> Path:
    """SBS4 fraction vs pack-years among ever-smokers."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.2 * len(cohorts), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, t) in zip(axes, cohorts.items(), strict=True):
        t = apply_filters(t, PRIMARY.min_snvs)
        e = t[t["smoking_ever_group"] == "ever"]
        py = pd.to_numeric(e["pack_years_smoked"], errors="coerce")
        ok = py.notna()
        ax.scatter(py[ok], e.loc[ok, SBS4], s=14, alpha=0.6, color=OI["blue"], edgecolor="none")
        ax.set_xlabel("pack-years smoked")
        ax.set_title(f"{name}\n{stats_rows.get(name, '')}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("SBS4 fraction (NNLS-attributed)")
    fig.suptitle("SBS4 fraction vs pack-years, ever-smokers", fontsize=11)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_melanoma(
    table: pd.DataFrame,
    ct_by_cohort: dict[str, pd.DataFrame],
    sig_order: Sequence[str],
    path: str | Path,
) -> Path:
    """Sorted stacked fractions (SBS7a-d summed vs others) plus C>T-at-dipyrimidine by cohort."""
    plt = _plt()
    t = apply_filters(table, PRIMARY.min_snvs).sort_values("uv_sbs7a_d", ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    palette = [OI["vermillion"], OI["sky"], OI["green"], OI["orange"], OI["purple"], OI["grey"]]
    bottom = np.zeros(len(t))
    parts = [("SBS7a-d (summed)", t["uv_sbs7a_d"].to_numpy())]
    for s in sig_order:
        if s not in UV_SIGS:
            parts.append((s, t[s].to_numpy()))
    for (name, vals), c in zip(parts, palette, strict=False):
        a1.bar(range(len(t)), vals, bottom=bottom, color=c, width=1.0, label=name)
        bottom += vals
    a1.set_xlabel(f"SKCM tumours sorted by summed SBS7a-d (n={len(t)})")
    a1.set_ylabel("fraction")
    a1.set_xticks([])
    a1.set_xlim(-0.5, len(t) - 0.5)
    a1.set_ylim(0, 1)
    a1.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    a1.set_title("Signature fractions per melanoma", fontsize=10)
    groups = {
        k: apply_filters(v, PRIMARY.min_snvs)["ct_dipyrimidine"].dropna().to_numpy()
        for k, v in ct_by_cohort.items()
    }
    colours = {k: (OI["vermillion"] if k == "SKCM" else OI["sky"]) for k in groups}
    _box_points(a2, groups, colours)
    a2.set_ylabel("fraction of SNVs that are C>T at a dipyrimidine")
    a2.set_title("Model-free check (raw SBS96, no fit)", fontsize=10)
    for lab in a2.get_xticklabels():
        lab.set_fontsize(7)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_colorectal(table: pd.DataFrame, stats_text: str, path: str | Path) -> Path:
    """Summed MMR fraction and total SNVs by MSI status."""
    plt = _plt()
    t = apply_filters(table[table["msi_group"].notna()], PRIMARY.min_snvs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
    colours = {"MSS": OI["sky"], "MSI": OI["vermillion"]}
    g = {k: t.loc[t["msi_group"] == k, "mmr_sum"].to_numpy(float) for k in ("MSS", "MSI")}
    _box_points(a1, g, colours)
    a1.set_ylabel("summed MMR fraction (SBS6+14+15+20+26+44)")
    a1.set_title("Summed MMR fraction\n" + stats_text, fontsize=9)
    g2 = {k: t.loc[t["msi_group"] == k, "total_snvs"].to_numpy(float) for k in ("MSS", "MSI")}
    _box_points(a2, g2, colours)
    a2.set_yscale("log")
    a2.set_ylabel("SBS96 SNVs per tumour (log scale)")
    a2.set_title("Total SNVs by MSI status (GDC label)", fontsize=9)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("COAD + READ, tumours with >= 50 SNVs", fontsize=11)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Generated summary (all numbers come from the tables)
# --------------------------------------------------------------------------------------

SUMMARY_COLS = [
    "cohort", "comparison", "group1", "label", "setting", "n1", "n2", "median1", "iqr1",
    "median2", "iqr2", "effect_name", "effect", "ci_low", "ci_high", "p", "min", "max",
    "median_cosine1", "median_cosine2", "notes",
]  # fmt: skip

TABLE_TITLES = {
    "lung_sbs4": "A. Lung: SBS4 fraction, never-smokers vs smokers (group 1 = smoker/ever)",
    "lung_dose_response": "A. Lung: SBS4 fraction vs pack-years (Spearman, ever-smokers)",
    "lung_by_smoking_group": "A. Lung: SBS4 fraction by smoking group (descriptive)",
    "melanoma_uv": "B. Melanoma: UV-signature fractions (descriptive; proportions with Wilson CI)",
    "melanoma_adjacent_check": "B. Melanoma: D5 adjacent_snvs check (paired)",
    "melanoma_ct_control": "B. Melanoma: model-free C>T control, SKCM vs other cohorts",
    "colorectal_mmr": "C. Colorectal: summed MMR fraction, MSI vs MSS",
    "colorectal_mmr_noSBS18": "C. Colorectal: summed MMR, fit without SBS18",
    "colorectal_mmr_per_signature": "C. Colorectal: each MMR signature, MSI vs MSS (descriptive)",
    "colorectal_hypermutation": "C. Colorectal: total SNVs by MSI status, and MSI share by case",
    "colorectal_sbs6_leakage": "C. Colorectal: SBS6 distribution and SBS1-SBS6 correlation",
    "colorectal_pole_like": "C. Colorectal: POLE-like tumours (SBS10a+10b+28 > 0.5), counts",
}
SUMMARY_ORDER = ["primary", "secondary", "sensitivity", "descriptive"]


def _pick(df: pd.DataFrame, **kw) -> pd.Series | None:
    sel = df
    for k, val in kw.items():
        sel = sel[sel[k] == val]
    return sel.iloc[0] if len(sel) else None


def _ci(r: pd.Series, nd: int = 2, label: bool = True) -> str:
    tag = "95% CI " if label else ""
    return f"{r['effect']:.{nd}f} ({tag}{r['ci_low']:.{nd}f}-{r['ci_high']:.{nd}f})"


def _pfmt(p: float) -> str:
    return "NA" if p is None or pd.isna(p) else f"{p:.2g}"


def _direction(r: pd.Series) -> str:
    """Rule: the AUC CI excludes 0.5 in the expected direction (group 1 higher)."""
    return "consistent with" if r["ci_low"] > 0.5 else "not consistent with"


def headline_lines(
    t: dict[str, pd.DataFrame], extras: dict[str, object] | None = None
) -> list[str]:
    """Rule-based headline bullets from the tables; only 'consistent with' wording."""
    extras = extras or {}
    out = []
    r = _pick(t["lung_sbs4"], cohort="LUAD", label="primary")
    if r is not None:
        out.append(
            f"- **Lung (LUAD), primary.** SBS4 fraction in smokers (current + reformed <=15 y; "
            f"n={int(r['n1'])}, median {r['median1']:.3f}, IQR {r['iqr1']}) vs lifelong "
            f"never-smokers (n={int(r['n2'])}, median {r['median2']:.3f}, IQR {r['iqr2']}): "
            f"AUC {_ci(r)}, Mann-Whitney p={_pfmt(r['p'])}. The AUC interval "
            f"{'excludes' if r['ci_low'] > 0.5 else 'does not exclude'} 0.5 in the direction "
            f"'smokers higher', which is {_direction(r)} higher SBS4 attribution in smokers."
        )
    out += _never_smoker_lines(t)
    r = _pick(t["melanoma_uv"], analysis="melanoma_uv_fraction", label="primary")
    if r is not None:
        lg = _pick(
            t["melanoma_uv"],
            analysis="melanoma_uv_proportion",
            setting="primary",
            comparison="SBS7a-d (summed) is the single largest component; "
            "profile adjacent_snvs exclude (default)",
        )
        gt = _pick(
            t["melanoma_uv"],
            analysis="melanoma_uv_proportion",
            setting="primary",
            comparison="summed SBS7a-d fraction > 50%; profile adjacent_snvs exclude (default)",
        )
        word = "consistent with" if r["median1"] > 0.5 else "not consistent with"
        out.append(
            f"- **Melanoma (SKCM), primary.** Summed SBS7a-d fraction: n={int(r['n1'])}, median "
            f"{r['median1']:.3f} (IQR {r['iqr1']}); summed SBS7a-d is the single largest "
            f"component in {round(lg['effect'] * lg['n1'])}/{int(lg['n1'])} tumours "
            f"({lg['effect'] * 100:.1f}%) and above 50% in "
            f"{round(gt['effect'] * gt['n1'])}/{int(gt['n1'])} ({gt['effect'] * 100:.1f}%); "
            "these two proportions are secondary measures; the counts coincide because a tumour "
            "above 50% is necessarily the largest component and here every largest-component "
            "tumour is also above 50%. "
            f"This is {word} UV-signature dominance (rule: median summed fraction > 0.5)."
        )
    r = _pick(t["colorectal_mmr"], cohort="COAD+READ", label="primary")
    if r is not None:
        out.append(
            f"- **Colorectal (COAD+READ), primary.** Summed MMR fraction (SBS6+14+15+20+26+44) in "
            f"MSI (n={int(r['n1'])}, median {r['median1']:.3f}, IQR {r['iqr1']}) vs MSS "
            f"(n={int(r['n2'])}, median {r['median2']:.3f}, IQR {r['iqr2']}): AUC {_ci(r)}, "
            f"p={_pfmt(r['p'])}. The AUC interval "
            f"{'excludes' if r['ci_low'] > 0.5 else 'does not exclude'} 0.5 in the direction "
            f"'MSI higher', which is {_direction(r)} higher "
            "mismatch-repair signature attribution in MSI-labelled tumours."
        )
    out += _msi_prevalence_lines(t)
    out += ["", "Key limiting and negative findings (same tables):", ""]
    out += _dose_response_lines(t["lung_dose_response"])
    sel = t["lung_sbs4"]
    sel = sel[
        (sel["cohort"] == "LUSC")
        & (sel["setting"] == "primary")
        & (sel["analysis"] == "lung_sbs4_primary_contrast")
        & sel["comparison"].str.endswith("fit with SBS18")
    ]
    if len(sel):
        r = sel.iloc[0]
        out.append(
            f"- LUSC is underpowered ({int(r['n2'])} lifelong never-smokers at >= 50 SNVs): AUC "
            f"{_ci(r)}, p={_pfmt(r['p'])}; reported descriptively."
        )
    r = _pick(t["colorectal_mmr"], cohort="COAD+READ", label="primary")
    sbs1 = t["colorectal_sbs6_leakage"]
    rho = sbs1[
        sbs1["comparison"].str.contains("SBS1 fraction vs SBS6") & (sbs1["group1"] == "SBS1")
    ]
    rho = rho[rho["comparison"].str.endswith("MSS tumours")]
    gt0 = sbs1[sbs1["comparison"] == "share of MSS tumours with SBS6 > 0"]
    if r is not None and len(rho) and len(gt0):
        out.append(
            f"- MSS tumours are not at zero: median summed MMR fraction {r['median2']:.3f} "
            f"(IQR {r['iqr2']}); {gt0.iloc[0]['effect'] * 100:.0f}% of MSS tumours have SBS6 > 0; "
            f"SBS1 vs SBS6 fraction in MSS Spearman rho {rho.iloc[0]['effect']:.2f}. "
            "This pattern is consistent with exposure trading between similar signatures "
            "(SBS1/SBS6) and the noise floor of NNLS refitting, so some MMR attribution in MSS "
            "tumours may not reflect a real mismatch-repair defect."
        )
    r = _pick(t["colorectal_mmr"], cohort="READ", setting="primary", label="secondary")
    if r is not None:
        out.append(
            f"- READ has only {int(r['n1'])} MSI tumours at >= 50 SNVs (vs {int(r['n2'])} MSS): "
            f"AUC {_ci(r)} is imprecise."
        )
    r = _pick(t["melanoma_adjacent_check"], label="sensitivity")
    if r is not None:
        mx = max(abs(r["min"]), abs(r["max"]))
        big = abs(r["effect"]) > MATERIAL_CHANGE
        out.append(
            f"- D5 check (adjacent SNVs counted as SBS vs excluded), summed SBS7a-d: paired "
            f"median difference {r['effect']:.3f}, largest single-tumour |difference| {mx:.3f} "
            f"(n={int(r['n1'])}); the median difference is "
            f"{'above' if big else 'below'} the {MATERIAL_CHANGE} reporting threshold "
            f"(the threshold applies to the median, not to individual tumours)."
        )
    out += _extras_lines(extras)
    return out


def _rho_phrase(r: pd.Series, count: bool = False) -> str:
    positive = (
        "association with SBS4 count, which also tracks total burden"
        if count
        else "consistent with a monotonic association"
    )
    word = (
        "not consistent with a monotonic dose-response"
        if r["ci_low"] <= 0 <= r["ci_high"]
        else positive
    )
    return (
        f"Spearman rho {r['effect']:.2f} (95% CI {r['ci_low']:.2f} to {r['ci_high']:.2f}), "
        f"p={_pfmt(r['p'])}, n={int(r['n1'])}: {word}"
    )


def _dose_response_lines(dr: pd.DataFrame) -> list[str]:
    """Pack-years vs SBS4 fraction (pre-specified) and vs SBS4 mutation count (added, D9)."""
    out = []
    ok = dr["comparison"].str.endswith("fit with SBS18") & (dr["setting"] == "primary")
    for c in ("LUAD", "LUSC"):
        fr = dr[ok & (dr["cohort"] == c) & (dr["analysis"] == "lung_sbs4_dose_response")]
        ct = dr[ok & (dr["cohort"] == c) & (dr["analysis"] == "lung_sbs4_count_dose_response")]
        if len(fr):
            out.append(
                f"- Pack-years vs SBS4 fraction in ever-smokers, {c}: {_rho_phrase(fr.iloc[0])}."
            )
        if len(ct):
            out.append(
                f"- Pack-years vs SBS4 mutation count (NNLS-attributed, added secondary test), "
                f"{c}: {_rho_phrase(ct.iloc[0], count=True)}."
            )
    if out:
        out.append(
            "- Context for the two dose-response tests: an exposure fraction is compositional "
            "(the fractions sum to 1 across the fitted signatures, so it depends on every other "
            "signature) and pack-years are self-reported and noisy, so a weak or absent "
            "dose-response was not expected to be a sensitive test. The SBS4 count scales with "
            "total mutational burden, so the count association may partly track burden. The "
            "count test is an added secondary test with no multiplicity correction, and it was "
            "added after the fraction result was seen (at Prithviraj's request)."
        )
    return out


def _never_smoker_lines(t: dict[str, pd.DataFrame]) -> list[str]:
    ns = t.get("lung_never_smoker_sbs4")
    if ns is None:
        return []
    luad = ns[ns["cohort"] == "LUAD"]
    share = luad[luad["comparison"].str.startswith("share of")]
    q3 = luad[luad["effect_name"] == "upper quartile (Q3)"]
    mx = luad[luad["effect_name"] == "maximum"]
    if not (len(share) and len(q3) and len(mx)):
        return []
    r = share.iloc[0]
    k = round(r["effect"] * r["n1"])
    return [
        "- A minority of reported never-smokers show tobacco-like SBS4 exposure. Possible reasons "
        "include misreported smoking history, second-hand smoke, or overlap between SBS4 and "
        "similar signatures; this analysis cannot tell these apart. In LUAD at >= 50 SNVs, "
        f"{k}/{int(r['n1'])} lifelong never-smokers ({r['effect'] * 100:.0f}%, Wilson 95% CI "
        f"{r['ci_low'] * 100:.0f}-{r['ci_high'] * 100:.0f}%) have SBS4 fraction > "
        f"{NEVER_SMOKER_SBS4_CUT} (a descriptive listing aid, not a test); the upper quartile of "
        f"their SBS4 fraction is {q3.iloc[0]['effect']:.2f} and the maximum "
        f"{mx.iloc[0]['effect']:.2f}. Context: Zhang et al. 2021 (Nat Genet) report no strong "
        "tobacco-smoking signatures in never-smoker lung cancers, even with second-hand smoke "
        "exposure (see References)."
    ]


def _msi_prevalence_lines(t: dict[str, pd.DataFrame]) -> list[str]:
    h = t.get("colorectal_hypermutation")
    if h is None:
        return []
    h = h[h["analysis"] == "colorectal_msi_fraction_by_case"]
    parts = {}
    for _, r in h.iterrows():
        k = int(r["n1"])
        n = k + int(r["n2"])
        parts[r["cohort"]] = (
            f"{r['effect'] * 100:.1f}% ({k}/{n}; Wilson 95% CI "
            f"{r['ci_low'] * 100:.1f}-{r['ci_high'] * 100:.1f}%)"
        )
    if not {"COAD", "READ"} <= set(parts):
        return []
    return [
        f"- MSI prevalence by case among cases with an MSI call: colon (COAD) {parts['COAD']}; "
        f"rectum (READ) {parts['READ']}. MSI is expected to be rarer in rectal than in colon "
        "cancer, so a lower READ share is expected. The 10-20% sanity check (TCGA Network 2012, "
        "Nature 487:330-337) applies to colon only."
    ]


def _extras_lines(extras: dict[str, object]) -> list[str]:
    out = []
    low = extras.get("msi_low_mmr")
    if isinstance(low, pd.DataFrame):
        if len(low):
            out.append(
                f"- {len(low)} MSI-labelled tumours (>= 50 SNVs) have summed MMR fraction below "
                f"{MSI_LOW_MMR_CUT} (a fixed listing cut, not a test): MSI score "
                f"{low['msi_score'].min():.3f}-{low['msi_score'].max():.3f}, total SNVs "
                f"{int(low['total_snvs'].min())}-{int(low['total_snvs'].max())}, "
                f"{int((low['pole_like_fraction_sbs10a_10b_28'] > 0.5).sum())} with SBS10a+10b+28 "
                "> 0.5. They are kept in all analyses and listed in "
                "validation_colorectal_msi_low_mmr.tsv. Low MMR in these may reflect MSIsensor2 "
                "false calls, low mutation counts, or POLE-like/unfitted signatures; no "
                "explanation is asserted."
            )
        else:
            out.append(
                f"- No MSI-labelled tumours (>= 50 SNVs) have summed MMR fraction below "
                f"{MSI_LOW_MMR_CUT}."
            )
    nu = extras.get("melanoma_non_uv")
    if isinstance(nu, pd.DataFrame):
        out.append(
            f"- {len(nu)} melanomas (>= 50 SNVs) are not UV-dominated (summed SBS7a-d <= "
            f"{NON_UV_CUT}); they are flagged in validation_melanoma_non_uv.tsv and kept in all "
            "analyses."
        )
    return out


def selection_section(t: dict[str, pd.DataFrame], flow: pd.DataFrame) -> list[str]:
    """Prose + table on how the >=50 SNV filter interacts with smoking status (LUAD)."""
    out = [
        "The SNV-count filter uses only each tumour's SNV total and never a smoking or MSI "
        "label (it is label-blind). It is nonetheless strongly correlated with smoking status "
        "in LUAD, because lifelong never-smokers have fewer mutations."
    ]
    fl = flow[flow["cohort"] == "LUAD"].set_index("group")
    if {"never", "smoker"} <= set(fl.index):
        nv, sm = fl.loc["never"], fl.loc["smoker"]
        out += [
            "",
            f"LUAD lifelong never-smokers: {int(nv['chosen_cases'])} chosen cases, "
            f"{int(nv['in_fit_table'])} with >= 20 SNVs (in the fit table), "
            f"{int(nv['snvs_ge_50'])} with >= 50, {int(nv['snvs_ge_100'])} with >= 100. "
            f"Smokers (current + reformed <=15 y): {int(sm['chosen_cases'])} chosen, "
            f"{int(sm['snvs_ge_50'])} ({sm['snvs_ge_50'] / sm['chosen_cases'] * 100:.0f}%) with "
            f">= 50 and {int(sm['snvs_ge_100'])} "
            f"({sm['snvs_ge_100'] / sm['chosen_cases'] * 100:.0f}%) with >= 100. Never-smokers "
            f"retained at >= 50: {nv['snvs_ge_50'] / nv['chosen_cases'] * 100:.0f}%.",
        ]
    lt = t["lung_sbs4"]
    lt = lt[
        (lt["cohort"] == "LUAD")
        & (lt["analysis"] == "lung_sbs4_primary_contrast")
        & lt["comparison"].str.endswith("fit with SBS18")
    ].set_index("setting")
    names = [("min_snvs_20", ">= 20 SNVs"), ("primary", ">= 50 SNVs (primary)"),
             ("min_snvs_100", ">= 100 SNVs")]  # fmt: skip
    if all(n in lt.index for n, _ in names):
        out += ["", "LUAD SBS4, smokers vs never-smokers, by SNV threshold (side by side):", ""]
        out += [
            "| threshold | n smokers | n never | AUC (95% CI) | p "
            "| median cosine (smokers / never) |",
            "|---|---|---|---|---|---|",
        ]
        for key, nm in names:
            r = lt.loc[key]
            out.append(
                f"| {nm} | {int(r['n1'])} | {int(r['n2'])} | {_ci(r, label=False)} | "
                f"{_pfmt(r['p'])} | {r['median_cosine1']:.2f} / {r['median_cosine2']:.2f} |"
            )
        aucs = [lt.loc[k, "effect"] for k, _ in names]
        lows = [lt.loc[k, "ci_low"] for k, _ in names]
        out += [
            "",
            f"Across the three thresholds the LUAD AUC ranges from {min(aucs):.2f} to "
            f"{max(aucs):.2f} and every 95% CI lower bound is above 0.5 "
            f"({'yes' if min(lows) > 0.5 else 'NO'}). The >= 50 SNV row is the primary; the "
            ">= 20 and >= 100 rows are sensitivity analyses shown in the same table, and the "
            "headline is never switched to whichever threshold looks better.",
        ]
        p = lt.loc["primary"]
        out += [
            "",
            f"Never-smoker fits are poorer at the primary setting (median reconstruction cosine "
            f"{p['median_cosine2']:.2f} vs {p['median_cosine1']:.2f} in smokers). The effect size "
            "changes with the threshold because the filter changes who is compared, not because "
            "the test changed. **The primary analysis stays at >= 50 SNVs as pre-specified** "
            "(D7); the other thresholds are sensitivity analyses and were not used to choose it.",
        ]
    out += ["", "Cohort flow (chosen cases to analysed tumours), all cohorts:", ""]
    cols = list(flow.columns)
    out.append(table_to_markdown(flow, cols))
    return out


REFERENCES = """\
- Alexandrov LB et al. (2020) The repertoire of mutational signatures in human cancer. Nature
  578:94-101.
- The Cancer Genome Atlas Network (2012) Comprehensive molecular characterization of human colon
  and rectal cancer. Nature 487:330-337.
- Zhang T et al. (2021) Genomic and evolutionary classification of lung cancer in never smokers.
  Nat Genet 53(9):1348-1359. doi:10.1038/s41588-021-00920-0. PMID 34493867. Its abstract reports
  that no strong tobacco smoking signatures were detected in never-smoker lung cancers, even with
  second-hand smoke exposure (context for the SBS4 overlap in never-smokers; not a result of
  this analysis).
"""


def supplementary_section(extras: dict[str, object]) -> list[str]:
    """Supplementary listings (flags only, nothing is excluded from any analysis)."""
    out = ["## Supplementary listings", ""]
    low = extras.get("msi_low_mmr")
    if isinstance(low, pd.DataFrame):
        out += [
            f"### MSI tumours with summed MMR fraction < {MSI_LOW_MMR_CUT}",
            "",
            f"The cut {MSI_LOW_MMR_CUT} was fixed before looking at the data and is a listing aid, "
            "not a test. Tumours with >= 50 SNVs only; MSI labels are bioinformatic. Count: "
            f"{len(low)}. Low MMR in these may reflect MSIsensor2 false calls, low mutation "
            "counts, or POLE-like/unfitted signatures; no explanation is asserted. Full list: "
            "`results/tables/validation_colorectal_msi_low_mmr.tsv`.",
            "",
        ]
        if len(low):
            out += [table_to_markdown(low, list(low.columns)), ""]
    nu = extras.get("melanoma_non_uv")
    if isinstance(nu, pd.DataFrame):
        out += [
            f"### Melanomas that are not UV-dominated (summed SBS7a-d <= {NON_UV_CUT})",
            "",
            f"Count: {len(nu)} (this is the complement of the secondary '> 50%' measure; the "
            "boundary is inclusive). Flagged, not excluded. Full list: "
            "`results/tables/validation_melanoma_non_uv.tsv`.",
            "",
        ]
        if len(nu):
            cols = [c for c in nu.columns if c not in ("case_id",)]
            out += [table_to_markdown(nu, cols), ""]
        fc = extras.get("non_uv_fields")
        if isinstance(fc, pd.DataFrame) and len(fc):
            out += [
                "Histology and site fields among non-UV melanomas vs all analysed SKCM tumours "
                "(counts):",
                "",
                table_to_markdown(fc, list(fc.columns)),
                "",
            ]
        bs = extras.get("non_uv_biopsy_site")
        if isinstance(bs, pd.DataFrame) and len(bs):
            out += [
                "Biopsy site (`site_of_resection_or_biopsy`) among non-UV melanomas vs all "
                "analysed SKCM tumours, split by sample type (01 primary, 06 metastatic; counts):",
                "",
                table_to_markdown(bs, list(bs.columns)),
                "",
            ]
        note = extras.get("skcm_fields_note")
        if note:
            out += [str(note), ""]
    return out


def limitations(drops: pd.DataFrame | None = None) -> str:
    items = [
        "**SBS1/SBS6 overlap.** SBS1 and SBS6 (and SBS5/SBS40a) are similar, so NNLS can trade "
        "exposure between them. Some SBS6 in MSS tumours may be leakage rather than a real MMR "
        "defect; see the SBS6 distribution and SBS1-SBS6 correlation above.",
        "**MSI labels are bioinformatic** (GDC MSIsensor2, tumour BAM only, 20% score cutoff), "
        "not clinical MSI/MMR tests.",
        "**Subset prevalences are approximate**: read from figure dots (Alexandrov 2020, ED "
        "Fig. 5).",
        "**SBS88 (colibactin), SBS36 and SBS10c/d are not fitted** in colorectal.",
        "**Fractions are NNLS-attributed**, not observed counts, and depend on the approved "
        "signature subset.",
        "**POLE-like hypermutators** (SBS10a/b/28 dominated) can be MSI-labelled yet score low "
        "on summed MMR; they are counted in the POLE table, not excluded.",
        "**Lower fit quality in never-smokers.** Reconstruction cosine is lower in LUAD "
        "never-smokers than in smokers (see the table above), and the >= 50 SNV filter removes "
        "many never-smokers; fractions of tumours with few SNVs are noisy.",
        "**Multiple testing.** There are three pre-specified primary analyses: LUAD SBS4 "
        "(never vs smoker), SKCM summed SBS7a-d fraction (a descriptive summary, no test), and "
        "COAD+READ summed MMR (MSI vs MSS). Secondary and sensitivity analyses are not "
        "independent tests and no multiplicity correction is applied.",
        "**Refitting, not machine learning.** Signatures are assigned by non-negative "
        "least-squares refitting to published COSMIC v3.4 signatures; nothing is trained on "
        "COSMIC data.",
        "**Research and education only; not for diagnosis.**",
        "Wording: results are described as consistent or not consistent with the expected "
        "biology; no causal claim is made.",
        "Group 1 is the group expected higher; AUC > 0.5 means group 1 tends higher. The "
        "rank-biserial correlation is 2*AUC-1. CIs are percentile bootstrap (2000 resamples, "
        "fixed seed).",
    ]
    if drops is not None and {"tri_allelic", "multiallelic_position"} <= set(drops.columns):
        tri = int(drops["tri_allelic"].sum())
        multi = int(drops["multiallelic_position"].sum())
        items.append(
            f"**D6/D6b filters did nothing on these data.** Across all five cohorts the profile "
            f"step dropped {tri} tri-allelic rows and {multi} multi-allelic-position rows "
            "(results/tables/profile_drops_summary.tsv), so those rules did not change any "
            "count. The dominant drops are non-SNP rows and (default mode) adjacent SNVs."
        )
    return "\n".join(f"- {i}" for i in items) + "\n"


def build_summary(
    tables: dict[str, pd.DataFrame],
    flow: pd.DataFrame,
    drops: pd.DataFrame | None = None,
    borderline: pd.DataFrame | None = None,
    extras: dict[str, object] | None = None,
) -> str:
    """Assemble the whole Markdown summary from the tables.

    ``extras`` may hold ``msi_low_mmr`` and ``melanoma_non_uv`` (supplementary listings),
    ``non_uv_fields`` (value counts) and ``skcm_fields_note``.
    """
    extras = extras or {}
    parts = [
        "# Validation summary",
        "",
        "Generated by `scripts/run_validation.py`; every number comes from the "
        "`results/tables/validation_*.tsv` tables. Primary analyses use tumours with >= 50 SNVs. "
        "Sensitivities: min_snvs 20 and 100, reconstruction cosine >= 0.8 (a quality filter that "
        "never uses smoking or MSI status), and the variants named in each row. Median cosine is "
        "the median reconstruction cosine of the tumours in the group. Descriptive rows list "
        "min/max in their own columns.",
        "",
        "## Headline",
        "",
        *headline_lines(tables, extras),
        "",
        "## Cohort flow and the SNV filter",
        "",
        *selection_section(tables, flow),
        "",
    ]
    if borderline is not None and len(borderline):
        parts += [
            "Borderline MSI scores (0.15-0.25, D4). 'clinical' counts are chosen cases with an "
            "MSI call; 'in fit table' counts need >= 20 SNVs; 'analysed' counts need >= 50 SNVs "
            "(the primary set). They differ because they are different denominators.",
            "",
            table_to_markdown(borderline, list(borderline.columns)),
            "",
        ]
    for label in SUMMARY_ORDER:
        parts += [f"## {label.capitalize()} results", ""]
        for name, df in tables.items():
            sub = df[df["label"] == label]
            if sub.empty:
                continue
            parts += [
                f"### {TABLE_TITLES.get(name, name)}",
                "",
                table_to_markdown(sub, SUMMARY_COLS),
                "",
            ]
    parts += supplementary_section(extras)
    parts += ["## Limitations and notes", "", limitations(drops)]
    parts += ["## References", "", REFERENCES]
    return "\n".join(parts)
