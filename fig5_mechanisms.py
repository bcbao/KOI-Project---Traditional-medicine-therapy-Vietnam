#!/usr/bin/env python3
"""
Figure 5 — Mechanisms and mediation (Aim 4).

    python fig5_mechanisms.py mindstroke_exam_data_wide.csv mechanisms.csv sessions.csv [-o fig5] [--boot 5000]
    python fig5_mechanisms.py --Lab mindstroke_exam_data_wide.csv mechanisms.csv sessions.csv

Đầu vào
  1) file wide kết cục chính:  id, arm, pss_w0, pss_w8 [, sex, site]
  2) mechanisms.csv  (một dòng = một người × một mốc):  id, week (0/4/8), tms, mastery, enjoyment
        tms        Toronto Mindfulness Scale (state), 0–52 ; tại W0 đo sau một phiên làm quen 10 phút
        mastery    tự đánh giá làm chủ kỹ năng 0–10 ; enjoyment 0–10
  3) sessions.csv    (một dòng = một người × một phiên):  id, session (1–19), mastery, enjoyment [, rhythm_index]
        rhythm_index   chỉ số nhịp nét bút từ MINDSTROKE (chỉ MS, MC; để trống ở MP/CG)

Mô hình (parallel mediation, X = contrast C1 mã hoá +0.5 cho MS/MC, −0.5 cho MP/CG; C2, C3 làm hiệp biến):
  a_i : M_i(W4) ~ X + C2 + C3 + M_i(W0) + pss_w0
  b_i : pss_w8 ~ X + C2 + C3 + ΣM_i(W4) + ΣM_i(W0) + pss_w0       (c' = hệ số của X)
  indirect_i = a_i × b_i ; CI percentile bootstrap (mặc định 5000 lần, theo người).
"""
import argparse, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import statsmodels.api as sm
from scipy import stats

ARMS = {"MS": ("Meaningful Sketching", "#D55E00"), "MC": ("Mindful Calligraphy", "#0072B2"),
        "MP": ("Music Practice", "#009E73"), "CG": ("Casual Gaming", "#CC79A7")}
MEDIATORS = [("tms", "M1 · state mindfulness (TMS, W4)"), ("mastery", "M2 · skill mastery (W4)"),
             ("enjoyment", "M3 · enjoyment (W4)")]
CODES = {"MS": (0.5, 0.5, 0), "MC": (0.5, -0.5, 0), "MP": (-0.5, 0, 0.5), "CG": (-0.5, 0, -0.5)}   # C1, C2, C3
INK, GREY, MUTE = "#1f2937", "#374151", "#6b7280"


# ---------------------------------------------------------------- data
def build(wide, mech):
    m = mech.pivot(index="id", columns="week", values=[k for k, _ in MEDIATORS])
    m.columns = [f"{v}_w{w}" for v, w in m.columns]
    d = wide.merge(m.reset_index(), on="id")
    for i, c in enumerate(("c1", "c2", "c3")):
        d[c] = d["arm"].map(lambda a: CODES[a][i])
    need = ["pss_w0", "pss_w8"] + [f"{k}_w{w}" for k, _ in MEDIATORS for w in (0, 4)]
    return d.dropna(subset=need).reset_index(drop=True)


def covariate_frame(d):
    X = d[["c1", "c2", "c3", "pss_w0"]].copy()
    if "sex" in d.columns: X["sex_M"] = (d["sex"] == "M").astype(float)
    if "site" in d.columns:
        for lv in sorted(d["site"].dropna().unique())[1:]: X[f"site_{lv}"] = (d["site"] == lv).astype(float)
    return X


def fit_paths(d):
    X_a = sm.add_constant(covariate_frame(d))
    a, b = {}, {}
    for k, _ in MEDIATORS:
        Xa = pd.concat([X_a, d[[f"{k}_w0"]]], axis=1)
        a[k] = sm.OLS(d[f"{k}_w4"], Xa).fit().params["c1"]
    Xb = sm.add_constant(pd.concat([covariate_frame(d), d[[f"{k}_w4" for k, _ in MEDIATORS] + [f"{k}_w0" for k, _ in MEDIATORS]]], axis=1))
    fb = sm.OLS(d["pss_w8"], Xb).fit()
    for k, _ in MEDIATORS:
        b[k] = fb.params[f"{k}_w4"]
    cprime = fb.params["c1"]
    ctot = sm.OLS(d["pss_w8"], X_a).fit().params["c1"]
    return a, b, cprime, ctot, fb


def total_effect_full(wide):
    """C1 (complete-case ANCOVA, cùng hiệp biến) trên toàn bộ người có PSS W8 — để đối chiếu với c của mẫu mediation."""
    d = wide.dropna(subset=["pss_w0", "pss_w8"]).copy()
    for i, c in enumerate(("c1", "c2", "c3")): d[c] = d["arm"].map(lambda a: CODES[a][i])
    f = sm.OLS(d["pss_w8"], sm.add_constant(covariate_frame(d))).fit()
    return float(f.params["c1"]), len(d)


def mediation(d, n_boot=5000, seed=1):
    a, b, cp, ct, fb = fit_paths(d)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k, _ in MEDIATORS}; boots["total_ind"] = []; boots["cprime"] = []
    for _ in range(n_boot):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        aa, bb, cpp, _, _ = fit_paths(s)
        tot = 0
        for k, _ in MEDIATORS:
            v = aa[k] * bb[k]; boots[k].append(v); tot += v
        boots["total_ind"].append(tot); boots["cprime"].append(cpp)
    rows = []
    for k, lab in MEDIATORS:
        ind = a[k] * b[k]; lo, hi = np.percentile(boots[k], [2.5, 97.5])
        rows.append(dict(path=lab, a=a[k], b=b[k], indirect=ind, ci_low=lo, ci_high=hi, prop_mediated=ind / ct if ct else np.nan))
    lo, hi = np.percentile(boots["total_ind"], [2.5, 97.5])
    rows.append(dict(path="Total indirect", a=np.nan, b=np.nan, indirect=sum(a[k] * b[k] for k, _ in MEDIATORS), ci_low=lo, ci_high=hi,
                     prop_mediated=sum(a[k] * b[k] for k, _ in MEDIATORS) / ct if ct else np.nan))
    lo, hi = np.percentile(boots["cprime"], [2.5, 97.5])
    rows.append(dict(path="Direct effect c′", a=np.nan, b=np.nan, indirect=cp, ci_low=lo, ci_high=hi, prop_mediated=np.nan))
    res = pd.DataFrame(rows)
    return res, a, b, cp, ct, fb, len(d)


# ---------------------------------------------------------------- figure
def path_diagram(ax, a, b, cp, ct, res, n, ct_full=None, n_full=None):
    ax.set_xlim(0, 10); ax.set_ylim(-0.9, 6); ax.axis("off")
    def box(x, y, w, h, text, fc="white", ec=GREY, tc=INK, fs=6.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")
    def arrow(x0, y0, x1, y1, label, lab_xy, sig):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.6 if sig else 1,
                                     color=INK if sig else "#9ca3af", connectionstyle="arc3,rad=0"))
        ax.text(*lab_xy, label, fontsize=6, color=INK if sig else MUTE, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.6))
    box(0.2, 2.4, 2.3, 1.2, "X · contrast C1\n(MS+MC) vs (MP+CG)", fc=GREY, ec=GREY, tc="white", bold=True)
    box(7.5, 2.4, 2.3, 1.2, "Y · PSS-10 Week 8\n(baseline-adjusted)", fc="#111827", ec="#111827", tc="white", bold=True)
    ys = [4.9, 3.0, 1.1]
    for (k, lab), y, r in zip(MEDIATORS, ys, res.itertuples()):
        box(3.6, y - 0.45, 2.8, 0.9, lab.replace(" · ", "\n"), fs=6)
        sig_ind = not (r.ci_low <= 0 <= r.ci_high)
        arrow(2.5, 3.0, 3.6, y, f"a = {a[k]:+.2f}", (3.05, (3.0 + y) / 2 + (0.25 if y > 3 else -0.25 if y < 3 else 0.38)), sig_ind)
        arrow(6.4, y, 7.5, 3.0, f"b = {b[k]:+.2f}", (6.95, (3.0 + y) / 2 + (0.25 if y > 3 else -0.25 if y < 3 else 0.38)), sig_ind)
    # direct effect along bottom
    ax.add_patch(FancyArrowPatch((1.35, 2.4), (8.65, 2.4), arrowstyle="-|>", mutation_scale=9, lw=1.2, color=INK,
                                 connectionstyle="arc3,rad=0.62", ls=(0, (4, 3))))
    ref = f"; C1 on all complete cases (n = {n_full}) = {ct_full:+.2f}" if ct_full is not None else ""
    ax.text(5.0, -0.7, f"c′ = {cp:+.2f}   (total c = {ct:+.2f} on mediation sample{ref})", fontsize=6.3, ha="center", color=INK)
    ax.text(0.2, 5.75, f"Parallel mediation, n = {n} (complete mediator & outcome data); mediators at W4 adjusted for W0; covariates: C2, C3, baseline PSS, sex, site",
            fontsize=5.6, color=MUTE, va="top")


def forest_ind(ax, res):
    y = np.arange(len(res))[::-1]
    ax.axvline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    for yi, r in zip(y, res.itertuples()):
        sig = not (r.ci_low <= 0 <= r.ci_high); col = INK if sig else "#9ca3af"
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=col, lw=1.4); ax.plot(r.indirect, yi, "s", ms=5, color=col)
        pm = f"  ({r.prop_mediated*100:.0f} % of total)" if not np.isnan(r.prop_mediated) else ""
        ax.text(0.5, yi + 0.3, f"{r.indirect:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]{pm}", transform=ax.get_yaxis_transform(),
                fontsize=5.6, color=MUTE, ha="center", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([r.path for r in res.itertuples()], fontsize=6.3)
    lim = max(abs(res["ci_low"].min()), abs(res["ci_high"].max())) * 1.3; ax.set_xlim(-lim, lim); ax.set_ylim(-0.6, len(res) - 0.3)
    ax.set_xlabel("effect on Week-8 PSS-10 (points) · bootstrap 95 % CI", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5); ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_title("Indirect and direct effects of C1", loc="left", fontsize=7.5, fontweight="bold")


def session_curves(ax, sess, var, title):
    for arm, (name, col) in ARMS.items():
        g = sess[sess["arm"] == arm].groupby("session")[var].agg(["mean", "count", "std"])
        ci = stats.t.ppf(0.975, g["count"] - 1) * g["std"] / np.sqrt(g["count"])
        ax.plot(g.index, g["mean"], color=col, lw=1.4, label=arm)
        ax.fill_between(g.index, g["mean"] - ci, g["mean"] + ci, color=col, alpha=0.12, lw=0)
    ax.set_xlabel("session", fontsize=6.5); ax.set_ylabel(f"{var} (0–10)", fontsize=6.5)
    ax.set_xticks([1, 5, 10, 15, 19]); ax.tick_params(labelsize=6.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower right")


def kinematics(ax, sess, wide):
    r = sess.dropna(subset=["rhythm_index"]).groupby(["id", "arm"])["rhythm_index"].agg(lambda v: v.iloc[-5:].mean()).reset_index()
    r = r.merge(wide[["id", "pss_w0", "pss_w8"]], on="id"); r["dpss"] = r["pss_w8"] - r["pss_w0"]; r = r.dropna()
    for arm in ("MS", "MC"):
        g = r[r["arm"] == arm]; col = ARMS[arm][1]
        ax.plot(g["rhythm_index"], g["dpss"], "o", ms=3, color=col, alpha=0.6, mec="none", label=f"{arm} (n={len(g)})")
        sl, ic, rv, p, _ = stats.linregress(g["rhythm_index"], g["dpss"])
        xs = np.linspace(g["rhythm_index"].min(), g["rhythm_index"].max(), 10); ax.plot(xs, ic + sl * xs, color=col, lw=1.3)
    sl, ic, rv, p, _ = stats.linregress(r["rhythm_index"], r["dpss"])
    ax.text(0.02, 0.03, f"pooled r = {rv:+.2f}, " + (f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"), transform=ax.transAxes, fontsize=5.8, color=MUTE)
    ax.set_xlabel("stroke-rhythm index, last 5 sessions (0–1)", fontsize=6.5); ax.set_ylabel("Δ PSS-10 (W8 − W0)", fontsize=6.5)
    ax.tick_params(labelsize=6.5); ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title("Stroke kinematics and stress change (MS, MC only)", loc="left", fontsize=7.5, fontweight="bold")
    return r


def draw(res, a, b, cp, ct, n, sess, wide, out, ct_full=None, n_full=None, dpi=300):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(17 / 2.54 * 1.6, 12.5 / 2.54 * 1.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.5, wspace=0.45, left=0.07, right=0.98, top=0.94, bottom=0.08)
    axA = fig.add_subplot(gs[0, :2]); axB = fig.add_subplot(gs[0, 2])
    axC1 = fig.add_subplot(gs[1, 0]); axC2 = fig.add_subplot(gs[1, 1]); axD = fig.add_subplot(gs[1, 2])
    path_diagram(axA, a, b, cp, ct, res, n, ct_full, n_full); axA.text(-0.02, 1.02, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")
    forest_ind(axB, res); axB.text(-0.75, 1.1, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
    session_curves(axC1, sess, "mastery", "Skill mastery across sessions"); axC1.text(-0.3, 1.1, "C", transform=axC1.transAxes, fontsize=12, fontweight="bold")
    session_curves(axC2, sess, "enjoyment", "Enjoyment across sessions")
    kinematics(axD, sess, wide); axD.text(-0.3, 1.1, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")
    for ext in ("svg", "png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Lab
def Lab(p_wide, p_mech, p_sess, seed=5):
    rng = np.random.default_rng(seed)
    w = pd.read_csv(p_wide)
    rows_m, rows_s = [], []
    tms_gain = {"MS": 4, "MC": 8, "MP": 3, "CG": 1}; mast_gain = {"MS": 1.6, "MC": 1.9, "MP": 2.3, "CG": 1.4}; enj = {"MS": 7.0, "MC": 6.6, "MP": 6.8, "CG": 7.4}
    for r in w.itertuples():
        u = rng.normal(0, 1)
        t0 = 24 + rng.normal(0, 5); m0 = 3.5 + rng.normal(0, 1); e0 = enj[r.arm] - 0.8 + rng.normal(0, 1)
        # liên hệ với thay đổi PSS: người tăng TMS nhiều giảm PSS nhiều hơn
        dp = (r.pss_w8 - r.pss_w0) if not np.isnan(r.pss_w8) else rng.normal(-5, 4)
        t4 = t0 + tms_gain[r.arm] - 0.45 * (dp + 5) + rng.normal(0, 4)
        m4 = m0 + mast_gain[r.arm] * 0.6 - 0.08 * (dp + 5) + rng.normal(0, 0.9)
        e4 = e0 + 0.6 - 0.05 * (dp + 5) + rng.normal(0, 0.9)
        t8 = t4 + 1.5 + rng.normal(0, 4); m8 = m0 + mast_gain[r.arm] + rng.normal(0, 0.9); e8 = e4 + rng.normal(0, 0.9)
        for wk, (t, m, e) in ((0, (t0, m0, e0)), (4, (t4, m4, e4)), (8, (t8, m8, e8))):
            miss = (wk == 8 and np.isnan(r.pss_w8)) or (wk == 4 and rng.random() < 0.05)
            rows_m.append(dict(id=r.id, week=wk, tms=np.nan if miss else round(float(np.clip(t, 0, 52)), 0),
                               mastery=np.nan if miss else round(float(np.clip(m, 0, 10)), 1), enjoyment=np.nan if miss else round(float(np.clip(e, 0, 10)), 1)))
        n_s = int(min(19, r.sessions_completed))
        rh0 = 0.45 + rng.normal(0, 0.08) - 0.015 * (dp + 5)
        for s in range(1, n_s + 1):
            rows_s.append(dict(id=r.id, arm=r.arm, session=s,
                               mastery=round(float(np.clip(m0 + mast_gain[r.arm] * (1 - np.exp(-s / 7)) + rng.normal(0, 0.8), 0, 10)), 1),
                               enjoyment=round(float(np.clip(e0 + 0.6 * (1 - np.exp(-s / 5)) + rng.normal(0, 0.8), 0, 10)), 1),
                               rhythm_index=round(float(np.clip(rh0 + 0.25 * (1 - np.exp(-s / 8)) + rng.normal(0, 0.05), 0, 1)), 3) if r.arm in ("MS", "MC") else np.nan))
    pd.DataFrame(rows_m).to_csv(p_mech, index=False); pd.DataFrame(rows_s).to_csv(p_sess, index=False)
    print(f"wrote Labd {p_mech}, {p_sess}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wide"); ap.add_argument("mech"); ap.add_argument("sess")
    ap.add_argument("-o", "--out", default="fig5_mechanisms"); ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--Lab", action="store_true")
    a = ap.parse_args()
    if a.Lab: Lab(a.wide, a.mech, a.sess); return
    wide = pd.read_csv(a.wide); mech = pd.read_csv(a.mech); sess = pd.read_csv(a.sess)
    if "arm" not in sess.columns: sess = sess.merge(wide[["id", "arm"]], on="id")
    d = build(wide, mech)
    res, aa, bb, cp, ct, fb, n = mediation(d, n_boot=a.boot)
    ct_full, n_full = total_effect_full(wide)
    res.loc[len(res)] = dict(path=f"Reference: C1 on all complete cases (n={n_full})", a=np.nan, b=np.nan, indirect=ct_full, ci_low=np.nan, ci_high=np.nan, prop_mediated=np.nan)
    res.to_csv(f"{a.out}_mediation.csv", index=False)
    draw(res.iloc[:-1], aa, bb, cp, ct, n, sess, wide, a.out, ct_full, n_full)
    print(res.round(3).to_string(index=False))
    print(f"c = {ct:.3f}, c' = {cp:.3f}, n = {n}; wrote {a.out}.svg/.png/.pdf, {a.out}_mediation.csv")


if __name__ == "__main__":
    main()
