#!/usr/bin/env python3
"""
Figure 6 — Immediate affective responses (Aim 3) and adherence / acceptability (Aim 5).

    python fig6_acute_adherence.py mindstroke_exam_data_wide.csv sessions.csv acceptability.csv [-o fig6]
    python fig6_acute_adherence.py --simulate mindstroke_exam_data_wide.csv sessions.csv acceptability.csv

Đầu vào
  1) wide:          id, arm, sessions_completed  (mẫu số ITT = mọi id trong file)
  2) sessions.csv:  id, session (1–19), stai_pre, stai_post, pa_pre, pa_post, na_pre, na_post [, mastery, enjoyment, rhythm_index]
                    STAI-S 20–80; PANAS PA/NA 10–50. Một dòng = một phiên đã hoàn thành.
  3) acceptability.csv: id, sus (0–100), continue_intent (0–10)   (đo ở W8; thiếu = không trả lời)

Mô hình: Δ ~ arm + session + (1 | id)  (MixedLM), contrasts C1–C3 trên hệ số arm.
"""
import argparse, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

ARMS = {"MS": ("Meaningful Sketching", "#D55E00"), "MC": ("Mindful Calligraphy", "#0072B2"),
        "MP": ("Music Practice", "#009E73"), "CG": ("Casual Gaming", "#CC79A7")}
CONTRASTS = [("C1 (MS+MC) − (MP+CG)", ["MS", "MC"], ["MP", "CG"]), ("C2 MS − MC", ["MS"], ["MC"]), ("C3 MP − CG", ["MP"], ["CG"])]
N_SESSIONS, ADHERENT_PCT, SUS_BENCH = 19, 80, 68
INK, GREY, MUTE = "#1f2937", "#374151", "#6b7280"


# ---------------------------------------------------------------- models
def mixed_contrasts(d, var):
    d = d.dropna(subset=[var]).copy()
    m = smf.mixedlm(f"{var} ~ C(arm, Treatment('CG')) + session", d, groups=d["id"]).fit(reml=True)
    names = m.params.index
    def L(arm):
        v = pd.Series(0.0, index=names)
        for nm in names:
            if nm.startswith("C(arm"): v[nm] = float(nm.endswith(f"[T.{arm}]"))
        return v
    out = []
    for label, A, B in CONTRASTS:
        c = sum(L(a) for a in A) / len(A) - sum(L(b) for b in B) / len(B)
        est = float(c @ m.params); se = float(np.sqrt(c @ m.cov_params().loc[names, names] @ c))
        out.append(dict(outcome=var, contrast=label, estimate=est, se=se, ci_low=est - 1.96 * se, ci_high=est + 1.96 * se,
                        p=2 * stats.norm.sf(abs(est / se)), n_id=d["id"].nunique(), n_obs=len(d)))
    slope = m.params["session"]; slope_p = m.pvalues["session"]
    return pd.DataFrame(out), slope, slope_p


def ci_mean(v):
    v = pd.Series(v).dropna(); n = len(v)
    return v.mean(), (stats.t.ppf(0.975, n - 1) * v.std() / np.sqrt(n) if n > 1 else np.nan), n


# ---------------------------------------------------------------- panels
def panel_a(ax, sess, res, slope, slope_p):
    for arm, (name, col) in ARMS.items():
        g = sess[sess["arm"] == arm].groupby("session")["d_stai"].agg(["mean", "std", "count"])
        ci = stats.t.ppf(0.975, g["count"] - 1) * g["std"] / np.sqrt(g["count"])
        ax.plot(g.index, g["mean"], color=col, lw=1.4, marker="o", ms=2.5, label=arm)
        ax.fill_between(g.index, g["mean"] - ci, g["mean"] + ci, color=col, alpha=0.12, lw=0)
    ax.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks([1, 5, 10, 15, 19]); ax.tick_params(labelsize=6.5)
    ax.set_xlabel("session", fontsize=6.5); ax.set_ylabel("Δ STAI-S, post − pre (points)", fontsize=6.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=5.8, ncol=4, loc="lower left")
    c1 = res.iloc[0]; ptxt = f"p = {c1.p:.3f}" if c1.p >= 0.001 else "p < 0.001"
    ax.text(0.01, 0.97, f"C1 (mixed model): {c1.estimate:+.2f} [{c1.ci_low:+.2f}, {c1.ci_high:+.2f}], {ptxt}   ·   "
            f"session slope {slope:+.3f}/session, " + (f"p = {slope_p:.3f}" if slope_p >= 0.001 else "p < 0.001"),
            transform=ax.transAxes, fontsize=5.6, color=MUTE, va="top")
    ax.set_title("Within-session change in state anxiety, every session (all arms)", loc="left", fontsize=7.5, fontweight="bold")


def panel_b(ax, sess, var, ylabel, title, res):
    for i, (arm, (name, col)) in enumerate(ARMS.items()):
        per = sess[sess["arm"] == arm].groupby("id")[var].mean()
        m, ci, n = ci_mean(per)
        x = i + np.random.default_rng(i).uniform(-0.18, 0.18, len(per))
        ax.plot(x, per, "o", ms=2, color=col, alpha=0.4, mec="none")
        ax.errorbar(i, m, yerr=ci, fmt="D", ms=4.5, color=INK, capsize=3, lw=1.2, zorder=5)
        ax.annotate(f"n={n}", (i, 0), xytext=(0, -2), textcoords="offset points", xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=5.5, color=col, annotation_clip=False)
    ax.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(range(4)); ax.set_xticklabels(list(ARMS), fontsize=6.5); ax.tick_params(axis="x", pad=9); ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel(ylabel, fontsize=6.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    c1 = res.iloc[0]; ptxt = f"p = {c1.p:.3f}" if c1.p >= 0.001 else "p < 0.001"
    neg = sess[var].mean() < 0
    ax.text(0.02, 0.03 if neg else 0.97, f"C1: {c1.estimate:+.2f} [{c1.ci_low:+.2f}, {c1.ci_high:+.2f}], {ptxt}", transform=ax.transAxes, fontsize=5.6, color=MUTE, va="bottom" if neg else "top")
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold")


def panel_c(ax1, ax2, wide, sess):
    last = sess.groupby("id")["session"].max()
    tab = []
    for arm, (name, col) in ARMS.items():
        ids = wide.loc[wide["arm"] == arm, "id"]; n = len(ids)
        ls = last.reindex(ids).fillna(0)
        surv = [(ls >= s).mean() * 100 for s in range(0, N_SESSIONS + 1)]
        ax1.step(range(0, N_SESSIONS + 1), surv, where="post", color=col, lw=1.5, label=arm)
        sc = wide.loc[wide["arm"] == arm, "sessions_completed"]
        adh = (sc >= ADHERENT_PCT / 100 * N_SESSIONS).mean() * 100
        tab.append(dict(arm=arm, n_itt=n, median_sessions=sc.median(), mean_sessions=sc.mean(), pct_adherent=adh, pct_any=(ls > 0).mean() * 100))
    ax1.set_ylim(0, 102); ax1.set_xticks([0, 5, 10, 15, 19]); ax1.tick_params(labelsize=6.5)
    ax1.set_xlabel("session", fontsize=6.5); ax1.set_ylabel("% of randomised still active", fontsize=6.5)
    ax1.axvline(ADHERENT_PCT / 100 * N_SESSIONS, color="#92400e", lw=0.8, ls=(0, (2, 2)))
    ax1.text(ADHERENT_PCT / 100 * N_SESSIONS + 0.3, 5, f"{ADHERENT_PCT} % threshold", fontsize=5.5, color="#92400e")
    for sp in ("top", "right"): ax1.spines[sp].set_visible(False)
    ax1.legend(frameon=False, fontsize=5.8, loc="lower left", ncol=2)
    ax1.set_title("Retention by session (ITT denominator)", loc="left", fontsize=7.5, fontweight="bold")
    # histogram sessions completed
    bins = np.arange(-0.5, N_SESSIONS + 1.5, 1)
    for arm, (name, col) in ARMS.items():
        sc = wide.loc[wide["arm"] == arm, "sessions_completed"]
        ax2.hist(sc, bins=bins, histtype="step", color=col, lw=1.3, label=arm)
    ax2.axvline(ADHERENT_PCT / 100 * N_SESSIONS, color="#92400e", lw=0.8, ls=(0, (2, 2)))
    ax2.set_xlabel("sessions completed (0–19)", fontsize=6.5); ax2.set_ylabel("participants", fontsize=6.5); ax2.tick_params(labelsize=6.5)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    t = pd.DataFrame(tab)
    txt = "\n".join(f"{r.arm}: median {r.median_sessions:.0f} · adherent {r.pct_adherent:.0f} %" for r in t.itertuples())
    ax2.text(0.02, 0.97, txt, transform=ax2.transAxes, fontsize=5.6, color=MUTE, va="top")
    ax2.set_title("Sessions completed", loc="left", fontsize=7.5, fontweight="bold")
    return t


def panel_d(ax, wide, acc):
    d = wide[["id", "arm"]].merge(acc, on="id", how="left")
    for i, (arm, (name, col)) in enumerate(ARMS.items()):
        v = d.loc[d["arm"] == arm, "sus"].dropna(); m, ci, n = ci_mean(v)
        x = i + np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        ax.plot(x, v, "o", ms=2, color=col, alpha=0.4, mec="none")
        ax.errorbar(i, m, yerr=ci, fmt="D", ms=4.5, color=INK, capsize=3, lw=1.2, zorder=5)
        ci_int = d.loc[d["arm"] == arm, "continue_intent"].dropna()
        ax.annotate(f"n={n}\nintent {ci_int.mean():.1f}/10", (i, 0), xytext=(0, -2), textcoords="offset points", xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=5.3, color=col, annotation_clip=False)
    ax.axhline(SUS_BENCH, color="#92400e", lw=0.8, ls=(0, (2, 2))); ax.text(3.45, SUS_BENCH + 1, f"SUS benchmark {SUS_BENCH}", fontsize=5.5, color="#92400e", ha="right")
    ax.set_ylim(0, 102); ax.set_xticks(range(4)); ax.set_xticklabels(list(ARMS), fontsize=6.5); ax.tick_params(axis="x", pad=16); ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("System Usability Scale (0–100)", fontsize=6.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    kw = stats.kruskal(*[d.loc[d["arm"] == a, "sus"].dropna() for a in ARMS])
    ax.text(0.02, 0.05, f"Kruskal–Wallis p = {kw.pvalue:.3f}", transform=ax.transAxes, fontsize=5.6, color=MUTE)
    ax.set_title("Usability and intention to continue (Week 8)", loc="left", fontsize=7.5, fontweight="bold")


def draw(wide, sess, acc, res_stai, slope, slope_p, res_pa, res_na, out, dpi=300):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(17 / 2.54 * 1.6, 12.5 / 2.54 * 1.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.55, wspace=0.5, left=0.07, right=0.98, top=0.94, bottom=0.1)
    axA = fig.add_subplot(gs[0, :2]); axB1 = fig.add_subplot(gs[0, 2]); axB2 = fig.add_subplot(gs[0, 3])
    axC1 = fig.add_subplot(gs[1, 0]); axC2 = fig.add_subplot(gs[1, 1]); axD = fig.add_subplot(gs[1, 2:])
    panel_a(axA, sess, res_stai, slope, slope_p); axA.text(-0.1, 1.1, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")
    panel_b(axB1, sess, "d_pa", "Δ positive affect (PANAS-PA)", "Positive affect, per-person mean Δ", res_pa); axB1.text(-0.32, 1.1, "B", transform=axB1.transAxes, fontsize=12, fontweight="bold")
    panel_b(axB2, sess, "d_na", "Δ negative affect (PANAS-NA)", "Negative affect, per-person mean Δ", res_na)
    tab = panel_c(axC1, axC2, wide, sess); axC1.text(-0.32, 1.1, "C", transform=axC1.transAxes, fontsize=12, fontweight="bold")
    panel_d(axD, wide, acc); axD.text(-0.15, 1.1, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")
    for ext in ("svg", "png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tab


# ---------------------------------------------------------------- simulate
def simulate(p_wide, p_sess, p_acc, seed=9):
    rng = np.random.default_rng(seed)
    w = pd.read_csv(p_wide)
    try:
        old = pd.read_csv(p_sess)
    except FileNotFoundError:
        old = None
    eff_stai = {"MS": -6.0, "MC": -7.0, "MP": -5.0, "CG": -3.5}; eff_pa = {"MS": 3.0, "MC": 2.5, "MP": 3.5, "CG": 2.0}; eff_na = {"MS": -2.5, "MC": -3.0, "MP": -2.0, "CG": -1.5}
    rows = []
    for r in w.itertuples():
        u = rng.normal(0, 1.5); n_s = int(min(19, r.sessions_completed))
        for s in range(1, n_s + 1):
            pre = 38 + rng.normal(0, 6); post = pre + eff_stai[r.arm] + u - 0.08 * s + rng.normal(0, 3.5)
            pa0 = 28 + rng.normal(0, 5); na0 = 16 + rng.normal(0, 4)
            rows.append(dict(id=r.id, session=s, stai_pre=round(float(np.clip(pre, 20, 80))), stai_post=round(float(np.clip(post, 20, 80))),
                             pa_pre=round(float(np.clip(pa0, 10, 50))), pa_post=round(float(np.clip(pa0 + eff_pa[r.arm] + rng.normal(0, 2.5), 10, 50))),
                             na_pre=round(float(np.clip(na0, 10, 50))), na_post=round(float(np.clip(na0 + eff_na[r.arm] + rng.normal(0, 2), 10, 50)))))
    new = pd.DataFrame(rows)
    if old is not None and "mastery" in old.columns:
        new = new.merge(old.drop(columns=[c for c in ("arm",) if c in old.columns]), on=["id", "session"], how="left")
    new.to_csv(p_sess, index=False)
    sus_m = {"MS": 74, "MC": 76, "MP": 71, "CG": 78}; intent = {"MS": 6.8, "MC": 7.2, "MP": 6.5, "CG": 6.0}
    acc = []
    for r in w.itertuples():
        if np.isnan(r.pss_w8) or rng.random() < 0.04: continue
        acc.append(dict(id=r.id, sus=round(float(np.clip(sus_m[r.arm] + rng.normal(0, 11), 0, 100)), 1), continue_intent=round(float(np.clip(intent[r.arm] + rng.normal(0, 2), 0, 10)), 1)))
    pd.DataFrame(acc).to_csv(p_acc, index=False)
    print(f"wrote simulated {p_sess} (STAI/PANAS cột thêm), {p_acc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wide"); ap.add_argument("sess"); ap.add_argument("acc")
    ap.add_argument("-o", "--out", default="fig6_acute_adherence"); ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    if a.simulate: simulate(a.wide, a.sess, a.acc); return
    wide = pd.read_csv(a.wide); sess = pd.read_csv(a.sess); acc = pd.read_csv(a.acc)
    need = {"id", "session", "stai_pre", "stai_post", "pa_pre", "pa_post", "na_pre", "na_post"}
    if need - set(sess.columns): sys.exit(f"sessions.csv thiếu cột: {sorted(need - set(sess.columns))}")
    sess = sess.merge(wide[["id", "arm"]], on="id", how="left", suffixes=("_x", ""))
    if "arm_x" in sess.columns: sess = sess.drop(columns="arm_x")
    sess["d_stai"] = sess["stai_post"] - sess["stai_pre"]; sess["d_pa"] = sess["pa_post"] - sess["pa_pre"]; sess["d_na"] = sess["na_post"] - sess["na_pre"]
    res_stai, slope, slope_p = mixed_contrasts(sess, "d_stai"); res_pa, _, _ = mixed_contrasts(sess, "d_pa"); res_na, _, _ = mixed_contrasts(sess, "d_na")
    allres = pd.concat([res_stai, res_pa, res_na]); allres.to_csv(f"{a.out}_stats.csv", index=False)
    tab = draw(wide, sess, acc, res_stai, slope, slope_p, res_pa, res_na, a.out)
    tab.to_csv(f"{a.out}_adherence.csv", index=False)
    print(allres[["outcome", "contrast", "estimate", "ci_low", "ci_high", "p"]].round(3).to_string(index=False)); print(tab.round(1).to_string(index=False))
    print(f"wrote {a.out}.svg/.png/.pdf, {a.out}_stats.csv, {a.out}_adherence.csv")


if __name__ == "__main__":
    main()
