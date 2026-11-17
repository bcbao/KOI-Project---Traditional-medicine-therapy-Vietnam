#!/usr/bin/env python3
"""
Figure 4 — Salivary cortisol outcomes (diurnal profile, slope, CAR, AUC_G, acute response).

    python fig4_cortisol.py cortisol_diurnal.csv cortisol_acute.csv [-o fig4]
    python fig4_cortisol.py --simulate cortisol_diurnal.csv cortisol_acute.csv   # tạo dữ liệu giả lập

CSV 1 — cortisol_diurnal.csv  (một dòng = một mẫu nước bọt)
    id, arm, week, day, sample, wake_time, time_since_wake_h, cortisol_nmol_l
    week   0 | 8          day 1 | 2        sample S1..S5
    wake_time "HH:MM"     time_since_wake_h  giờ thực tế lấy mẫu kể từ khi thức (app-timestamped)
    cortisol_nmol_l       để trống nếu mẫu hỏng/thiếu
CSV 2 — cortisol_acute.csv   (một dòng = một phiên có lấy mẫu, sub-sample)
    id, arm, week, session_start, cortisol_pre, cortisol_post
    week 1 | 4 | 8   pre = −10 min, post = +20 min sau phiên

Chỉ số (tính cho từng ngày, rồi trung bình các ngày hợp lệ):
    valid day   : có S1, S2, S5 và S2 lấy trong 0.33–0.75 h sau thức (≈ 20–45 min)
    CAR (AUC_I) : (S2 − S1) × Δt / 2            [nmol/L·h]  (chỉ ngày có S2 trong 25–40 min = "CAR-valid")
    AUC_G       : hình thang S1→S5               [nmol/L·h]
    slope       : hệ số hồi quy log(cortisol) ~ giờ, dùng S1, S3, S4, S5 (bỏ S2)  [log-nmol/L per h]
Mô hình: ANCOVA chỉ số_W8 ~ arm + chỉ số_W0 + wake_time + sex + site (sex/site nếu có); contrasts C1–C3.
Acute: Δ = log(post) − log(pre); MixedLM Δ ~ arm + week + (1 | id).
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
CONTRASTS = [("C1 (MS+MC) − (MP+CG)", ["MS", "MC"], ["MP", "CG"]),
             ("C2 MS − MC", ["MS"], ["MC"]), ("C3 MP − CG", ["MP"], ["CG"])]
SAMPLES = ["S1", "S2", "S3", "S4", "S5"]
NOMINAL_H = {"S1": 0, "S2": 0.5, "S3": 4, "S4": 8, "S5": 15}
INK, GREY, MUTE = "#1f2937", "#374151", "#6b7280"


# ---------------------------------------------------------------- indices
def hhmm_to_h(s):
    h, m = str(s).split(":"); return int(h) + int(m) / 60


def day_indices(g):
    g = g.dropna(subset=["cortisol_nmol_l"]).set_index("sample")
    have = set(g.index)
    if not {"S1", "S2", "S5"} <= have:
        return None
    t2 = g.loc["S2", "time_since_wake_h"]
    if not (0.33 <= t2 <= 0.75):
        return None
    t = g["time_since_wake_h"]; c = g["cortisol_nmol_l"]
    car = (c["S2"] - c["S1"]) * (t["S2"] - t["S1"]) / 2
    car_valid = 0.42 <= t2 <= 0.67
    order = [s for s in SAMPLES if s in have]
    aucg = np.trapezoid(c[order].values, t[order].values)
    sl = [s for s in ("S1", "S3", "S4", "S5") if s in have]
    slope = np.polyfit(t[sl].values, np.log(c[sl].values), 1)[0] if len(sl) >= 3 else np.nan
    return dict(car=car if car_valid else np.nan, aucg=aucg, slope=slope, bedtime=np.log(c["S5"]),
                wake_h=hhmm_to_h(g["wake_time"].iloc[0]))


def person_indices(diu):
    rows = []
    for (pid, arm, week), g in diu.groupby(["id", "arm", "week"]):
        days = [d for d in (day_indices(gd) for _, gd in g.groupby("day")) if d]
        if not days:
            continue
        m = pd.DataFrame(days).mean(numeric_only=True)
        rows.append(dict(id=pid, arm=arm, week=week, n_valid_days=len(days), **m.to_dict()))
    return pd.DataFrame(rows)


def profile_means(diu):
    """mean ± CI của log-cortisol theo nhóm × tuần × mẫu (trung bình 2 ngày trong người trước)."""
    d = diu.dropna(subset=["cortisol_nmol_l"]).copy()
    d["logc"] = np.log(d["cortisol_nmol_l"])
    per = d.groupby(["id", "arm", "week", "sample"]).agg(logc=("logc", "mean"), t=("time_since_wake_h", "mean")).reset_index()
    out = per.groupby(["arm", "week", "sample"]).agg(n=("logc", "size"), m=("logc", "mean"), sd=("logc", "std"), t=("t", "mean")).reset_index()
    out["ci"] = stats.t.ppf(0.975, out["n"] - 1) * out["sd"] / np.sqrt(out["n"])
    return out


# ---------------------------------------------------------------- models
def ancova_contrasts(wide, var, covars):
    d = wide.dropna(subset=[f"{var}_w8", f"{var}_w0"]).copy()
    rhs = f"{var}_w0" + "".join(f" + {c}" for c in covars if c in d.columns and d[c].nunique() > 1)
    m = smf.ols(f"{var}_w8 ~ C(arm, Treatment('CG')) + {rhs}", data=d).fit()
    names = m.params.index
    def L(arm):
        v = pd.Series(0.0, index=names); v["Intercept"] = 1
        for nm in names:
            if nm == "Intercept": continue
            if nm.startswith("C(arm"): v[nm] = float(nm.endswith(f"[T.{arm}]"))
            elif nm.startswith("C("):
                col = nm[2:nm.index(")")]; lvl = nm.split("[T.")[1][:-1]; v[nm] = (d[col] == lvl).mean()
            else: v[nm] = d[nm].mean()
        return v
    out = []
    for label, A, B in CONTRASTS:
        c = sum(L(a) for a in A) / len(A) - sum(L(b) for b in B) / len(B)
        est = float(c @ m.params); se = float(np.sqrt(c @ m.cov_params() @ c)); df = m.df_resid
        tc = stats.t.ppf(0.975, df); p = 2 * stats.t.sf(abs(est / se), df)
        out.append(dict(outcome=var, contrast=label, estimate=est, se=se, ci_low=est - tc * se, ci_high=est + tc * se,
                        p=p, hedges_g=est / np.sqrt(m.mse_resid) * (1 - 3 / (4 * df - 1)), n=len(d)))
    adj = {a: float(L(a) @ m.params) for a in ARMS}
    return pd.DataFrame(out), adj, m


def acute_model(ac):
    d = ac.dropna(subset=["cortisol_pre", "cortisol_post"]).copy()
    d["delta"] = np.log(d["cortisol_post"]) - np.log(d["cortisol_pre"])
    m = smf.mixedlm("delta ~ C(arm, Treatment('CG')) + C(week)", d, groups=d["id"]).fit(reml=True)
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
        out.append(dict(outcome="acute_delta", contrast=label, estimate=est, se=se, ci_low=est - 1.96 * se,
                        ci_high=est + 1.96 * se, p=2 * stats.norm.sf(abs(est / se)), hedges_g=np.nan, n=d["id"].nunique()))
    summ = d.groupby(["arm", "week"]).agg(n=("delta", "size"), m=("delta", "mean"), sd=("delta", "std")).reset_index()
    summ["ci"] = stats.t.ppf(0.975, summ["n"] - 1) * summ["sd"] / np.sqrt(summ["n"])
    return pd.DataFrame(out), summ, d


# ---------------------------------------------------------------- figure
def forest(ax, res, title, xlabel, unit_note=""):
    y = np.arange(len(res))[::-1]
    ax.axvline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    for yi, r in zip(y, res.itertuples()):
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=INK, lw=1.4); ax.plot(r.estimate, yi, "s", ms=5, color=INK)
        ptxt = f"p = {r.p:.3f}" if r.p >= 0.001 else "p < 0.001"
        ax.text(0.5, yi + 0.32, f"{r.estimate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  {ptxt}", transform=ax.get_yaxis_transform(),
                fontsize=5.6, color=MUTE, ha="center", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([r.contrast for r in res.itertuples()], fontsize=6.5)
    lim = max(abs(res["ci_low"].min()), abs(res["ci_high"].max())) * 1.25
    ax.set_xlim(-lim, lim); ax.set_ylim(-0.7, len(res) - 0.2)
    ax.set_xlabel(xlabel, fontsize=6.5); ax.tick_params(axis="x", labelsize=6.5); ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold")


def dots(ax, wide, var, title, ylabel, adj=None):
    for i, (arm, (name, col)) in enumerate(ARMS.items()):
        v = (wide.loc[wide["arm"] == arm, f"{var}_w8"] - wide.loc[wide["arm"] == arm, f"{var}_w0"]).dropna()
        x = i + np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        ax.plot(x, v, "o", ms=2.2, color=col, alpha=0.45, mec="none")
        m = v.mean(); ci = stats.t.ppf(0.975, len(v) - 1) * v.std() / np.sqrt(len(v))
        ax.errorbar(i, m, yerr=ci, fmt="D", ms=4.5, color=INK, capsize=3, lw=1.2, zorder=5)
        ax.text(i, ax.get_ylim()[0] if False else np.nan, "")
        ax.annotate(f"n={len(v)}", (i, 0), xytext=(0, -2), textcoords="offset points", xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=5.5, color=col, annotation_clip=False)
    ax.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(range(4)); ax.set_xticklabels(list(ARMS), fontsize=6.5); ax.tick_params(axis="x", pad=9)
    ax.set_ylabel(ylabel, fontsize=6.5); ax.tick_params(axis="y", labelsize=6.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold")


def draw(prof, wide, res_slope, adj_slope, res_car, res_aucg, acute_summ, res_acute, out, dpi=300):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(17 / 2.54 * 1.6, 13.5 / 2.54 * 1.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.5, wspace=0.5, left=0.07, right=0.98, top=0.93, bottom=0.1)
    # ---- A profiles
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, (arm, (name, col)) in zip(axes, ARMS.items()):
        for wk, ls, lab in ((0, (0, (4, 3)), "Week 0"), (8, "-", "Week 8")):
            s = prof[(prof["arm"] == arm) & (prof["week"] == wk)].set_index("sample").reindex(SAMPLES)
            ax.errorbar(s["t"], s["m"], yerr=s["ci"], color=col, ls=ls, lw=1.4, marker="o", ms=3, capsize=2,
                        alpha=0.55 if wk == 0 else 1, label=lab)
        n0 = int(wide.loc[wide["arm"] == arm, "slope_w0"].notna().sum()); n8 = int(wide.loc[wide["arm"] == arm, "slope_w8"].notna().sum())
        ax.set_title(f"{name}\nevaluable n: W0 {n0} · W8 {n8}", fontsize=6.8, color=col, loc="left")
        ax.set_xticks([0, 4, 8, 12, 15]); ax.tick_params(labelsize=6.5)
        ax.set_xlabel("hours since awakening", fontsize=6.5)
        yt = np.log([1, 2, 5, 10, 20]); ax.set_yticks(yt); ax.set_yticklabels([1, 2, 5, 10, 20])
        ax.set_ylim(np.log(0.8), np.log(28)); ax.grid(axis="y", color="#eeeeee", lw=0.6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("salivary cortisol, nmol/L (log scale)", fontsize=6.5)
            ax.legend(frameon=False, fontsize=6, loc="upper right")
    axes[0].text(-0.3, 1.18, "A", transform=axes[0].transAxes, fontsize=12, fontweight="bold")
    # ---- B slope forest
    axB = fig.add_subplot(gs[1, 0])
    forest(axB, res_slope, "Diurnal slope at Week 8 (key index)", "adjusted difference, log-nmol/L per h\n(negative = steeper in first arm(s))")
    axB.text(-0.55, 1.12, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
    axB.text(0, -0.36, "ANCOVA: slope_W8 ~ arm + slope_W0 + wake time + sex + site\nadjusted W8 slope: " +
             " · ".join(f"{a} {adj_slope[a]:.3f}" for a in ARMS), transform=axB.transAxes, fontsize=5.5, color=MUTE, va="top")
    # ---- C CAR & AUCg change
    axC1 = fig.add_subplot(gs[1, 1]); axC2 = fig.add_subplot(gs[1, 2])
    dots(axC1, wide, "car", "CAR (AUC$_I$), change W8 − W0", "Δ nmol/L·h")
    dots(axC2, wide, "aucg", "AUC$_G$ (S1–S5), change W8 − W0", "Δ nmol/L·h")
    for ax, r in ((axC1, res_car), (axC2, res_aucg)):
        c1 = r.iloc[0]; ptxt = f"p = {c1.p:.3f}" if c1.p >= 0.001 else "p < 0.001"
        ax.text(0.02, 0.97, f"C1: {c1.estimate:+.2f} [{c1.ci_low:+.2f}, {c1.ci_high:+.2f}], {ptxt}", transform=ax.transAxes,
                fontsize=5.6, color=MUTE, va="top")
    axC1.text(-0.35, 1.12, "C", transform=axC1.transAxes, fontsize=12, fontweight="bold")
    # ---- D acute
    axD = fig.add_subplot(gs[1, 3])
    off = {"MS": -0.24, "MC": -0.08, "MP": 0.08, "CG": 0.24}
    for arm, (name, col) in ARMS.items():
        s = acute_summ[acute_summ["arm"] == arm]
        axD.errorbar(s["week"] + off[arm], s["m"] * 100, yerr=s["ci"] * 100, color=col, marker="o", ms=3.5, lw=1.3, capsize=2, label=arm)
    axD.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    axD.set_xticks([1, 4, 8]); axD.set_xticklabels(["W1", "W4", "W8"], fontsize=6.5); axD.tick_params(axis="y", labelsize=6.5)
    axD.set_ylabel("Δ log-cortisol post − pre (%)", fontsize=6.5)
    for sp in ("top", "right"): axD.spines[sp].set_visible(False)
    axD.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower left")
    n_ac = int(res_acute["n"].iloc[0]); c1 = res_acute.iloc[0]
    axD.set_title(f"Acute pre→post session response (sub-sample n = {n_ac})", loc="left", fontsize=7.5, fontweight="bold")
    ptxt = f"p = {c1.p:.3f}" if c1.p >= 0.001 else "p < 0.001"
    axD.text(0.02, 0.97, f"C1 (mixed model): {c1.estimate*100:+.1f} % [{c1.ci_low*100:+.1f}, {c1.ci_high*100:+.1f}], {ptxt}",
             transform=axD.transAxes, fontsize=5.6, color=MUTE, va="top")
    axD.text(-0.35, 1.12, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")
    for ext in ("svg", "png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- simulate
def simulate(p_diu, p_ac, seed=11):
    rng = np.random.default_rng(seed)
    try:
        base = pd.read_csv("mindstroke_exam_data_wide.csv")[["id", "arm", "sex", "site"]]
    except FileNotFoundError:
        ids = []; k = 0
        for arm, nA in {"MS": 45, "MC": 45, "MP": 43, "CG": 46}.items():
            for _ in range(nA):
                k += 1; ids.append(dict(id=f"P{k:03d}", arm=arm, sex=rng.choice(["F", "M"]), site=rng.choice(list("ABC"))))
        base = pd.DataFrame(ids)
    eval_w0 = {"MS": 44, "MC": 45, "MP": 42, "CG": 46}; eval_w8 = {"MS": 39, "MC": 40, "MP": 36, "CG": 41}
    slope_eff = {"MS": -0.020, "MC": -0.025, "MP": -0.010, "CG": 0.0}   # thay đổi slope W8 (log/h), âm = dốc hơn
    rows = []
    for arm, g in base.groupby("arm"):
        ids = list(g["id"]); rng.shuffle(ids)
        ok0 = set(ids[:eval_w0[arm]]); ok8 = set(ids[:eval_w8[arm]])
        for pid in ids:
            u_level = rng.normal(0, 0.25); u_slope = rng.normal(0, 0.015)
            for wk, okset in ((0, ok0), (8, ok8)):
                for day in (1, 2):
                    wake = 6.5 + rng.normal(0.6, 0.5); wake_s = f"{int(wake):02d}:{int((wake % 1) * 60):02d}"
                    l1 = np.log(12) + u_level + rng.normal(0, 0.15)
                    car = 0.45 + rng.normal(0, 0.18)
                    sl = -0.135 + u_slope + (slope_eff[arm] if wk == 8 else 0) + rng.normal(0, 0.012)
                    t2 = 0.5 + rng.normal(0, 0.06)
                    for s in SAMPLES:
                        t = {"S1": 0 + abs(rng.normal(0, 0.03)), "S2": t2, "S3": 4 + rng.normal(0, 0.3), "S4": 8 + rng.normal(0, 0.4), "S5": 15 + rng.normal(0, 0.8)}[s]
                        logc = l1 + car * (1 if s == "S2" else 0) + sl * t * (0 if s == "S2" else 1) + rng.normal(0, 0.12)
                        c = round(float(np.exp(logc)), 2)
                        if pid not in okset or rng.random() < 0.025: c = np.nan
                        rows.append(dict(id=pid, arm=arm, week=wk, day=day, sample=s, wake_time=wake_s,
                                         time_since_wake_h=round(t, 2), cortisol_nmol_l=c))
    pd.DataFrame(rows).to_csv(p_diu, index=False)
    # acute sub-sample
    sub = pd.concat([g.sample(13, random_state=seed) for _, g in base.groupby("arm")]).reset_index(drop=True)
    eff = {"MS": -0.20, "MC": -0.24, "MP": -0.12, "CG": -0.05}
    rows = []
    for r in sub.itertuples():
        u = rng.normal(0, 0.06)
        for wk in (1, 4, 8):
            if rng.random() < 0.08: continue
            pre = float(np.exp(np.log(5.0) + rng.normal(0, 0.3)))
            post = float(pre * np.exp(eff[r.arm] + u - 0.03 * (wk - 1) / 7 + rng.normal(0, 0.09)))
            hh = int(rng.integers(15, 18)); rows.append(dict(id=r.id, arm=r.arm, week=wk, session_start=f"{hh}:{int(rng.integers(0, 60)):02d}",
                                                              cortisol_pre=round(pre, 2), cortisol_post=round(post, 2)))
    pd.DataFrame(rows).to_csv(p_ac, index=False)
    print(f"wrote simulated {p_diu}, {p_ac}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("diurnal"); ap.add_argument("acute")
    ap.add_argument("-o", "--out", default="fig4_cortisol"); ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    if a.simulate: simulate(a.diurnal, a.acute); return
    diu = pd.read_csv(a.diurnal); ac = pd.read_csv(a.acute)
    for df, need in ((diu, {"id", "arm", "week", "day", "sample", "wake_time", "time_since_wake_h", "cortisol_nmol_l"}),
                     (ac, {"id", "arm", "week", "cortisol_pre", "cortisol_post"})):
        if need - set(df.columns): sys.exit(f"thiếu cột: {sorted(need - set(df.columns))}")
    idx = person_indices(diu)
    wide = idx.pivot(index="id", columns="week", values=["car", "aucg", "slope", "bedtime", "wake_h", "n_valid_days"])
    wide.columns = [f"{v}_w{w}" for v, w in wide.columns]; wide = wide.reset_index()
    wide = wide.merge(diu.drop_duplicates("id")[["id", "arm"]], on="id")
    try:
        cov = pd.read_csv("mindstroke_exam_data_wide.csv")[["id", "sex", "site"]]; wide = wide.merge(cov, on="id", how="left")
    except FileNotFoundError:
        pass
    covs = ["wake_h_w8", "C(sex)", "C(site)"]
    res_slope, adj_slope, _ = ancova_contrasts(wide, "slope", covs)
    res_car, _, _ = ancova_contrasts(wide, "car", covs)
    res_aucg, _, _ = ancova_contrasts(wide, "aucg", covs)
    res_bed, _, _ = ancova_contrasts(wide, "bedtime", covs)
    res_acute, acute_summ, _ = acute_model(ac)
    allres = pd.concat([res_slope, res_car, res_aucg, res_bed, res_acute]); allres.to_csv(f"{a.out}_stats.csv", index=False)
    wide.to_csv(f"{a.out}_indices.csv", index=False)
    prof = profile_means(diu)
    draw(prof, wide, res_slope, adj_slope, res_car, res_aucg, acute_summ, res_acute, a.out)
    print(allres[["outcome", "contrast", "estimate", "ci_low", "ci_high", "p", "n"]].round(4).to_string(index=False))
    print(f"wrote {a.out}.svg/.png/.pdf, {a.out}_stats.csv, {a.out}_indices.csv")


if __name__ == "__main__":
    main()
