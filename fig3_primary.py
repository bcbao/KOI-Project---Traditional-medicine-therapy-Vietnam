#!/usr/bin/env python3
"""
Figure 3 — Primary outcome (PSS-10) from a long-format CSV.

    python fig3_primary.py data_pss.csv                -> fig3_primary.svg/.png/.pdf + fig3_primary_stats.csv
    python fig3_primary.py data_pss.csv -o fig3
    python fig3_primary.py data_pss.csv    

CSV yêu cầu (một dòng = một người × một mốc):
    id, arm, week, pss [, sex, site]
    id    mã người tham gia (chuỗi/số), lặp lại ở mỗi week
    arm   MS | MC | MP | CG
    week  0 | 4 | 8   (phải có dòng week=0 cho mọi id; week thiếu -> để trống pss hoặc bỏ dòng)
    pss   điểm PSS-10 (0–40); để trống nếu thiếu
    sex   F | M      (tuỳ chọn; dùng làm hiệp biến ANCOVA)
    site  A | B | C  (tuỳ chọn; dùng làm hiệp biến ANCOVA)

Panel A: trung bình ± 95 % CI quan sát được theo nhóm tại mỗi mốc (complete-case).
Panel B: hiệu số trung bình hiệu chỉnh tại tuần 8 cho các contrast định trước
         (ANCOVA: pss_w8 ~ arm + pss_w0 [+ sex + site]), 95 % CI, Hedges g, p.
Ghi chú: hình dùng complete-case; phân tích chính dùng multiple imputation — ghi trong caption.
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# ======================================================================
#  CẤU HÌNH
# ======================================================================
ARMS = {"MS": ("Meaningful Sketching", "#D55E00"),
        "MC": ("Mindful Calligraphy", "#0072B2"),
        "MP": ("Music Practice", "#009E73"),
        "CG": ("Casual Gaming", "#CC79A7")}
WEEKS = [0, 4, 8]
ENDPOINT = 8
MCID = 3.0                       # minimal clinically important difference (PSS-10 points)
CONTRASTS = [                    # (nhãn, nhóm "trừ", nhóm "bị trừ") — hiệu số = mean(first) − mean(second)
    ("C1  (MS + MC) − (MP + CG)", ["MS", "MC"], ["MP", "CG"]),
    ("C2  MS − MC",               ["MS"],       ["MC"]),
    ("C3  MP − CG",               ["MP"],       ["CG"]),
]
PAIRWISE_EXPLORATORY = True      # thêm 6 cặp exploratory (màu xám) vào panel B
# ======================================================================

INK, GREY, MUTE = "#1f2937", "#374151", "#6b7280"


# ---------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path)
    need = {"id", "arm", "week", "pss"}
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"CSV thiếu cột: {sorted(miss)}")
    df["arm"] = df["arm"].astype(str).str.strip().str.upper()
    bad = set(df["arm"]) - set(ARMS)
    if bad:
        sys.exit(f"Mã nhóm lạ: {sorted(bad)} (chỉ nhận {list(ARMS)})")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["pss"] = pd.to_numeric(df["pss"], errors="coerce")
    if ((df["pss"] < 0) | (df["pss"] > 40)).any():
        print("  ⚠ có giá trị PSS ngoài 0–40", file=sys.stderr)
    dup = df.duplicated(["id", "week"]).sum()
    if dup:
        sys.exit(f"{dup} dòng trùng id×week")
    wide = df.pivot(index="id", columns="week", values="pss")
    arm_of = df.drop_duplicates("id").set_index("id")["arm"]
    if (df.groupby("id")["arm"].nunique() > 1).any():
        sys.exit("một id có nhiều hơn một arm")
    wide["arm"] = arm_of
    for c in ("sex", "site"):
        if c in df.columns:
            wide[c] = df.drop_duplicates("id").set_index("id")[c]
    if 0 not in wide.columns or wide[0].isna().any():
        print("  ⚠ có id thiếu PSS baseline (week 0) — bị loại khỏi ANCOVA", file=sys.stderr)
    return wide


def summary(wide):
    rows = []
    for arm in ARMS:
        for w in WEEKS:
            v = wide.loc[wide["arm"] == arm, w].dropna() if w in wide.columns else pd.Series(dtype=float)
            n = len(v)
            if n:
                m, sd = v.mean(), v.std(ddof=1)
                ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
            else:
                m = sd = ci = np.nan
            rows.append(dict(arm=arm, week=w, n=n, mean=m, sd=sd, ci95=ci))
    return pd.DataFrame(rows)


def ancova(wide):
    d = wide.rename(columns={0: "pss0", ENDPOINT: "pss8"}).dropna(subset=["pss0", "pss8"]).copy()
    cov = " + pss0"
    for c in ("sex", "site"):
        if c in d.columns and d[c].notna().all() and d[c].nunique() > 1:
            cov += f" + C({c})"
    model = smf.ols(f"pss8 ~ C(arm, Treatment('CG')){cov}", data=d).fit()
    # adjusted means via emmeans-style prediction at mean covariates
    names = model.params.index
    def coef_vec(arm):
        v = pd.Series(0.0, index=names)
        v["Intercept"] = 1.0
        v["pss0"] = d["pss0"].mean()
        for c in ("sex", "site"):
            for nm in names:
                if nm.startswith(f"C({c})"):
                    lvl = nm.split("[T.")[1][:-1]
                    v[nm] = (d[c] == lvl).mean()
        k = f"C(arm, Treatment('CG'))[T.{arm}]"
        if k in names:
            v[k] = 1.0
        return v
    L = {a: coef_vec(a) for a in ARMS}
    sd_pooled = np.sqrt(model.mse_resid)
    out = []
    def contrast(label, A, B, kind):
        c = sum(L[a] for a in A) / len(A) - sum(L[b] for b in B) / len(B)
        est = float(c @ model.params)
        se = float(np.sqrt(c @ model.cov_params() @ c))
        dfr = model.df_resid
        t = est / se
        p = 2 * stats.t.sf(abs(t), dfr)
        tcrit = stats.t.ppf(0.975, dfr)
        g = est / sd_pooled * (1 - 3 / (4 * dfr - 1))
        out.append(dict(contrast=label, kind=kind, estimate=est, se=se, ci_low=est - tcrit * se,
                        ci_high=est + tcrit * se, t=t, df=dfr, p=p, hedges_g=g))
    for label, A, B in CONTRASTS:
        contrast(label, A, B, "confirmatory")
    if PAIRWISE_EXPLORATORY:
        keys = list(ARMS)
        for i in range(4):
            for j in range(i + 1, 4):
                contrast(f"{keys[i]} − {keys[j]}", [keys[i]], [keys[j]], "exploratory")
    res = pd.DataFrame(out)
    # Holm for C2, C3 (C1 tested alone at α=0.05 after omnibus)
    sec = res.index[(res["kind"] == "confirmatory") & (res["contrast"].str.startswith(("C2", "C3")))]
    if len(sec):
        p = res.loc[sec, "p"].values
        order = np.argsort(p)
        adj = np.empty_like(p)
        m = len(p)
        running = 0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * p[idx])
            adj[idx] = min(1.0, running)
        res.loc[sec, "p_holm"] = adj
    adj_means = {a: float(L[a] @ model.params) for a in ARMS}
    omnibus = model.f_test(" = ".join(f"C(arm, Treatment('CG'))[T.{a}]" for a in ["MS", "MC", "MP"]) + " = 0")
    return res, adj_means, model, omnibus, len(d)


# ---------------------------------------------------------------- multiple imputation (primary, ITT)
def ancova_mi(wide, m=50, seed=2026):
    """MICE (statsmodels) trên toàn bộ người được ngẫu nhiên hoá; gộp theo Rubin."""
    from statsmodels.imputation import mice
    d = wide.rename(columns={0: "pss0", 4: "pss4", ENDPOINT: "pss8"}).copy()
    d = d[["arm", "pss0", "pss4", "pss8"] + [c for c in ("sex", "site") if c in d.columns]].reset_index(drop=True)
    # biến giả cho arm/sex/site để MICE dùng được
    X = d[["pss0", "pss4", "pss8"]].copy()
    for a_ in ARMS:
        if a_ != "CG": X[f"arm_{a_}"] = (d["arm"] == a_).astype(float)
    if "sex" in d.columns: X["sex_M"] = (d["sex"] == "M").astype(float)
    if "site" in d.columns:
        for lv in sorted(d["site"].dropna().unique())[1:]: X[f"site_{lv}"] = (d["site"] == lv).astype(float)
    cov = ["pss0"] + [c for c in X.columns if c.startswith(("sex_", "site_"))]
    arm_cols = [f"arm_{a_}" for a_ in ARMS if a_ != "CG"]
    rhs = " + ".join(arm_cols + cov)
    imp = mice.MICEData(X, perturbation_method="gaussian")
    for c in X.columns:
        if c in ("pss8", "pss4"):
            imp.set_imputer(c, formula=" + ".join([x for x in X.columns if x not in (c,)] ))
    np.random.seed(seed)
    fit = mice.MICE("pss8 ~ " + rhs, sm.OLS, imp).fit(n_burnin=10, n_imputations=m)
    names = list(fit.model.exog_names)
    params = pd.Series(np.asarray(fit.params), index=names)
    covm = pd.DataFrame(np.asarray(fit.cov_params()), index=names, columns=names)
    def L(arm):
        v = pd.Series(0.0, index=names); v["Intercept"] = 1.0
        for nm in names:
            if nm.startswith("arm_"): v[nm] = float(nm == f"arm_{arm}")
            elif nm != "Intercept": v[nm] = X[nm].mean()
        return v
    out = []
    for label, A, B in CONTRASTS:
        c = sum(L(a) for a in A) / len(A) - sum(L(b) for b in B) / len(B)
        est = float(c @ params); se = float(np.sqrt(c @ covm @ c))
        p = 2 * stats.norm.sf(abs(est / se))
        out.append(dict(contrast=label, kind="confirmatory", estimate=est, se=se, ci_low=est - 1.96 * se, ci_high=est + 1.96 * se, p=p, method="MI"))
    res = pd.DataFrame(out)
    sec = res.index[res["contrast"].str.startswith(("C2", "C3"))]
    pv = res.loc[sec, "p"].values; order = np.argsort(pv); adj = np.empty_like(pv); run = 0
    for rank, idx in enumerate(order):
        run = max(run, (len(pv) - rank) * pv[idx]); adj[idx] = min(1.0, run)
    res.loc[sec, "p_holm"] = adj
    return res, len(d), m


# ---------------------------------------------------------------- figure
def draw(summ, res, adj_means, n_ancova, omnibus, out, res_mi=None, n_itt=None, m_imp=None, dpi=300):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(17 / 2.54 * 1.5, 8 / 2.54 * 1.5),
                                   gridspec_kw=dict(width_ratios=[1.05, 1.25], wspace=0.45))
    # ---- A trajectories
    off = {"MS": -0.25, "MC": -0.08, "MP": 0.08, "CG": 0.25}
    # vị trí nhãn n= tại W8, tách tối thiểu 0.45 điểm để không chồng chữ
    ends = {arm: float(summ[(summ["arm"] == arm) & (summ["week"] == ENDPOINT)]["mean"].iloc[0]) for arm in ARMS}
    lab_y = dict(sorted(ends.items(), key=lambda kv: kv[1]))
    prev = None
    for arm in lab_y:
        if prev is not None and lab_y[arm] - prev < 0.45: lab_y[arm] = prev + 0.45
        prev = lab_y[arm]
    for arm, (name, col) in ARMS.items():
        s = summ[summ["arm"] == arm]
        x = s["week"] + off[arm]
        axA.errorbar(x, s["mean"], yerr=s["ci95"], color=col, lw=1.6, marker="o", ms=4.5,
                     capsize=2.5, label=name)
        n0 = int(s.loc[s["week"] == 0, "n"].iloc[0]); n8 = int(s.loc[s["week"] == ENDPOINT, "n"].iloc[0])
        axA.text(ENDPOINT + 0.45, lab_y[arm], f"n={n8}", fontsize=6, color=col, va="center")
    axA.set_xticks(WEEKS); axA.set_xticklabels([f"Week {w}" for w in WEEKS], fontsize=8)
    axA.set_xlim(-0.8, ENDPOINT + 1.6)
    axA.set_ylabel("PSS-10 score (mean ± 95 % CI)", fontsize=8)
    axA.tick_params(axis="y", labelsize=7.5)
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)
    axA.grid(axis="y", color="#e5e7eb", lw=0.6)
    axA.legend(frameon=False, fontsize=6.5, loc="lower left")
    axA.set_title("Perceived stress by arm — observed cases (descriptive)", loc="left", fontsize=8.5, fontweight="bold")
    axA.text(-0.12, 1.06, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")

    # ---- B forest
    conf = res[res["kind"] == "confirmatory"].reset_index(drop=True)
    expl = res[res["kind"] == "exploratory"]
    rows = list(conf.itertuples()) + ([None] if len(expl) else []) + list(expl.itertuples())
    y = np.arange(len(rows))[::-1]
    axB.axvspan(-MCID, MCID, color="#fef3c7", zorder=0)
    axB.axvline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    labels = []
    for yi, r in zip(y, rows):
        if r is None:
            labels.append(""); continue
        if r.kind == "confirmatory" and res_mi is not None:
            mi = res_mi.iloc[r.Index]
            # MI = primary (black, filled); complete-case = sensitivity (grey, open, offset below)
            axB.plot([mi.ci_low, mi.ci_high], [yi + 0.12, yi + 0.12], color=INK, lw=1.5)
            axB.plot(mi.estimate, yi + 0.12, marker="s", ms=5, color=INK)
            axB.plot([r.ci_low, r.ci_high], [yi - 0.22, yi - 0.22], color="#9ca3af", lw=1.1)
            axB.plot(r.estimate, yi - 0.22, marker="s", ms=4, mfc="white", mec="#9ca3af", color="#9ca3af")
            ptxt = f"p = {mi.p:.3f}" if mi.p >= 0.001 else "p < 0.001"
            if not np.isnan(getattr(mi, "p_holm", np.nan)): ptxt += f" (Holm {mi.p_holm:.3f})"
            axB.text(1.02, yi + 0.12, f"MI: {mi.estimate:+.2f} [{mi.ci_low:+.2f}, {mi.ci_high:+.2f}]  {ptxt}",
                     transform=axB.get_yaxis_transform(), fontsize=5.8, color=INK, va="center")
            axB.text(1.02, yi - 0.22, f"CC: {r.estimate:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]  g = {r.hedges_g:+.2f}",
                     transform=axB.get_yaxis_transform(), fontsize=5.6, color="#6b7280", va="center")
        else:
            col = INK if r.kind == "confirmatory" else "#9ca3af"
            axB.plot([r.ci_low, r.ci_high], [yi, yi], color=col, lw=1.4)
            axB.plot(r.estimate, yi, marker="s" if r.kind == "confirmatory" else "o", ms=5, color=col)
            ptxt = f"p = {r.p:.3f}" if r.p >= 0.001 else "p < 0.001"
            if r.kind == "confirmatory" and not np.isnan(getattr(r, "p_holm", np.nan)):
                ptxt += f" (Holm {r.p_holm:.3f})"
            axB.text(1.02, yi, f"{r.estimate:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]  g = {r.hedges_g:+.2f}  {ptxt}",
                     transform=axB.get_yaxis_transform(), fontsize=6, color=col, va="center")
        labels.append(r.contrast)
    axB.set_yticks(y); axB.set_yticklabels(labels, fontsize=7)
    axB.set_xlabel("Adjusted mean difference in Week-8 PSS-10 (points)\n← favours first arm(s)   favours second arm(s) →", fontsize=7)
    axB.tick_params(axis="x", labelsize=7)
    lim = max(abs(res["ci_low"].min()), abs(res["ci_high"].max()), MCID + 1) * 1.1
    axB.set_xlim(-lim, lim)
    for sp in ("top", "right", "left"): axB.spines[sp].set_visible(False)
    axB.tick_params(axis="y", length=0)
    axB.text(0, -0.9, f"MCID band ± {MCID:g} points", fontsize=6, color="#92400e", ha="center", va="top")
    axB.set_ylim(-1.2, len(rows) - 0.5)
    ttl = f"Pre-specified contrasts — primary: ITT, MI m = {m_imp} (n = {n_itt}); sensitivity: complete-case (n = {n_ancova})" if res_mi is not None else f"Pre-specified contrasts, ANCOVA (n = {n_ancova})"
    axB.set_title(ttl, loc="left", fontsize=7.6, fontweight="bold")
    axB.text(-0.35, 1.06, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
    om = f"Omnibus F({int(omnibus.df_num)}, {int(omnibus.df_denom)}) = {float(omnibus.fvalue):.2f}, p = {float(omnibus.pvalue):.3f}"
    am = " · ".join(f"{a} {adj_means[a]:.1f}" for a in ARMS)
    fig.text(0.5, -0.04, f"Model: Week-8 PSS-10 ~ arm + baseline PSS-10 (+ sex, site). Primary analysis = intention-to-treat with multiple imputation "
             f"(filled squares); complete-case ANCOVA shown as sensitivity (open squares; {om}; adjusted W8 means {am}). "
             f"Exploratory pairwise contrasts (grey circles) are complete-case, unadjusted for multiplicity.", ha="center", fontsize=6.0, color=MUTE, wrap=True)
    for ext in ("svg", "png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- simulate
def simulate(path, seed=7):
    rng = np.random.default_rng(seed)
    alloc = {"MS": 45, "MC": 45, "MP": 43, "CG": 46}
    eff8 = {"MS": -6.0, "MC": -6.5, "MP": -4.0, "CG": -3.0}   # giả lập giảm điểm ở tuần 8
    rows = []; k = 0
    for arm, nA in alloc.items():
        for _ in range(nA):
            k += 1; pid = f"P{k:03d}"
            sex = rng.choice(["F", "M"], p=[0.6, 0.4]); site = rng.choice(["A", "B", "C"], p=[.44, .32, .24])
            b = rng.normal(24, 3.2); b = float(np.clip(round(b), 20, 36))
            u = rng.normal(0, 2.2)
            w4 = b + eff8[arm] * 0.55 + u + rng.normal(0, 2.5)
            w8 = b + eff8[arm] + u + rng.normal(0, 2.8)
            miss8 = rng.random() < {"MS": 3/45, "MC": 3/45, "MP": 4/43, "CG": 2/46}[arm]
            miss4 = rng.random() < 0.05
            rows.append((pid, arm, 0, b, sex, site))
            rows.append((pid, arm, 4, np.nan if miss4 else float(np.clip(round(w4), 0, 40)), sex, site))
            rows.append((pid, arm, 8, np.nan if miss8 else float(np.clip(round(w8), 0, 40)), sex, site))
    pd.DataFrame(rows, columns=["id", "arm", "week", "pss", "sex", "site"]).to_csv(path, index=False)
    print(f"wrote simulated {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="fig3_primary")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("-m", type=int, default=50, help="số bộ imputation")
    a = ap.parse_args()
    if a.simulate:
        simulate(a.csv); return
    wide = load(a.csv)
    summ = summary(wide)
    res, adj_means, model, omnibus, n_anc = ancova(wide)
    res["method"] = "complete-case"
    res_mi, n_itt, m_imp = ancova_mi(wide, m=a.m)
    summ.to_csv(f"{a.out}_summary.csv", index=False)
    pd.concat([res_mi, res]).to_csv(f"{a.out}_stats.csv", index=False)
    draw(summ, res, adj_means, n_anc, omnibus, a.out, res_mi=res_mi, n_itt=n_itt, m_imp=m_imp)
    print("PRIMARY (ITT, MI):"); print(res_mi[["contrast", "estimate", "ci_low", "ci_high", "p"]].round(3).to_string(index=False))
    print(summ.pivot(index="arm", columns="week", values="mean").round(2))
    print(res[["contrast", "estimate", "ci_low", "ci_high", "hedges_g", "p"]].round(3).to_string(index=False))
    print(f"wrote {a.out}.svg/.png/.pdf, {a.out}_summary.csv, {a.out}_stats.csv")


if __name__ == "__main__":
    main()
