#!/usr/bin/env python3
"""
Sinh counts.json cho consort_flow.py TỪ DỮ LIỆU, để Fig 1 và Fig 3–6 luôn cùng một nguồn.

    python make_counts.py participants_wide.csv enrolment.json [--cortisol fig4_cortisol_indices.csv] [-o counts.json]

participants_wide.csv (một dòng = một người được ngẫu nhiên hoá):
    id, arm, site, sex, pss_w0, pss_w8, sessions_completed
    + tuỳ chọn:  started (1/0), not_started_reason,
                 w8_status  = assessed | lost | discontinued_assessed | discontinued_lost
                 w8_reason  (lý do lost / discontinued)
    Nếu thiếu w8_status: pss_w8 trống -> "lost", có -> "assessed" (và cảnh báo).
enrolment.json: các số TRƯỚC ngẫu nhiên hoá (không có trong file người tham gia):
    sites, recruitment_period, assessed, pss_cutoff, excluded_total, not_meeting_criteria,
    not_meeting_reasons, declined, other_excluded, baseline_entered, withdrew_pre_rand,
    withdrew_pre_rand_reasons, acute_subsample, show_sites
Các ô theo nhánh được TÍNH: allocated, started, not_started (+lý do), lost, discontinued (+lý do),
discontinued_but_assessed, assessed_w8, adherent_80, itt, outcome_observed, pp, cortisol_w0/w8.
"""
import argparse, json, sys
import pandas as pd

ARMS = ["MS", "MC", "MP", "CG"]
N_SESSIONS, ADH = 19, 0.8


def reasons(series):
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return {k: int(v) for k, v in s.value_counts().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wide"); ap.add_argument("enrol")
    ap.add_argument("--cortisol", default=None, help="fig4_cortisol_indices.csv (cột slope_w0, slope_w8)")
    ap.add_argument("-o", "--out", default="counts.json")
    a = ap.parse_args()
    w = pd.read_csv(a.wide)
    enrol = json.load(open(a.enrol, encoding="utf-8"))
    if "w8_status" not in w.columns:
        print("  ⚠ không có cột w8_status: coi thiếu pss_w8 = lost to follow-up", file=sys.stderr)
        w["w8_status"] = w["pss_w8"].notna().map({True: "assessed", False: "lost"})
    if "started" not in w.columns:
        w["started"] = 1
    if "not_started_reason" not in w.columns: w["not_started_reason"] = ""
    if "w8_reason" not in w.columns: w["w8_reason"] = ""
    cort = pd.read_csv(a.cortisol).set_index("id") if a.cortisol else None
    out = dict(enrol)
    out["randomised"] = enrol.get("randomised", int(len(w)))
    if out["randomised"] != len(w):
        print(f"  ⚠ randomised trong enrolment.json ({out['randomised']}) ≠ số dòng file ({len(w)})", file=sys.stderr)
        out["randomised"] = int(len(w))
    thr = ADH * N_SESSIONS
    for arm in ARMS:
        g = w[w["arm"] == arm]
        assessed = g["w8_status"].isin(["assessed", "discontinued_assessed"])
        has_w8 = g["pss_w8"].notna()
        if (assessed != has_w8).any():
            print(f"  ⚠ {arm}: w8_status và pss_w8 không khớp ở {int((assessed != has_w8).sum())} người", file=sys.stderr)
        lost = g["w8_status"] == "lost"
        disc = g["w8_status"].str.startswith("discontinued")
        adh = g["sessions_completed"] >= thr
        c0 = c8 = None
        if cort is not None:
            ci = cort.reindex(g["id"])
            c0 = int(ci["slope_w0"].notna().sum()); c8 = int((ci["slope_w0"].notna() & ci["slope_w8"].notna()).sum())
        out[arm] = dict(
            allocated=int(len(g)),
            started=int((g["started"] == 1).sum()), not_started=int((g["started"] != 1).sum()),
            not_started_reasons=reasons(g.loc[g["started"] != 1, "not_started_reason"]),
            not_started_assessed=int(((g["started"] != 1) & has_w8).sum()),
            not_started_lost=int(((g["started"] != 1) & lost).sum()),
            not_started_discontinued=int(((g["started"] != 1) & disc).sum()),
            lost_to_followup=int(lost.sum()), lost_reasons=reasons(g.loc[lost, "w8_reason"]),
            discontinued=int(disc.sum()), discontinued_reasons=reasons(g.loc[disc, "w8_reason"]),
            discontinued_but_assessed=int((g["w8_status"] == "discontinued_assessed").sum()),
            assessed_w8=int(has_w8.sum()), adherent_80=int(adh.sum()),
            itt=int(len(g)), outcome_observed=int(has_w8.sum()), pp=int((adh & has_w8).sum()),
            cortisol_w0=c0, cortisol_w8=c8,
        )
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {a.out}")
    print(pd.DataFrame({a_: {k: out[a_][k] for k in ("allocated", "assessed_w8", "adherent_80", "pp", "cortisol_w8")} for a_ in ARMS}).T)


if __name__ == "__main__":
    main()
