#!/usr/bin/env python3
"""
CONSORT 2010 flow diagram — MINDSTROKE 4-arm, multi-centre RCT.

    python consort_flow.py                  # dùng khối COUNTS dưới đây -> consort_flow.svg/.png/.pdf
    python consort_flow.py -o fig1          # đổi tên file xuất
    python consort_flow.py --strict         # bản CONSORT thuần (bỏ dải can thiệp & ô phân tích)
    python consort_flow.py counts.json      # hoặc đọc số từ JSON cùng cấu trúc COUNTS

Quy ước: None -> in "[ ]".  Lý do (reasons) là dict {"lý do": n}; {} -> in "reasons: ___".
Script KHÔNG tự tính số dẫn xuất; nó chỉ KIỂM TRA cộng dồn và cảnh báo nếu lệch.
"""
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ======================================================================
#  ĐIỀN SỐ Ở ĐÂY
# ======================================================================
def arm(**kw):
    base = dict(
        allocated=None,
        started=None,                 # started allocated intervention (≥ 1 session)
        not_started=None, not_started_reasons={},
        not_started_assessed=None, not_started_lost=None, not_started_discontinued=None,  # người không bắt đầu đi đâu ở W8
        lost_to_followup=None, lost_reasons={},
        discontinued=None, discontinued_reasons={},
        discontinued_but_assessed=None,   # số discontinued vẫn được đo tuần 8
        assessed_w8=None,             # có kết cục chính tuần 8
        adherent_80=None,             # hoàn thành ≥ 80 % phiên
        itt=None,                     # = allocated (ITT, multiple imputation)
        outcome_observed=None,        # = assessed_w8
        pp=None,                      # ≥ 80 % phiên VÀ có kết cục tuần 8
        cortisol_w0=None, cortisol_w8=None,   # hồ sơ cortisol hợp lệ ở W0 / cả W0+W8
    )
    base.update(kw); return base

COUNTS = {
    # --- trung tâm tuyển chọn (đa trung tâm) -------------------------
    "sites": {
        "Site A · Univ. of Health Sciences": dict(screened=None, randomised=None),
        "Site B · [University]":             dict(screened=None, randomised=None),
        "Site C · [University]":             dict(screened=None, randomised=None),
    },
    "show_sites": False,               # True = vẽ hàng ô từng site; False = chỉ ghi "multi-centre, k sites"
    "recruitment_period": "",          # vd "Mar–Sep 2027"; để "" nếu không in

    # --- enrolment ---------------------------------------------------
    "assessed": None,                  # tổng sàng lọc (= tổng screened các site)
    "pss_cutoff": "≥ 20",              # ngưỡng stress cao (Cohen); đổi "≥ 14" nếu dùng mức trung bình
    "excluded_total": None,
    "not_meeting_criteria": None,      # tổng không đạt tiêu chí
    "not_meeting_reasons": {           # tiểu mục của dòng trên (phải cộng = not_meeting_criteria)
        "PSS-10 below cut-off":                               None,
        "Clinical-range symptoms, referred":  None,   # PHQ-9/GAD-7 ≥ 15
        "Medical / HPA-axis exclusion":       None,   # endocrine, steroids, shift work
        "Regular art, music or gaming practice":              None,
        "Other criteria":                                     None,
    },
    "declined": None,
    "other_excluded": None,

    "baseline_entered": None,          # vào đánh giá baseline
    "withdrew_pre_rand": None,         # rút lui / baseline không hoàn chỉnh trước ngẫu nhiên hoá
    "withdrew_pre_rand_reasons": {},   # vd {"incomplete saliva sampling": 3, "withdrew consent": 2}
    "randomised": None,

    # --- 4 nhánh -----------------------------------------------------
    "MS": arm(),
    "MC": arm(),
    "MP": arm(),
    "CG": arm(),

    # --- sub-sample lấy mẫu cấp quanh phiên (để None nếu không dùng) --
    "acute_subsample": None,
}
# ======================================================================

ARMS = [
    ("MS", "Meaningful Sketching", "#D55E00"),
    ("MC", "Mindful Calligraphy", "#0072B2"),
    ("MP", "Music Practice",      "#009E73"),
    ("CG", "Casual Gaming",       "#CC79A7"),
]
INK, GREY, SOFT, MUTE = "#1f2937", "#374151", "#eef2f7", "#4b5563"


# ------------------------------------------------------------------ helpers
def n(v):
    return "[ ]" if v is None else f"{v:,}"


def reasons(d, blank="reasons: ___"):
    if not d:
        return blank
    return "; ".join(f"{k} {n(v)}" for k, v in d.items())


def warn(msg):
    print("  ⚠ " + msg, file=sys.stderr)


def check(c):
    """Kiểm tra cộng dồn; chỉ cảnh báo, không dừng."""
    def ok(*v): return all(x is not None for x in v)
    s = c["sites"].values()
    if ok(c["assessed"]) and all(ok(x["screened"]) for x in s) and sum(x["screened"] for x in s) != c["assessed"]:
        warn("tổng screened các site ≠ assessed")
    if ok(c["randomised"]) and all(ok(x["randomised"]) for x in s) and sum(x["randomised"] for x in s) != c["randomised"]:
        warn("tổng randomised các site ≠ randomised")
    if ok(c["excluded_total"], c["not_meeting_criteria"], c["declined"], c["other_excluded"]) and \
       c["not_meeting_criteria"] + c["declined"] + c["other_excluded"] != c["excluded_total"]:
        warn("not_meeting + declined + other ≠ excluded_total")
    r = c["not_meeting_reasons"].values()
    if ok(c["not_meeting_criteria"]) and all(ok(x) for x in r) and sum(r) != c["not_meeting_criteria"]:
        warn("tiểu mục not_meeting_reasons ≠ not_meeting_criteria")
    if ok(c["assessed"], c["excluded_total"], c["baseline_entered"]) and c["assessed"] - c["excluded_total"] != c["baseline_entered"]:
        warn("assessed − excluded ≠ baseline_entered")
    if ok(c["baseline_entered"], c["withdrew_pre_rand"], c["randomised"]) and c["baseline_entered"] - c["withdrew_pre_rand"] != c["randomised"]:
        warn("baseline_entered − withdrew_pre_rand ≠ randomised")
    tot = 0
    for code, _, _ in ARMS:
        a = c[code]
        if ok(a["allocated"]): tot += a["allocated"]
        if ok(a["allocated"], a["started"], a["not_started"]) and a["started"] + a["not_started"] != a["allocated"]:
            warn(f"{code}: started + not_started ≠ allocated")
        if ok(a["allocated"], a["lost_to_followup"], a["discontinued"], a["discontinued_but_assessed"], a["assessed_w8"]) and \
           a["allocated"] - a["lost_to_followup"] - (a["discontinued"] - a["discontinued_but_assessed"]) != a["assessed_w8"]:
            warn(f"{code}: allocated − lost − (discontinued không đo) ≠ assessed_w8 — 'did not start' phải nằm trong lost/discontinued/assessed, không cộng thêm")
        if ok(a["not_started"], a["not_started_assessed"], a["not_started_lost"], a["not_started_discontinued"]) and \
           a["not_started_assessed"] + a["not_started_lost"] + a["not_started_discontinued"] != a["not_started"]:
            warn(f"{code}: phân bổ 'did not start' (assessed+lost+discontinued) ≠ not_started")
        if ok(a["itt"], a["allocated"]) and a["itt"] != a["allocated"]:
            warn(f"{code}: itt ≠ allocated (ITT phải = allocated)")
        if ok(a["outcome_observed"], a["assessed_w8"]) and a["outcome_observed"] != a["assessed_w8"]:
            warn(f"{code}: outcome_observed ≠ assessed_w8")
        if ok(a["pp"], a["adherent_80"]) and a["pp"] > a["adherent_80"]:
            warn(f"{code}: pp > adherent_80")
        if ok(a["cortisol_w8"], a["cortisol_w0"]) and a["cortisol_w8"] > a["cortisol_w0"]:
            warn(f"{code}: cortisol_w8 > cortisol_w0")
        for k in ("lost", "discontinued", "not_started"):
            tot_key = {"lost": "lost_to_followup"}.get(k, k)
            rs = a[f"{k}_reasons"]
            if rs and ok(a[tot_key]) and all(ok(v) for v in rs.values()) and sum(rs.values()) != a[tot_key]:
                warn(f"{code}: lý do {k} cộng ≠ {tot_key}")
    if ok(c["randomised"]) and tot and tot != c["randomised"]:
        warn("tổng allocated 4 nhánh ≠ randomised")


# ------------------------------------------------------------------ drawing
def box(ax, x, y, w, h, lines, *, fill="white", edge=GREY, lw=1.1, head=None,
        head_color=None, fs=7, align="center", bold_first=True, dashed=False):
    ax.add_patch(FancyBboxPatch((x, -y - h), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec=edge, lw=lw, ls=(0, (4, 3)) if dashed else "-"))
    top = y
    if head:
        ax.add_patch(FancyBboxPatch((x, -y - 0.4), w, 0.4, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=head_color, ec=head_color))
        ax.text(x + w / 2, -y - 0.2, head, ha="center", va="center", color="white",
                fontsize=fs + 0.8, fontweight="bold")
        top = y + 0.4
    step = (h - (top - y)) / (len(lines) + 1)
    for i, line in enumerate(lines):
        yy = -(top + step * (i + 1))
        bold = bold_first and i == 0
        if isinstance(line, tuple):            # (text, {"indent":..,"color":..,"bold":..})
            line, opt = line
            bold = opt.get("bold", bold)
            col = opt.get("color", INK); ind = opt.get("indent", 0)
        else:
            col, ind = INK, 0
        if align == "left":
            ax.text(x + 0.15 + ind, yy, line, ha="left", va="center", fontsize=fs, color=col,
                    fontweight="bold" if bold else "normal")
        else:
            ax.text(x + w / 2, yy, line, ha="center", va="center", fontsize=fs, color=col,
                    fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, -y0), (x1, -y1), arrowstyle="-|>", mutation_scale=10,
                                 lw=1.1, color=GREY, shrinkA=0, shrinkB=0))


def elbow(ax, x0, y0, ym, x1, y1):
    ax.plot([x0, x0, x1], [-y0, -ym, -ym], color=GREY, lw=1.1)
    arrow(ax, x1, ym, x1, y1)


def phase(ax, y0, y1, label):
    ax.add_patch(FancyBboxPatch((0.2, -y1), 0.9, y1 - y0, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=GREY, ec=GREY))
    ax.text(0.65, -(y0 + y1) / 2, label, rotation=90, ha="center", va="center", color="white",
            fontsize=8 if (y1 - y0) > 1.6 else 6.5, fontweight="bold")


def draw(c, out, strict=False, dpi=300):
    W = 17.0
    y_alloc, h_alloc = 6.1, 1.5
    y_band, h_band = 7.85, 0.7
    y_fu = 8.8 if not strict else 7.85
    h_fu = 2.35
    y_an = y_fu + h_fu + 0.45
    h_an = 1.55
    y_fn = y_an + h_an + 0.3
    n_notes = 3 if strict else 4
    H = y_fn + 0.3 * n_notes + 0.25
    fig, ax = plt.subplots(figsize=(W / 2.54 * 2.1, H / 2.54 * 2.1))
    y_top0 = 0 if c.get("show_sites") else -1.25
    fig.set_size_inches(W / 2.54 * 2.1, (H + y_top0) / 2.54 * 2.1)
    ax.set_xlim(0, W); ax.set_ylim(-H, y_top0); ax.axis("off")
    plt.rcParams["font.family"] = "DejaVu Sans"
    cx = 8.6

    # ---------------- sites row (optional) ----------------------------
    sites = list(c["sites"].items())
    y_top = 1.45
    if c.get("show_sites"):
        sw, sg = 3.3, 0.25
        total_w = len(sites) * sw + (len(sites) - 1) * sg
        sx0 = cx - total_w / 2
        for i, (name, s_) in enumerate(sites):
            x = sx0 + i * (sw + sg)
            box(ax, x, 0.25, sw, 0.78,
                [name, f"screened {n(s_['screened'])} · randomised {n(s_['randomised'])}"], fs=6.4)
            arrow(ax, x + sw / 2, 1.03, x + sw / 2, 1.2)
        ax.plot([sx0 + sw / 2, sx0 + total_w - sw / 2], [-1.2, -1.2], color=GREY, lw=1.1)
        arrow(ax, cx, 1.2, cx, 1.45)
        if c["recruitment_period"]:
            ax.text(sx0 + total_w + 0.2, -0.64, f"Recruitment\n{c['recruitment_period']}", fontsize=6.2,
                    color=MUTE, va="center")
        site_line = None
    else:
        k = len(sites)
        site_line = f"multi-centre recruitment, {k} university sites"
        if c["recruitment_period"]:
            site_line += f" · {c['recruitment_period']}"

    # ---------------- enrolment --------------------------------------
    phase(ax, 0.25 if c.get("show_sites") else 1.45, 4.5, "Enrolment")
    lines = [f"Assessed for eligibility (n = {n(c['assessed'])})",
             f"healthy university students · age 18–25 · PSS-10 {c['pss_cutoff']} at screening"]
    if site_line:
        lines.append((site_line, {"color": MUTE}))
    box(ax, cx - 2.6, 1.45, 5.2, 0.85 + (0.22 if site_line else 0), lines, fs=6.8)
    ex = [f"Excluded (n = {n(c['excluded_total'])})",
          f"Not meeting inclusion criteria (n = {n(c['not_meeting_criteria'])})"]
    ex += [(f"– {k} (n = {n(v)})", {"indent": 0.3, "color": MUTE}) for k, v in c["not_meeting_reasons"].items()]
    ex += [f"Declined to participate (n = {n(c['declined'])})",
           f"Other reasons (n = {n(c['other_excluded'])})"]
    box(ax, 11.6, 1.45, 5.2, 0.3 * len(ex) + 0.3, ex, align="left", fs=6.4)
    arrow(ax, cx + 2.6, 1.85, 11.6, 1.85)

    box(ax, cx - 2.6, 2.75 + (0.1 if site_line else 0), 5.2, 0.85,
        [f"Entered baseline assessment, Week 0 (n = {n(c['baseline_entered'])})",
         "PSS-10 · STAI · PANAS · PSQI · FFMQ · salivary cortisol (2 d × 5)"], fs=6.8)
    arrow(ax, cx, 2.3 + (0.22 if site_line else 0), cx, 2.75)
    nr = [f"Not randomised (n = {n(c['withdrew_pre_rand'])})"]
    nr += [(f"– {k} (n = {n(v)})", {"indent": 0.2, "color": MUTE})
           for k, v in c["withdrew_pre_rand_reasons"].items()] or ["reasons: ___"]
    h_nr = 0.3 * len(nr) + 0.3
    box(ax, 1.4, 3.75, 4.3, h_nr, nr, align="left", fs=6.4)
    arrow(ax, cx, 3.75 + h_nr / 2, 5.7, 3.75 + h_nr / 2)

    # ---------------- allocation -------------------------------------
    phase(ax, 4.6, 7.7, "Allocation")
    box(ax, cx - 2.6, 4.7, 5.2, 0.95,
        [f"Randomised 1:1:1:1 (n = {n(c['randomised'])})",
         "computer-generated permuted blocks within strata",
         "(site, sex, baseline PSS-10) · central concealed allocation"], fill=SOFT, fs=6.6)
    ax.plot([cx, cx], [-3.6, -4.7], color=GREY, lw=1.1); arrow(ax, cx, 4.5, cx, 4.7)

    col_w, gap, x_start = 3.55, 0.3, 1.5
    xs = [x_start + i * (col_w + gap) for i in range(4)]
    phase(ax, y_alloc + h_alloc + 0.15, y_an - 0.25, "Follow-up (8 wk)")
    phase(ax, y_an - 0.05, H - 0.1, "Analysis")

    for (code, name, color), x in zip(ARMS, xs):
        a = c[code]; xc = x + col_w / 2
        elbow(ax, cx, 5.65, 5.85, xc, y_alloc)
        box(ax, x, y_alloc, col_w, h_alloc,
            [f"Allocated (n = {n(a['allocated'])})",
             f"Started allocated intervention (n = {n(a['started'])})",
             f"Did not start (n = {n(a['not_started'])}) — " + reasons(a["not_started_reasons"], "reasons: ___"),
             (f"→ at W8: assessed {n(a['not_started_assessed'])} · lost {n(a['not_started_lost'])} · discontinued {n(a['not_started_discontinued'])}",
              {"indent": 0.2, "color": MUTE})],
            head=name, head_color=color, edge=color, lw=1.6, fs=6.0, align="left")
        if strict:
            arrow(ax, xc, y_alloc + h_alloc, xc, y_fu)
        else:
            arrow(ax, xc, y_alloc + h_alloc, xc, y_band)
            arrow(ax, xc, y_band + h_band, xc, y_fu)
        box(ax, x, y_fu, col_w, h_fu,
            [f"Assessed at Week 8 (n = {n(a['assessed_w8'])})",
             f"Lost to follow-up (n = {n(a['lost_to_followup'])})",
             (reasons(a["lost_reasons"]), {"indent": 0.2, "color": MUTE}),
             f"Discontinued intervention (n = {n(a['discontinued'])}),",
             (f"of whom assessed at W8 (n = {n(a['discontinued_but_assessed'])})", {"indent": 0.2}),
             (reasons(a["discontinued_reasons"]), {"indent": 0.2, "color": MUTE}),
             f"Adherent, ≥ 80 % sessions (n = {n(a['adherent_80'])})"],
            align="left", fs=6.1)
        arrow(ax, xc, y_fu + h_fu, xc, y_an)
        box(ax, x, y_an, col_w, h_an,
            [f"Analysed — ITT (n = {n(a['itt'])})ᵃ",
             f"primary outcome observed (n = {n(a['outcome_observed'])})",
             f"Per-protocol (n = {n(a['pp'])})ᵇ",
             f"Cortisol-evaluable W0 / W0+W8 (n = {n(a['cortisol_w0'])} / {n(a['cortisol_w8'])})ᶜ"],
            fs=6.3)

    x0, x1 = xs[0], xs[-1] + col_w
    if not strict:
        ax.add_patch(FancyBboxPatch((x0, -y_band - h_band), x1 - x0, h_band,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#f3f4f6", ec="#9ca3af", ls=(0, (4, 3))))
        ax.text((x0 + x1) / 2, -y_band - 0.25,
                "Intervention: 8 weeks · 45 min/session · every 3 days (≈19 sessions) · MINDSTROKE platform",
                ha="center", va="center", fontsize=7, fontweight="bold", color=INK)
        sub = "pre/post-session STAI-S & PANAS every session · stroke kinematics logged"
        if c["acute_subsample"] is not None:
            sub += f" · acute cortisol sub-sample (n = {n(c['acute_subsample'])})"
        ax.text((x0 + x1) / 2, -y_band - 0.5, sub, ha="center", va="center", fontsize=6.3, color=MUTE)

    # ---------------- footnotes -------------------------------------
    notes = [
        "ᵃ ITT = all randomised; missing Week-8 outcomes handled by multiple imputation (ANCOVA on Week-8 PSS-10, "
        "adjusted for baseline PSS-10, sex and site).",
        "ᵇ Per-protocol = completed ≥ 80 % of scheduled sessions AND provided the Week-8 primary outcome. "
        "Participants who did not start are counted again in the Week-8 box of their arm (the '→ at W8' line shows where); "
        "hence allocated − lost − discontinued-not-assessed = assessed, and 'did not start' is not added to that sum.",
        "ᶜ Cortisol-evaluable = ≥ 1 valid sampling day (S1, S2 within 30 ± 5 min, and S5 present) at the time point(s) stated.",
    ]
    if not strict:
        notes.append("Pre-specified contrasts: C1 (MS+MC) vs (MP+CG) · C2 MS vs MC · C3 MP vs CG. "
                     "Outcome assessors and statistician blinded to allocation.")
    for i, t in enumerate(notes):
        ax.text(x0, -(y_fn + 0.3 * i), t, fontsize=5.9, color=MUTE, va="top")

    for ext in ("svg", "png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main
def merge(user):
    c = json.loads(json.dumps(COUNTS))
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict) and not k.endswith("_reasons"):
            c[k].update(v)
        else:
            c[k] = v
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("counts", nargs="?")
    p.add_argument("-o", "--out", default="consort_flow")
    p.add_argument("--strict", action="store_true", help="bản CONSORT thuần")
    p.add_argument("--template", action="store_true", help="in JSON trống")
    a = p.parse_args()
    if a.template:
        json.dump(COUNTS, sys.stdout, indent=2, ensure_ascii=False); print(); return
    c = COUNTS
    if a.counts:
        with open(a.counts, encoding="utf-8") as f:
            c = merge(json.load(f))
    check(c)
    draw(c, a.out, strict=a.strict)
    print(f"wrote {a.out}.svg / .png / .pdf")


if __name__ == "__main__":
    main()
