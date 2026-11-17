#!/usr/bin/env python3
"""
Figure 2 — Study design and measurement schedule (MINDSTROKE RCT).

    python fig2_design.py            -> fig2_design.svg / .png / .pdf
    python fig2_design.py -o fig2

Ba panel:
  A  Dòng thời gian 8 tuần: phiên can thiệp, mốc đánh giá, cửa sổ lấy cortisol, sub-sample cấp
  B  Cấu trúc một phiên 45 phút và các điểm đo
  C  Lịch lấy mẫu cortisol trong một ngày (giờ kể từ khi thức dậy)
Mọi thông số thời gian nằm trong khối PARAMS; vẽ lại khi đổi thiết kế.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ======================================================================
#  THÔNG SỐ THIẾT KẾ
# ======================================================================
PARAMS = dict(
    weeks=8,                       # thời gian can thiệp
    session_every_days=3,          # tần suất phiên
    session_min=45,                # thời lượng phiên (phút)
    checkin_min=5, checkout_min=5, # phần đầu/cuối phiên (bảng hỏi)
    breath_cue_at=[0, 17],         # phút (tính từ đầu hoạt động) có cue thở 1 phút
    assess_weeks=[0, 4, 8],        # mốc đánh giá bảng hỏi (PSS-10, PSQI, FFMQ …)
    cortisol_weeks=[0, 8],         # cửa sổ cortisol nhịp ngày đêm (2 ngày × 5 mẫu)
    cortisol_days=2,
    acute_weeks=[1, 4, 8],         # phiên lấy mẫu cấp (sub-sample)
    acute_pre_min=-10, acute_post_min=20,   # mốc nước bọt quanh phiên
    screening_weeks=(-2, 0),       # cửa sổ sàng lọc + consent
    followup_week=12,              # theo dõi tuỳ chọn; None để ẩn
    diurnal=[("S1", "awakening", 0.0), ("S2", "+30 min", 0.5), ("S3", "+4 h", 4.0),
             ("S4", "+8 h", 8.0), ("S5", "bedtime", 15.0)],
    acute_session_window="15:00–18:00",
)
# ======================================================================

ARMS = [("Meaningful Sketching", "#D55E00"), ("Mindful Calligraphy", "#0072B2"),
        ("Music Practice", "#009E73"), ("Casual Gaming", "#CC79A7")]
INK, GREY, MUTE, SOFT = "#1f2937", "#374151", "#6b7280", "#eef2f7"
C_Q, C_CORT, C_ACUTE, C_SESS = "#111827", "#b45309", "#0072B2", "#9ca3af"


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")


# ---------------------------------------------------------------- panel A
def panel_a(ax, p):
    days = p["weeks"] * 7
    x0 = p["screening_weeks"][0] * 7
    x1 = (p["followup_week"] * 7 if p["followup_week"] else days) + 3
    ax.set_xlim(x0 - 1, x1); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title("Study timeline", loc="left", fontsize=9, fontweight="bold", color=INK)

    # screening band
    ax.add_patch(Rectangle((x0, 0.55), -x0, 4.6, fc="#f3f4f6", ec="none"))
    ax.text(x0 / 2, 5.0, "Screening\n& consent", ha="center", va="top", fontsize=6.5, color=MUTE)
    # intervention band
    ax.add_patch(Rectangle((0, 0.55), days, 4.6, fc="#fafafa", ec="#d1d5db", lw=0.8))
    ax.text(days / 2, 5.0, f"Intervention · {p['weeks']} weeks · {p['session_min']} min/session · "
            f"every {p['session_every_days']} days", ha="center", va="top", fontsize=6.8, color=INK,
            fontweight="bold")
    if p["followup_week"]:
        fx = p["followup_week"] * 7
        ax.add_patch(Rectangle((days, 0.55), fx - days, 4.6, fc="none", ec="#d1d5db", lw=0.8, ls=(0, (3, 3))))
        ax.text((days + fx) / 2, 5.0, "Optional\nfollow-up", ha="center", va="top", fontsize=6.5, color=MUTE)

    # week axis
    ax.plot([x0, x1 - 3], [0.55, 0.55], color=GREY, lw=1)
    for w in range(p["screening_weeks"][0], (p["followup_week"] or p["weeks"]) + 1):
        if w > p["weeks"] and w != p["followup_week"]:
            continue
        ax.plot([w * 7, w * 7], [0.55, 0.4], color=GREY, lw=0.8)
        ax.text(w * 7, 0.3, f"W{w}", ha="center", va="top", fontsize=6.5, color=GREY)

    # rows
    rows = {"sessions": 4.45, "q": 3.2, "cort": 1.95, "acute": 0.95}
    ax.text(x0 - 0.5, rows["sessions"], "Activity sessions", ha="right", va="center", fontsize=6.6, color=INK)
    ax.text(x0 - 0.5, rows["q"], "Questionnaires", ha="right", va="center", fontsize=6.6, color=INK)
    ax.text(x0 - 0.5, rows["cort"], "Diurnal cortisol", ha="right", va="center", fontsize=6.6, color=INK)
    ax.text(x0 - 0.5, rows["acute"], "Acute sub-sample", ha="right", va="center", fontsize=6.6, color=INK)

    # sessions
    sess = list(range(0, days + 1, p["session_every_days"]))
    sess = [d for d in sess if d <= days]
    for d in sess:
        for i, (_, col) in enumerate(ARMS):
            ax.plot([d], [rows["sessions"] + (i - 1.5) * 0.14], marker="|", ms=5, mew=1.2, color=col)
    ax.text(days + 0.8, rows["sessions"], f"{len(sess)} sessions\n(all arms)", va="center", fontsize=6, color=MUTE)

    # questionnaires
    for w in p["assess_weeks"]:
        ax.plot(w * 7, rows["q"], marker="D", ms=5, color=C_Q)
        lab = {0: "baseline", p["weeks"]: "primary endpoint"}.get(w, "mid-point")
        ax.text(w * 7, rows["q"] + 0.22, f"{lab}\nPSS-10 · STAI · PANAS · PSQI · FFMQ",
                ha="center", va="bottom", fontsize=5.4, color=MUTE, linespacing=1.1)
    if p["followup_week"]:
        ax.plot(p["followup_week"] * 7, rows["q"], marker="D", ms=5, mfc="white", mec=C_Q)
        ax.text(p["followup_week"] * 7, rows["q"] + 0.22, "PSS-10 · PSQI", ha="center", va="bottom",
                fontsize=5.4, color=MUTE)

    # cortisol windows
    for w in p["cortisol_weeks"]:
        d0 = w * 7 - (p["cortisol_days"] if w == p["weeks"] else 0)
        ax.add_patch(FancyBboxPatch((d0 - 0.3, rows["cort"] - 0.22), p["cortisol_days"] + 0.6, 0.44,
                                    boxstyle="round,pad=0,rounding_size=0.1", fc=C_CORT, ec="none"))
        ax.text(w * 7 - (p["cortisol_days"] / 2 if w == p["weeks"] else -p["cortisol_days"] / 2),
                rows["cort"] - 0.32, f"{p['cortisol_days']} days × 5 samples\n(see C)",
                ha="center", va="top", fontsize=5.4, color=MUTE)

    # acute sub-sample
    for w in p["acute_weeks"]:
        d = min(w * 7, days)
        d = max([s for s in sess if s <= d] or [d])
        ax.plot(d, rows["acute"], marker="o", ms=5, color=C_ACUTE)
    ax.text(days + 0.8, rows["acute"], "saliva pre / post\none session (see B)", va="center",
            fontsize=6, color=MUTE)

    # per-session measures note
    ax.text(0.5, rows["sessions"] - 0.42, "every session: pre/post STAI-S & PANAS · enjoyment/mastery ratings · "
            "stroke kinematics logged", fontsize=5.6, color=MUTE, va="top")
    # arm legend
    for i, (name, col) in enumerate(ARMS):
        ax.plot([x1 - 14 + i * 0.1], [6.0], marker="|", ms=0)  # keep xlim
    lx = x0
    for name, col in ARMS:
        ax.plot(lx, 5.85, marker="s", ms=4, color=col)
        ax.text(lx + 0.8, 5.85, name, va="center", fontsize=5.8, color=INK)
        lx += 14


# ---------------------------------------------------------------- panel B
def panel_b(ax, p):
    pre, post = p["checkin_min"], p["checkout_min"]
    core = p["session_min"] - pre - post
    tmin, tmax = p["acute_pre_min"] - 4, p["session_min"] + p["acute_post_min"] + 6
    ax.set_xlim(tmin, tmax); ax.set_ylim(0, 3.4); ax.axis("off")
    ax.set_title("Session structure and measurement points (identical across arms)", loc="left",
                 fontsize=9, fontweight="bold", color=INK)
    y, h = 1.5, 0.8
    ax.add_patch(Rectangle((0, y), pre, h, fc=C_SESS, ec="none"))
    ax.add_patch(Rectangle((pre, y), core, h, fc=GREY, ec="none"))
    ax.add_patch(Rectangle((pre + core, y), post, h, fc=C_SESS, ec="none"))
    ax.text(pre / 2, y + h / 2, f"Check-in\n{pre} min", ha="center", va="center", fontsize=6, color="white")
    ax.text(pre + core / 2, y + h / 2, f"Arm-specific core activity · {core} min",
            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax.text(pre + core + post / 2, y + h / 2, f"Check-out\n{post} min", ha="center", va="center",
            fontsize=6, color="white")
    for t in p["breath_cue_at"]:
        ax.plot([pre + t, pre + t], [y, y + h], color="white", lw=1, ls=(0, (2, 2)))
    ax.text(pre + p["breath_cue_at"][-1] + 1, y + h + 0.08, "1-min breathing cue (all arms)", fontsize=5.4,
            color=MUTE, va="bottom")
    # time axis
    ax.plot([0, p["session_min"]], [y - 0.15, y - 0.15], color=GREY, lw=1)
    for t in (0, pre, pre + core, p["session_min"]):
        ax.plot([t, t], [y - 0.15, y - 0.28], color=GREY, lw=0.8)
        ax.text(t, y - 0.33, f"{t}", ha="center", va="top", fontsize=6, color=GREY)
    ax.text(p["session_min"] / 2, y - 0.62, "minutes", ha="center", va="top", fontsize=6, color=GREY)
    # measurement markers
    def mark(t, label, col, yy):
        ax.plot(t, yy, marker="v", ms=6, color=col)
        ax.text(t, yy + 0.16, label, ha="center", va="bottom", fontsize=5.6, color=col, linespacing=1.1)
    mark(pre / 2, "STAI-S · PANAS\n(pre)", C_Q, y + h + 0.35)
    mark(pre + core + post / 2, "STAI-S · PANAS\n(post) · TMS\nmastery · enjoyment", C_Q, y + h + 0.35)
    mark(p["acute_pre_min"], f"saliva\n{p['acute_pre_min']} min", C_CORT, y + h + 0.35)
    mark(p["session_min"] + p["acute_post_min"], f"saliva\n+{p['acute_post_min']} min", C_CORT, y + h + 0.35)
    ax.text(tmin + 1, 0.35, f"Saliva around sessions only in the acute sub-sample at W{', W'.join(map(str, p['acute_weeks']))}; "
            f"sessions scheduled {p['acute_session_window']} to limit circadian variation.",
            fontsize=5.6, color=MUTE, va="bottom")


# ---------------------------------------------------------------- panel C
def panel_c(ax, p):
    d = p["diurnal"]
    tmax = d[-1][2] + 1.5
    ax.set_xlim(-0.8, tmax); ax.set_ylim(-0.5, 3.2); ax.axis("off")
    ax.set_title("Diurnal salivary cortisol sampling (each sampling day)", loc="left",
                 fontsize=9, fontweight="bold", color=INK)
    y = 1.5
    ax.plot([0, d[-1][2]], [y, y], color=GREY, lw=1)
    for i, (code, lab, t) in enumerate(d):
        ax.plot(t, y, marker="o", ms=6, mfc="white", mec=C_CORT, mew=1.6)
        ax.text(t, y + 0.3, code, ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=C_CORT)
        ax.text(t, y - 0.3 - (0.3 if i == 1 else 0), lab, ha="center", va="top", fontsize=6, color=INK)
    ax.text(d[-1][2] / 2, y - 0.8, "hours since awakening (app-timestamped)", ha="center", va="top",
            fontsize=6, color=GREY)
    # brackets for indices
    ax.annotate("", xy=(d[1][2], y + 0.85), xytext=(d[0][2], y + 0.85),
                arrowprops=dict(arrowstyle="|-|", color=C_CORT, lw=0.8, mutation_scale=3))
    ax.text((d[0][2] + d[1][2]) / 2, y + 0.95, "CAR (AUC$_I$)", ha="center", va="bottom", fontsize=5.8, color=C_CORT)
    ax.annotate("", xy=(d[-1][2], y + 0.85), xytext=(d[1][2], y + 0.85),
                arrowprops=dict(arrowstyle="|-|", color=GREY, lw=0.8, mutation_scale=3))
    ax.text((d[1][2] + d[-1][2]) / 2, y + 0.95, "diurnal slope (key index) · AUC$_G$ over S1–S5",
            ha="center", va="bottom", fontsize=5.8, color=GREY)
    ax.text(-0.8, -0.45, "Rules: no food, drink, caffeine or tooth-brushing ≥ 30 min before; no vigorous exercise ≥ 1 h; "
            "S2 within 30 ± 5 min of waking;\n2 consecutive weekdays outside exam weeks; samples kept at −20 °C then −80 °C; "
            "a participant's W0 and W8 samples assayed on the same plate.",
            fontsize=5.4, color=MUTE, va="bottom", linespacing=1.3)


def draw(p, out, dpi=300):
    fig = plt.figure(figsize=(17 / 2.54 * 1.6, 13 / 2.54 * 1.6))
    plt.rcParams["font.family"] = "DejaVu Sans"
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.25, 1.15], hspace=0.3,
                          left=0.12, right=0.97, top=0.95, bottom=0.03)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    panel_a(axA, p); panel_label(axA, "A")
    panel_b(axB, p); panel_label(axB, "B")
    panel_c(axC, p); panel_label(axC, "C")
    for ext in ("svg", "png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="fig2_design")
    a = ap.parse_args()
    draw(PARAMS, a.out)
    print(f"wrote {a.out}.svg / .png / .pdf")
