# MINDSTROKE — reproducible figures for a 4-arm RCT

Code that generates Figures 1–6 of the MINDSTROKE trial (Meaningful Sketching, Mindful Calligraphy, Music Practice, Casual Gaming; perceived stress and salivary cortisol in healthy university students with elevated stress) **directly from the study data**, so that every number on every figure comes from one source and is cross-checked automatically.

| Figure | Content | Script | Inputs |
|---|---|---|---|
| 1 | CONSORT flow diagram (full and `--strict` CONSORT-only version) | `scripts/consort_flow.py` ← `scripts/make_counts.py` | `participants_wide.csv`, `enrolment.json`, cortisol indices from Fig 4 |
| 2 | Study design and measurement schedule | `scripts/fig2_design.py` | `PARAMS` block in the script |
| 3 | Primary outcome: PSS-10 trajectories; pre-specified contrasts, ITT with multiple imputation (primary) and complete-case (sensitivity) | `scripts/fig3_primary.py` | `data_pss.csv` |
| 4 | Salivary cortisol: diurnal profiles, slope, CAR, AUC_G, acute response | `scripts/fig4_cortisol.py` | `cortisol_diurnal.csv`, `cortisol_acute.csv` |
| 5 | Mechanisms: parallel mediation, per-session mastery/enjoyment, stroke kinematics | `scripts/fig5_mechanisms.py` | `participants_wide.csv`, `mechanisms.csv`, `sessions.csv` |
| 6 | Within-session affect, adherence, usability | `scripts/fig6_acute_adherence.py` | `participants_wide.csv`, `sessions.csv`, `acceptability.csv` |

All data shipped in `data/UHSVNU/` are upload url and exist only to exercise the code. The figures in `figures/` were produced from them.

## Quick start

```bash
pip install -r requirements.txt
./run_all.sh                      # example data → outputs/
./run_all.sh data/real outputs    # your data
```

`run_all.sh` runs the scripts in dependency order (Fig 4 before Fig 1) and ends with a cross-check: participants assessed at Week 8 must agree between Fig 1 and Fig 3, and adherence between Fig 1 and Fig 6; it exits non-zero on any mismatch.

Google Colab: open `notebooks/MINDSTROKE_figures_colab.ipynb`, upload the repository zip when prompted, Runtime → Run all.

## Data files

One row = one unit noted below; missing values are left **blank** (never 0, 99 or NA). Arm codes `MS | MC | MP | CG`; sample codes `S1–S5`; cortisol in nmol/L (µg/dL × 27.59). Header-only templates are in `data/templates/`.

| File | One row = | Columns |
|---|---|---|
| `participants_wide.csv` | randomised participant | `id, arm, site, sex, age, pss_w0, pss_w4, pss_w8, sessions_completed, started (1/0), not_started_reason, w8_status (assessed / lost / discontinued_assessed / discontinued_lost), w8_reason` |
| `enrolment.json` | — | pre-randomisation counts: screened, exclusions by reason, baseline entered, not randomised by reason, recruitment period, PSS cut-off, sites |
| `data_pss.csv` | participant × visit | `id, arm, week (0/4/8), pss, sex, site` |
| `cortisol_diurnal.csv` | saliva sample | `id, arm, week (0/8), day (1/2), sample, wake_time (HH:MM), time_since_wake_h, cortisol_nmol_l` |
| `cortisol_acute.csv` | sampled session | `id, arm, week (1/4/8), session_start, cortisol_pre, cortisol_post` |
| `sessions.csv` | participant × session | `id, session (1–19), stai_pre, stai_post, pa_pre, pa_post, na_pre, na_post, mastery, enjoyment, rhythm_index (MS/MC only)` |
| `mechanisms.csv` | participant × visit | `id, week (0/4/8), tms, mastery, enjoyment` |
| `acceptability.csv` | participant | `id, sus, continue_intent` |

### Rules implemented in the code

* **CONSORT boxes** per arm are derived, never typed: allocated − lost − (discontinued and not assessed) = assessed at W8; participants who did not start are classified in the W8 box by their follow-up status (shown on the "→ at W8" line). `make_counts.py` and `consort_flow.py` warn on any inconsistency.
* **Primary analysis**: ANCOVA on W8 PSS-10 (arm + baseline PSS-10 + sex + site), ITT with MICE (m = 50) pooled by Rubin's rules; complete-case and per-protocol (≥ 80 % sessions and W8 outcome) as sensitivity. Contrasts: C1 (MS+MC) − (MP+CG) primary; C2 MS − MC and C3 MP − CG with Holm.
* **Cortisol**: a sampling day is valid if S1, S2 (20–45 min after waking) and S5 are present; CAR only if S2 within 25–40 min; slope = regression of log-cortisol on hours since awakening using S1, S3, S4, S5; AUC_G trapezoidal S1–S5; indices averaged over valid days; W8 analysed by ANCOVA adjusted for W0, waking time, sex and site.
* **Session data**: mixed models Δ ~ arm + session + (1 | id).
* **Mediation**: parallel, X = C1 (±0.5) with C2, C3, baseline PSS, sex, site as covariates; mediators at W4 adjusted for W0; percentile bootstrap CIs.

Each script prints its statistics and writes `*_stats.csv` / `*_summary.csv` next to the figure (SVG, PNG 300 dpi, PDF).

## Repository layout

```
scripts/      figure and counting scripts
data/UHSVNU/ Lab data (seeded); data/templates/ header-only CSVs; put real data in data/real/ (git-ignored)
figures/      figures rendered from the example data
notebooks/    Colab notebook
docs/         statistics teaching exercise built on the example data (Vietnamese)
run_all.sh    end-to-end build with cross-check
```

## Citation

If you use this code, cite the MINDSTROKE protocol [reference to be added] and this repository.
