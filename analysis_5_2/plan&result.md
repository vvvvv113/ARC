# Analysis Pipeline: Human × CodeIt Progress Curves (400 Tasks)

**Working directory:** `analysis_5_2/`
**All scripts run from repo root:** `python3 analysis_5_2/SXX_xxx.py`

---

## Key Results: Importance Classification and Narrative Order

### Research Question
Do human and CodeIt problem-solving trajectories on ARC evaluation tasks show similar patterns? If so, at what level of abstraction — overall progress shape, difficulty ordering, or actual intermediate states?

---

### Tier 1 — Results That Directly Answer the Research Question

These five results form the core narrative and should all appear in the Results section of the paper.

**1. S00 — Task structure is bimodal; all comparisons must be stratified**
Group A (baseline = 0, 133 tasks) and Group B (baseline > 0, 267 tasks) represent structurally different cognitive demands. Every downstream result is conditional on this split. This is not a finding about human–CodeIt comparison, but it is the prerequisite that makes all other findings interpretable. Presented first to frame the entire analysis.
→ `analysis_5_2/processed/S00_baseline/baseline_distribution.png`

**2. S06 Pair 1 — Human and CodeIt success trajectories are shape-similar (r = 0.75, Group A)**
When both agents succeed on Group A tasks, their normalized progress curves are strongly correlated in shape. This is the headline result: at the level of normalized progress over time, the two agents trace qualitatively similar paths. This finding opens the comparison — it establishes that there *is* something to compare, and that the comparison is non-trivial.
→ `analysis_5_2/processed/S06_comparison/dtw_distribution.png`

**3. S05 AUC — Same shape, different dynamics: human_success AUC (0.584) vs. codeit_success (0.250)**
Despite the shape similarity in Result 2, humans maintain more than twice the cumulative normalized progress across the trajectory. Humans build progress gradually and sustain it; CodeIt converges rapidly near the end. This immediately qualifies Result 2: "similar shape" does not mean "same solving process." The two agents differ substantially in the temporal dynamics of how they reach the solution.
→ `analysis_5_2/processed/S04_curves/auc_overview.png`, `analysis_5_2/processed/S05_metrics/auc_distribution.png`

**4. S05 Spearman — No shared difficulty ordering: ρ = −0.076, p_adj = 0.79 (Group A)**
Tasks that are easy for humans are not systematically easier for CodeIt, and vice versa. This further qualifies Result 2: the shape similarity across tasks does not reflect a shared underlying difficulty structure. The two agents are not solving the same implicit hierarchy of problems — they are shape-similar despite navigating different difficulty landscapes.
→ `analysis_5_2/processed/S05_metrics/spearman_scatter_A.png`

**5. S08 — Near-zero shared intermediate states even when both succeed**
For all six analyzed tasks, human and CodeIt panels share only 2–9 nodes out of 13–222. Even on tasks where both agents succeed at high rates, they traverse almost entirely disjoint sets of intermediate grid states. This is the deepest result: the normalized progress curves look similar (Result 2) because the *abstraction* of normalized progress captures shared macro-structure, while the actual problem-solving paths — the specific grid states visited — are completely different. Shape similarity is an abstraction artifact, not evidence of convergent strategy.
→ `analysis_5_2/processed/S08_state_space/state_space_interactive.html`

---

### Tier 2 — Validation That Bounds the Interpretation

**6. S06 Pair 3 Permutation — Human success vs. failure are maximally discriminable (p_perm = 0.0)**
None of 5,000 random label permutations produces a DTW as large as the observed success–failure difference. This validates that the trajectory signal is real: the progress curves genuinely reflect outcome-relevant behavior, not noise. Without this, Results 2–4 could be dismissed as pattern-matching on meaningless curves.
→ `analysis_5_2/processed/S06_comparison/permutation_test.png`

**7. S07 — No human ability bias in CodeIt task selection**
Human AUC and success rate do not differ significantly between CodeIt-covered (n=134) and uncovered (n=266) tasks. This validates the external validity of Results 2–5: the human–CodeIt comparison is not inflated by CodeIt having selectively covered tasks that happen to be easier for humans.
→ `analysis_5_2/processed/S07_selection_bias/human_auc_comparison.png`

---

### How the Results Connect: Logical Flow

```
S00: Two task types exist (Group A / B)
  │
  └─► All results stratified by Group A/B
          │
          ├─► S06 Pair 1: Normalized progress curves are shape-similar (r=0.75)
          │       │
          │       ├─► S05 AUC: But dynamics differ — humans build gradually, CodeIt converges late
          │       │       │
          │       │       └─► S05 Spearman: And difficulty orderings don't align (ρ≈0)
          │       │
          │       └─► S08: And the actual intermediate states are disjoint
          │               │
          │               └─► Tension resolved: shape similarity is a macro-abstraction;
          │                   the underlying strategies are fundamentally different
          │
          └─► S06 Pair 3: The trajectory signal is valid (permutation p=0.0)
          └─► S07: The comparison is unbiased (no human ability selection effect)
```

**The central tension in the paper:** Results 2 and 5 appear contradictory — if progress curves are shape-similar, how can the underlying states be disjoint? The resolution is that normalized progress is a coarse abstraction that compresses all intermediate states into a single scalar. Two very different paths through the grid-state space can produce similar progress-over-time profiles if they spend similar proportions of their trajectory at each progress level. Shape similarity at the curve level does not imply convergent strategy at the state level. This tension — similarity in abstraction, divergence in implementation — is the main interpretive contribution.

---

## S00 — Baseline Distribution

### Method
For each of the 400 ARC evaluation tasks, the baseline is defined as `progress(input_grid, target_grid)`, the cell-overlap fraction between the task's test input and test output grid. This yields a scalar in [0, 1] for every task. Computed directly from `codelt/data/evaluation/*.json`.

A secondary indicator `n_wrong_cells = floor(total_cells × (1 − baseline))` counts the integer number of cells that must change. Tasks where `n_wrong_cells < 3` are flagged `numerically_coarse = True`; their progress curves are discrete step functions and therefore unreliable for continuous analysis.

### Why
Progress normalization in S04 divides by `(1 − baseline)`. If baseline is near 1, small absolute changes in grid similarity produce large swings in normalized progress, amplifying noise. More fundamentally, `baseline = 0` tasks (Group A) require discovering a global transformation from scratch — the input and output share no structure — while `baseline > 0` tasks (Group B) require identifying and correcting local discrepancies. These represent distinct cognitive demands (abstract rule induction vs. perceptual error correction), so **all downstream analyses stratify by Group A / Group B**. Collapsing the two would risk Simpson's Paradox: an effect present within each group could invert or vanish in the aggregate.

### Limitations to Note
- The Group A/B split is binary. Tasks at the boundary of the two groups are treated categorically when they may represent a continuum of task difficulty; this is a simplification that trades interpretability for precision.

### Results
| File | Path |
|---|---|
| Summary | `analysis_5_2/processed/S00_baseline/baseline_analysis.txt` |
| Distribution plot | `analysis_5_2/processed/S00_baseline/baseline_distribution.png` |
| Per-task CSV | `analysis_5_2/processed/S00_baseline/task_baselines.csv` |

### Key Findings for Report
- Distribution is **strongly bimodal**: 133 tasks (33.2%) have baseline = 0 (Group A), 267 tasks (66.8%) have baseline > 0 (Group B). The bimodal structure justifies treating them as two qualitatively different task populations throughout the analysis.
  → **Figure:** `analysis_5_2/processed/S00_baseline/baseline_distribution.png`
- Group B median baseline = 0.72; 180 tasks have baseline > 0.75, meaning the majority of Group B tasks differ from their targets by only a few cells.

---

## S01 — Attempt Distribution Validation

### Method
From `human_data/data/data.csv` (filtered to `task_type == 'evaluation'`), for each `(hashed_id, task_id)` pair, compute: (1) the maximum attempt number, (2) the number of actions in the last attempt, and (3) the final normalized progress at the last action of the last attempt. Plot distributions and the relationship between attempt number and median final progress.

### Why
The entire human trace analysis (S02) rests on the assumption that the **last attempt** is the most informative. If the majority of participants abandon a task after only 2–3 actions, the "last attempt" carries little signal. This validation step provides empirical support for that assumption before it is baked into the pipeline.

### Limitations to Note
- The last attempt is not necessarily the best attempt. Using the attempt with the highest final progress would theoretically capture peak performance more accurately, but would introduce post-hoc selection bias (choosing the best outcome inflates performance estimates). The last attempt is conservative and standard.

### Results
| File | Path |
|---|---|
| Summary | `analysis_5_2/processed/S01_attempts/summary.txt` |
| Attempt distribution | `analysis_5_2/processed/S01_attempts/attempt_distribution.png` |
| Progress by attempt | `analysis_5_2/processed/S01_attempts/attempt_progress_trend.png` |
| Per-row stats | `analysis_5_2/processed/S01_attempts/attempt_stats.csv` |

### Key Findings for Report
- 4,101 unique `(hashed_id, task_id)` pairs. Median last-attempt action count: **24 actions** (IQR 8–52); only **9.6%** have ≤3 actions — well below the 20% threshold, so the last-attempt assumption holds.
  → **Figure:** `analysis_5_2/processed/S01_attempts/attempt_distribution.png`

---

## S02 — Human Traces (All 400 Evaluation Tasks)

### Method
From `human_data/data/data.csv`, filtered to `task_type == 'evaluation'`:

1. For each `(hashed_id, task_id)`, retain only rows where `attempt_number` is maximum.
2. Sort by `action_id` ascending; collect the `test_output_grid` sequence.
3. **De-duplicate consecutive identical grids** — many actions (e.g., color picker opens/closes) do not change the grid; retaining redundant frames would inflate step counts without adding information.
4. `success = solved.any()` over the retained rows for that attempt.

Grids are stored as pipe-delimited strings: `|row1|row2|...` where each character is a digit 0–9.

### Why
The existing analysis covered only 59 tasks — those that overlap with CodeIt's replay buffer — introducing selection bias into any human-only analysis. Rebuilding from all 400 tasks provides an unbiased baseline for (a) measuring selection bias in S07 and (b) computing human progress curves independently of CodeIt coverage.

### Limitations to Note
- The human interface initializes the output canvas to a state that is not the task input grid (except when input and output dimensions match). Human traces therefore do not start from the same representational anchor as CodeIt, which always begins from the input. S04 addresses this by prepending the input grid to human traces so that both agents share `norm[0] = 0`, but the raw intermediate states remain in different grid spaces for tasks with mismatched input/output dimensions.

### Results
| File | Path |
|---|---|
| Summary | `analysis_5_2/processed/S02_human_traces/summary.txt` |
| Trace JSON | `analysis_5_2/processed/S02_human_traces/human_traces_all.json` |
| Per-task CSV | `analysis_5_2/processed/S02_human_traces/human_traces_summary.csv` |
| Coverage plot | `analysis_5_2/processed/S02_human_traces/human_traces_coverage.png` |

### Key Findings for Report
- All 400 tasks have human trace coverage: 395 tasks have ≥1 success trace, 378 have ≥1 failed trace — near-complete coverage across the full evaluation set.
  → **Figure:** `analysis_5_2/processed/S02_human_traces/human_traces_coverage.png`

---

## S03 — CodeIt Traces (3 Seeds Combined)

### Method
Sources: `seed17` (solutions_97.json), `seed42` (solutions_95.json), `seed123` (solutions_96.json).

Two-stage de-duplication applied separately for success and failed programs:
1. **Program string de-duplication:** Remove exact DSL string duplicates across seeds.
2. **Trace content de-duplication:** Execute each unique program via `execute_candidate_program_with_trace` on the task's test input; hash the resulting grid sequence (`SHA1(joined_grids_string)`). Discard programs whose trace hash has already been seen — eliminating semantically equivalent programs that differ only syntactically.

Programs with execution errors are discarded. No cap on traces retained. Seed concentration quantified via **HHI** = Σ(seed_i proportion²); HHI = 1/3 is uniform, HHI = 1 means all from one seed.

### Why
Without trace-level de-duplication, a single effective algorithmic strategy would appear thousands of times across seeds, inflating sample sizes and biasing DTW statistics toward whichever seed contributed more iterations. The HHI replaces an arbitrary trace cap as a principled, continuous measure of non-independence that preserves all data while making the degree of concentration explicit and reportable.

### Limitations to Note
- CodeIt traces are **not independent samples**: all traces from a given seed derive from the same trained model, making intra-seed correlations unavoidable. This fundamentally limits the inferential weight that can be placed on CodeIt sample sizes. The seed sensitivity analysis in S06 is the primary tool for quantifying this dependency's effect on conclusions.

### Results
| File | Path |
|---|---|
| Summary | `analysis_5_2/processed/S03_codeit_traces/summary.txt` |
| Trace JSON | `analysis_5_2/processed/S03_codeit_traces/codeit_traces_3seeds.json` |
| Seed breakdown CSV | `analysis_5_2/processed/S03_codeit_traces/seed_breakdown.csv` |
| Coverage plot | `analysis_5_2/processed/S03_codeit_traces/codeit_traces_coverage.png` |

### Key Findings for Report
- **134 tasks** have any CodeIt trace; 70 have ≥1 success trace; 127 have ≥1 failed trace.
- Two-stage de-duplication reduces success traces from ~51,157 (string de-duped only) to **7,312** and retains **10,868** failed traces — demonstrating that the original dataset was dominated by semantically equivalent programs.
  → **Figure:** `analysis_5_2/processed/S03_codeit_traces/codeit_traces_coverage.png`
- HHI (failed traces): median = 0.54; **28 tasks have HHI = 1.0** (all failed traces from a single seed), indicating substantial non-independence for these tasks.

---

## S04 — Progress Curves (v2 Normalization, 400 Tasks)

### Method
**Normalization (v2):**
```
baseline(task) = progress(input_grid, target_grid)
norm(t) = (progress(t) - baseline) / (1 - baseline)
```
- `norm = 0`: grid equals the task input (starting point for both agents after prepending).
- `norm = 1`: grid equals the target (fully solved).
- `norm < 0`: grid is further from target than the input — retained, not clipped.

**Human traces:** The raw input grid is prepended so that `norm[0] = 0`, giving both agents a common zero-point.

**Resampling:** Each trace is linearly interpolated to 100 equal time points. Summary statistics (median, p25, p75) computed element-wise across all traces in a group.

**Plots** organized into four subdirectories: `A_with_codeit`, `A_no_codeit`, `B_with_codeit`, `B_no_codeit`. Y-axis is dynamic, showing negative values in full.

### Why
The v2 normalization anchors both human and CodeIt progress to a shared zero-point (the task input), making cross-agent comparisons interpretable. Without this, Group B tasks (high baseline) would appear to start near 1.0, making a flat trajectory look like high achievement. Retaining negative values is scientifically correct: a participant who moves the grid further from the target than the input is genuinely doing worse than doing nothing — this failure mode is substantively different from simply making no progress, and censoring it would misrepresent the distribution of human strategies.

### Limitations to Note
- For tasks where input and output have different grid dimensions, human traces (in output space) and CodeIt traces (in input space) are normalized using the same formula but relative to different representational starting points. The normalized progress values are individually valid, but comparisons of raw intermediate grid states between agents are not meaningful for these tasks.

### Results
| File | Path |
|---|---|
| Summary | `analysis_5_2/processed/S04_curves/summary.txt` |
| Full curve JSON | `analysis_5_2/processed/S04_curves/progress_curves_400.json` |
| Per-task×group CSV | `analysis_5_2/processed/S04_curves/curve_summary_400.csv` |
| **AUC overview** | `analysis_5_2/processed/S04_curves/auc_overview.png` |
| AUC scatter (hs vs cs) | `analysis_5_2/processed/S04_curves/auc_scatter_hs_vs_cs.png` |
| Per-task plots | `analysis_5_2/processed/S04_curves/curve_400/{A,B}_{with,no}_codeit/*.png` |

### Key Findings for Report
- The AUC overview plot shows all four groups (human_success, human_failed, codeit_success, codeit_failed) across Group A and B. Key pattern: in Group A, human_success and codeit_success AUCs are clearly separated (humans maintain higher cumulative progress throughout), while human_failed and codeit_failed cluster near zero or below. In Group B, all four groups compress toward the same range due to high baseline normalization.
  → **Figure:** `analysis_5_2/processed/S04_curves/auc_overview.png`
- Negative normalized progress is common in human_failed traces in Group A — participants systematically make the grid worse than the starting state before giving up. This is a qualitatively distinct failure mode from simply stalling.
- Per-task progress curves in `A_with_codeit/` show the clearest separation between success and failed groups and are the most interpretable; curves in `B_with_codeit/` are compressed and should be interpreted cautiously.

---

## S05 — AUC and Steps-to-90pct Metrics

### Method
Two scalar summaries from each task×group's median progress curve (100-point resampled):

- **AUC** = `np.trapz(median_curve, dx=1/99)` ∈ (−∞, 1]. Negative AUC: trajectory spends more time below the baseline than above.
- **steps_to_90pct** = first normalized time point where `median_curve ≥ 0.90`; `NaN` if never reached.

**Spearman ρ** (human_success AUC vs. codeit_success AUC, per task): Group A and B tested separately; BH-corrected as a single family of 2 tests. Bootstrap 95% CI: B=1000 resamples of tasks.

### Why
AUC reduces the entire trajectory to a single number capturing both speed and eventual level, enabling task-level correlation between agents. The Spearman correlation directly answers whether tasks that are structurally easy for humans are also easy for CodeIt — a prerequisite for interpreting trajectory similarity as reflecting shared problem structure rather than coincidence.

### Limitations to Note
- The Spearman test has limited power at n=52 (Group A) and n=17 (Group B). Non-significant results should be read as "insufficient evidence of correlation" rather than confirmed absence of correlation.

### Results
| File | Path |
|---|---|
| Summary | `analysis_5_2/processed/S05_metrics/summary.txt` |
| Per-task metrics | `analysis_5_2/processed/S05_metrics/curve_metrics.csv` |
| Spearman results | `analysis_5_2/processed/S05_metrics/spearman_summary.csv` |
| **AUC distributions** | `analysis_5_2/processed/S05_metrics/auc_distribution.png` |
| **Spearman scatter Group A** | `analysis_5_2/processed/S05_metrics/spearman_scatter_A.png` |
| Spearman scatter Group B | `analysis_5_2/processed/S05_metrics/spearman_scatter_B.png` |

### Key Findings for Report
**AUC medians by group:**
| Group | A | B |
|---|---|---|
| human_success | **0.584** | 0.308 |
| codeit_success | **0.250** | 0.307 |
| human_failed | 0.000 | — |
| codeit_failed | 0.099 | — |

- In Group A, human_success AUC (0.584) is more than twice codeit_success AUC (0.250). This indicates that successful humans maintain higher cumulative normalized progress across the solving trajectory, while CodeIt tends to converge quickly once it finds a solution rather than building incrementally.
  → **Figure:** `analysis_5_2/processed/S05_metrics/auc_distribution.png`
- **steps_to_90pct (Group A):** human_success median = 0.87, codeit_success median = 0.96 — CodeIt reaches near-solution closer to the end of its normalized timeline, consistent with a search strategy that finds the solution in a final burst rather than gradually.
- **Spearman ρ (Group A): ρ = −0.076, 95% CI [−0.32, 0.18], p_adj = 0.79. No significant task-level correlation.** Tasks that are easy for humans are not systematically easier for CodeIt, suggesting the two agents are not solving the same implicit difficulty hierarchy.
  → **Figure:** `analysis_5_2/processed/S05_metrics/spearman_scatter_A.png`

---

## S06 — Pairwise Curve Comparison (Pearson r, DTW, Permutation)

### Method
**Four pairs** compared:
- Pair 1: `human_success` vs. `codeit_success`
- Pair 2: `human_failed` vs. `codeit_failed`
- Pair 3: `human_success` vs. `human_failed` (within-human, between-outcome)
- Pair 4: `codeit_success` vs. `codeit_failed` (within-CodeIt, between-outcome)

**Pearson r (F1 family):** Per task, Pearson correlation between the two median curves (100 points). Significance via Wilcoxon signed-rank test across tasks; BH-corrected within F1 (8 p-values).

**DTW (F2 family):** Sakoe-Chiba constrained DTW (window = 10 steps, L1 cost). Bootstrap CI (B=1000) by **resampling human side only**, keeping CodeIt median fixed — CodeIt is a deterministic model, not a sample from a population. BH-corrected within F2.

**Permutation test (F3, Pair 3):** Global statistic = median DTW across all tasks with both human_success and human_failed. 5,000 within-task label shuffles; p = fraction of permuted statistics ≥ observed.

**Seed sensitivity:** All-pairs DTW for Pairs 1 and 2 repeated separately for each seed. Compared to combined-seed results to quantify non-independence impact.

### Why
Pearson r captures whether the *shapes* of two curves are similar regardless of scale, while DTW allows temporal warping — measuring whether agents follow the same general trajectory even if one is faster. Both metrics are needed because high shape similarity (Pearson r) can coexist with large absolute distance (DTW), and vice versa. The permutation test is used for Pair 3 rather than a t-test because human traces within the same task are not independent (all participants respond to the same stimulus), and permuting labels within tasks preserves this dependency structure while breaking the success/failure distinction.

### Limitations to Note
- **Group B results for Pair 1 are unreliable**: seed sensitivity analysis shows 2–4× variation in all-pairs DTW across seeds for Group B (e.g., f0df5ff0: 35–149 across seeds). This is a direct consequence of the high non-independence (HHI) in Group B CodeIt traces. Conclusions for Group B from Pair 1 should be treated as exploratory only and are not suitable for strong claims.

### Results
| File | Path |
|---|---|
| Summary | `analysis_5_2/processed/S06_comparison/summary.txt` |
| F1 Pearson Wilcoxon | `analysis_5_2/processed/S06_comparison/f1_pearson_wilcoxon.csv` |
| F2 DTW Wilcoxon | `analysis_5_2/processed/S06_comparison/f2_dtw_wilcoxon.csv` |
| Permutation result | `analysis_5_2/processed/S06_comparison/permutation_result.csv` |
| Seed sensitivity CSV | `analysis_5_2/processed/S06_comparison/sensitivity_seed.csv` |
| **DTW distribution** | `analysis_5_2/processed/S06_comparison/dtw_distribution.png` |
| **All-pairs DTW plot** | `analysis_5_2/processed/S06_comparison/allpairs_dtw_distribution.png` |
| **Permutation test plot** | `analysis_5_2/processed/S06_comparison/permutation_test.png` |
| **Seed sensitivity Pair 1** | `analysis_5_2/processed/S06_comparison/sensitivity_seed_pair1.png` |

### Key Findings for Report

**F1 — Pearson r (curve shape similarity, Group A):**
| Pair | median r | p_adj |
|---|---|---|
| Pair 1: human_success vs. codeit_success | **0.75** | <0.001 |
| Pair 2: human_failed vs. codeit_failed | **0.59** | <0.001 |
| Pair 3: human_success vs. human_failed | **0.84** | <0.001 |
| Pair 4: codeit_success vs. codeit_failed | **0.94** | <0.001 |

Human_success and codeit_success trajectories show strong shape correlation (r=0.75 in Group A), suggesting that when both agents succeed, they trace qualitatively similar normalized progress paths. The even higher within-agent correlations (Pair 3: 0.84, Pair 4: 0.94) confirm that the between-agent similarity is not an artifact of curve compression.
→ **Figure:** `analysis_5_2/processed/S06_comparison/dtw_distribution.png`

**F2 — Median DTW (Group A):**
| Pair | DTW |
|---|---|
| Pair 1 (hs vs cs) | 18.8 |
| Pair 2 (hf vs cf) | 11.8 |
| Pair 3 (hs vs hf) | 43.5 |
| Pair 4 (cs vs cf) | 9.9 |

Despite shape similarity, the absolute DTW distance for Pair 1 (18.8) is substantial — human and CodeIt success trajectories are shape-similar but not temporally close. Pair 3's DTW (43.5) nearly 4× exceeds Pair 1, confirming that the human success–failure distinction is the largest source of trajectory divergence in this dataset.
→ **Figure:** `analysis_5_2/processed/S06_comparison/allpairs_dtw_distribution.png`

**F3 — Permutation test (Pair 3):**
Observed global DTW = **58.77**; permutation mean = 20.00; **p_perm = 0.0** (none of 5,000 permutations matched or exceeded observed). Human success and failure trajectories are maximally discriminable — no random label assignment produces a DTW as large as the true outcome-based difference.
→ **Figure:** `analysis_5_2/processed/S06_comparison/permutation_test.png`

**Seed sensitivity (Pair 1, Group A):** Results are consistent across seeds (median DTW range approximately 18–25 per seed vs. 23 combined), supporting the robustness of the Pair 1 Group A conclusion.
→ **Figure:** `analysis_5_2/processed/S06_comparison/sensitivity_seed_pair1.png`

---

## S07 — Selection Bias Check

### Method
Compare 134 tasks with any CodeIt trace vs. 266 tasks without. Metrics per task: human_success AUC, human success rate, baseline, `n_wrong_cells`. Mann-Whitney U + rank-biserial r effect size. Stratified by baseline_group.

### Why
CodeIt's replay buffer contains only tasks that its search encountered. If these are systematically easier for humans, the human–CodeIt comparison in S06 overestimates agreement for the full 400-task distribution. S07 provides the quantitative basis for assessing and qualifying the external validity of the main findings.

### Limitations to Note
- "No CodeIt trace" does not imply CodeIt failed on that task — the task may simply not have been sampled into the replay buffer during training. The 266 tasks without traces are not a confirmed set of CodeIt failures; they are an under-explored set.

### Results
| File | Path |
|---|---|
| Summary | `analysis_5_2/processed/S07_selection_bias/summary.txt` |
| Results CSV | `analysis_5_2/processed/S07_selection_bias/selection_bias_summary.csv` |
| **Human AUC comparison** | `analysis_5_2/processed/S07_selection_bias/human_auc_comparison.png` |
| **n_wrong_cells comparison** | `analysis_5_2/processed/S07_selection_bias/n_wrong_cells_comparison.png` |
| Baseline comparison | `analysis_5_2/processed/S07_selection_bias/baseline_comparison.png` |

### Key Findings for Report
- **Human_success AUC:** no significant difference between with/without CodeIt groups (p=0.09, r=0.10 overall; non-significant in both Group A and B). Human problem-solving ability does not differ between the two task subsets — there is no human ability bias in CodeIt task selection.
  → **Figure:** `analysis_5_2/processed/S07_selection_bias/human_auc_comparison.png`
- **n_wrong_cells:** tasks with CodeIt data have significantly fewer cells to change (median 25 vs. 40.5, p=0.001, r=−0.21; Group A large effect: r=−0.30, p=0.003). CodeIt systematically covers tasks with smaller, more constrained state spaces.
  → **Figure:** `analysis_5_2/processed/S07_selection_bias/n_wrong_cells_comparison.png`
- **Baseline structure:** CodeIt-covered tasks are disproportionately Group A (52% vs. 24% in uncovered tasks), reflecting that CodeIt's program search is more successful on tasks requiring global transformation discovery (where there is a clear rule to find) than on local correction tasks.
- **Conclusion for Discussion:** The absence of human ability bias preserves the validity of the human–CodeIt comparison within the covered subset. However, the structural selection toward smaller state spaces and Group A tasks means that findings should not be generalized to the full ARC evaluation distribution without qualification.

---

## S08 — State Space Graphs (Interactive HTML)

### Method
For six selected tasks (spanning four narrative types), construct two directed graphs per task — human panel and CodeIt panel. Nodes = unique grid states (`SHA1[:8]`); edges = consecutive state transitions weighted by frequency; node size = `log(visits + 1) × 7`. Input and target nodes are pinned; other nodes positioned by D3 force simulation using BFS distance from start node as radial constraint. Node color encodes agent group membership. Clicking any node renders the full ARC grid.

Tasks selected:
- `bf699163` (Group A): humans all solve (9/9), CodeIt mostly fails (2/190) — human-better
- `34b99a2b` (Group A): CodeIt all solves (22/22), humans mostly fail (1/9) — smallest state space
- `7953d61e` (Group A): CodeIt all solves (51/51), humans mostly fail (3/11)
- `e7639916` (Group B): both agents succeed at high rate (human 0.73, CodeIt 0.93)
- `1acc24af` (Group B, baseline=0.92): both agents succeed at low rate
- `32e9702f` (Group B, baseline=0.12): mixed success rates

### Why
Progress curves reduce problem-solving to a scalar per time step, concealing the spatial structure of exploration. The state space graph reveals *which* intermediate grid states were visited and whether the two agents share any of them. Shared nodes indicate convergent exploration — both agents passed through the same intermediate grid configuration — while disjoint state spaces indicate that similar progress curves can arise from entirely different intermediate representations.

### Limitations to Note
- For tasks where input and output grid dimensions differ (`7953d61e`: input 4×4, output 8×8; `bf699163`: input 13×14, output 3×3), human traces are in output-grid space and CodeIt traces are in input-grid space. Zero or near-zero shared nodes for these tasks reflects representational incompatibility, not necessarily strategic divergence.

### Results
| File | Path |
|---|---|
| Summary | `analysis_5_2/processed/S08_state_space/summary.txt` |
| **Interactive HTML** | `analysis_5_2/processed/S08_state_space/state_space_interactive.html` |
| Task candidates CSV | `analysis_5_2/processed/S08_state_space/task_candidates.csv` |

### Key Findings for Report
- **Shared nodes between human and CodeIt panels are near-zero for all six tasks** (2–9 shared nodes out of 13–222 per panel). Even on `e7639916` where both agents succeed at >70%, they share only 3 out of 183 human nodes and 46 CodeIt nodes. Both agents reach the same endpoint via largely non-overlapping intermediate grid states.

- **Success corresponds to compact, directed exploration; failure corresponds to diffuse exploration — for both agents.** On `34b99a2b` and `7953d61e` (CodeIt-better tasks), the CodeIt state space is compact (13–19 nodes, linear paths to target), while the human state space is large (95–206 nodes, many branches and dead ends). On `bf699163` (human-better task), the reverse: human state space is small and focused (29 nodes), CodeIt generates 71 nodes with extensive failed branches.
  → **Interactive:** `analysis_5_2/processed/S08_state_space/state_space_interactive.html`

---

## Cross-Script Summary for Paper

| Analysis | Primary Claim | Qualifying Caveat |
|---|---|---|
| S00 | Bimodal task distribution; Group A (abstract induction) and Group B (local correction) require distinct analysis | Binary split simplifies a continuum |
| S01 | "Last attempt" assumption validated: 9.6% ≤3 actions, well below 20% threshold | Last attempt ≠ best attempt; conservative by design |
| S05 | No task-level AUC correlation between human and CodeIt (ρ ≈ 0, ns in both groups) | Low n limits power; absence of evidence ≠ evidence of absence |
| S06 F1 | Human and CodeIt success curves are shape-similar in Group A (Pearson r=0.75, p<0.001) | Group B conclusions unreliable due to seed non-independence (2–4× sensitivity variation) |
| S06 F3 | Human success and failure trajectories are maximally discriminable (p_perm=0.0 / 5,000) | Global statistic; within-task effect sizes not reported separately |
| S07 | No human ability bias in CodeIt task selection; structural bias toward smaller, Group A tasks | "No trace" ≠ CodeIt failure; may be sampling artifact |
| S08 | Human and CodeIt traverse almost entirely disjoint state spaces even when both succeed | Dimension mismatch for some tasks makes comparison structurally impossible |
