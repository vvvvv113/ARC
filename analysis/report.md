# Analysis Report: Human vs CodeIt on ARC Evaluation Tasks

**Date:** 2026-04-23  
**Data sources:**  
- `human_data/data/summary_data.csv` — 15,736 attempt-level rows (800 tasks, 422+ participants)  
- `human_data/data/data.csv` — 586,266 action-level rows with intermediate grid states  
- `codelt/data/solutions_100.json` — CodeIt solutions after 100 meta-iterations  
- Processed outputs in `analysis/processed/`

**Scripts:**  
- `analysis/01_human_vs_ai_difficulty.py` → `processed/task_difficulty.csv`  
- `analysis/02_solving_effort_correlation.py` → `processed/solving_effort.csv`  
- `analysis/03_task_overlap.py` → `processed/task_overlap.csv`  
- `analysis/04_human_grid_traces.py` → `processed/human_traces.json`  
- `analysis/05_codeit_grid_traces.py` → `processed/codeit_traces.json`  
- `analysis/06_progress_curves.py` → `processed/progress_curves.json` + `processed/curves/*.png`  
- `analysis/07_curve_metrics.py` → `processed/curve_metrics.csv` + scatter plots

---

## Q1: Human Solve Rate vs CodeIt — Which Tasks Are Hard for Whom?

### Method
- **Human solve rate per task**: fraction of participants who solved the task in at least one attempt.
- **CodeIt difficulty proxy**: the earliest iteration (out of 99) at which any solution program was found, extracted from the `new_task_key` field (e.g., `"00576224_it81__21"` → iteration 81). A lower iteration means CodeIt found a solution earlier = easier for CodeIt.
- Tasks are classified into four quadrants using the **median** as the threshold (median human solve rate = **66.7%**, median CodeIt first iteration = **31**).

### Results

| Category | Count (all 59) | Count (solving_effort, 58 tasks with human solvers) | Description |
|---|---|---|---|
| **Easy for both** | 18 | 18 | Humans solve well (≥67%) AND CodeIt finds solution early (≤iter 31) |
| **Hard for both** | 16 | 16 | Humans struggle (<67%) AND CodeIt needs many iterations (>iter 31) |
| **Only hard for AI** | 13 | 13 | Humans solve well but CodeIt takes many iterations |
| **Only hard for humans** | 12 | 11 | CodeIt finds solution early but humans rarely solve it |

> Note: `solving_effort.csv` has 58 rows (excludes `31d5ba1a`, the one task with zero human solvers), so "Only hard for humans" drops from 12 to 11 in that file.

### Notable Examples

**Easy for both** (e.g., `3194b014`, `19bb5feb`, `070dd51e`):  
These tasks have high human solve rates (75–100%) and CodeIt found solutions by iteration 8–24. They likely involve simpler spatial transformations.

**Hard for both** (e.g., `0934a4d8` — 11% human rate, iter 67; `0c9aba6e` — 11% human rate, iter 35):  
These are genuinely difficult tasks; even among the 59 that CodeIt eventually solves, a subset took both humans and the model significant effort.

**Only hard for AI** (e.g., `62ab2642` — 83% human rate, but CodeIt only found it at iter 73):  
Humans perceive the pattern intuitively, but it is hard to express in the DSL or requires lucky program sampling.

**Only hard for humans** (e.g., `31d5ba1a` — **0% human solve rate**, but CodeIt found it at iter 11; `60c09cac` — 56% human rate, iter 6):  
CodeIt's DSL-based enumeration can find solutions that humans consistently fail at, suggesting some tasks require systematic transformation sequences that humans find non-intuitive.

---

## Q2: Human Effort vs CodeIt Iteration — Is There a Correlation?

### Method
- **Human effort metrics per task** (only among participants who eventually solved the task):
  - `avg_actions_to_solve`: average total number of edit actions taken across all attempts up to and including the first successful attempt.
  - `avg_attempts_to_solve`: average number of attempts before first solve.
  - `overall_avg_actions`: average actions per attempt across all participants (including those who never solved it).
- Spearman rank correlation with `codeit_first_iter` (n = 58 tasks with both human and CodeIt data).

### Results

| Human metric | Spearman ρ | p-value | Significant? |
|---|---|---|---|
| `avg_actions_to_solve` | **+0.444** | 0.0005 | ✓ Yes |
| `avg_attempts_to_solve` | +0.098 | 0.4660 | ✗ No |
| `overall_avg_actions` | **+0.412** | 0.0013 | ✓ Yes |

**Average effort across tasks:**  
- Humans spend on average **66 actions** and **1.4 attempts** to solve a task (among solvers).  
- CodeIt's average first-solution iteration is **35.6** (range: 4–95).

### Interpretation

There is a **moderate, statistically significant positive correlation** (ρ ≈ 0.44) between how many actions humans spend on a task and how late in training CodeIt first solves it. Tasks that are action-intensive for humans tend to require more CodeIt iterations — suggesting that both agents are tracking an underlying task difficulty signal.

However, the **number of attempts** does not correlate with CodeIt iterations (ρ = 0.10, p = 0.47). This makes sense: humans often switch strategies (new attempts) on tasks they find conceptually puzzling, which is a qualitatively different dimension of difficulty than raw action count.

**In short**: raw action count is a shared difficulty signal between humans and CodeIt, but strategic re-attempts by humans do not map onto CodeIt's search depth.

---

## Q3: Are the Tasks Humans Solve and CodeIt Solves the Same Set?

### Method
- Used the full human summary data to determine, for each of the 400 ARC evaluation tasks, whether **any participant** solved it.
- Compared against the 59 CodeIt test-split solutions.
- Fisher's exact test to check if there is a statistically significant association between the two sets.

### Results

| Category | Count |
|---|---|
| **Both solved** (human ✓, CodeIt ✓) | 58 |
| **Only humans solved** (human ✓, CodeIt ✗) | 337 |
| **Only CodeIt solved** (human ✗, CodeIt ✓) | **1** (`31d5ba1a`) |
| **Neither solved** | 4 |

**Fisher's exact test**: odds ratio = 0.69, p = 0.55 → **not significant**.

### Interpretation

Humans are remarkably capable on the ARC evaluation set: **395 out of 400 tasks** were solved by at least one participant. This creates a ceiling effect — nearly everything humans can solve and CodeIt solves is already in the human-solved pool, making statistical association impossible to detect.

The more meaningful finding is the **asymmetry**:

- **58 of 59 CodeIt-solved tasks were also solved by humans** — CodeIt is largely working within the human-solvable space.  
- **1 task (`31d5ba1a`) was solved by CodeIt but by zero humans** — this is a unique case where systematic DSL enumeration succeeds where human intuition fails entirely (also noted in Q1 as "only hard for humans" with 0% human solve rate).  
- **337 tasks were solved by humans but not CodeIt** — these represent the large gap remaining for CodeIt: tasks where human spatial reasoning succeeds but DSL program synthesis does not.

**In short**: the two solving sets are not fundamentally different populations — CodeIt mostly solves a subset of what humans can also solve, with one notable exception. The bottleneck for CodeIt is not targeting the wrong tasks; it is failing to synthesize correct programs for the 337 tasks humans handle easily.

---

## Q4: Progress Curves — Convergence Speed (AUC)

### Method

For each of the 59 tasks, intermediate grid states were captured during solving:
- **Human traces** (script 04): each participant's last attempt; `test_output_grid` per action, deduplicated consecutive frames. 603 trajectories total (367 success, 236 failed).
- **CodeIt traces** (script 05): DSL programs executed line-by-line; each line whose output is a valid grid is recorded. 29,278 programs traced across all three splits.

**Progress** at each step = fraction of cells matching the target output grid.

**Normalisation** (per trajectory):
`norm(t) = (progress(t) − progress(0)) / (1 − progress(0))`
Each trajectory is resampled to 100 evenly-spaced points, then shifted and scaled so it starts at 0. This removes the baseline advantage from tasks where the input grid already resembles the target. Negative values are meaningful: they indicate the solver moved *further from* the target than where they started (e.g., grid reset).

**Aggregation**: element-wise **median** across all normalised trajectories within a group per task. The median is used instead of the mean because some trajectories have high starting progress (denominator near 0), causing extreme amplification under individual normalisation; the median is robust to these outliers.

Four median curves per task (any may be absent):
1. Human success — participants whose last attempt solved the task
2. Human failed — participants who never solved the task
3. CodeIt success — programs with `test_performance = True`
4. CodeIt failed — programs with `test_performance = False`

**AUC** = area under the median normalised curve (trapezoidal integration): higher = faster convergence.

### Results

**Spearman ρ (human_success AUC ~ codeit_success AUC, n = 58 tasks):** ρ = **+0.069**, p = 0.60 → **no significant correlation**

| Difficulty category | Human success AUC | CodeIt success AUC | Human failed AUC | CodeIt failed AUC |
|---|---|---|---|---|
| Easy for both | 0.539 | 0.413 | −0.022 | 0.147 |
| Hard for both | 0.495 | 0.311 | −0.023 | 0.316 |
| Only hard for AI | 0.597 | 0.139 | −0.051 | 0.200 |
| Only hard for humans | 0.550 | 0.330 | +0.002 | 0.160 |

### Interpretation

**Human success AUC is roughly stable (~0.50–0.60) across difficulty categories.** Once a human commits to a successful attempt, their convergence speed is largely independent of task difficulty.

**CodeIt success AUC drops sharply with difficulty**: from 0.413 ("Easy for both") to 0.139 ("Only hard for AI"). Even when CodeIt eventually synthesises a correct program, the program navigates the grid state space less efficiently on harder tasks — DSL lines produce less incremental progress per step.

**Human failed AUC is negative (−0.02 to −0.05)**: failed participants on average end up further from the target than where they started — driven by grid resets and wrong-direction edits. This is qualitatively different from CodeIt failed programs, which show positive AUC (0.15–0.32): failed programs still make partial progress even without reaching the answer.

**No correlation between human and CodeIt convergence speed** (ρ = +0.069, p = 0.60): how efficiently a human solves a task is unrelated to how efficiently CodeIt's program reaches the answer on the same task. This is consistent with the two agents using fundamentally different search strategies.

---

## Q5: Pairwise Curve Similarity — L2 Distance and Pearson r

### Method

For each task, the four median normalised curves are compared in all four within-agent and between-agent pairs:

| Pair | Question |
|---|---|
| human_success vs codeit_success | Do agents take similar paths when both succeed? |
| human_failed vs codeit_failed | Do agents fail in the same way? |
| human_success vs human_failed | How different is human success from human failure? |
| codeit_success vs codeit_failed | How different is CodeIt success from CodeIt failure? |

Two metrics per pair per task:
- **L2 distance** = `sqrt(sum_t (curve_A(t) − curve_B(t))^2)` — absolute path dissimilarity over all 100 time steps. Larger = curves are further apart in progress space.
- **Pearson r** — correlation between the two 100-point curves. Measures trend similarity: r ≈ 1 = same learning-rate profile; r < 0 = one rises while the other falls.

### Results

Median L2 distance and Pearson r across tasks by difficulty category:

**human_success vs codeit_success**

| Difficulty category | Median L2 | Median Pearson r |
|---|---|---|
| Easy for both | 2.87 | +0.878 |
| Hard for both | 3.95 | +0.657 |
| Only hard for AI | 4.14 | +0.687 |
| Only hard for humans | 2.25 | +0.894 |

**human_failed vs codeit_failed**

| Difficulty category | Median L2 | Median Pearson r |
|---|---|---|
| Easy for both | 2.28 | −0.587 |
| Hard for both | 4.31 | −0.522 |
| Only hard for AI | 3.79 | +0.324 |
| Only hard for humans | 2.31 | +0.283 |

**human_success vs human_failed**

| Difficulty category | Median L2 | Median Pearson r |
|---|---|---|
| Easy for both | 5.79 | −0.404 |
| Hard for both | 6.07 | −0.627 |
| Only hard for AI | 7.66 | −0.008 |
| Only hard for humans | 6.52 | +0.057 |

**codeit_success vs codeit_failed**

| Difficulty category | Median L2 | Median Pearson r |
|---|---|---|
| Easy for both | 3.91 | +0.940 |
| Hard for both | 1.69 | +0.905 |
| Only hard for AI | 2.10 | +0.972 |
| Only hard for humans | 2.78 | +0.943 |

### Interpretation

**Human success and CodeIt success share similar trajectory shapes** (Pearson r ≈ +0.66–0.89 across difficulty categories), despite the absolute path difference (L2 ≈ 2–4). Both agents generally move monotonically toward the target when they succeed — the similarity is highest on tasks that are "easy for both" or "only hard for humans."

**Human failure and CodeIt failure are structurally opposite on easy and hard-for-both tasks** (Pearson r ≈ −0.52 to −0.59): human failed participants tend to regress (negative progress over time) while CodeIt failed programs still make partial upward progress. This reversal disappears on "Only hard for AI" tasks (r ≈ +0.32), where failed CodeIt programs may also stagnate near zero — more similar to human failure.

**Human success vs human failure shows the largest L2 distance of any pair** (median 5.8–7.7), with strongly negative Pearson r on easy and hard-for-both tasks (r ≈ −0.4 to −0.6). The two human groups take qualitatively opposite paths: solvers climb toward the target while non-solvers drift away. The correlation is near zero on "Only hard for AI" tasks (r ≈ 0.0), where even successful humans take irregular non-monotone paths.

**CodeIt success vs CodeIt failure have very similar curve shapes** (Pearson r ≈ +0.90–0.97) despite measurable L2 distance. Both groups make progress in the same direction — the difference is in *how far* they get, not *how* they move. This reflects the deterministic, program-based nature of CodeIt: a failed program often executes most of the correct transformations but makes an error at one step.

---

## Summary

| Question | Key Finding |
|---|---|
| **Q1** | Tasks split roughly evenly across 4 difficulty quadrants. 12 tasks are uniquely hard for humans (CodeIt solves them early); 13 are uniquely hard for AI (humans solve intuitively but DSL search takes long). |
| **Q2** | Significant positive correlation between human action count and CodeIt iteration (ρ=0.44, p<0.001). Number of attempts is not correlated. Both agents share a common difficulty signal at the action/search level. |
| **Q3** | Almost complete overlap: CodeIt solves 58/59 tasks that humans also solve. 337 human-solved tasks remain unsolved by CodeIt. Only 1 task (`31d5ba1a`) is uniquely solved by CodeIt. |
| **Q4** | No correlation between human and CodeIt convergence speed (ρ=+0.069, p=0.60). Human success AUC is stable across difficulty (~0.50–0.60); CodeIt success AUC drops sharply for harder tasks (0.41 → 0.14). Human failed solvers go backward (negative AUC); CodeIt failed programs still partially converge. |
| **Q5** | Human and CodeIt success curves share similar trend shapes (Pearson r ≈ 0.66–0.89) but differ in absolute path. Human failure and CodeIt failure are trend-opposite on easy tasks (r ≈ −0.55): humans regress while CodeIt programs partially progress. CodeIt success vs failure are nearly identical in shape (r ≈ 0.94) — they differ in endpoint, not strategy. |
