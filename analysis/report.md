# Analysis Report: Human vs CodeIt on ARC Evaluation Tasks

**Date:** 2026-04-24  
**Data sources:**  
- `human_data/data/summary_data.csv` — 15,736 attempt-level rows (800 tasks, 422+ participants)  
- `human_data/data/data.csv` — 586,266 action-level rows with intermediate grid states  
- `codelt/data/solutions_100.json` — CodeIt solutions after 100 meta-iterations  
- Processed outputs in `analysis/processed/`

**Scripts:**  
- `analysis/01_human_vs_ai_difficulty.py` → `processed/01_difficulty/task_difficulty.csv`  
- `analysis/02_solving_effort_correlation.py` → `processed/02_effort/solving_effort.csv`  
- `analysis/03_task_overlap.py` → `processed/03_overlap/task_overlap.csv`  
- `analysis/04_human_grid_traces.py` → `processed/04_human_traces/human_traces.json`  
- `analysis/05_codeit_grid_traces.py` → `processed/05_codeit_traces/codeit_traces.json`  
- `analysis/06_progress_curves_v2.py` → `processed/06_curves/progress_curves_v2.json` + `curve_v2/*.png` (v2 normalization; baseline = progress(input_grid, target_grid))  
- `analysis/07_curve_metrics.py` → `processed/07_metrics/curve_metrics.csv` + scatter plots  
- `analysis/08_curve_comparison.py` → `processed/08_curve_comparison/pair_metrics_*.csv` + strip plots

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

**Normalisation v2** (unified task-level baseline):
```
norm_v2(t) = (progress(t) − baseline) / (1 − baseline)
baseline   = progress(input_grid, target_grid)
```
This sets `norm = 0` at the task input grid for both agents and `norm = 1` at the target, giving a shared coordinate system. Under this scheme:
- **CodeIt** always starts at exactly 0 (its first trace frame is the input grid by construction).
- **Humans** start from a blank canvas (all-zeros output), which maps to a *negative* norm value on any task where the input already partially matches the target. Positive v2 AUC means a human trajectory exceeded the input baseline level on average; negative AUC means it stayed or fell below.

Each trajectory is resampled to 100 evenly-spaced points before normalisation.

**Aggregation**: element-wise **median** across all normalised trajectories within a group per task.

Four median curves per task (any may be absent):
1. Human success — participants whose last attempt solved the task
2. Human failed — participants who never solved the task
3. CodeIt success — programs with `test_performance = True`
4. CodeIt failed — programs with `test_performance = False`

**AUC** = area under the median normalised v2 curve (trapezoidal integration).

### Results

**Spearman ρ (human_success AUC ~ codeit_success AUC, n = 58 tasks):** ρ = **−0.301**, p = 0.022 → **significant negative correlation**

| Difficulty category | Human success AUC | CodeIt success AUC | Human failed AUC | CodeIt failed AUC |
|---|---|---|---|---|
| Easy for both | +0.354 | 0.413 | −0.330 | 0.147 |
| Hard for both | +0.109 | 0.311 | +0.255 | 0.316 |
| Only hard for AI | −0.044 | 0.139 | −0.681 | 0.200 |
| Only hard for humans | +0.614 | 0.330 | +0.277 | 0.160 |

### Interpretation

**Human success AUC varies strongly by difficulty category.** On "Only hard for humans" tasks (where CodeIt finds a solution early), the small fraction of humans who do succeed converge very efficiently (+0.614). On "Only hard for AI" tasks (where humans solve intuitively but CodeIt takes many iterations), successful humans actually average *below* the input baseline (AUC = −0.044) — they spend much of their trajectory on a blank canvas that falls below the input-grid level, reflecting the H-ARC blank-start design.

**CodeIt success AUC drops monotonically with difficulty**: from 0.413 ("Easy for both") to 0.139 ("Only hard for AI"). Even when CodeIt eventually synthesises a correct program, it navigates the grid state space less efficiently on harder tasks.

**Human failed AUC is highly negative on "Only hard for AI" tasks (−0.681)**: these participants start from blank (already below baseline) and fail to recover, spending most of the trajectory far from the target. In contrast, failed humans on "Hard for both" and "Only hard for humans" tasks show positive AUC (+0.255, +0.277), indicating they at least climb above the input-grid level before failing to reach the target.

**Significant negative correlation between human and CodeIt convergence speed** (ρ = −0.301, p = 0.022): on tasks where humans converge efficiently (high human AUC), CodeIt tends to converge less efficiently, and vice versa. This reflects the complementary nature of the two search strategies — tasks where human intuition generates rapid progress are often precisely those where the DSL search space is poorly aligned with the answer, and vice versa.

---

## Q5: Pairwise Curve Similarity — L2 Distance and Pearson r

### Method

For each task, the four median v2-normalised curves are compared in all four within-agent and between-agent pairs:

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
| Easy for both | 2.77 | +0.878 |
| Hard for both | 5.15 | +0.731 |
| Only hard for AI | 7.24 | +0.679 |
| Only hard for humans | 2.43 | +0.894 |

**human_failed vs codeit_failed**

| Difficulty category | Median L2 | Median Pearson r |
|---|---|---|
| Easy for both | 2.67 | −0.279 |
| Hard for both | 3.14 | +0.268 |
| Only hard for AI | 3.57 | +0.339 |
| Only hard for humans | 3.03 | +0.199 |

**human_success vs human_failed**

| Difficulty category | Median L2 | Median Pearson r |
|---|---|---|
| Easy for both | 5.79 | +0.232 |
| Hard for both | 4.52 | +0.362 |
| Only hard for AI | 7.87 | +0.156 |
| Only hard for humans | 4.23 | +0.025 |

**codeit_success vs codeit_failed**

| Difficulty category | Median L2 | Median Pearson r |
|---|---|---|
| Easy for both | 3.91 | +0.940 |
| Hard for both | 1.69 | +0.905 |
| Only hard for AI | 2.10 | +0.972 |
| Only hard for humans | 2.78 | +0.943 |

### Interpretation

**Human success and CodeIt success share similar trajectory shapes** (Pearson r ≈ +0.68–0.89 across difficulty categories). The trend similarity holds across all difficulty categories, but the absolute path gap (L2) grows sharply for harder tasks — from 2.77 on "Easy for both" to 7.24 on "Only hard for AI." On these hardest-for-AI tasks, humans climb quickly while CodeIt programs take many small steps, producing the same upward trend but over very different scales.

**Human failure and CodeIt failure are structurally opposite only on easy tasks** under v2 (Pearson r = −0.279 for "Easy for both"): human failed participants start below the input baseline (blank canvas) and remain below or near it, while CodeIt failed programs start at 0 and make partial upward progress. This contrast weakens and reverses on harder tasks (r > 0 for "Hard for both" and beyond), where even failed CodeIt programs may stagnate similarly to human failed participants.

**Human success vs human failure share a positive Pearson r across all categories** (+0.025 to +0.362) under v2 normalisation. Unlike v1 (where per-trajectory normalisation forced both groups to start at 0 and produced a negative trend contrast), v2 reveals that both successful and failed humans start from the same negative v2 value (blank canvas below the input-baseline) and initially rise together. The groups diverge only near the end — success trajectories reach the target, failures plateau or drift. The largest L2 distances (up to 7.87 on "Only hard for AI") confirm the endpoint separation is substantial even when the overall trend is shared.

**CodeIt success vs CodeIt failure have nearly identical curve shapes** (Pearson r ≈ +0.90–0.97) with moderate L2 distance. Both groups move in the same direction from the same starting point (norm = 0); the separation is in how far they travel, not in strategy. This reflects the deterministic structure of DSL programs: a failed program typically executes most of the correct transformations but terminates one or more steps short of the target.

---

## Summary

| Question | Key Finding |
|---|---|
| **Q1** | Tasks split roughly evenly across 4 difficulty quadrants. 12 tasks are uniquely hard for humans (CodeIt solves them early); 13 are uniquely hard for AI (humans solve intuitively but DSL search takes long). |
| **Q2** | Significant positive correlation between human action count and CodeIt iteration (ρ=0.44, p<0.001). Number of attempts is not correlated. Both agents share a common difficulty signal at the action/search level. |
| **Q3** | Almost complete overlap: CodeIt solves 58/59 tasks that humans also solve. 337 human-solved tasks remain unsolved by CodeIt. Only 1 task (`31d5ba1a`) is uniquely solved by CodeIt. |
| **Q4** | Significant negative correlation between human and CodeIt convergence speed (ρ=−0.301, p=0.022): efficient human trajectories tend to coincide with inefficient CodeIt programs, and vice versa. Human success AUC varies strongly by category (−0.044 on "Only hard for AI" to +0.614 on "Only hard for humans"); CodeIt success AUC drops monotonically from 0.41 to 0.14. Human failed AUC is severely negative (−0.681) on "Only hard for AI" tasks due to blank-canvas starting point. |
| **Q5** | Human and CodeIt success curves share similar trend shapes (Pearson r ≈ 0.68–0.89) but grow far apart in absolute terms on harder tasks (L2 up to 7.24 on "Only hard for AI"). Under v2 normalisation, human success vs failure show positive Pearson r (both groups start from the same blank-canvas baseline), differing in endpoint rather than direction. CodeIt success vs failure are nearly identical in shape (r ≈ 0.94) — they differ in endpoint, not strategy. |
