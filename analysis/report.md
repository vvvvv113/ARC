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

## Q4: Progress Curves — How Do Humans and CodeIt Navigate the Grid State Space?

### Method

For each of the 59 tasks, intermediate grid states were captured during solving:
- **Human traces** (script 04): each participant's last attempt; `test_output_grid` per action, deduplicated consecutive frames. 603 trajectories total (367 success, 236 failed).
- **CodeIt traces** (script 05): DSL programs executed line-by-line; each line whose output is a valid grid is recorded. 29,278 programs traced across all three splits.

**Progress** at each step = fraction of cells matching the target output grid.  
**Normalisation**: each trajectory is resampled to 100 evenly-spaced points, then shifted and scaled so it starts at 0:  
`norm(t) = (progress(t) − progress(0)) / (1 − progress(0))`  
This removes the baseline advantage from tasks where the input grid already resembles the target.

For each task, four mean curves are computed (any may be absent):
1. Human success — participants whose last attempt solved the task
2. Human failed — participants who never solved the task
3. CodeIt success — programs with `test_performance = True`
4. CodeIt failed — programs with `test_performance = False`

**AUC** (area under the normalised curve, via trapezoidal integration) measures convergence speed: higher = faster progress.  
**Steps-to-90%** = normalised x position where the mean curve first reaches 0.9; NaN if never reached.

### Results

**Spearman correlation between human_success AUC and codeit_success AUC across 59 tasks:**  
ρ = **−0.018**, p = 0.89 → **no correlation**

| Difficulty category | Human success AUC | CodeIt success AUC | Human failed AUC | CodeIt failed AUC |
|---|---|---|---|---|
| Easy for both | 0.465 | 0.383 | −0.015 | 0.121 |
| Hard for both | 0.467 | 0.269 | −0.032 | 0.086 |
| Only hard for AI | 0.468 | 0.114 | −0.026 | 0.075 |
| Only hard for humans | 0.452 | 0.338 | — | 0.060 |

### Interpretation

**Human success AUC is nearly constant (~0.46–0.47) across all difficulty categories.** Once humans commit to a successful solve, their convergence speed is similar regardless of task difficulty quadrant.

**CodeIt success AUC varies dramatically by category**: highest for tasks "Easy for both" (0.383), dropping to 0.114 for tasks "Only hard for AI". Even when CodeIt eventually produces a correct program for a hard task, it tends to converge on the solution grid more slowly — the programs make less incremental progress per DSL line.

**Human failed AUC is negative (around −0.02 to −0.03)**: participants who never solve a task on average move the grid *away* from the target over time, likely because they reset the grid or explore wrong transformations. This is structurally different from CodeIt failed programs, which show small positive AUC (~0.06–0.12) — failed programs still tend to partially match the target even without fully solving it.

**No correlation between human and CodeIt convergence speed** (ρ = −0.018): the speed at which humans solve a task is completely unrelated to how efficiently CodeIt's programs navigate to the answer. This suggests the two agents are using fundamentally different strategies — humans converge incrementally through interactive edits, while CodeIt's programs jump to the answer in a single execution with variable intermediate quality.

---

## Summary

| Question | Key Finding |
|---|---|
| **Q1** | Tasks split roughly evenly across 4 difficulty quadrants. 12 tasks are uniquely hard for humans (CodeIt solves them early); 13 are uniquely hard for AI (humans solve intuitively but DSL search takes long). |
| **Q2** | Significant positive correlation between human action count and CodeIt iteration (ρ=0.44, p<0.001). Number of attempts is not correlated. Both agents share a common difficulty signal at the action/search level. |
| **Q3** | Almost complete overlap: CodeIt solves 58/59 tasks that humans also solve. 337 human-solved tasks remain unsolved by CodeIt. Only 1 task (`31d5ba1a`) is uniquely solved by CodeIt. |
| **Q4** | No correlation between human and CodeIt convergence speed (ρ=−0.018, p=0.89). Human success AUC is stable across difficulty categories (~0.46); CodeIt success AUC drops sharply for harder tasks (0.38 → 0.11). Human failed solvers go backward (negative AUC); CodeIt failed programs still partially converge. |
