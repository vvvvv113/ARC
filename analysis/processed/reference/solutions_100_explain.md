solutions_100.json is the saved state of everything CodeIt has successfully solved after running 100 meta-iterations of training.

Structure:


{
  "policy": {
    "seen_example":       { task_id → { program → metadata }, ... },  # 114 tasks
    "task_demonstration": { task_id → { program → metadata }, ... },  #  46 tasks
    "test":               { task_id → { program → metadata }, ... }   #  59 tasks
  }
}
Each key under "policy" represents a different performance tier:

seen_example (114 tasks): Programs that correctly reproduce the output on the same input-output examples the model was shown during training. The weakest criterion — the program may just be memorizing.
task_demonstration (46 tasks): Programs that generalize to the held-out demonstration pairs of the task (the extra examples not used during training). Stronger evidence of generalization.
test (59 tasks): Programs that produce the correct output on the hidden test input — the one input that has no provided output. This is the only split that counts as truly "solved" in the ARC benchmark sense.
What each entry looks like:


"00576224": {
  "x1 = hconcat(I, I)\nO = vmirror(x1)": {
    "program": "x1 = hconcat(I, I)\nO = vmirror(x1)",
    "new_task_key": "00576224_it81__21",
    "seen_example_performance": [1.0, 1.0],
    "task_demonstration_performance": [true, true],
    "test_performance": [true]
  },
  ... (331 more programs for this task)
}
program: the DSL code string that solves the task
new_task_key: encodes which task and which iteration it was found — {task_id}_it{N}__{sample_index} means this solution was first discovered at meta-iteration N
seen_example_performance: per-example accuracy on training examples (floats, 0–1)
task_demonstration_performance: pass/fail on each demonstration pair (booleans)
test_performance: pass/fail on the hidden test input (boolean) — [true] means solved
In short: it is a snapshot of CodeIt's solution bank at the end of training. For each solved task, it stores every program variant that passed the given tier's evaluation criterion, not just the best one — which is why a single task like 00576224 can have 332 different solution programs.

