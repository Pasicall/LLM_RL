# LLM_RL

Self-study implementation of CS336 Spring 2026 Assignment 5: Alignment.
Based on the [Stanford starter repository](https://github.com/stanford-cs336/assignment5-alignment).
The assignment implementations are in progress.

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf)

The completely optional supplement on safety alignment, instruction tuning, and RLHF is available at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf).

Datasets, assignment PDFs, environments, and training outputs are kept locally and
excluded from this repository. Obtain the required datasets from the Stanford
starter repository and place them under `data/` before running experiments.

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install the base dependencies for local development:
```
uv sync --frozen
```

2. Run the required unit tests:

``` sh
uv run --no-sync pytest tests/test_grpo.py
```

Unimplemented adapters will fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

