"""A small reference experiment: a seeded Monte-Carlo estimate. Deterministic, stdlib only, writes one artifact."""
from __future__ import annotations

from research.experiments.spec import ExperimentSpec

CODE = '''import json, os, random
cfg = json.load(open("config.json"))
random.seed(int(os.environ["EXPERIMENT_SEED"]))
n = int(cfg["samples"])
inside = sum(1 for _ in range(n) if random.random() ** 2 + random.random() ** 2 <= 1.0)
estimate = 4.0 * inside / n
with open("out/estimate.txt", "w") as f:
    f.write(f"{estimate:.6f}\\n")
print("RESULT " + json.dumps({"pi_estimate": estimate, "abs_error": abs(estimate - 3.141592653589793), "samples": n}))
'''


def example_spec(seed: int = 7, samples: int = 20000, hypothesis_id: str | None = None) -> ExperimentSpec:
    return ExperimentSpec(name="monte-carlo-pi", code=CODE, seed=seed, config={"samples": samples},
                          hypothesis_id=hypothesis_id, timeout_s=30)
