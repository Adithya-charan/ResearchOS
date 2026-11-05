"""Optional formal-verification integration point (Z3). A satisfiability result concerns the formal constraints given, under
the assumptions written in them - it says nothing about novelty or about whether the constraints model reality."""
from __future__ import annotations

from importlib import util


def z3_available() -> bool:
    return util.find_spec("z3") is not None


def check_smt2(smt2: str, timeout_ms: int = 5000) -> dict:
    """Solve an SMT-LIB2 problem. Returns {'available', 'result': sat|unsat|unknown|error, ...}; never raises on bad input."""
    if not z3_available():
        return {"available": False, "result": "unavailable", "note": "z3-solver is not installed (pip install z3-solver)"}
    import z3  # type: ignore[import-not-found]
    solver = z3.Solver()
    solver.set("timeout", int(timeout_ms))
    try:
        solver.from_string(smt2)
        res = str(solver.check())
    except z3.Z3Exception as e:
        return {"available": True, "result": "error", "note": str(e)[:300]}
    return {"available": True, "result": res,
            "note": "Result applies only to the given constraints and their stated assumptions."}
