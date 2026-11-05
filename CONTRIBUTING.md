# Contributing to ResearchOS

Thanks for helping. This project values correctness and honesty over feature count.

## Ground rules

1. **Inspect before you edit** and keep working code working.
2. **Every behaviour change ships with tests.** A feature is not done until it has tests, a reproducible run and a documented result
   (see `docs/PHASE_REPORTS.md`).
3. **Never present unfinished work as complete.** If something is partial, say so in the code, the docs and the PR.
4. **Never weaken a limitation notice** or make a claim the system cannot support. The only novelty statement allowed is:
   "No close match was identified in the indexed corpus searched for this run. This is not proof that the idea has never been published."
5. **No pretrained weights, hosted AI APIs, or external model runtimes.** Models are trained from random initialisation here.
6. **Security by default:** deny-by-default permissions, validate untrusted input, scope every lookup to its owner, no secrets in the repo.

## Workflow

```bash
git checkout -b feat/short-description
pip install -e ".[dev,api,corpus]"        # after installing PyTorch for your platform
python -m pytest                          # must pass before you open a PR
```

* Branch from `main`; open a pull request; CI must be green.
* Use [Conventional Commits](https://www.conventionalcommits.org/): `feat(api): ...`, `fix(corpus): ...`, `docs: ...`, `test: ...`,
  `ci: ...`, `chore: ...`. Keep commits focused and the subject under ~72 characters.
* Do not commit datasets, checkpoints, databases or `.env` files (the `hygiene` CI job rejects them).
* Update `README.md` / `docs/PHASE_REPORTS.md` when behaviour or limits change.

## Code style

Python 3.11+, type hints on public functions, small focused modules, comments that explain *why*. Prefer pydantic contracts at trust
boundaries and explicit errors over silent fallbacks.
