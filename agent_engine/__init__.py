"""RESEARCHOS agent engine: typed contracts, registry, task graph, scheduler, persisted state, policies.

Agents are software roles, not separate models. They call deterministic tools; the locally trained language model is
an optional, budgeted, untrusted helper. Everything an agent returns is validated before it is persisted.
"""
