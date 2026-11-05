# Security Policy

ResearchOS is experimental software. It handles user documents, runs approved code in a sandbox and serves a web API, so
security reports are welcome.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting
(*Security* tab -> *Report a vulnerability*) on this repository. Include steps to reproduce, the affected version or commit,
and the impact you expect. You will get an acknowledgement and a fix or mitigation plan as soon as practical.

## Scope notes

* The non-isolating subprocess experiment fallback is **not a security boundary**; it is for development only and is disabled
  unless `RESEARCHOS_ALLOW_WEAK_SANDBOX=true`.
* The Docker sandbox path is implemented but has not been exercised on every platform; treat it as defence in depth, not proof.
* Malware scanning of uploads is a hook, not a bundled scanner.
* Deployments should run behind TLS (`RESEARCHOS_COOKIE_SECURE=true`) and configure `RESEARCHOS_ALLOWED_ORIGINS`.

## Supported versions

Only the latest commit on `main` receives fixes.
