# Engineering Team Onboarding & Deployment Standards

## 1. Branching Strategy & Code Review
All production repositories follow Trunk-Based Development with short-lived feature branches:
* Feature branches must branch off `main` and be named `feat/<issue-id>-short-desc` or `fix/<issue-id>-short-desc`.
* Pull requests require at least **2 approvals from senior team members** before merging.
* Merges to `main` must use squash commits with a clear conventional commit message (`feat:`, `fix:`, `refactor:`, `perf:`).

## 2. CI/CD Pipeline & Automated Quality Gates
Every PR triggers the automated GitHub Actions CI pipeline:
* **Unit & Integration Tests:** Test suite must pass with a minimum of **85% code coverage**. Coverage decreases by more than 1% block merging.
* **Linter & Type Checking:** Ruff and Mypy must pass with zero warnings in strict mode.
* **Security Scanning:** Trivy runs container vulnerability scans, and Gitleaks checks for committed API keys or tokens.

## 3. Canary Deployments & Rollback Protocol
Production deployments happen automatically on merge to `main` via Kubernetes Argo Rollouts:
* **Canary Phase:** 5% of global traffic is routed to the new version for 15 minutes while DataDog monitors the error rate.
* **Automated Rollback:** If the HTTP 5xx error rate exceeds 0.5% or latency p99 degrades by more than 20% compared to baseline, the rollout aborts immediately and reverts traffic to the previous stable release.
