# Cloud Operations: Severity Levels & Incident Management SOP

## 1. Incident Severity Classification Matrix

### Severity 0 (P0 - Critical Outage)
* **Definition:** Core customer-facing functionality is completely offline or massive data integrity risk affecting >15% of active users (e.g. checkout service down, primary database failover stuck).
* **Response SLA:** Incident Commander acknowledged within **5 minutes**. Status page updated every 15 minutes.
* **Escalation:** PagerDuty pages the on-call engineer, VP of Engineering, and Head of Product immediately.

### Severity 1 (P1 - Major Degradation)
* **Definition:** Major feature failure or severe performance degradation with no immediate workaround (e.g. search latency >5 seconds, background job queue delayed by 2 hours).
* **Response SLA:** Incident Commander acknowledged within **15 minutes**.
* **War Room:** A dedicated Slack channel `#incident-YYYYMMDD-sev1` and Google Meet bridge are created immediately.

### Severity 2 (P2 - Minor Incident)
* **Definition:** Non-critical feature bug with an existing workaround affecting a subset of users. Addressed within standard business hours.

## 2. Post-Mortem & Blameless RCA Protocol
Every P0 and P1 incident requires a formal Post-Mortem meeting:
* Meeting must occur within **48 hours** of incident resolution.
* Must follow a **blameless culture**: focus on architectural shortcomings, automated guardrails, and missing alerts rather than human error.
* All action items (P0/P1 corrective tickets) must be prioritized into the next immediate engineering sprint.
