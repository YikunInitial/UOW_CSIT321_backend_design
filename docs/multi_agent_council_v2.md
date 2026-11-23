# Multi-Agent Council v2

## Purpose

The council tests whether safety-valid multi-strategy competition provides a better quality/runtime trade-off than committing to one optimiser strategy in advance. It is not intended to claim that multiple agent labels alone improve optimisation.

The research mechanism is:

```text
same products and global budget
  -> strategy-specialised candidate agents
  -> independent Validator / SafetyAgent veto
  -> role-based evaluation of actual outputs
  -> deterministic Coordinator ranking
  -> best available safety-valid anytime solution
```

## Candidate specialisation

- `FastGreedyAgent`: fast balance-aware feasible baseline
- `DestinationGroupingAgent`: destination-first operational clarity
- `BalanceAwareAgent`: destination grouping plus balance-aware scoring
- `AnnealingAgent`: higher-cost SA refinement when the total budget supports it

Council candidates set `allow_policy_strategy_override=false`. The optimiser may still stop an expensive stage to respect a deadline, but it cannot silently turn a non-SA candidate into an SA candidate or change the candidate's research identity.

## Fair budget policy

When a global council budget is supplied, candidate agents receive weighted shares of the remaining budget:

- FastGreedyAgent: 0.10
- DestinationGroupingAgent: 0.20
- BalanceAwareAgent: 0.20
- AnnealingAgent: 0.50

Unused time remains available to later candidates. The larger AnnealingAgent share reflects its deliberately higher search cost. Budget weights are part of the experimental configuration and must remain fixed within a comparison.

The frozen v2.0.0 runtime reserves 10% of the global budget for independent
validation and coordination. Deadline enforcement remains cooperative and every
run reports `deadline_met`, `overrun_seconds`, and `overrun_ratio`; the system
does not claim a process-isolated hard real-time guarantee.

## Evaluation and safety

Every candidate is independently revalidated against the original product ids. Safety failure or execution failure causes a veto and makes the candidate ineligible for selection.

Non-vetoed candidates are scored from actual outputs:

- EfficiencyAgent: wagon count and area utilisation
- BalanceAgent: front/rear, left/right, and diagonal regional penalties
- OperationsAgent: actual mixed-destination wagons
- TimeAgent: runtime relative to the candidate's allocated budget
- SafetyAgent: mandatory validity gate

The deterministic Coordinator ranks by weighted score, then lower objective, lower runtime, and stable candidate name.

## Reproducible artifacts

Each council run writes:

- `multi_agent_council/council_decision.json`
- `multi_agent_council/council_evaluation_matrix.json`
- `multi_agent_council/council_evaluation_matrix.csv`

The artifacts are hashed in `run_manifest.json`. The audit JSON and Markdown include the selected candidate, diversity summary, and veto summary.

## Frozen-data diagnostic baseline

On the 62-product project dataset with a 15-second global budget, the corrected council produced:

- 4 completed candidates
- 4 distinct applied strategies
- 3 distinct outcomes
- 0 safety vetoes
- selected candidate: `BalanceAwareAgent`

The diagnostic run confirmed that `AnnealingAgent` executed SA and that the Coordinator preferred the faster balance-aware result when SA did not provide enough additional value under the evaluator weights. This is a diagnostic result, not a paper conclusion; formal experiments require repeated datasets, scenarios, budgets, and seeds.

## Research questions enabled

- Does council competition improve objective quality or recovery success over the best fixed single strategy?
- How frequently does the winning strategy change across instance scale, event severity, and time budget?
- How much candidate diversity is required before the council adds value?
- What is the latency overhead of council evaluation?
- Under what conditions does AnnealingAgent justify its larger budget?
- How often does SafetyAgent veto prevent an invalid candidate from being selected?

## Current limitation

Candidate execution is sequential and uses cooperative in-process deadlines. Parallel isolated workers are intentionally deferred until the sequential experimental contract is stable; parallel execution would change resource contention and therefore requires its own controlled experiment.
