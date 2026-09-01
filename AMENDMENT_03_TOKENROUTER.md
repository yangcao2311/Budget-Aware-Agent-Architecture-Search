# AMENDMENT 03 — TokenRouter route identity

Status: **FROZEN BEFORE ANY TOKENROUTER WORKFLOW OBSERVATION** on 2026-09-01.
One minimal connectivity request preceded this amendment and is excluded from
all analysis. It used the prompt `Reply exactly OK.`, a 16-token completion
cap, and produced no visible answer because the cap was consumed by reasoning.

The available key was created for `https://api.tokenrouter.com/v1`, not the
Z.AI first-party endpoint assumed in Amendment 02. The authenticated `/models`
endpoint exposed one GLM route, `z-ai/glm-5.3-free`; a connectivity response
reported its model as `glm-5.3`. TokenRouter did not expose metadata proving
that this route is identical to the first-party product named
`glm-5.3-flash`.

Accordingly:

- the replication uses the exact routed ID `z-ai/glm-5.3-free`;
- the manuscript and artifacts will call it the **TokenRouter GLM-5.3 route**,
  not GLM-5.3-Flash, unless independently documented mapping evidence is
  obtained before submission;
- all experimental choices, tasks, seeds/replicate indices, three provenance
  arms, verifier/refinement graph, outcomes, and analysis rules in Amendment
  02 remain unchanged;
- `reasoning_effort=low` remains fixed because the compatibility request was
  accepted;
- the route is exposed as `free`, so accounting records zero nominal dollar
  price while still reporting provider-returned input/output tokens and call
  counts. No compute- or dollar-matched cross-provider claim is made.

The first successful workflow response, rather than the connectivity request,
starts observation of the replication campaign.
