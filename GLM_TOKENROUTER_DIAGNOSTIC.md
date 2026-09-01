# TokenRouter GLM route diagnostic

Date: 2026-09-01. This is an implementation diagnostic, not an experimental
result and not evidence used in the paper's accuracy comparisons.

## Observed route identity

- Authenticated endpoint: `https://api.tokenrouter.com/v1`.
- `/models` exposed `z-ai/glm-5.3-free` as the only model ID containing
  `glm` for the supplied token.
- Successful responses requested through that route reported
  `model = glm-5.3`.
- A request for the exact ID `z-ai/glm-5.3-flash` returned HTTP 403: the token
  has no access to that model.

These observations do not establish that `z-ai/glm-5.3-free` is, or is not,
the product called GLM-5.3-Flash. It must not be labelled Flash in the paper
without provider documentation or access to both exact routes.

## Behavioral-probe outcome

The same deterministic prompt and parameters were used for the available
`free` route and attempted `flash` route. The prompt was the 5 L / 3 L jug
problem with a 100 mL/s leak in the 5 L jug and a 200 mL/s faucet.

- At a 2,048-token completion cap, the `free` route returned 2,048 reasoning
  tokens and no visible final answer.
- At an 8,192-token completion cap, it again returned exactly 8,192 reasoning
  tokens and no visible final answer.
- The `flash` route could not be probed because it returned HTTP 403.
- A request to disable thinking was rejected for the substantive prompt with
  HTTP 400 (`GLM-5.3 does not support disabling thinking`).

Therefore no paired behavioral comparison was possible. Even a successful
single-answer comparison would be suggestive rather than model-identity
proof, because routing, system prompts, decoding, and serving revisions are
uncontrolled.

## Workflow-smoke outcome and decision

The TokenRouter route completed one code and one math baseline request, but
both exhausted the existing output reservation before a reliable final
answer. The code task used 916 output tokens and 45.10 seconds; the math task
used 1,536 output tokens and 175.47 seconds. Both failed held-out grading, and
the math output contained no boxed answer.

Amendment 02 requires gradeable final answers and no reservation mismatch
before the full study. The route therefore fails the frozen compatibility
gate. The full GLM replication is not run with this route. A future run would
require either documented access to the intended exact model or a separately
preregistered protocol with larger reasoning-aware budgets; the latter would
not be compute-comparable to the existing campaign.
