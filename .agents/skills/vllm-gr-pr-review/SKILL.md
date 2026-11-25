---
name: vllm-gr-pr-review
description: Review or re-review a vllm-gr pull request, commit, branch diff, or local changes for concrete regressions. Trace affected GR contracts and evaluate the evidence for the reviewed revision.
---

# Review a vLLM-GR change

## Establish the review target

- For a PR, record its base and head SHAs. Read the complete merge-base-to-head
  diff, relevant discussion and checks for that revision. If either ref moves
  during collection, refresh the affected evidence. Read source, tests and
  design documents at the reviewed revision.
- For a local target, honor the requested scope. Compare a committed branch
  with `git diff <base>...<head>`. For uncommitted work, inspect staged and
  unstaged changes and the contents of relevant untracked files.
- On re-review, compare the previously reviewed head with the new head.
  Recheck earlier findings at their current call sites, inspect newly affected
  paths and account for base changes or rebases before carrying evidence forward.

## Follow the affected behavior

Read the [project instructions](../../../AGENTS.md). Use
[development navigation](../../../docs/contributing/development.md) to locate
the producer, its live consumers, the design contract and regression tests.
For a documentation-only change, verify the changed claims, links or commands
against their owners. Select the following checks when the diff warrants them.

### Upstream compatibility and runtime patches

Resolve the supported upstream versions from the reviewed
[compatibility declarations](../../../vllm_gr/platforms/compat.py). Compare the
patched symbol and its callers at that upstream revision. For runtime evidence,
confirm the imported package and provider. Trace plugin initialization into the
affected main, EngineCore or worker process. When a patch moves or disappears,
check the [ownership inventory](../../../docs/design/patch_inventory.yaml) for
activation, contracts, tests and removal conditions. For example, removing an
attention selector wrapper requires tracing both explicitly selected and
marker-restored custom backends on the affected platforms.

### Beam scheduling, worker execution and results

Follow one discriminating request through the changed boundary using the
[Beam batch contract](../../../docs/design/feature/beam_batch_execution.md).
For example, two prompts at width four occupy two native request slots and
eight Decode rows. For a KV reorder change, follow the second request's parent
index from operator output to its request-local pool slot and check where the
offset is applied. Compare with the
[multi-request reorder test](../../../tests/v1/beam/test_beam_decision.py).
For asynchronous lifecycle changes, follow dispatch, receipt and retirement
using the [scheduling contract](../../../docs/design/feature/async_scheduling.md).

### Validation and performance claims

Choose a test whose inputs reach the changed path and whose assertions would
fail for the suspected regression. Check its fixtures,
[level and platform](../../../docs/contributing/ci/test_levels.md), runner and
actual outcome. Output parity, graph replay, Recall and latency each require
different observations. For a replay claim, look for the counter for the
claimed mode. For a performance or Recall comparison, use the
[benchmark acceptance guide](../../../docs/contributing/ci/benchmark_acceptance.md)
to check revisions, workload identity, runtime and measurement conditions.
Identify the commit CI tested, including a synthetic merge when applicable.

## Report findings

Lead with actionable regressions introduced by the diff, ordered by severity.
Give each a priority (P0-P3), a concise title, a changed path and line, and a
reachable scenario explaining the impact. Check the surrounding code and tests
before attributing a defect. If none qualify, report `No findings.`
Summarize material validation gaps separately. For a re-review, also give the
current disposition of prior findings. Deliver the review in chat by default.
Post to the PR only when the user requests publication.
