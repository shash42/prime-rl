# EffectPred dependency patches

`verifiers-sdk-retries.patch` exposes OpenAI SDK retry configuration for native
renderer clients and the null harness. It applies to the pinned verifiers
submodule. EffectPred's `scripts/prime_rl/setup.sh` applies it before syncing.
For a standalone PRIME checkout, run:

```bash
git -C deps/verifiers apply ../../patches/verifiers-sdk-retries.patch
```

The patch is already applied if its reverse passes `git apply --reverse --check`.
