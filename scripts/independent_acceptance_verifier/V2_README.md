# V2 corrected DiffAct inpainting candidate generator

V2 is candidate generation only. It never accepts a repair; V1 remains the
independent acceptor. This review package implements B0 sampler validity and does
not authorize sampling, B1 oracle analysis, or outer-test access.

## Frozen sampler

- Core intervals are the exact frozen 5%-budget OOF selector spans. Overlapping
  or touching intervals are merged before sampling.
- A halo of `{0,8,16}` frames expands the sampling-free region. Core, halo, and
  clamped exterior are disjoint and exhaustive. Halo frames are a
  generation/postprocessing buffer only: after official median-15 probability
  postprocessing, every non-core frame is restored to the incumbent label.
- The incumbent is represented as a hard one-hot class sequence in DiffAct's
  normalized latent range. Context at diffusion time `t` is produced with the
  model's exact `alphas_cumprod[t]`, never an approximate step index.
- Restarts use `t_start ∈ {250,500,750,999}`. The remaining reverse path follows
  the released 25-step DDIM grid below that exact starting time. Pure noise at
  `t=999` is the fifth initialization.
- One fixed context-noise tensor defines the clamped context trajectory across
  every reverse step. DDIM step noise remains sequential and seeded.
- Each setting draws at most 15 sequential samples. After at least seven, it may
  stop after five consecutive samples add no new collapsed trace.
- Candidates are cluster medoids under normalized Levenshtein distance between
  collapsed postprocessed traces, with connected-component threshold 0.25.
  Per-frame voting is forbidden.

## B0 validity gate

Every one of the 12 outer×inner OOF checkpoints must pass:

1. exterior invariance after official postprocessing and the explicit non-core
   restoration contract;
2. exact empty-mask identity without invoking the diffusion model; and
3. exact seeded replay for probabilities and deployed labels.

Any failure blocks B1. The review study is fail-closed and cannot run B0 until a
Fable approval digest is embedded in a newly generated production study.

## Approved B0 → B1 execution

Fable approved B0 and the pre-registered B1 candidate oracle on 2026-08-01.
All twelve actual-checkpoint validity tasks must pass before any B1 sampling.
After V1 has claimed its first free GPU, three V2 waiters claim distinct
remaining devices using two consecutive free checks; they never preempt Phase B
or V1.

B1 samples each OOF video jointly over its frozen selected cores for halo widths
0/8/16 and restart times 250/500/750/999 (999 is the pure-noise arm). Each arm
uses sequential k≤15 sampling with the reviewed early-stop rule. Exact samples
are deduplicated, traces are clustered at normalized segmental edit distance
0.25, and only cluster medoids propose labels; per-frame voting is forbidden.
For compatibility with the reviewed V0/V1 candidate schema, a medoid proposes
its deterministic modal label within each selected core. B1 reports the OOF
best-of-k oracle, wrong-mass candidate availability, and incremental oracle over
the visual candidate pool under the frozen kill bars.

The currently running V1 study is never mutated. If B1 passes the join bars,
the candidates are packaged for Fable review and a separately versioned V1
extension; they are not silently inserted into an underway verifier run. Outer
test remains closed.

## B1, still blocked

On OOF spans only, best-of-k below +1.4 Acc closes V2. Samples join V1's
candidate corpus only at best-of-k ≥+1.75 Acc, correct candidates for ≥35% of
flagged wrong-frame mass, and ≥+0.5 incremental oracle over the visual-head
candidates. No outer data is used anywhere in V2.
