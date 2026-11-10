# Algorithm Translation

Studying whether a neural network can **translate the internal state of one
metaheuristic into another** — turning a PSO swarm (positions + **velocities** +
personal bests) into an equivalent CMA-ES search distribution (mean +
**covariance** + step-size + evolution paths), and back.

When a dynamic-algorithm-selection controller switches optimizers mid-run, the
hand-off is normally *lossy*: the shared population is carried but every
algorithm-specific structure (velocities, covariance) is discarded. This project
**learns** that hand-off instead.

## Idea

Each optimizer exposes a warm-start state via `get_data()` / `set_data()`. That
state splits into:

- **shared** fields — positions, values, best-so-far — the fixed "circumstances"
  of a switch, carried verbatim and never translated;
- **specific** fields — the algorithm's own machinery, which a bidirectional
  translator learns to map between algorithms.

Translatable optimizers (`algorithm_translation/optimizers/`): **PSO** and its
variants `SPSO`/`SPSOL`/`CPSO` (per-particle velocities + personal bests),
**CMA-ES** (mean + covariance + step-size + evolution paths), **MadDE** (Biswas
et al., CEC 2021 — a self-adaptive DE whose translated field is its **external
archive**, a variable-size *point-set*), and **BOBYQA** (a model-based trust
region, translating its quadratic model's gradient/Hessian and its radius).
The archive is a distinct `POINT_SET` field type: the encoder pools it
(set-equivariant), and the decoder regenerates it as `round(2.3·N)` points
seeded from the carried population. Any pair can be translated. (`IPSO` is in
the portfolio but not translatable: it restarts from one particle and grows.)

CMA-ES's search distribution is **computed** rather than decoded from scratch
(`algorithm_translation/models/prior.py`): the mean is CMA-ES's own weighted recombination of the
source's best points (a swarm's personal bests, otherwise its population), and
the covariance has the form of its rank-μ update, blended 10% with its isotropic
part. Its decoder head only predicts a residual around that — a mean shift, a
covariance factor (`--cov-rank`), a step-size multiplier and a blend shift, i.e.
`(1 + rank)·D + 2` numbers — so a zero action reproduces the computed state.

A `TranslatorPair` holds an encoder + decoder head per algorithm. Translating
`A -> B` reuses A's population and only decodes B's specific fields. It is trained
with reinforcement learning (PPO or TD3, below) on the downstream optimization
reward.

The network is **set-equivariant**: permutation-invariant over particles
(DeepSets pooling) and permutation-equivariant over coordinate axes (attention
over per-dimension tokens with a covariance-derived bias). One model spans all
problem dimensions `D` and population sizes `N`. The covariance is decoded as
`L Lᵀ + diag(d)`, PSD by construction.

## Install

Python 3.11, dependencies via [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Logging (Weights & Biases)

Both trainers can log metrics to W&B. Copy the example env file and fill in your
credentials (the file is git-ignored and loaded automatically):

```bash
cp .env.example .env      # then edit WANDB_API_KEY / WANDB_PROJECT / WANDB_ENTITY
```

Then pass `--wandb` (optionally `--wandb-project` / `--wandb-run-name`) to
`uv run train` or `uv run train-td3`. Per-update metrics are logged. Set
`WANDB_MODE=offline` to sync later, or `disabled` to turn it off. Without
`--wandb` nothing W&B-related is touched.

## Train with reinforcement learning (PPO or TD3)

We want the translated state to make the target optimizer *optimize well*, which
is non-differentiable. The RL trainers optimize that true objective directly over
a Gymnasium environment, with the set-equivariant translator network as the
policy. Two interchangeable trainers are provided; they share their environment,
CLI, and W&B/checkpoint plumbing (`algorithm_translation/cli/common.py`) and differ only in the
learning algorithm. Each is a console script (see `[project.scripts]`), run with
`uv run`:

```bash
# on-policy, stochastic-action PPO
uv run train PSO CMAES -d 2 3 5 --n-individuals 12 \
    --fe-multiplier 2000 --updates 100 --rollout-steps 1024

# off-policy, deterministic-action TD3 (twin critics + replay buffer)
uv run train-td3 PSO CMAES -d 2 3 5 --n-individuals 12 \
    --fe-multiplier 2000 --updates 100 --rollout-steps 1024
```

Shared across both trainers:

- **Environment** (`algorithm_translation/rl/env.py`, a `gymnasium.Env`): one episode is a BBOB run
  with a single hand-off at a log-sampled switch point (the target then runs to
  the end of the budget; `--fixed-order` makes every switch `algo_a → algo_b`,
  otherwise the direction is random per episode); the observation is the source optimizer's
  state, the action is the target optimizer's state (the translation). Each algo
  may run its own population size (`--n-individuals-a` / `--n-individuals-b`); the
  action is decoded onto the source's `N` and the assembled warm-start resized to
  the target's, so the action dimension stays a pure function of `(algo, D, N)`.
- **Reward** (`--reward-mode`): every mode is `score(translated run) −
  score(baseline run)`. Each run's segment is scored by **AOCC** (area over the
  convergence curve, relative to the gap to the BBOB optimum at the switch), or
  by the log-scaled best-so-far improvement with `--endpoint-reward`. Modes
  differ only in the baseline:
  - `noswitch` **(default)** — the source optimizer simply continuing. Signed:
    positive when switching (with the translated state) beats staying,
    negative when it does worse.
  - `absolute` — no baseline (0): the translated run's own score.
  - `relative` — the hand-off available without a network: the computed warm
    start into CMA-ES (see above), the lossy default for other targets.
  - `coldtarget` — the target optimizer run cold for the whole budget.

  A comma-separated list sums several modes (`mixed` = `noswitch,relative`).
  Each non-`absolute` mode runs one extra counterfactual optimizer per step
  (~2× cost each). Except for a pure `absolute` reward, a per-direction mean
  reward is logged (`reward/PSO->CMAES`, `reward/CMAES->PSO`).
- **Action**: the decoded target fields, with the covariance emitted in factor
  space → stays PSD. PPO puts a Gaussian over these coordinates; TD3 acts
  deterministically and adds exploration noise (see below).
- **Critic**: its own encoder (decoupled from the actor). It additionally
  consumes a side-input of the 2 progress scalars (running-normalized), widened
  by `--scale-feature` / `--use-ela` (PPO only). This
  feeds **only the critic** (the actor/translator stays a function of the
  optimizer state alone, so the evaluator (`uv run eval`) is unaffected).
- **Normalization** (on by default): rewards are scaled by the running std of the
  discounted return, and the scalar context observation is standardized by its
  running mean/std — both persisted across updates so value targets and
  advantages stay at ~unit scale. The structured optimizer state is already
  per-sample normalized inside the network (`NormContext`). Disable with
  `--no-norm-reward` / `--no-norm-obs`. (Logged `return` is always the raw value.)
- **Grouping**: both loops are custom (no SB3) so they handle variable `D`/`N`
  and the structured covariance action; minibatches are grouped by
  `(target_algo, D, N)` so every batch is shape-homogeneous.

**PPO** (`algorithm_translation/rl/ppo/`, on-policy): clipped-surrogate + GAE over a
sliding-window rollout buffer, with PPO value-clipping and a Huber value loss so
a value-function spike can't swamp the policy gradient. Exploration is a learned
Gaussian whose log-std is **per action segment per target algorithm** (e.g.
CMA-ES's mean shift, its covariance factor, and PSO's per-particle
velocities each get their own std) — a single global scalar can't express the
different natural scales of those fields — plus a bounded per-dimension residual
read from the encoder, so exploration adapts to the population. The base
log-stds sit in their own optimizer param group at `lr × --std-lr-mult`, since
these few scalars need a far larger step than the rest of the network to track
their optimum. Saves
`models/PSO_CMAES_ppo.pt`.

**TD3** (`algorithm_translation/rl/td3/`, off-policy): a deterministic actor (the translator's
mean action, no learned std) with twin critics — `min(Q1, Q2)` targets, delayed
& Polyak-averaged target networks, and target-policy smoothing — trained from a
replay buffer. Exploration is fixed-scale Gaussian noise added to the action
during rollout (`--expl-noise`); there is no decay schedule, so the
exploration/exploitation balance is constant over training. Because the action
dimension varies with `(algo, D, N)`, the critic can't concatenate a fixed-size
`(state, action)`: instead the action is decoded back into a `CanonicalState`
(parameter-free, so replayed off-policy actions reconstruct identically) and
encoded with the target algo's own `StateEncoder` — an "action-as-state" critic.
Saves `models/PSO_CMAES_td3.pt`.

Either checkpoint is consumed directly by `uv run eval`
(`--model models/PSO_CMAES_ppo.pt` or `..._td3.pt`).

**Train/test scenario** (both trainers, `--scenario`): every run is either
`lopo` (**leave-some-problems-out**, default: held-out BBOB *functions*, chosen
by `--split easy|hard|random`) or `loio` (**leave-some-instances-out**: the same
functions, `--functions` default all 24, trained on `--instances` and tested on
held-out `--test-instances`). Envs only ever see the train problems; the split
is stored in every checkpoint (`ckpt["split"]`), and `uv run eval` /
`uv run eval-switch` default to that held-out set and always drop training
problems.

**Autosave + evaluation**: every `--save-every` updates (default 50; 0
disables) the weights are saved to `<out>_upd00050.pt`, ... and that checkpoint
is evaluated on up to `--eval-max-problems` held-out problems per direction
(same episodes every time, fixed by `--eval-seed`). The scores are stored in the
checkpoint (`ckpt["eval"]`) and, with `--wandb`, logged as `eval/*`
(`log10_prec/{translated,lossy,cold}`, `win_rate/translated_vs_{lossy,cold}`,
per direction and averaged under `eval/mean/*`).

## Evaluate (the headline metric)

```bash
uv run eval PSO CMAES -d 2 3 5 --n-individuals 12 --max-problems 60
```

On the BBOB **test** split, warms up the source optimizer to a switch point
sampled log-uniformly over the budget (one switch per problem, drawn
independently), then switches to the target under three hand-offs and reports
the best objective reached by each:

- **translated** — source state run through the trained translator;
- **lossy** — carry only the shared population (current default);
- **cold** — fresh target, ignores the source.

Success = translated ≥ lossy on average.

## Layout

```
algorithm_translation/
├── optimizers/      # vendored SubOptimizer base + PSO family, CMA-ES, MadDE, BOBYQA
├── suite/           # IOH/BBOB problem loader + train/test splits
├── state/           # StateSpec schema + native<->canonical conversion, batching
├── models/          # norm, set-equivariant layers, encoder, heads, translator
├── train_loop.py    # device resolution + checkpoint io
├── rl/              # RL core: env (gymnasium), reward, policy, normalize
│   ├── ppo/         #   PPO: config / rollout / update / loop
│   └── td3/         #   TD3: config / networks / buffer / rollout / update / loop
└── cli/             # console-script entry points + shared CLI/env plumbing
    ├── common.py    #   shared RL CLI args, env/logger construction, autosave
    ├── scenario.py  #   lopo/loio train/test split flags + checkpoint split
    ├── train_ppo.py #   `uv run train`        <ALG_A> <ALG_B>
    ├── train_td3.py #   `uv run train-td3`    <ALG_A> <ALG_B>
    ├── evaluate.py  #   `uv run eval`         translated vs lossy vs cold hand-off
    └── evaluate_switcher.py  # `uv run eval-switch`  multi-switch evaluation
tests/               # round-trip, equivariance/PSD, RL env + PPO + TD3
```

## Acknowledgements

The optimizer implementations and the BBOB run/switch loop are adapted from the
companion **DynamicAlgorithmSelection2** project.
