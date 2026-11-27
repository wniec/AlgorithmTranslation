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
region, translating its quadratic model's gradient/Hessian and its radius), and
two estimation-of-distribution algorithms: **UMDA** (continuous UMDA_c with
smoothed updates, translating its diagonal Gaussian: mean + per-coordinate std)
and **AMALGAM** (AMaLGaM, Bosman et al. 2008/2013 — a full Gaussian whose
covariance has memory, plus its anticipated mean shift and distribution
multiplier `c_mult`). Plain UMDA_c has no memory (its distribution is a function
of the last population, which is already shared), hence the smoothing.
The archive is a distinct `POINT_SET` field type: the encoder pools it
(set-equivariant), and the decoder regenerates it as `round(2.3·N)` points
seeded from the carried population. Any pair can be translated. (`IPSO` is in
the portfolio but not translatable: it restarts from one particle and grows.)

CMA-ES's search distribution, BOBYQA's quadratic model and the EDAs'
distributions are **computed**
rather than decoded from scratch (`algorithm_translation/models/prior/`):

- CMA-ES: the mean is CMA-ES's own weighted recombination of the source's best
  points (a swarm's personal bests, otherwise its population), and the
  covariance has the form of its rank-μ update, blended 10% with its isotropic
  part.
- BOBYQA: the quadratic model BOBYQA itself fits to the handed-over positions
  and values (minimum-norm interpolation, centred on the best point), so it is
  the same whichever optimizer produced them; the radius is the population's
  mean per-coordinate std. A residual survives BOBYQA's own refit only in the
  part of the model the points leave open.
- UMDA / AMALGAM: the distribution the EDA itself estimates from the
  handed-over points (maximum likelihood over the best 30% / 35%), with
  `ams_shift = 0` and `c_mult = 1`. Their residuals use two shift columns (mean
  + per-coordinate log-std for UMDA, which can therefore narrow a single
  coordinate; mean + anticipated mean shift for AMaLGaM).

For CMA-ES and BOBYQA, the decoder head only predicts a residual around the computed state: a
per-coordinate shift (mean / gradient), a low-rank factor added to the
matrix (`--cov-rank`; PSD `F Fᵀ` for the covariance, signed `F·diag(+1, −1, …)·Fᵀ`
for the Hessian, which may be indefinite, so it needs rank ≥ 2 to remove
curvature) and two global log-scale shifts (step size and blend /
curvature scale and radius), i.e. `(1 + rank)·D + 2` numbers. A zero action
reproduces the computed state.

A `TranslatorPair` holds an encoder + decoder head per algorithm. Translating
`A -> B` reuses A's population and only decodes B's specific fields. It is trained
with reinforcement learning (PPO or GRPO, below) on the downstream optimization
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
`uv run train` or `uv run train-grpo`. Per-update metrics are logged. Set
`WANDB_MODE=offline` to sync later, or `disabled` to turn it off. Without
`--wandb` nothing W&B-related is touched.

## Train with reinforcement learning (PPO or GRPO)

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

# critic-free GRPO: a group of sampled translations per switch state
uv run train-grpo PSO CMAES -d 2 3 5 --n-individuals 12 \
    --fe-multiplier 2000 --updates 100 --groups-per-update 128 --group-size 8
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
    start into CMA-ES, BOBYQA, UMDA or AMALGAM (see above), the lossy default
    for other targets.
  - `coldtarget` — the target optimizer run cold for the whole budget.

  A comma-separated list sums several modes (`mixed` = `noswitch,relative`).
  Each non-`absolute` mode runs one extra counterfactual optimizer per step
  (~2× cost each). Except for a pure `absolute` reward, a per-direction mean
  reward is logged (`reward/PSO->CMAES`, `reward/CMAES->PSO`).
- **Action**: the decoded target fields, with the covariance emitted in factor
  space → stays PSD. Both trainers put a Gaussian over these coordinates (same
  policy network, `ActorCritic`).
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

**GRPO** (`algorithm_translation/rl/grpo/`, critic-free): with a single switch
per episode the task is a contextual bandit, so instead of a learned value
baseline every switch state gets a group of `--group-size` sampled translations.
Each is scored from that same state under the same post-switch seeds
(`TranslationEnv.evaluate_action`, which does not advance the episode), and its
advantage is its reward relative to the group: mean-centered and divided by the
group std, or only mean-centered with `--no-adv-std-scale`. Between-problem and
seed variance cancel inside a group. The action-independent reward baselines
(`noswitch` / `relative` / `coldtarget`) cancel exactly, so `--reward-mode` only
affects what is logged; each baseline is run once per group and cached. The
update is PPO's clipped surrogate over these advantages (`--epochs`,
`--minibatch-size`, `--clip`), on-policy, with no critic, value loss, GAE or
reward/observation normalization. Logged extras: `group_reward_std` (how much
the action matters within a state), `zero_adv_group_frac`, `approx_kl`,
`clip_frac`. Cost per update: `--groups-per-update` warmups plus
`groups × group-size × --reward-seeds` target runs plus one set of baselines per
group. Saves `models/PSO_CMAES_grpo.pt`.

Either checkpoint is consumed directly by `uv run eval`
(`--model models/PSO_CMAES_ppo.pt` or `..._grpo.pt`).

**Train/test scenario** (both trainers, `--scenario`): every run is either
`lopo` (**leave-some-problems-out**, default: held-out BBOB *functions*, chosen
by `--split easy|hard|random`; instances are ignored, both sides use all 15
unless `--instances` restricts them) or `loio` (**leave-some-instances-out**: the
same functions, `--functions` default all 24, trained on `--instances`, default
1-5 and 71-75, and tested on the held-out `--test-instances`, default 76-80). Envs only ever see the train problems; the split
is stored in every checkpoint (`ckpt["split"]`), and `uv run eval` /
`uv run eval-switch` default to that held-out set and always drop training
problems.

**Autosave + evaluation**: every `--save-every` updates (default 1; 0
disables) the weights are saved to `<out>_upd00001.pt`, `<out>_upd00002.pt`, ... and that checkpoint
is evaluated on up to `--eval-max-problems` held-out problems per direction
(same episodes every time, fixed by `--eval-seed`), each re-run under
`--eval-reps` optimizer seeds (default: `--reward-seeds`) and averaged. The scores are stored in the
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
├── optimizers/      # vendored SubOptimizer base + PSO family, CMA-ES, MadDE, BOBYQA, EDAs
├── suite/           # IOH/BBOB problem loader + train/test splits
├── state/           # StateSpec schema + native<->canonical conversion, batching
├── models/          # norm, set-equivariant layers, encoder, heads, translator
├── runs.py          # building / running an optimizer segment (shared everywhere)
├── seeding.py       # derive_seed: every random stream from (seed, stream tag, ...)
├── train_loop.py    # device resolution + checkpoint io
├── rl/              # RL core
│   ├── env.py       #   TranslationEnv (gymnasium): one switch per episode
│   ├── scoring.py   #   SwitchScorer: translated run(s) minus the reward baselines
│   ├── reward.py    #   scoring one segment (AOCC / endpoint)
│   ├── policy.py    #   ActorCritic (translator as actor + critic)
│   ├── normalize.py #   running mean/std (PPO reward/obs normalization)
│   ├── workers.py   #   process pool for parallel rollouts
│   ├── ppo/         #   PPO: config / rollout / update / loop
│   └── grpo/        #   GRPO: config / rollout / update / loop
├── evaluation/      # evaluating a trained translator (no CLI)
│   ├── heldout.py   #   one switch per held-out problem (eval + checkpoint eval)
│   ├── switcher.py  #   random multi-switch schedules (eval-switch)
│   └── anytime.py   #   budget AOCC, ERT, ranks
└── cli/             # console-script entry points + shared CLI/env plumbing
    ├── common.py    #   shared RL CLI args, env/logger construction, autosave
    ├── scenario.py  #   lopo/loio train/test split flags + checkpoint split
    ├── train_ppo.py #   `uv run train`        <ALG_A> <ALG_B>
    ├── train_grpo.py #  `uv run train-grpo`   <ALG_A> <ALG_B>
    ├── evaluate.py  #   `uv run eval`         translated vs lossy vs cold hand-off
    └── evaluate_switcher.py  # `uv run eval-switch`  multi-switch evaluation
tests/               # round-trip, equivariance/PSD, RL env + PPO + GRPO
```

## Acknowledgements

The optimizer implementations and the BBOB run/switch loop are adapted from the
companion **DynamicAlgorithmSelection2** project.
