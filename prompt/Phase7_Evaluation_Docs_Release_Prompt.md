# PHASE 7 — Final evaluation, ablations, documentation, demo-readiness and release (v2.0)

> **Prerequisites:** Phases 1–6 are merged into `v2` (Phase 1 message-driven core; 2 React UI;
> 3 learning environment; 4 imitation model; 5 PPO model; 6 MCTS + Brain panel). The user gives
> you the base commit. Names follow the earlier prompts; **the merged code wins** — adapt and
> list adaptations in the final report.

Paste this whole file into Antigravity (Planning mode, strongest model).

---

## 0. Context and goal

This is the last phase. Nothing new is invented; the job is to **prove, explain, package and
rehearse** — so that a four-person student team can pass two graded reviews:

* **Review 1** — PEAS (3), environment & agent analysis (3), algorithmic modelling & search (3), Q&A / presentation (1).
* **Review 2** — tool/package selection & setup (3), multi-agent execution & interaction (3), demo quality & **testing** (3), code structure & scalability (1).

(Course 23CSE401 *Fundamentals of AI*, textbook AIMA 4th ed.; the evaluation is a live demo plus viva by examiners.)

You will deliver:

1. a **rigorous final evaluation** (all strategies, regimes, scenarios, scales; paired statistics; robustness; failure analysis) and **ablations**;
2. **documentation** that a grader and a viva examiner can read in 20 minutes: technical report, updated design/testing docs, a **rubric-to-evidence map**, viva Q&A, demo script, presentation outline;
3. **quality hardening**: a final test/CI/lint/type pass, a fresh-clone offline smoke test, scalability evidence, performance profile;
4. **release**: merge to `main`, tag `v2.0`, clean repository — **without pushing**.

Honesty is the core requirement: results are reported as measured, including losses. A claim without
a table, a seed range and a command to reproduce it does not go in the docs.

---

## 1. Working agreement

* Branch `phase-7-release` from `v2`. Small commits. **Do not push anything. Do not delete branches or tags.** The user pushes.
* Stay within the evaluation protocols below: **TEST seeds (8500–8599) are used once per final table.** Any exploratory analysis uses VAL seeds (8000–8049) or fresh seeds ≥ 9000 (declare them in the report). Never tune on TEST.
* If a result contradicts an earlier phase's claim, report the contradiction prominently — do not hide or "fix" it by changing the protocol.
* Do not change model weights, the feature schema or engine behaviour. Bug fixes are allowed only if they have a regression test, are listed in the report, and **do not** invalidate earlier results (if they do, re-run the affected tables and say so).
* Python 3.12 / `uv`; `ruff` clean; JS via `pnpm`. Write a plan first.

---

## 2. Final evaluation protocol (pre-registered — implement exactly)

### 2.1 Strategies compared

`nearest_car` (reflex baseline) · `collective` (LOOK, strong classical) · `cnp_astar` (Contract Net + A*) · `full` (the v1 champion: + SA, parking, adaptive weights; **the teacher**) · `liftzero_bc` (imitation) · `liftzero_ppo` (RL) · `liftzero_mcts` (RL + look-ahead). (Also `liftzero_bc_mcts` in the ablation only.)

### 2.2 Experiments (all in the real simulator; multiprocessing; deterministic seeds)

| ID | What | Seeds | Ticks |
| --- | --- | --- | --- |
| E1 | **Four regimes** (`up_peak, down_peak, two_way, interfloor`; 15 floors × 4 cars) × 7 strategies | TEST 8500–8599 (100) | 900 |
| E2 | **Nine scenario YAMLs** × 7 strategies | 20 seeds (9000–9019) | scenario duration |
| E3 | **Scaling**: building sizes 6×2, 10×3, 15×4, 25×6, 40×8, 60×12, 80×16 at constant relative load (`rate = 0.04·cars·u`, `u = 1.2`), × {`nearest_car, collective, full, liftzero_ppo, liftzero_mcts`} | 20 seeds each | 900 |
| E4 | **Load sweep**: `u ∈ {0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2}` on 15×4 two-way, × the same 5 strategies | 30 seeds | 900 |
| E5 | **Distribution shift / robustness**: (a) rate ×0.5 and ×1.5 relative to training mid-range; (b) unseen capacity 4 and 20; (c) `seconds_per_floor` ∈ {1, 4}; (d) more cars than any training run (10, 12 on 25 floors); (e) car fault at t=300 repaired at t=600; (f) fire alarm at t=400 cleared at t=500; (g) a 30-person rush at the lobby at t=200; (h) priority passengers 10 % | 30 seeds each | 900 |
| E6 | **Safety soak**: ≥ 1 000 randomised runs per learned strategy (random building, traffic, disturbances) checking `model.violations()` **every tick**, delivery after drain, and that ineligible cars never win | seeds 20000+ | 600–1800 |
| E7 | **Determinism**: 20 runs run twice (separate processes) ⇒ bit-identical `Metrics` and message counts, for every strategy | — | 900 |
| E8 | **Latency & compute**: per-decision latency (net / search / total; median, p95, p99), `compute_ms_per_tick`, wall-clock per simulated hour per strategy, memory (RSS) — on the machine used; state CPU model | 10 seeds | 3600 |

### 2.3 Metrics and statistics

Metrics: average wait, p95 wait, max wait, long-wait % (> 60 s), average ride, average system time, throughput (passengers/h), energy, delivered/arrived, messages per call, `nodes_expanded`, decision latency, **fairness**: Gini coefficient of per-car served passengers and the standard deviation of waits per floor.

For every comparison that appears in the report: **paired differences on identical seeds**, 95 % bootstrap CI (10 000 resamples), Wilcoxon signed-rank p-value, win-rate, Cohen's d_z; **Holm–Bonferroni** across the 4 regimes (E1) and across scenarios (E2) for each headline claim. Report effect sizes in **percent** relative to the reference strategy, always with the CI. Mark cells where the strategy is **worse** in red and say so in the text.

### 2.4 Headline claims (pre-registered, reported pass/fail)

* **H1** — `liftzero_ppo` or `liftzero_mcts` has lower mean avg-wait than `full` with Holm-adjusted p < 0.05 in ≥ 2 of 4 regimes (E1) with p95 and long-wait % not worse by > 10 %.
* **H2** — `full` (the classical champion) beats `nearest_car` in avg wait in ≥ 3 of 4 regimes (reproduces v1's claim under the new protocol; v1 documented one loss to `collective` in down-peak — report whether it persists).
* **H3** — Zero safety violations across E1–E7 for every strategy.
* **H4** — Scaling: LiftZero's per-decision latency grows sub-quadratically with fleet size and stays ≤ 50 ms (search) / ≤ 5 ms (net) up to 16 cars; performance does not collapse relative to `full` at sizes larger than training (report the size at which it degrades, if any).
* **H5** — Determinism holds (E7).

---

## 3. Ablations (A1–A10, on VAL seeds, 30 seeds, the four regimes, paired vs the full LiftZero pipeline)

| ID | Removed / varied | Question answered |
| --- | --- | --- |
| A1 | No imitation warm start (PPO from scratch, same budget) | does imitation help RL? |
| A2 | Imitation only (`liftzero_bc`) vs + DAgger rounds 0/1/2 | what does DAgger buy? |
| A3 | PPO without the KL anchor to the teacher (β = 0) | is the anchor needed for stability? |
| A4 | PPO with symmetric critic (no privileged info) | does CTDE asymmetry help? |
| A5 | No MCTS vs sims ∈ {4, 8, 16, 32, 64, 128} | anytime curve / latency–quality Pareto |
| A6 | MCTS leaf = rollout vs value; CRN on/off; margin gate on/off; horizon D=1/2/3 | which search ingredients matter? |
| A7 | Feature-group knock-out at *training* time (retrain smoke-size models without: doors / plan / load / pattern / call-relation) | which information does the policy need? |
| A8 | Model size: d ∈ {32, 64, 128}, layers ∈ {2, 4, 6} (smoke budget) | is 0.27 M the right size? |
| A9 | Training data size: 5 %, 20 %, 50 %, 100 % of expert decisions (learning curve) | data efficiency |
| A10 | Domain randomisation off (train only on 15×4) vs on, tested on E3 scaling | does randomisation buy generalisation? |

Budgets: use the `smoke`/`default` presets from Phases 4–5; if a full-budget retrain of an ablation is infeasible on the user's machine, run it at 20 % budget **for all arms equally** and say so. Every ablation table has the full pipeline as reference and CIs. Provide a one-paragraph takeaway per ablation, including "no measurable effect" results.

---

## 4. Explainability & failure analysis (required sections)

* **Where LiftZero loses**: list every (regime, metric) where it is worse than `full`/`collective`/`nearest_car` with CI excluding 0; for each, sample the 5 worst seeds, replay them, and explain (with decision traces from `AuctionRound.reason`, Brain logs, screenshots) *why*. Candidate causes to test explicitly: down-peak lobby congestion, car bunching, door-time model mismatch, stale `waiting` counts, imitation bias on near-ties, twin–real routing mismatch.
* **What the policy learned**: bid-landscape plots (score vs distance/load/plan_end_eta) for BC, PPO and teacher; attribute groups by occlusion averaged over 5 000 decisions per regime (bar chart); a "did it rediscover X" table: *prefers cars already heading toward the call* (`call_on_route`), *avoids full cars*, *spreads cars in down-peak*, *keeps a car near the lobby in up-peak* — each measured as a statistic over decisions, not an anecdote.
* **Decision vignettes**: 6 annotated real decisions (3 where LiftZero beats the teacher, 3 where it loses) with the board state drawn, bids, trace, and counterfactual outcomes.
* **Limitations section**: what is still classical (A* routing, eligibility, SA reassignment, parking), what is learned (the bid), training-distribution limits, twin assumptions, public-information-only features, compute of search, anything that did not replicate.

---

## 5. Deliverable documents (all in `docs/`, cross-linked, no stale numbers)

1. **`docs/REPORT.md`** — the technical report (≈ 12–18 pages when rendered): abstract; problem & PEAS; environment properties; architecture (agents, protocol, status board, safety); classical algorithms (A*, Contract Net, SA, hill climbing, minimax/alpha-beta, forward-chaining rules, EWMA); LiftZero (features, network, imitation, DAgger, PPO/CTDE, MCTS); experimental setup (protocol §2); results (E1–E8) with figures and tables; ablations; failure analysis; limitations; ethics/safety notes; reproducibility; references (AIMA 4e chapters: 2, 3, 4, 5, 6/7/9 for rules, 11/17 for planning/decision, 22–23 for learning/RL, plus the original papers for Contract Net (Smith 1980), FIPA-ACL, PPO (Schulman et al. 2017), GAE (2016), AlphaZero (Silver et al. 2018), PUCT, DAgger (Ross et al. 2011), Set Transformer (Lee et al. 2019), MuZero normalisation). Generate figures with matplotlib in a **consistent, accessible palette** (colour-blind safe, readable in grayscale; 200 dpi; saved to `reports/final/`). All figures reproducible from `scripts/make_figures.py`.
2. **`docs/RUBRIC_MAP.md`** — a table mapping **every rubric line** (all 8 above) to concrete evidence: file/section, command to demonstrate it live, screenshot, test names. Include a "30-second answer" for each. This is the examiners' index.
3. **`docs/DESIGN.md`** — updated end to end (architecture diagram incl. status board and LiftZero; PEAS for every agent incl. the Brain; environment classification with justifications; agent-type ladder showing which agent is simple-reflex / model-based / goal / utility / **learning** and why; algorithm cards; message protocol; scalability section with the E3 numbers).
4. **`docs/TESTING.md`** — all layers (unit, property, integration, API, golden, safety soak, determinism, fidelity, UI unit/e2e/a11y), counts, coverage, how to run each, flakiness policy, bugs found by testing (carry forward the v1 table, add v2's).
5. **`docs/VIVA_QA.md`** — **≥ 60** Q&As in 8 groups (problem & PEAS; environment; agents & multi-agent justification; search & optimisation; learning (imitation/RL/CTDE/MCTS); testing & evaluation; scalability & engineering; limitations & ethics), each answer 2–5 sentences with the pointer to evidence; include "trick" questions (*"isn't this just a central controller?"*, *"why does the learned policy not beat the baseline in X?"*, *"how do you know it didn't overfit to the simulator?"*, *"is attention an explanation?"*, *"what happens if the model file is missing?"*, *"why a Transformer for 4 cars?"*).
6. **`docs/DEMO_SCRIPT.md`** — a minute-by-minute **7-minute** demo with exact clicks/keys, what to say, what the audience should notice, and a fallback for each step if something misbehaves; plus a 2-minute short version. Matches Story mode's beats.
7. **`docs/PRESENTATION_OUTLINE.md`** — a 14-slide outline (title, problem, why multi-agent, PEAS, environment, architecture, classical algorithms, LiftZero idea, training pipeline, results, demo, testing, limitations/future, Q&A) with speaker notes per slide, the exact figure files to use from `reports/final/`, and a 4-person speaking split (who covers which slides, matching the work split in `docs/DESIGN.md §9`).
8. **`README.md`** — rewrite: one-paragraph pitch, screenshot, 3-command quick start (`uv sync`, `uv run elevator serve`, open the browser), feature list, project layout, how to run tests, how to reproduce results, how to retrain the model (pointers), troubleshooting, license/credits placeholder.
9. **`docs/CHANGELOG.md`** (v1.0 → v2.0, per phase) and a **`docs/ARCHITECTURE_DECISIONS.md`** (ADR-style, one page per decision: why Mesa, message bus + status board, A* routing kept classical, set-Transformer, imitation→RL→search ordering, twin simulator, ONNX runtime, what we did *not* do and why).
10. **`AGENTS.md`** — refresh contributor guidance for the new structure (don't remove the existing rules).

Style: plain, precise, no marketing words, numbers with units and CIs, every table caption states the seeds and the command.

---

## 6. Quality hardening

* **Static quality**: `ruff check` and `ruff format --check` clean; add and pass `pyright` (basic mode) or `mypy` on `src/elevator_mas` (configure in `pyproject.toml`; fix real typing bugs, allow `# type: ignore[code]` only with a comment); `pnpm typecheck && pnpm lint`.
* **Coverage**: report per-package coverage; core engine ≥ 90 %, learning ≥ 85 %, API ≥ 85 %; list uncovered critical branches and either test them or justify.
* **CI**: add `.github/workflows/ci.yml` (matrix: ubuntu-latest, Python 3.12): jobs `lint`, `test` (`uv run pytest -m "not slow"`), `ui` (pnpm install, typecheck, lint, vitest), `e2e` (Playwright Chromium against the built app), and a nightly-style `slow` job. Cache `uv` and `pnpm`. The workflow must pass locally with `act`-like reasoning (validate YAML with `yamllint`/schema; you cannot push, so also provide `scripts/ci_local.sh` that runs the same steps).
* **Fresh-clone smoke test** (`scripts/fresh_clone_check.sh`): in a temp dir, `git clone` the repo from the local path → `uv sync` (no dev groups, no Node) → `uv run elevator verify` → start `elevator serve` on a free port → `curl /api/health`, `/`, `/classic`, `/api/brain` → run a 120-tick scenario via `/api/run` for each learned strategy → shut down. Must pass **with network blocked after `uv sync`** (use `unshare -n` or an equivalent where available; otherwise document the manual offline test and run it once in the browser DevTools offline mode).
* **Repository hygiene**: size report (largest files; the repo without `.git` should stay < 25 MB incl. `src/elevator_mas/web/dist` and `models/*.onnx`); no stray `data/`, `runs/`, `node_modules`, `__pycache__`; `.gitignore` correct (the bundle and models ARE tracked; `data/`, `runs/` are not); licences of dependencies listed in `docs/THIRD_PARTY.md` (auto-generate with `pip-licenses` / `license-checker`, review for copyleft surprises).
* **Performance profile**: `py-spy`/`cProfile` of a 900-tick `full` and `liftzero_mcts` run; top-10 hotspots table; fix only trivial wins with tests (≤ 10 % effort), document the rest.
* **Scalability evidence** (E3) drawn as a plot: wall-clock per simulated hour and per-decision latency vs fleet size, for each strategy.
* **Robustness to missing artefacts**: delete/rename `models/*.onnx` → server still starts, learned strategies show `available:false` with a reason in `/api/meta` and the Brain panel, classical strategies unaffected (tests).
* **Accessibility & browser**: Lighthouse a11y ≥ 90 on Mission Control and Theory (record scores); verify at 1280×720, 1440×900, 1920×1080; test in Chromium and Firefox (Playwright) for the golden path.

---

## 7. Demo rehearsal automation

* `scripts/demo_check.py`: starts the server on a free port, drives the **Story mode flow** headlessly through the API (reset → rush → fault → fire → compare), asserts each beat's condition occurs within its timeout, prints a PASS/FAIL table with timings, and exits non-zero on failure. Run it 20 times (different ports/seeds where applicable) and report the pass rate; investigate any flake and fix its root cause (not by lengthening timeouts blindly).
* A **"safe demo preset"**: `uv run elevator serve --scenario demo_story --seed 7` must produce the same story every time (document the seed in the demo script) and the dist bundle must be present. Provide `scripts/pre_demo.sh` that checks: `uv`, Python, port free, models present, bundle present, tests-smoke (30 s), and prints a green/red checklist.
* Record **fallback assets** so the demo survives a laptop failure: a set of PNG screenshots + a 90-second screen-capture description (do not attempt video if you cannot record; instead produce the Playwright screenshot sequence `docs/img/demo/NN_*.png` and a `docs/DEMO_FALLBACK.md` slide-style walkthrough).

---

## 8. Release procedure

1. Ensure `v2` contains everything (merge any unmerged phase branches the user lists; resolve conflicts carefully — expected only in docs; re-run the full test suite after each merge).
2. Run the complete verification: `scripts/ci_local.sh`, `scripts/fresh_clone_check.sh`, `scripts/demo_check.py` (20×), the E1–E8 evaluation (or load the cached results with their hashes), figure generation.
3. Update `pyproject.toml` version to `2.0.0`, `docs/CHANGELOG.md`, the `/api/version` payload.
4. Commit on `phase-7-release`; merge into `v2` (`--ff-only` if possible); then merge `v2` into `main` with `git merge --no-ff v2 -m "Release v2.0.0"`; create an **annotated tag** `v2.0` (`git tag -a v2.0 -m "LiftZero: learned multi-agent bidding with look-ahead"`). Keep tag `v1-stable` untouched. **Do not push; do not delete branches.** Print the exact push commands for the user (`git push origin main v2 --tags`) in the final report, and warn that the `.gitignore`d artefacts (`data/`, `runs/`) are intentionally not in the repo.
5. After tagging, run the fresh-clone check once more **from the tag**.

---

## 9. Evidence package (what the final report must contain)

* The pre-registered claim table H1–H5 with PASS/FAIL and the numbers.
* Result tables E1–E8 (compact versions; full versions linked) with CIs; figure list.
* Ablation takeaways A1–A10 (one line each) with the link to full tables.
* Test inventory by layer with counts; coverage by package; CI/local CI result; fresh-clone check log; demo-check pass rate.
* Repository stats (size, LOC by package, #tests), dependency list + licences summary.
* Deviations from this prompt (and why), anything that failed or did not replicate, open risks for the demo.
* The exact commands for the user to verify and to present:

```
git checkout main && git log --oneline -5 && git tag -l
uv sync
uv run pytest -m "not slow"
uv run elevator verify
bash scripts/pre_demo.sh
uv run elevator serve --scenario demo_story --seed 7        # open http://127.0.0.1:8000 → Present
```

---

## 10. Acceptance checklist

- [ ] E1–E8 executed per the protocol; TEST seeds used once; every table has paired stats; H1–H5 reported pass/fail honestly.
- [ ] Ablations A1–A10 done (equal budgets stated), with takeaways.
- [ ] Failure analysis + learned-behaviour analysis + 6 vignettes + limitations written.
- [ ] `docs/REPORT.md`, `RUBRIC_MAP.md`, `DESIGN.md`, `TESTING.md`, `VIVA_QA.md` (≥ 60 Q&As), `DEMO_SCRIPT.md`, `PRESENTATION_OUTLINE.md`, `README.md`, `CHANGELOG.md`, `ARCHITECTURE_DECISIONS.md`, `AGENTS.md`, `THIRD_PARTY.md` complete and mutually consistent (no stale counts — grep for old numbers like "182" and fix).
- [ ] Figures reproducible via `scripts/make_figures.py`; accessible palette.
- [ ] ruff + type checker + frontend lint/typecheck clean; coverage targets met; CI workflow + `ci_local.sh` pass.
- [ ] Fresh-clone offline check passes (from the `v2.0` tag); missing-model robustness tests pass.
- [ ] `demo_check.py` passes 20/20 (or flakes root-caused); `pre_demo.sh` works; fallback screenshots committed.
- [ ] Repo < 25 MB (excl. `.git`), hygiene verified; version 2.0.0; `main` merged `--no-ff`; annotated tag `v2.0`; **nothing pushed**.

## 11. Final report format

One page of headline results (H1–H5 + the single most important table), then the evidence package, deviations and risks, then
the verification/presentation commands above and the push commands for the user.
