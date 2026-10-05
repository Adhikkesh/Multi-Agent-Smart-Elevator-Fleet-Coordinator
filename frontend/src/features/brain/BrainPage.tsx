import { BrainCircuit, Play, Search, Sparkles } from "lucide-react";
import React, { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";
import type { BrainDecision, BrainResponse, ExplainResponse } from "../../api/schemas";

const POLL_MS = 1000;

function fmt(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined || !Number.isFinite(x) ? "—" : x.toFixed(digits);
}

const Stat: React.FC<{ label: string; value: string; hint?: string }> = ({
  label,
  value,
  hint,
}) => (
  <div className="rounded-xl border border-border bg-card p-3 shadow-sm">
    <div className="text-xs text-muted-foreground">{label}</div>
    <div className="mt-1 font-mono text-xl font-semibold text-foreground">{value}</div>
    {hint ? <div className="mt-0.5 text-[11px] text-muted-foreground">{hint}</div> : null}
  </div>
);

/** A thin horizontal bar; `value` in [0, 1]. Identity is always also shown as text. */
const Bar: React.FC<{ value: number; tone?: "primary" | "muted" }> = ({
  value,
  tone = "primary",
}) => (
  <div className="h-2 w-full rounded-full bg-muted" aria-hidden="true">
    <div
      className={`h-2 rounded-full ${tone === "primary" ? "bg-primary" : "bg-muted-foreground/60"}`}
      style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }}
    />
  </div>
);

export const DecisionDetail: React.FC<{
  decision: BrainDecision;
  explanation: ExplainResponse | null;
  onExplain: () => void;
  explaining: boolean;
}> = ({ decision, explanation, onExplain, explaining }) => {
  const search = decision.search;
  const proposers = decision.candidates.filter((c) => !c.refused);
  const maxScore = Math.max(1e-6, ...proposers.map((c) => c.net_score ?? 0));
  const totalVisits = search
    ? Math.max(
        1,
        search.visits.reduce((a, b) => a + b, 0),
      )
    : 1;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold text-foreground">
          Call {decision.call.floor}
          {decision.call.direction === "UP" ? "↑" : "↓"} at t={decision.tick}s
        </span>
        <span className="rounded-full bg-secondary px-2 py-0.5 text-xs">
          awarded to car {decision.chosen}
        </span>
        <span className="rounded-full bg-secondary px-2 py-0.5 text-xs">
          network choice car {decision.net_choice ?? "—"}
        </span>
        {decision.teacher_choice !== null ? (
          <span className="rounded-full bg-secondary px-2 py-0.5 text-xs">
            A* teacher car {decision.teacher_choice}
          </span>
        ) : null}
        {search?.overridden ? (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800 dark:bg-amber-900/40 dark:text-amber-200">
            look-ahead override
          </span>
        ) : null}
      </div>

      <table className="w-full text-xs" aria-label="Candidate cars">
        <thead className="text-left text-muted-foreground">
          <tr>
            <th className="py-1 pr-2">Car</th>
            <th className="py-1 pr-2">Bid</th>
            <th className="py-1 pr-2 w-1/4">Network log-cost (lower wins)</th>
            <th className="py-1 pr-2 w-1/5">Attention</th>
            {search ? <th className="py-1 pr-2 w-1/5">Search visits</th> : null}
            {search ? <th className="py-1 pr-2">Q</th> : null}
          </tr>
        </thead>
        <tbody>
          {decision.candidates.map((c) => {
            const k = search ? search.candidates.indexOf(c.car_id) : -1;
            return (
              <tr key={c.car_id} className="border-t border-border">
                <td className="py-1.5 pr-2 font-mono">
                  {c.car_id}
                  {c.car_id === decision.chosen ? " ★" : ""}
                </td>
                <td className="py-1.5 pr-2 font-mono">
                  {c.refused ? `refused (${c.reason ?? "?"})` : fmt(c.bid)}
                </td>
                <td className="py-1.5 pr-2">
                  {c.refused ? null : (
                    <div className="flex items-center gap-2">
                      <Bar value={(c.net_score ?? 0) / maxScore} tone="muted" />
                      <span className="font-mono">{fmt(c.net_score, 2)}</span>
                    </div>
                  )}
                </td>
                <td className="py-1.5 pr-2">
                  {c.attention !== null ? (
                    <div className="flex items-center gap-2">
                      <Bar value={c.attention} />
                      <span className="font-mono">{fmt(c.attention * 100, 0)}%</span>
                    </div>
                  ) : (
                    "—"
                  )}
                </td>
                {search ? (
                  <td className="py-1.5 pr-2">
                    {k >= 0 ? (
                      <div className="flex items-center gap-2">
                        <Bar value={(search.visits[k] ?? 0) / totalVisits} />
                        <span className="font-mono">{search.visits[k]}</span>
                      </div>
                    ) : (
                      "—"
                    )}
                  </td>
                ) : null}
                {search ? (
                  <td className="py-1.5 pr-2 font-mono">{k >= 0 ? fmt(search.q[k], 2) : "—"}</td>
                ) : null}
              </tr>
            );
          })}
        </tbody>
      </table>

      {search ? (
        <p className="text-xs text-muted-foreground">
          PUCT look-ahead: {search.sims_done} simulations to depth {search.depth_max} in{" "}
          {fmt(search.time_ms, 0)} ms ({search.leaf} leaves). Q = negative team cost over the next
          minute of sampled futures.
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">
          Not searched: the network's choice was clear (or the strategy has no look-ahead).
        </p>
      )}

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onExplain}
          disabled={explaining}
          className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-medium hover:bg-secondary disabled:opacity-50"
        >
          <Search className="h-3.5 w-3.5" /> Why? (occlusion)
        </button>
      </div>
      {explanation ? (
        <table className="w-full text-xs" aria-label="Feature-group importance">
          <thead className="text-left text-muted-foreground">
            <tr>
              <th className="py-1 pr-2">Hidden feature group</th>
              <th className="py-1 pr-2">Δ score of chosen car</th>
              <th className="py-1 pr-2">Δ runner-up</th>
              <th className="py-1 pr-2">Flips decision?</th>
            </tr>
          </thead>
          <tbody>
            {explanation.groups.map((g) => (
              <tr key={g.group} className="border-t border-border">
                <td className="py-1.5 pr-2">{g.group.replace("_", " ")}</td>
                <td className="py-1.5 pr-2 font-mono">{fmt(g.delta_score_chosen, 3)}</td>
                <td className="py-1.5 pr-2 font-mono">{fmt(g.delta_score_runner_up, 3)}</td>
                <td className="py-1.5 pr-2">{g.flips_decision ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </div>
  );
};

export const BrainPage: React.FC = () => {
  const [brain, setBrain] = useState<BrainResponse | null>(null);
  const [decisions, setDecisions] = useState<BrainDecision[]>([]);
  const [selected, setSelected] = useState<string | number | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sims, setSims] = useState(32);
  const [budget, setBudget] = useState(50);

  const refresh = useCallback(async () => {
    try {
      const b = await api.brain();
      setBrain(b);
      setDecisions(b.available ? await api.brainDecisions(50) : []);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const current = decisions.find((d) => d.conversation_id === selected) ?? decisions[0] ?? null;

  const switchTo = async (strategy: string) => {
    await api.reset({ strategy });
    setSelected(null);
    setExplanation(null);
    await refresh();
  };

  const applySearch = async () => {
    await api.setBrainConfig({ sims, time_budget_ms: budget });
    await refresh();
  };

  const explain = async () => {
    if (!current) return;
    setExplaining(true);
    try {
      setExplanation(await api.explain(current.conversation_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExplaining(false);
    }
  };

  const stats = brain?.stats;
  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <BrainCircuit className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold text-foreground">LiftZero Brain</h1>
          {brain?.active.model ? (
            <span className="text-xs text-muted-foreground">
              {brain.active.model.name} v{brain.active.model.version} ·{" "}
              {brain.active.model.param_count.toLocaleString()} parameters
            </span>
          ) : null}
        </div>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Learned strategies">
          {brain?.strategies.map((s) => (
            <button
              key={s.name}
              type="button"
              disabled={!s.available}
              title={s.available ? s.label : s.reason}
              onClick={() => void switchTo(s.name)}
              className={`rounded-lg border px-3 py-1.5 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40 ${
                brain.active.strategy === s.name
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-border bg-background hover:bg-secondary"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {error ? (
        <div
          role="alert"
          className="rounded-lg border border-red-300 bg-red-50 p-2 text-xs text-red-800"
        >
          {error}
        </div>
      ) : null}

      {brain && !brain.available ? (
        <div className="rounded-xl border border-border bg-card p-6 text-sm text-muted-foreground">
          <Sparkles className="mb-2 h-5 w-5 text-primary" />
          The running strategy ({brain.active.label}) bids classically. Choose a LiftZero strategy
          above to watch the network decide, live.
        </div>
      ) : null}

      {brain?.available ? (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Stat label="Learned decisions" value={String(stats?.decisions ?? 0)} />
            <Stat label="Award = network choice" value={`${fmt(stats?.agree_net_pct)}%`} />
            <Stat
              label="Agreement with A* teacher"
              value={stats?.agree_teacher_pct == null ? "—" : `${fmt(stats.agree_teacher_pct)}%`}
              hint="needs the shadow teacher"
            />
            <Stat
              label="Searched / overridden"
              value={`${fmt(stats?.searched_pct, 0)}% / ${fmt(stats?.overridden_pct, 0)}%`}
            />
            <Stat label="Search time p95" value={`${fmt(stats?.search_ms_p95, 0)} ms`} />
          </div>

          {brain.active.search ? (
            <div className="flex flex-wrap items-end gap-4 rounded-xl border border-border bg-card p-3 text-xs">
              <label className="flex flex-col gap-1">
                Simulations per decision: <span className="font-mono">{sims}</span>
                <input
                  type="range"
                  min={0}
                  max={128}
                  value={sims}
                  onChange={(e) => setSims(Number(e.target.value))}
                />
              </label>
              <label className="flex flex-col gap-1">
                Time budget: <span className="font-mono">{budget} ms</span>
                <input
                  type="range"
                  min={10}
                  max={200}
                  step={10}
                  value={budget}
                  onChange={(e) => setBudget(Number(e.target.value))}
                />
              </label>
              <button
                type="button"
                onClick={() => void applySearch()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 font-medium text-primary-foreground"
              >
                <Play className="h-3.5 w-3.5" /> Apply
              </button>
              <span className="text-muted-foreground">
                Live: {brain.active.search.sims} sims, {brain.active.search.time_budget_ms} ms,
                horizon {brain.active.search.horizon} s, {brain.active.search.leaf} leaves
              </span>
            </div>
          ) : null}

          <div className="grid min-h-0 grid-cols-1 gap-4 lg:grid-cols-[18rem_1fr]">
            <ul
              className="max-h-[32rem] overflow-y-auto rounded-xl border border-border bg-card p-2"
              aria-label="Recent decisions"
            >
              {decisions.length === 0 ? (
                <li className="p-2 text-xs text-muted-foreground">
                  No decisions yet — press play on Mission Control.
                </li>
              ) : null}
              {decisions.map((d) => (
                <li key={String(d.conversation_id)}>
                  <button
                    type="button"
                    onClick={() => {
                      setSelected(d.conversation_id);
                      setExplanation(null);
                    }}
                    className={`flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left text-xs ${
                      current?.conversation_id === d.conversation_id
                        ? "bg-secondary"
                        : "hover:bg-secondary/60"
                    }`}
                  >
                    <span className="font-mono">
                      t={d.tick} · {d.call.floor}
                      {d.call.direction === "UP" ? "↑" : "↓"} → car {d.chosen}
                    </span>
                    <span className="text-muted-foreground">
                      {d.search ? (d.search.overridden ? "override" : "searched") : "net"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            <div className="rounded-xl border border-border bg-card p-4">
              {current ? (
                <DecisionDetail
                  decision={current}
                  explanation={explanation}
                  onExplain={() => void explain()}
                  explaining={explaining}
                />
              ) : (
                <p className="text-sm text-muted-foreground">Select a decision.</p>
              )}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
