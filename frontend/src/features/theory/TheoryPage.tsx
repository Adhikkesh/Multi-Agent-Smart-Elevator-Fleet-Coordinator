import { useQuery } from "@tanstack/react-query";
import { ChevronDown } from "lucide-react";
import { cleanDocstring } from "../../lib/docstring";
import React, { useState } from "react";
import { api } from "../../api/client";
import { PERFORMATIVE_COLORS } from "../../lib/colors";

export const TheoryPage: React.FC = () => {
  const { data: meta } = useQuery({
    queryKey: ["meta"],
    queryFn: () => api.getMeta(),
  });

  const [activeAgentTab, setActiveAgentTab] = useState(0);
  const [openVivaIdx, setOpenVivaIdx] = useState<number | null>(null);

  const vivaQuestions = [
    {
      q: "1. Why is this environment partially observable?",
      a: "A passenger's destination floor is completely private until they board and press a car button. The dispatcher must commit a car without knowing where that passenger will travel.",
    },
    {
      q: "2. Why is the Contract Net Protocol (CNP) preferred over centralized dispatch?",
      a: "Centralized controllers create a single point of failure and fail to scale. CNP distributes route sequencing and valuation to the individual car agents, allowing local autonomy and fault resilience.",
    },
    {
      q: "3. What makes the routing heuristic admissible and consistent?",
      a: "h(n) is the sum of two disjoint lower bounds: weighted passenger travel (straight-line distance to each stop) and the minimum energy line span. Neither can overestimate true cost, satisfying admissibility and the triangle inequality.",
    },
    {
      q: "4. What is the agent type of each agent in this system?",
      a: "Passenger: Simple Reflex; Floor: Model-based Reflex; Elevator: Utility-based; Dispatcher: Goal/Utility-based auctioneer; Traffic Monitor: Learning Agent; Safety: Knowledge-based Agent.",
    },
    {
      q: "5. How does Simulated Annealing avoid getting trapped in local minima?",
      a: "Unlike Hill Climbing which only accepts strictly improving moves, Simulated Annealing probabilistically accepts worse assignments with probability exp(-ΔE / T), allowing it to escape suboptimal basins.",
    },
    {
      q: "6. Why formulate elevator parking as a minimax game?",
      a: "During calm traffic, demand arrival locations are unpredictable. Minimax treats nature/passengers as an adversary placing requests at the worst possible floors, minimizing maximum pickup latency.",
    },
    {
      q: "7. What pruning does Alpha-Beta achieve in the parking game?",
      a: "Alpha-beta pruning eliminates branches where the minimum score already guaranteed to MIN is worse than the current MAX alternative, reducing evaluated nodes by up to 50% without altering the optimal choice.",
    },
    {
      q: "8. How does the Safety Agent override autonomous car plans?",
      a: "Safety rules (R1-R7) operate via forward-chaining production rules with high salience. In fire or overload events, R1 issues mandatory FIRE_RECALL orders that cancel all car goals.",
    },
    {
      q: "9. How does the Traffic Monitor learn?",
      a: "It computes exponentially weighted moving averages (EWMA) of trip origins and inter-floor traffic, dynamically tuning the dispatch utility weights (W1-W4) to prioritize wait time vs ride time.",
    },
    {
      q: "10. What is collective control in elevator engineering?",
      a: "Cars collect passengers travelling in their current direction without taking passengers backwards past their destinations, sweeping up and down efficiently.",
    },
    {
      q: "11. How does the blackboard/shared status board ensure agent privacy?",
      a: "Cars publish only public facts (position, load, availability, commitments) to the status board. No agent reads or writes another agent's private search state or memory.",
    },
    {
      q: "12. What does LiftZero (Phase 6) add over classical AI?",
      a: "It uses deep neural networks trained on expert A* search demonstrations to replace hand-crafted heuristics with learned valuation functions and MCTS tree search.",
    },
  ];

  return (
    <div className="flex h-full flex-col overflow-y-auto p-4 space-y-6">
      {/* Title */}
      <div className="border-b border-border pb-3">
        <h1 className="text-xl font-bold text-foreground">
          Fundamentals of AI (AIMA 4e) Theoretical Reference
        </h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Curated viva cheat-sheet, environment analysis, PEAS tables, algorithm proofs, and
          multi-agent protocols.
        </p>
      </div>

      {/* 1. System PEAS */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold uppercase tracking-wider text-primary">
          1. System-Level PEAS Formulation
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="rounded-lg border border-border bg-card p-3 shadow-xs">
            <span className="font-bold text-primary uppercase text-[10px]">Performance (P)</span>
            <p className="mt-1 text-muted-foreground">{meta?.system_peas.performance}</p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3 shadow-xs">
            <span className="font-bold text-primary uppercase text-[10px]">Environment (E)</span>
            <p className="mt-1 text-muted-foreground">{meta?.system_peas.environment}</p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3 shadow-xs">
            <span className="font-bold text-primary uppercase text-[10px]">Actuators (A)</span>
            <p className="mt-1 text-muted-foreground">{meta?.system_peas.actuators}</p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3 shadow-xs">
            <span className="font-bold text-primary uppercase text-[10px]">Sensors (S)</span>
            <p className="mt-1 text-muted-foreground">{meta?.system_peas.sensors}</p>
          </div>
        </div>
      </section>

      {/* 2. Individual Agents PEAS Tabs */}
      <section className="space-y-3">
        <h2 className="text-sm font-bold uppercase tracking-wider text-primary">
          2. Agent PEAS Specifications
        </h2>
        <div className="flex flex-wrap gap-1.5 border-b border-border pb-2">
          {meta?.agents.map((agent, idx) => (
            <button
              key={agent.name}
              onClick={() => setActiveAgentTab(idx)}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                activeAgentTab === idx
                  ? "bg-primary text-primary-foreground shadow-xs"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              }`}
            >
              {agent.name}
            </button>
          ))}
        </div>

        {meta?.agents[activeAgentTab] && (
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm space-y-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-mono text-sm font-bold text-foreground">
                {meta.agents[activeAgentTab]?.name}
              </span>
              <span className="rounded bg-primary/10 px-2 py-0.5 font-semibold text-primary uppercase text-[10px]">
                {meta.agents[activeAgentTab]?.agent_type}
              </span>
            </div>
            <p className="text-muted-foreground italic">
              {cleanDocstring(meta.agents[activeAgentTab]?.docstring ?? "")}
            </p>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 pt-2">
              <div className="rounded border border-border/60 bg-muted/20 p-2.5">
                <span className="font-bold text-primary text-[10px] uppercase">Performance</span>
                <p className="mt-1 text-muted-foreground">
                  {meta.agents[activeAgentTab]?.peas.performance}
                </p>
              </div>
              <div className="rounded border border-border/60 bg-muted/20 p-2.5">
                <span className="font-bold text-primary text-[10px] uppercase">Environment</span>
                <p className="mt-1 text-muted-foreground">
                  {meta.agents[activeAgentTab]?.peas.environment}
                </p>
              </div>
              <div className="rounded border border-border/60 bg-muted/20 p-2.5">
                <span className="font-bold text-primary text-[10px] uppercase">Actuators</span>
                <p className="mt-1 text-muted-foreground">
                  {meta.agents[activeAgentTab]?.peas.actuators}
                </p>
              </div>
              <div className="rounded border border-border/60 bg-muted/20 p-2.5">
                <span className="font-bold text-primary text-[10px] uppercase">Sensors</span>
                <p className="mt-1 text-muted-foreground">
                  {meta.agents[activeAgentTab]?.peas.sensors}
                </p>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* 3. Environment Classification */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold uppercase tracking-wider text-primary">
          3. Environment Classification (AIMA 4e Table 2.1)
        </h2>
        <div className="overflow-x-auto rounded-lg border border-border/60 bg-card">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border bg-muted/40 font-semibold text-foreground">
              <tr>
                <th className="p-2.5">Property</th>
                <th className="p-2.5">Classification</th>
                <th className="p-2.5">System Justification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {meta?.environment.map((env) => (
                <tr key={env.property} className="hover:bg-muted/20">
                  <td className="p-2.5 font-semibold text-foreground">{env.property}</td>
                  <td className="p-2.5 font-mono text-primary font-bold">{env.value}</td>
                  <td className="p-2.5 text-muted-foreground">{env.justification}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 4. Why Multi-Agent */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold uppercase tracking-wider text-primary">
          4. Why Multi-Agent Systems (MAS)?
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div className="rounded-lg border border-border bg-card p-3 shadow-xs">
            <span className="font-bold text-foreground">Distributed Sensing & Control</span>
            <p className="mt-1 text-muted-foreground">
              Sensors on floors and encoders in cars process local state independently without
              bottlenecking a centralized server.
            </p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3 shadow-xs">
            <span className="font-bold text-foreground">Fault Tolerance & Self-Healing</span>
            <p className="mt-1 text-muted-foreground">
              If car 1 breaks down, the fleet immediately re-allocates its orphaned commitments via
              instant Contract Net re-auctioning.
            </p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3 shadow-xs">
            <span className="font-bold text-foreground">Scalability & Modularity</span>
            <p className="mt-1 text-muted-foreground">
              Adding or removing cars does not require reprogramming logic; agents negotiate through
              standard FIPA-ACL protocols.
            </p>
          </div>
        </div>
      </section>

      {/* 5. Protocol Diagram & Performatives */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold uppercase tracking-wider text-primary">
          5. FIPA-ACL Contract Net Protocol & Performatives
        </h2>
        <div className="rounded-xl border border-border bg-card p-4 shadow-sm space-y-3 text-xs">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {Object.entries(PERFORMATIVE_COLORS).map(([perf, color]) => (
              <div
                key={perf}
                className="flex items-center gap-2 rounded border border-border/40 p-2 font-mono"
              >
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
                <span className="font-bold">{perf}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 6. Likely Viva Questions Accordion */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold uppercase tracking-wider text-primary">
          6. Likely Examiner Viva Questions & Answers
        </h2>
        <div className="space-y-2">
          {vivaQuestions.map((vq, idx) => {
            const isOpen = openVivaIdx === idx;
            return (
              <div
                key={idx}
                className="rounded-lg border border-border bg-card overflow-hidden transition-colors"
              >
                <button
                  onClick={() => setOpenVivaIdx(isOpen ? null : idx)}
                  className="flex w-full items-center justify-between p-3 text-left text-xs font-semibold text-foreground hover:bg-muted/40"
                >
                  <span>{vq.q}</span>
                  <ChevronDown
                    className={`h-4 w-4 shrink-0 transition-transform ${
                      isOpen ? "rotate-180 text-primary" : "text-muted-foreground"
                    }`}
                  />
                </button>
                {isOpen && (
                  <div className="border-t border-border/60 bg-muted/20 p-3 text-xs text-muted-foreground">
                    {vq.a}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
};
