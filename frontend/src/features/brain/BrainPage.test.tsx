import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { BrainDecisionSchema, BrainSchema, ExplainSchema } from "../../api/schemas";
import { DecisionDetail } from "./BrainPage";

const decision = {
  tick: 120,
  conversation_id: "c12",
  call: { floor: 7, direction: "UP" },
  candidates: [
    {
      car_id: 0,
      refused: false,
      reason: null,
      bid: 12.5,
      wait: 10,
      ride: 1,
      crowding: 0.5,
      energy: 1,
      net_score: 2.6,
      attention: 0.6,
      teacher_bid: 11,
    },
    {
      car_id: 1,
      refused: false,
      reason: null,
      bid: 14.1,
      wait: 12,
      ride: 1,
      crowding: 0.5,
      energy: 0.6,
      net_score: 2.7,
      attention: 0.3,
      teacher_bid: 9,
    },
    {
      car_id: 2,
      refused: true,
      reason: "full",
      bid: null,
      wait: null,
      ride: null,
      crowding: null,
      energy: null,
      net_score: null,
      attention: 0.1,
      teacher_bid: null,
    },
  ],
  chosen: 1,
  net_choice: 0,
  teacher_choice: 1,
  search: {
    chosen: 1,
    net_choice: 0,
    candidates: [0, 1],
    prior: [0.52, 0.48],
    visits: [6, 16],
    q: [-5.1, -4.2],
    root_value: -4.5,
    sims_done: 22,
    depth_max: 3,
    time_ms: 48.2,
    nn_ms: 4.1,
    leaf: "rollout",
    overridden: true,
    fallback: false,
  },
};

describe("Brain API schemas", () => {
  it("parse a searched decision with refusals and null (non-finite) values", () => {
    expect(() => BrainDecisionSchema.parse(decision)).not.toThrow();
  });

  it("parse the brain summary for a classical strategy", () => {
    const brain = {
      available: false,
      strategies: [{ name: "liftzero_bc", label: "LiftZero", available: true, reason: "" }],
      active: {
        strategy: "full",
        label: "Full",
        bidder: "classical",
        arbiter: "min_bid",
        model: null,
        search: null,
        shadow_teacher: false,
      },
      stats: null,
    };
    expect(BrainSchema.parse(brain).available).toBe(false);
  });

  it("reject a malformed explanation", () => {
    expect(() =>
      ExplainSchema.parse({ conversation_id: 1, method: "occlusion", groups: [{ group: "x" }] }),
    ).toThrow();
  });
});

describe("DecisionDetail", () => {
  it("shows candidates, the refusal reason, the override and the search summary", () => {
    const onExplain = vi.fn();
    render(
      <DecisionDetail
        decision={BrainDecisionSchema.parse(decision)}
        explanation={null}
        onExplain={onExplain}
        explaining={false}
      />,
    );
    expect(screen.getByText(/awarded to car 1/)).toBeInTheDocument();
    expect(screen.getByText(/network choice car 0/)).toBeInTheDocument();
    expect(screen.getByText(/A\* teacher car 1/)).toBeInTheDocument();
    expect(screen.getByText(/look-ahead override/)).toBeInTheDocument();
    expect(screen.getByText(/refused \(full\)/)).toBeInTheDocument();
    expect(screen.getByText(/22 simulations to depth 3/)).toBeInTheDocument();
    screen.getByRole("button", { name: /Why\?/ }).click();
    expect(onExplain).toHaveBeenCalledOnce();
  });

  it("renders an occlusion explanation table", () => {
    const explanation = ExplainSchema.parse({
      conversation_id: "c12",
      method: "occlusion",
      groups: [
        {
          group: "traffic_pattern",
          delta_score_chosen: 0.41,
          delta_score_runner_up: 0.1,
          flips_decision: true,
        },
      ],
    });
    render(
      <DecisionDetail
        decision={BrainDecisionSchema.parse(decision)}
        explanation={explanation}
        onExplain={() => {}}
        explaining={false}
      />,
    );
    expect(screen.getByText("traffic pattern")).toBeInTheDocument();
    expect(screen.getByText("yes")).toBeInTheDocument();
  });
});

describe("Theory page docstring cleaning", () => {
  it("strips markdown emphasis and the trailing PEAS block", async () => {
    const { cleanDocstring } = await import("../../lib/docstring");
    expect(
      cleanDocstring(
        "A person. **AIMA agent type: simple reflex** (§2.4.2). *if here* board. PEAS - **P** wait",
      ),
    ).toBe("A person. AIMA agent type: simple reflex (§2.4.2). if here board.");
  });
});
