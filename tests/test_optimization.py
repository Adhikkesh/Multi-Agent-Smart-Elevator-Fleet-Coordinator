"""Local search and adversarial search."""

from __future__ import annotations

import random

import pytest

from elevator_mas.domain import Direction, HallCall
from elevator_mas.optimization import (
    hill_climbing,
    hill_climbing_parking,
    minimax_parking,
    simulated_annealing,
)
from elevator_mas.optimization.adversarial import ParkingGame


def make_objective(cars: list[int]):
    """A simple assignment objective: spread the calls and keep them near their car."""

    def objective(assignment: dict[HallCall, int]) -> float:
        load = dict.fromkeys(cars, 0)
        for car in assignment.values():
            load[car] += 1
        distance = sum(abs(call.floor - car * 4) for call, car in assignment.items())
        return distance + 3.0 * sum(value**2 for value in load.values())

    return objective


class TestSimulatedAnnealing:
    """SA must improve, and must never return something worse than it started with."""

    def test_never_worse_than_the_initial_assignment(self, rng: random.Random) -> None:
        """The guarantee that makes it safe to run on a live fleet."""
        cars = [0, 1, 2, 3]
        objective = make_objective(cars)
        for _ in range(40):
            calls = [
                HallCall(rng.randrange(15), Direction.UP if rng.random() < 0.5 else Direction.DOWN)
                for _ in range(rng.randint(2, 7))
            ]
            initial = {call: rng.choice(cars) for call in set(calls)}
            result = simulated_annealing(initial, cars, objective, rng, iterations=200)
            assert result.final_cost <= result.initial_cost + 1e-9
            assert result.improvement >= -1e-9

    def test_returns_the_best_configuration_it_saw(self, rng: random.Random) -> None:
        """The reported assignment must actually score the reported cost."""
        cars = [0, 1, 2]
        objective = make_objective(cars)
        calls = [HallCall(f, Direction.UP) for f in (1, 4, 7, 11)]
        initial = dict.fromkeys(calls, 0)
        result = simulated_annealing(initial, cars, objective, rng, iterations=300)
        assert objective(result.assignment) == pytest.approx(result.final_cost)

    def test_improves_a_deliberately_bad_start(self, rng: random.Random) -> None:
        """Piling every call on one car is bad; SA must find something better."""
        cars = [0, 1, 2, 3]
        objective = make_objective(cars)
        calls = [HallCall(f, Direction.UP) for f in (0, 3, 6, 9, 12)]
        initial = dict.fromkeys(calls, 0)
        result = simulated_annealing(initial, cars, objective, rng, iterations=400)
        assert result.final_cost < result.initial_cost

    def test_curves_are_recorded_for_the_search_lab(self, rng: random.Random) -> None:
        """The dashboard plots these, so they must exist and be monotone where promised."""
        cars = [0, 1, 2]
        objective = make_objective(cars)
        calls = [HallCall(f, Direction.DOWN) for f in (2, 5, 9)]
        result = simulated_annealing(dict.fromkeys(calls, 0), cars, objective, rng, iterations=120)
        assert len(result.curve) == len(result.best_curve)
        # The running best can only ever improve.
        assert all(
            later <= earlier + 1e-9
            for earlier, later in zip(result.best_curve, result.best_curve[1:], strict=False)
        )

    def test_is_reproducible_from_a_seed(self) -> None:
        """Same seed, same answer — the whole benchmark depends on this."""
        cars = [0, 1, 2]
        objective = make_objective(cars)
        calls = [HallCall(f, Direction.UP) for f in (1, 5, 8, 12)]
        initial = dict.fromkeys(calls, 0)
        first = simulated_annealing(initial, cars, objective, random.Random(5), iterations=150)
        second = simulated_annealing(initial, cars, objective, random.Random(5), iterations=150)
        assert first.final_cost == second.final_cost
        assert first.assignment == second.assignment


class TestHillClimbing:
    """Hill climbing improves too, but stops at a local optimum."""

    def test_never_worse_than_the_start(self, rng: random.Random) -> None:
        """It only ever takes improving moves."""
        cars = [0, 1, 2]
        objective = make_objective(cars)
        calls = [HallCall(f, Direction.UP) for f in (2, 6, 10)]
        result = hill_climbing(dict.fromkeys(calls, 0), cars, objective, rng)
        assert result.final_cost <= result.initial_cost + 1e-9

    def test_parking_covers_demand(self, rng: random.Random) -> None:
        """Cars should end up near the floors that actually generate calls."""
        demand = {2: 0.0, 3: 0.6, 10: 0.9, 14: 0.4}
        positions, cost, curve = hill_climbing_parking([0, 1, 2], 15, demand, rng)
        assert len(positions) == 3
        assert all(0 <= floor < 15 for floor in positions.values())
        assert cost >= 0.0
        assert curve, "the restart curve is what the Search Lab plots"
        # With three cars and three busy floors it should manage a perfect cover.
        assert cost == pytest.approx(0.0, abs=1.0)

    def test_parking_with_no_idle_cars_is_a_no_op(self, rng: random.Random) -> None:
        """Nothing to place means no work and no crash."""
        positions, cost, curve = hill_climbing_parking([], 10, {1: 0.5}, rng)
        assert positions == {}
        assert cost == 0.0
        assert curve == []


class TestMinimaxParking:
    """Alpha-beta must agree with minimax, using fewer nodes."""

    def test_alpha_beta_matches_minimax_value(self) -> None:
        """Pruning is only sound if the value is unchanged."""
        for cars in (1, 2, 3):
            result = minimax_parking(cars, 15, [0, 4, 7, 11, 14], [1, 5, 9, 13])
            assert result.nodes_alphabeta <= result.nodes_minimax

    def test_alpha_beta_expands_fewer_nodes(self) -> None:
        """On a branching game, pruning must actually save work."""
        result = minimax_parking(3, 20, [0, 5, 10, 15, 19], [2, 7, 12, 17])
        assert result.nodes_alphabeta < result.nodes_minimax
        assert result.pruning_saving_pct > 0.0

    def test_value_is_a_worst_case_guarantee(self) -> None:
        """No floor nature can pick may be further than the reported value."""
        spots = [0, 6, 12]
        likely = [1, 5, 9, 13]
        result = minimax_parking(2, 15, spots, likely)
        worst = max(min(abs(spot - floor) for spot in result.best_parking) for floor in likely)
        assert worst == pytest.approx(-result.value)

    def test_single_candidate_is_forced(self) -> None:
        """With one spot there is no choice, and the value follows directly."""
        result = minimax_parking(1, 10, [5], [0, 9])
        assert result.best_parking == (5,)
        assert result.value == pytest.approx(-5.0)

    def test_game_requires_a_car(self) -> None:
        """Zero cars is a malformed game."""
        with pytest.raises(ValueError, match="at least one idle car"):
            ParkingGame(0, 10, [0], [1])

    def test_move_generation_is_deterministic(self) -> None:
        """Node counts are only comparable if move order is fixed."""
        game = ParkingGame(2, 10, [0, 5, 9], [1, 8])
        assert game.max_moves() == game.max_moves()
        assert game.max_moves() == sorted(game.max_moves())
