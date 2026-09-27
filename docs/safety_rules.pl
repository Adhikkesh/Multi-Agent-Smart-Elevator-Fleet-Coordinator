% Safety rules as Horn clauses — the Prolog mirror of src/elevator_mas/rules/safety_rules.py
%
% Same knowledge, declared for a resolution engine instead of the forward-chaining
% engine in the Python package. Running the queries at the bottom shows the two
% formulations entail the same conclusions, which is the point: the SafetyAgent's
% knowledge is data, independent of the inference procedure (AIMA 4e ch. 7 & 9).
%
% Run:  swipl -q -s docs/safety_rules.pl
%
% SWI-Prolog is NOT required to run this project — this file is the optional bonus
% deliverable, provided so the safety knowledge can be checked independently of the
% Python engine. Install it with `apt install swi-prolog` (Ubuntu) or
% `brew install swi-prolog` (macOS) if you want to execute it.

:- discontiguous recall_to_lobby/1.

% ---------------------------------------------------------------- example world
% Facts a run would assert. Swap these to test other situations.
car(0). car(1). car(2). car(3).
lobby(0).
capacity(10).

fire_alarm.                  % comment out to model the alarm being clear
car_fault(2).
load(0, 12).                 % car 0 is carrying 12 people -> overloaded
load(1, 4).
load(2, 0).
load(3, 7).
at_floor(1, 0).              % car 1 has already reached the lobby
door_blocked(3, 9).
door_obstruction_limit(6).

% ------------------------------------------------------------------- the rules

% R1  IF fire_alarm AND car in service THEN recall the car to the lobby.
recall_to_lobby(Car) :-
    fire_alarm,
    car(Car),
    \+ out_of_service(Car).

% R2  IF fire_alarm AND the car is at the lobby THEN hold its doors open.
doors_held_open(Car) :-
    fire_alarm,
    recall_to_lobby(Car),
    lobby(L),
    at_floor(Car, L).

% R3  IF fire_alarm THEN block every hall call.
hall_calls_blocked :-
    fire_alarm.

% R4  IF a car has faulted THEN it is out of service and its calls are re-auctioned.
out_of_service(Car) :-
    car_fault(Car).

reauction_calls(Car) :-
    out_of_service(Car).

% R5  IF the load exceeds capacity THEN hold the doors and refuse boarding.
overloaded(Car) :-
    load(Car, Load),
    capacity(Cap),
    Load > Cap.

refuse_boarding(Car) :-
    overloaded(Car).

% R6  IF the doors have been blocked too long THEN re-open them.
reopen_doors(Car) :-
    door_blocked(Car, Ticks),
    door_obstruction_limit(Limit),
    Ticks > Limit.

% R7  IF the alarm is clear THEN normal service resumes.
normal_service :-
    \+ fire_alarm.

% A car may take a new hall call only when nothing safety-related forbids it.
may_serve_hall_call(Car) :-
    car(Car),
    \+ out_of_service(Car),
    \+ hall_calls_blocked,
    \+ overloaded(Car).

% ----------------------------------------------------------------- expected answers
%
%   ?- recall_to_lobby(X).        X = 0 ; 1 ; 3          (2 is out of service)
%   ?- doors_held_open(X).        X = 1                   (only car 1 is at the lobby)
%   ?- hall_calls_blocked.        true
%   ?- out_of_service(X).         X = 2
%   ?- overloaded(X).             X = 0
%   ?- reopen_doors(X).           X = 3                   (blocked 9 > limit 6)
%   ?- may_serve_hall_call(X).    false                   (fire mode blocks everything)
%   ?- normal_service.            false

:- initialization(main).

main :-
    format("rules-fired report~n"),
    forall(recall_to_lobby(C),   format("  R1 recall car ~w to lobby~n", [C])),
    forall(doors_held_open(C),   format("  R2 car ~w holds doors open~n", [C])),
    ( hall_calls_blocked -> format("  R3 hall calls blocked~n") ; true ),
    forall(out_of_service(C),    format("  R4 car ~w out of service, calls re-auctioned~n", [C])),
    forall(refuse_boarding(C),   format("  R5 car ~w refuses boarding (overloaded)~n", [C])),
    forall(reopen_doors(C),      format("  R6 car ~w re-opens obstructed doors~n", [C])),
    ( normal_service -> format("  R7 normal service~n") ; format("  (fire mode active)~n") ),
    halt.
