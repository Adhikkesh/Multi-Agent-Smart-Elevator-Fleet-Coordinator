/**
 * LiftZero API and WebSocket Types
 * Reflects the backend schemas and Mesa model representations.
 */

export type Direction = "UP" | "DOWN" | "IDLE";

export type DoorState = "closed" | "opening" | "open" | "closing";

export type CarMotionState =
  | "idle"
  | "moving_up"
  | "moving_down"
  | "doors"
  | "out_of_service"
  | "fire_recall";

export type Performative =
  | "REQUEST"
  | "CFP"
  | "PROPOSE"
  | "REFUSE"
  | "ACCEPT_PROPOSAL"
  | "REJECT_PROPOSAL"
  | "INFORM"
  | "CANCEL"
  | "FAILURE";

export interface RouteStop {
  floor: number;
  kind: "pickup" | "dropoff";
  direction: Direction;
  weight?: number;
}

export interface CarState {
  car_id: number;
  floor: number;
  direction: Direction;
  door: DoorState;
  state: CarMotionState;
  load: number;
  capacity: number;
  out_of_service: boolean;
  fire_mode: boolean;
  route: RouteStop[];
  car_calls: number[];
  assigned: { floor: number; direction: Direction }[];
  progress: number; // 0..1
  plan_cost: number | null;
  plan_nodes: number;
}

export interface FloorState {
  floor: number;
  up: boolean;
  down: boolean;
  waiting_up: number;
  waiting_down: number;
  assigned_up: number | null;
  assigned_down: number | null;
  eta_up: number | null;
  eta_down: number | null;
  escalations: number;
  oldest_wait: number;
}

export interface Metrics {
  tick: number;
  arrived: number;
  delivered: number;
  waiting: number;
  riding: number;
  avg_wait: number;
  p95_wait: number;
  max_wait: number;
  avg_ride: number;
  avg_system: number;
  long_wait_pct: number;
  throughput: number;
  energy: number;
  floors_travelled: number;
  stops: number;
  reversals: number;
  messages: number;
  messages_per_call: number;
  calls: number;
  nodes_expanded: number;
  replans: number;
  compute_ms_per_tick: number;
  rules_fired: number;
}

export interface Bid {
  car_id: number;
  total: number | null; // null if refused!
  wait: number;
  ride: number;
  crowding: number;
  energy: number;
  eta: number | null;
  refused: boolean;
  reason: string;
}

export interface Auction {
  tick: number;
  conversation_id: string;
  floor: number;
  direction: "UP" | "DOWN";
  bids: Bid[];
  winner: number | null;
  reason: string;
}

export interface Message {
  seq: number;
  tick: number;
  performative: Performative;
  sender: string;
  receiver: string;
  conversation_id: string;
  content: any;
}

export interface RuleFired {
  tick: number;
  rule: string;
  binding: Record<string, any>;
  effect: string;
}

export interface TrafficWeights {
  wait: number;
  ride: number;
  crowding: number;
  energy: number;
}

export interface TrafficInfo {
  pattern: string;
  reason: string;
  observed: number;
  true_pattern: string;
  weights: TrafficWeights;
}

export interface SimEvent {
  tick: number;
  kind: string;
  floor?: number;
  count?: number;
  car?: number;
}

export interface BrainFrame {
  model: string;
  params: number;
  latency_ms: number;
  decision?: {
    call: { floor: number; direction: "UP" | "DOWN" };
    scores: { car_id: number; score: number; teacher?: number }[];
    chosen: number;
    teacher_choice?: number;
    mcts?: { visits: number[]; value: number };
  };
}

export interface BuildingConfig {
  floors: number;
  cars: number;
  capacity: number;
  lobby: number;
}

export interface SessionInfo {
  running: boolean;
  speed: number;
  duration: number;
}

export interface Snapshot {
  tick: number;
  scenario: string;
  strategy: string;
  seed: number;
  building: BuildingConfig;
  cars: CarState[];
  floors: FloorState[];
  metrics: Metrics;
  auction: Auction | null;
  messages: Message[];
  rules: RuleFired[];
  traffic: TrafficInfo;
  events: SimEvent[];
  fire_alarm: boolean;
  reassignment: any | null;
  parking: Record<string, number>;
  session: SessionInfo;
  brain?: BrainFrame | null;
}

/* =========================== REST Responses =========================== */

export interface StrategyMeta {
  name: string;
  description: string;
  learning?: boolean;
}

export interface AgentInfo {
  name: string;
  agent_type: string;
  peas: {
    performance: string;
    environment: string;
    actuators: string;
    sensors: string;
  };
  docstring: string;
}

export interface EnvironmentProperty {
  property: string;
  value: string;
  justification: string;
}

export interface SafetyRuleMeta {
  name: string;
  salience: number;
  description: string;
}

export interface MetaResponse {
  scenarios: string[];
  strategies: StrategyMeta[];
  agents: AgentInfo[];
  environment: EnvironmentProperty[];
  system_peas: {
    performance: string;
    environment: string;
    actuators: string;
    sensors: string;
  };
  rules: SafetyRuleMeta[];
}

export interface SearchResultItem {
  algorithm: string;
  found: boolean;
  cost: number;
  nodes_expanded: number;
  nodes_generated: number;
  max_frontier: number;
  runtime_ms: number;
  depth: number;
  route: string[];
  optimal?: boolean;
}

export interface SearchLabRoutingResponse {
  source: string;
  car_id: number;
  current_floor: number;
  stops: { floor: number; kind: string; direction: string; weight: number }[];
  h_at_start: number;
  results: SearchResultItem[];
}

export interface LocalSearchResult {
  initial_cost: number;
  final_cost: number;
  improvement: number;
  iterations: number;
  accepted: number;
  curve: number[];
  best_curve?: number[];
  assignment: Record<string, number>;
}

export interface AnnealingLabResponse {
  available: boolean;
  synthetic?: boolean;
  reason?: string;
  calls?: number;
  cars?: number;
  simulated_annealing?: LocalSearchResult;
  hill_climbing?: LocalSearchResult;
}

export interface MinimaxLabResponse {
  idle_cars: number;
  cars_placed: number;
  candidate_spots: number[];
  likely_floors: number[];
  best_parking: Record<string, number>;
  value: number;
  minimax_nodes: number;
  alphabeta_nodes: number;
  pruning_saving_pct: number;
}

export interface BenchmarkSummaryRow {
  scenario: string;
  strategy: string;
  runs: number;
  avg_wait_mean: number;
  avg_wait_std: number;
  p95_wait_mean: number;
  p95_wait_std: number;
  long_wait_pct_mean: number;
  long_wait_pct_std: number;
  energy_mean: number;
  energy_std: number;
  throughput_mean: number;
  throughput_std: number;
  violations: number;
}

export interface BenchmarkComparisonRow {
  scenario: string;
  strategy: string;
  avg_wait_delta_pct: number;
  p95_wait_delta_pct: number;
  long_wait_pct_delta_pct: number;
  energy_delta_pct: number;
  wins_on: string[];
}

export interface BenchmarkResponse {
  runs: number;
  scenarios: string[];
  strategies: string[];
  seeds: number[];
  ticks: number;
  summary: BenchmarkSummaryRow[];
  comparison: BenchmarkComparisonRow[];
  wins: Record<string, number>;
}

export interface RunSeries {
  tick: number[];
  avg_wait: number[];
  p95_wait: number[];
  waiting: number[];
  riding: number[];
  delivered: number[];
  energy: number[];
  long_wait_pct: number[];
  messages: number[];
}

export interface RunResponse {
  scenario: string;
  strategy: string;
  seed: number;
  ticks: number;
  runtime_s: number;
  final: Metrics;
  series: RunSeries;
  rules_fired: string[];
  violations: string[];
}

export interface BoardCarStatus {
  car_id: number;
  tick: number;
  floor: number;
  direction: Direction;
  load: number;
  capacity: number;
  space: number;
  available: boolean;
  out_of_service: boolean;
  fire_mode: boolean;
  door: string;
  assigned_calls: string[];
  car_calls: number[];
  riders: number;
  plan_end_floor: number | null;
  plan_end_eta: number;
  planned_stops: number;
  park_target: number | null;
}

export interface BoardPolicy {
  pattern: string;
  weights: TrafficWeights;
  published_by: string;
  tick: number;
}

export interface BoardResponse {
  cars: BoardCarStatus[];
  policy: BoardPolicy;
  writes: number;
}

export interface VersionResponse {
  app: string;
  git: string | null;
}

export interface AgentDetailResponse {
  address: string;
  agent_type: string;
  peas: {
    performance: string;
    environment: string;
    actuators: string;
    sensors: string;
  };
  [key: string]: any;
}
