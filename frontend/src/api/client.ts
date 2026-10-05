import {
  AnnealingLabSchema,
  BenchmarkSchema,
  BoardResponseSchema,
  type BrainConfigUpdate,
  BrainDecisionsSchema,
  type BrainDecision,
  type BrainResponse,
  BrainSchema,
  ExplainSchema,
  type ExplainResponse,
  MetaSchema,
  MinimaxLabSchema,
  RunResponseSchema,
  SearchLabRoutingSchema,
} from "./schemas";
import type {
  AgentDetailResponse,
  AnnealingLabResponse,
  Auction,
  BenchmarkResponse,
  BoardResponse,
  Message,
  MetaResponse,
  MinimaxLabResponse,
  RunResponse,
  SearchLabRoutingResponse,
  Snapshot,
  VersionResponse,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API Error ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && typeof options.body === "string" && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errJson = await response.json();
      if (errJson && typeof errJson.detail === "string") {
        detail = errJson.detail;
      } else if (errJson && errJson.detail) {
        detail = JSON.stringify(errJson.detail);
      }
    } catch {
      // Use statusText
    }
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  async brain(): Promise<BrainResponse> {
    return BrainSchema.parse(await request<unknown>("/api/brain"));
  },

  async brainDecisions(limit = 50): Promise<BrainDecision[]> {
    const raw = await request<unknown>(`/api/brain/decisions?limit=${limit}`);
    return BrainDecisionsSchema.parse(raw).decisions;
  },

  async setBrainConfig(update: BrainConfigUpdate): Promise<unknown> {
    return request<unknown>("/api/brain/config", {
      method: "POST",
      body: JSON.stringify(update),
    });
  },

  async explain(conversationId: string | number): Promise<ExplainResponse> {
    const raw = await request<unknown>("/api/brain/explain", {
      method: "POST",
      body: JSON.stringify({ conversation_id: conversationId }),
    });
    return ExplainSchema.parse(raw);
  },

  async getState(): Promise<Snapshot> {
    return request<Snapshot>("/api/state");
  },

  async getMeta(): Promise<MetaResponse> {
    const raw = await request<MetaResponse>("/api/meta");
    return MetaSchema.parse(raw) as MetaResponse;
  },

  async play(): Promise<{ running: boolean }> {
    return request<{ running: boolean }>("/api/play", { method: "POST" });
  },

  async pause(): Promise<{ running: boolean }> {
    return request<{ running: boolean }>("/api/pause", { method: "POST" });
  },

  async step(ticks = 1): Promise<{ tick: number }> {
    return request<{ tick: number }>("/api/step", {
      method: "POST",
      body: JSON.stringify({ ticks }),
    });
  },

  async setSpeed(speed: number): Promise<{ speed: number }> {
    return request<{ speed: number }>("/api/speed", {
      method: "POST",
      body: JSON.stringify({ speed }),
    });
  },

  async reset(params: {
    scenario?: string;
    strategy?: string;
    seed?: number;
    floors?: number;
    cars?: number;
  }): Promise<Snapshot> {
    return request<Snapshot>("/api/reset", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  async addPassenger(params: {
    origin: number;
    destination?: number;
    priority?: boolean;
  }): Promise<any> {
    return request<any>("/api/passenger", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  async inject(params: {
    kind: string;
    car?: number;
    floor?: number;
    count?: number;
  }): Promise<any> {
    return request<any>("/api/inject", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  async inspectAgent(address: string): Promise<AgentDetailResponse> {
    return request<AgentDetailResponse>(`/api/agent/${encodeURIComponent(address)}`);
  },

  async getMessages(limit = 120): Promise<{ messages: Message[]; total: number }> {
    return request<{ messages: Message[]; total: number }>(`/api/messages?limit=${limit}`);
  },

  async getAuctions(limit = 20): Promise<{ auctions: Auction[] }> {
    return request<{ auctions: Auction[] }>(`/api/auctions?limit=${limit}`);
  },

  async getSearchLabRouting(car?: number): Promise<SearchLabRoutingResponse> {
    const query = car !== undefined ? `?car=${car}` : "";
    const raw = await request<SearchLabRoutingResponse>(`/api/search-lab${query}`);
    return SearchLabRoutingSchema.parse(raw) as SearchLabRoutingResponse;
  },

  async getAnnealingLab(): Promise<AnnealingLabResponse> {
    const raw = await request<AnnealingLabResponse>("/api/search-lab/annealing");
    return AnnealingLabSchema.parse(raw) as AnnealingLabResponse;
  },

  async getMinimaxLab(): Promise<MinimaxLabResponse> {
    const raw = await request<MinimaxLabResponse>("/api/search-lab/minimax");
    return MinimaxLabSchema.parse(raw) as MinimaxLabResponse;
  },

  async benchmark(
    params: {
      scenarios?: string[];
      strategies?: string[];
      seeds: number;
      ticks: number;
    },
    signal?: AbortSignal,
  ): Promise<BenchmarkResponse> {
    const raw = await request<BenchmarkResponse>("/api/benchmark", {
      method: "POST",
      body: JSON.stringify(params),
      signal,
    });
    return BenchmarkSchema.parse(raw) as BenchmarkResponse;
  },

  async run(
    params: {
      scenario: string;
      strategy: string;
      seed: number;
      ticks?: number;
      sample_every?: number;
    },
    signal?: AbortSignal,
  ): Promise<RunResponse> {
    const raw = await request<RunResponse>("/api/run", {
      method: "POST",
      body: JSON.stringify(params),
      signal,
    });
    return RunResponseSchema.parse(raw) as RunResponse;
  },

  async getBoard(): Promise<BoardResponse> {
    const raw = await request<BoardResponse>("/api/board");
    return BoardResponseSchema.parse(raw) as BoardResponse;
  },

  async getVersion(): Promise<VersionResponse> {
    return request<VersionResponse>("/api/version");
  },

  async getHealth(): Promise<{ status: string; tick: number; scenario: string; strategy: string }> {
    return request<{ status: string; tick: number; scenario: string; strategy: string }>(
      "/api/health",
    );
  },
};
