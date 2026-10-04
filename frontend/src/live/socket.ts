/**
 * Resilient WebSocket client with exponential backoff and rAF coalescing.
 */

import { validateSnapshotShape } from "../api/schemas";
import type { Snapshot } from "../api/types";
import { useLiveStore } from "../store/liveStore";

let socket: WebSocket | null = null;
let reconnectTimer: any = null;
let reconnectDelay = 500;
const MAX_RECONNECT_DELAY = 5000;
let isIntentionallyClosed = false;

// rAF coalescer state
let pendingSnapshot: Snapshot | null = null;
let rafId: number | null = null;

function flushSnapshot() {
  if (pendingSnapshot) {
    useLiveStore.getState().updateSnapshot(pendingSnapshot);
    pendingSnapshot = null;
  }
  rafId = null;
}

function scheduleSnapshotUpdate(snapshot: Snapshot) {
  // If tab is hidden, drop frame to save CPU
  if (typeof document !== "undefined" && document.hidden) {
    return;
  }

  pendingSnapshot = snapshot;
  if (rafId === null) {
    rafId = requestAnimationFrame(flushSnapshot);
  }
}

export function getWsUrl(): string {
  if (typeof window === "undefined") return "ws://localhost:8000/ws";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

export function connectWebSocket(url = getWsUrl()): void {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  isIntentionallyClosed = false;
  useLiveStore.getState().setConnecting(true);

  try {
    socket = new WebSocket(url);

    socket.onopen = () => {
      reconnectDelay = 500;
      useLiveStore.getState().setConnected(true);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (validateSnapshotShape(data)) {
          scheduleSnapshotUpdate(data as Snapshot);
        }
      } catch {
        // Ignore unparseable frames
      }
    };

    socket.onclose = () => {
      useLiveStore.getState().setConnected(false);
      if (!isIntentionallyClosed) {
        scheduleReconnect(url);
      }
    };

    socket.onerror = () => {
      useLiveStore.getState().setConnected(false);
      try {
        socket?.close();
      } catch {
        // ignore
      }
    };
  } catch {
    useLiveStore.getState().setConnected(false);
    scheduleReconnect(url);
  }
}

function scheduleReconnect(url: string) {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    reconnectDelay = Math.min(MAX_RECONNECT_DELAY, reconnectDelay * 1.5);
    connectWebSocket(url);
  }, reconnectDelay);
}

export function disconnectWebSocket(): void {
  isIntentionallyClosed = true;
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  pendingSnapshot = null;
  if (socket) {
    socket.close();
    socket = null;
  }
  useLiveStore.getState().setConnected(false);
}
