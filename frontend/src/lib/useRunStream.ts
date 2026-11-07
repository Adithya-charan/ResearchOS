import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { connectSse, type SseStatus } from "./sse";
import type { EngineEvent } from "./types";

const MAX_EVENTS = 300;

/** Subscribes to the real run stream while `enabled`. Events are only what the server sends; nothing is simulated.
 *  Query caches for the run and its tasks are refreshed (throttled) when events arrive. The connection is closed on unmount. */
export function useRunStream(runId: string | undefined, enabled: boolean) {
  const qc = useQueryClient();
  const [events, setEvents] = useState<EngineEvent[]>([]);
  const [status, setStatus] = useState<SseStatus>("closed");
  const [info, setInfo] = useState<string | undefined>();
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setEvents([]);
    if (!runId || !enabled) { setStatus("closed"); return; }
    const refresh = () => {
      if (timer.current) return;
      timer.current = setTimeout(() => {
        timer.current = null;
        void qc.invalidateQueries({ queryKey: ["run", runId] });
        void qc.invalidateQueries({ queryKey: ["tasks", runId] });
      }, 400);
    };
    const stop = connectSse({
      path: `/runs/${runId}/stream`,
      onStatus: (s, i) => { setStatus(s); setInfo(i); if (s === "ended" || s === "closed") refresh(); },
      onMessage: (m) => {
        if (m.event === "state") { refresh(); return; }
        const d = m.data as Partial<EngineEvent> | null;
        if (d && typeof d === "object" && "type" in d) {
          setEvents((prev) => [...prev, { ...(d as EngineEvent), id: m.id ?? (d as EngineEvent).id }].slice(-MAX_EVENTS));
          refresh();
        }
      },
    });
    return () => { stop(); if (timer.current) { clearTimeout(timer.current); timer.current = null; } };
  }, [runId, enabled, qc]);

  return { events, status, info };
}
