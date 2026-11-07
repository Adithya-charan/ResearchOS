import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import type { Run, Task, TaskDetail } from "./types";
import { useRuns } from "../project";

/** Gaps, hypotheses and experiment plans are not stored as separate records by the backend: they are the output of the
 *  gap / hypothesis / experiment_design agent tasks of a run. This reads that output, read-only, for a chosen run
 *  (default: the most recent run that has such a task). Nothing is invented or edited here. */
export function useAgentOutput(projectId: string, agentType: string, fixedRunId?: string) {
  const runs = useRuns(projectId);
  const [picked, setPicked] = useState<string | null>(null);
  const runList: Run[] = runs.data ?? [];
  const runId = fixedRunId ?? picked ?? runList[0]?.id ?? null;
  const tasks = useQuery({ queryKey: ["tasks", runId], enabled: !!runId, queryFn: ({ signal }) => api.get<Task[]>(`/runs/${runId}/tasks`, { signal }) });
  const task = (tasks.data ?? []).filter((t) => t.agent_type === agentType).sort((a, b) => b.seq - a.seq)[0];
  const detail = useQuery({
    queryKey: ["task", runId, task?.id], enabled: !!runId && !!task && task.status === "succeeded",
    queryFn: ({ signal }) => api.get<TaskDetail>(`/runs/${runId}/tasks/${task!.id}`, { signal }),
  });
  return { runs, runList, runId, setRunId: setPicked, tasks, task, detail, loading: runs.isLoading || tasks.isLoading || detail.isLoading, error: runs.error ?? tasks.error ?? detail.error };
}
