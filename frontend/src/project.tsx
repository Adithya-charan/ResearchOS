import { useQuery } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { PROJECT_KEY, useAuth } from "./auth";
import { api } from "./lib/api";
import type { Project, Run } from "./lib/types";

interface Ctx { projects: Project[]; loading: boolean; error: unknown; project: Project | null; projectId: string | null; setProjectId: (id: string | null) => void; refetch: () => void }
const ProjectCtx = createContext<Ctx | null>(null);

export const qk = {
  projects: (uid?: string) => ["projects", uid] as const,
  runs: (pid: string) => ["runs", pid] as const,
};

export function ProjectProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const q = useQuery({ queryKey: qk.projects(user?.id), queryFn: ({ signal }) => api.get<Project[]>("/projects", { signal }), enabled: !!user });
  const [stored, setStored] = useState<string | null>(() => { try { return localStorage.getItem(PROJECT_KEY); } catch { return null; } });
  const projects = useMemo(() => q.data ?? [], [q.data]);
  const projectId = projects.find((p) => p.id === stored)?.id ?? projects[0]?.id ?? null;
  useEffect(() => { try { if (projectId) localStorage.setItem(PROJECT_KEY, projectId); } catch { /* storage unavailable */ } }, [projectId]);
  const setProjectId = useCallback((id: string | null) => setStored(id), []);
  const value = useMemo<Ctx>(() => ({
    projects, loading: q.isLoading, error: q.error, project: projects.find((p) => p.id === projectId) ?? null, projectId, setProjectId, refetch: () => void q.refetch(),
  }), [projects, projectId, setProjectId, q]);
  return <ProjectCtx.Provider value={value}>{children}</ProjectCtx.Provider>;
}
export function useProject() { const c = useContext(ProjectCtx); if (!c) throw new Error("useProject outside ProjectProvider"); return c; }

export function useRuns(projectId: string | null) {
  return useQuery({ queryKey: qk.runs(projectId ?? "-"), queryFn: ({ signal }) => api.get<Run[]>(`/projects/${projectId}/runs`, { signal }), enabled: !!projectId });
}
