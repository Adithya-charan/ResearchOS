import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useProject } from "../project";
import type { Project } from "../lib/types";
import { Button, EmptyState, ErrorState, SkeletonBlock } from "./ui";

/** Renders loading / error / content for a query. Empty handling stays with the caller (it knows what "empty" means). */
export function Q<T>({ q, label, children, rows }: { q: UseQueryResult<T>; label: string; rows?: number; children: (data: T) => ReactNode }) {
  if (q.isLoading) return <SkeletonBlock label={`Loading ${label}`} rows={rows} />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  if (q.data === undefined) return null;
  return <>{children(q.data)}</>;
}

/** Pages that need a selected project render through this. */
export function RequireProject({ children }: { children: (p: Project) => ReactNode }) {
  const { project, projects, loading, error, refetch } = useProject();
  if (loading) return <SkeletonBlock label="Loading projects" />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;
  if (!project) {
    return <EmptyState title={projects.length ? "Select a project" : "No projects yet"} hint="This page shows data for one project. Create a project to begin."
      action={<Link to="/projects"><Button variant="primary">Go to projects</Button></Link>} />;
  }
  return <>{children(project)}</>;
}

export const LEXICAL_NOTE = "Retrieval is lexical (BM25 term matching over the uploaded library). It is not semantic search: it will miss synonyms and paraphrases, and a match is not proof of a claim.";
export const NOVELTY_NOTE = "Novelty is assessed only against the documents searched in this project's library. It is not proof that an idea has never been published.";
