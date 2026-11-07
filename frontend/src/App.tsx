import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth";
import { Shell } from "./components/Shell";
import { SkeletonBlock } from "./components/ui";
import { ProjectProvider } from "./project";
import Login from "./pages/Login";

// Route-level code splitting: charts, the graph library and markdown are only fetched by the pages that use them.
const Overview = lazy(() => import("./pages/Overview"));
const Projects = lazy(() => import("./pages/Projects"));
const Workspace = lazy(() => import("./pages/Workspace"));
const Agents = lazy(() => import("./pages/Agents"));
const Library = lazy(() => import("./pages/Library"));
const Gaps = lazy(() => import("./pages/Gaps"));
const Hypotheses = lazy(() => import("./pages/Hypotheses"));
const Experiments = lazy(() => import("./pages/Experiments"));
const Evidence = lazy(() => import("./pages/Evidence"));
const Novelty = lazy(() => import("./pages/Novelty"));
const Reports = lazy(() => import("./pages/Reports"));
const ModelLab = lazy(() => import("./pages/ModelLab"));
const Activity = lazy(() => import("./pages/Activity"));
const Settings = lazy(() => import("./pages/Settings"));

export default function App() {
  const { state } = useAuth();
  if (state.status === "loading") return <div className="p-8"><SkeletonBlock label="Restoring your session" /></div>;
  if (state.status === "anon") return <Login reason={state.reason} />;
  return (
    <ProjectProvider>
      <Suspense fallback={<div className="p-6"><SkeletonBlock label="Loading page" /></div>}>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<Overview />} />
            <Route path="projects" element={<Projects />} />
            <Route path="projects/:projectId" element={<Projects />} />
            <Route path="workspace" element={<Workspace />} />
            <Route path="workspace/:runId" element={<Workspace />} />
            <Route path="agents" element={<Agents />} />
            <Route path="agents/:runId" element={<Agents />} />
            <Route path="library" element={<Library />} />
            <Route path="gaps" element={<Gaps />} />
            <Route path="hypotheses" element={<Hypotheses />} />
            <Route path="experiments" element={<Experiments />} />
            <Route path="evidence" element={<Evidence />} />
            <Route path="novelty" element={<Novelty />} />
            <Route path="reports" element={<Reports />} />
            <Route path="reports/:reportId" element={<Reports />} />
            <Route path="model-lab" element={<ModelLab />} />
            <Route path="activity" element={<Activity />} />
            <Route path="settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </ProjectProvider>
  );
}
