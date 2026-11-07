import clsx from "clsx";
import {
  Activity, Bell, BookOpen, Boxes, ChevronRight, Cpu, FileText, FlaskConical, FolderKanban, Gauge, LayoutDashboard, Lightbulb, LogOut, Menu, Microscope, Moon,
  PanelLeftClose, PanelLeftOpen, Radar, Search, Settings, ShieldCheck, Sparkles, Sun, Target, User, Users, X,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "../lib/api";
import type { Approval } from "../lib/types";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth";
import { useProject } from "../project";
import { Button, Select } from "./ui";

interface NavItem { to: string; label: string; icon: ReactNode; end?: boolean }
const ic = "h-4 w-4 shrink-0";
export const NAV: { group: string; items: NavItem[] }[] = [
  { group: "Workspace", items: [
    { to: "/", label: "Overview", icon: <LayoutDashboard className={ic} aria-hidden />, end: true },
    { to: "/projects", label: "Projects", icon: <FolderKanban className={ic} aria-hidden /> },
    { to: "/workspace", label: "Research Workspace", icon: <Microscope className={ic} aria-hidden /> },
    { to: "/agents", label: "Agent Command Center", icon: <Users className={ic} aria-hidden /> },
  ] },
  { group: "Literature & analysis", items: [
    { to: "/library", label: "Literature Library", icon: <BookOpen className={ic} aria-hidden /> },
    { to: "/gaps", label: "Research Gaps", icon: <Target className={ic} aria-hidden /> },
    { to: "/hypotheses", label: "Hypotheses", icon: <Lightbulb className={ic} aria-hidden /> },
    { to: "/experiments", label: "Experiments", icon: <FlaskConical className={ic} aria-hidden /> },
  ] },
  { group: "Evidence & output", items: [
    { to: "/evidence", label: "Evidence Verification", icon: <ShieldCheck className={ic} aria-hidden /> },
    { to: "/novelty", label: "Novelty Assessment", icon: <Radar className={ic} aria-hidden /> },
    { to: "/reports", label: "Research Reports", icon: <FileText className={ic} aria-hidden /> },
  ] },
  { group: "Platform", items: [
    { to: "/model-lab", label: "Model Lab", icon: <Cpu className={ic} aria-hidden /> },
    { to: "/activity", label: "Activity & Audit", icon: <Activity className={ic} aria-hidden /> },
    { to: "/settings", label: "Settings", icon: <Settings className={ic} aria-hidden /> },
  ] },
];
const CRUMBS: Record<string, string> = Object.fromEntries(NAV.flatMap((g) => g.items.map((i) => [i.to.slice(1), i.label])));

export function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => { try { return localStorage.getItem("researchos.theme") === "dark" ? "dark" : "light"; } catch { return "light"; } });
  useEffect(() => { document.documentElement.dataset.theme = theme; try { localStorage.setItem("researchos.theme", theme); } catch { /* ignore */ } }, [theme]);
  return [theme, setTheme] as const;
}

function Breadcrumbs() {
  const { pathname } = useLocation();
  const parts = pathname.split("/").filter(Boolean);
  const first = parts[0] ?? "";
  return (
    <nav aria-label="Breadcrumb" className="min-w-0 text-sm text-muted">
      <ol className="flex items-center gap-1 truncate">
        <li><Link to="/" className="hover:text-ink">RESEARCHOS</Link></li>
        {first && <li className="flex items-center gap-1"><ChevronRight className="h-3 w-3" aria-hidden /><span aria-current={parts.length === 1 ? "page" : undefined} className="text-ink">{CRUMBS[first] ?? first}</span></li>}
        {parts[1] && <li className="flex items-center gap-1"><ChevronRight className="h-3 w-3" aria-hidden /><span aria-current="page" className="max-w-[10rem] truncate font-mono text-xs text-ink">{parts[1]}</span></li>}
      </ol>
    </nav>
  );
}

export function Shell() {
  const { user, logout } = useAuth();
  const { projects, projectId, setProjectId } = useProject();
  const [collapsed, setCollapsed] = useState(() => { try { return localStorage.getItem("researchos.nav") === "collapsed"; } catch { return false; } });
  const [drawer, setDrawer] = useState(false);
  const [menu, setMenu] = useState(false);
  const [theme, setTheme] = useTheme();
  const [q, setQ] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const approvals = useQuery({ queryKey: ["approvals", "pending"], queryFn: ({ signal }) => api.get<Approval[]>("/approvals", { signal, query: { status: "pending" } }), refetchInterval: 60_000 });
  const pending = approvals.data?.length ?? 0;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.key === "/" && !e.ctrlKey && !e.metaKey && t && !/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) && !t.isContentEditable) { e.preventDefault(); searchRef.current?.focus(); }
    };
    document.addEventListener("keydown", onKey); return () => document.removeEventListener("keydown", onKey);
  }, []);
  const nav = useNavigate(); const loc = useLocation();
  const menuRef = useRef<HTMLDivElement>(null);
  useEffect(() => { setDrawer(false); setMenu(false); }, [loc.pathname]);
  useEffect(() => { try { localStorage.setItem("researchos.nav", collapsed ? "collapsed" : "open"); } catch { /* ignore */ } }, [collapsed]);
  useEffect(() => {
    if (!menu) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setMenu(false); };
    const onClick = (e: MouseEvent) => { if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenu(false); };
    document.addEventListener("keydown", onKey); document.addEventListener("mousedown", onClick);
    return () => { document.removeEventListener("keydown", onKey); document.removeEventListener("mousedown", onClick); };
  }, [menu]);

  const sidebar = (
    <nav aria-label="Main" className="flex h-full flex-col gap-3 overflow-y-auto px-2 py-3">
      <div className={clsx("flex items-center gap-2 px-2 pb-1 text-ink", collapsed && "justify-center px-0")}><Sparkles className="h-4 w-4 shrink-0 text-accent" aria-hidden />{!collapsed && <span className="text-[13px] font-semibold tracking-wide">RESEARCHOS</span>}</div>
      {NAV.map((g) => (
        <div key={g.group}>
          {!collapsed && <p className="px-2 pb-1 text-[10.5px] font-medium uppercase tracking-wider text-muted">{g.group}</p>}
          {collapsed && <div className="mx-2 mb-1 border-t border-line" aria-hidden />}
          <ul className="space-y-px">
            {g.items.map((i) => (
              <li key={i.to}>
                <NavLink to={i.to} end={i.end} title={collapsed ? i.label : undefined} aria-label={collapsed ? i.label : undefined}
                  className={({ isActive }) => clsx("relative flex h-7 items-center gap-2 rounded-md px-2 text-[13px]", isActive ? "bg-surface font-medium text-ink ring-1 ring-line" : "text-muted hover:bg-surface/70 hover:text-ink", collapsed && "justify-center px-0")}>
                  {({ isActive }) => (<>
                    {isActive && <span className="absolute -left-2 top-1.5 h-4 w-0.5 rounded bg-accent" aria-hidden />}
                    <span className={isActive ? "text-accent" : ""}>{i.icon}</span>{!collapsed && <span className="truncate">{i.label}</span>}
                  </>)}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );

  return (
    <div className="flex h-full bg-bg text-ink">
      <a href="#main" className="skip-link">Skip to content</a>
      <aside className={clsx("hidden shrink-0 border-r border-line bg-side md:block", collapsed ? "w-14" : "w-56")}>{sidebar}</aside>
      {drawer && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 bg-black/50" onClick={() => setDrawer(false)} />
          <aside className="relative h-full w-72 border-r border-line bg-side"><button className="absolute right-2 top-2 rounded p-1 text-ink" aria-label="Close navigation" onClick={() => setDrawer(false)}><X className="h-5 w-5" aria-hidden /></button>{sidebar}</aside>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 items-center gap-2 border-b border-line bg-surface px-3">
          <Button variant="ghost" size="sm" className="md:hidden" aria-label="Open navigation" onClick={() => setDrawer(true)}><Menu className="h-4 w-4" aria-hidden /></Button>
          <Button variant="ghost" size="sm" className="hidden md:inline-flex" aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} title={collapsed ? "Expand sidebar" : "Collapse sidebar"} onClick={() => setCollapsed((c) => !c)}>
            {collapsed ? <PanelLeftOpen className="h-4 w-4" aria-hidden /> : <PanelLeftClose className="h-4 w-4" aria-hidden />}
          </Button>
          <div className="hidden min-w-0 flex-1 sm:block"><Breadcrumbs /></div>
          <form role="search" className="ml-auto" onSubmit={(e) => { e.preventDefault(); if (q.trim()) nav(`/library?q=${encodeURIComponent(q.trim())}`); }}>
            <label htmlFor="global-search" className="sr-only">Search the literature library (press / to focus)</label>
            <div className="relative"><Search className="pointer-events-none absolute left-2 top-2 h-3.5 w-3.5 text-muted" aria-hidden />
              <input id="global-search" ref={searchRef} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search library" title="Lexical search of the current project library. Shortcut: /"
                className="h-7 w-32 rounded-lg border border-line-strong bg-surface pl-7 pr-6 text-sm sm:w-52" />
              <kbd className="pointer-events-none absolute right-1.5 top-1.5 hidden rounded border border-line px-1 text-[10px] text-muted sm:block" aria-hidden>/</kbd></div>
          </form>
          <label htmlFor="project-switch" className="sr-only">Current project</label>
          <Select id="project-switch" className="hidden h-7 w-40 py-0 sm:block" value={projectId ?? ""} onChange={(e) => setProjectId(e.target.value || null)}>
            {projects.length === 0 && <option value="">No projects</option>}
            {projects.map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
          </Select>
          <Link to="/activity" className="relative inline-flex h-7 w-7 items-center justify-center rounded-lg text-muted hover:bg-surface-2 hover:text-ink" aria-label={pending ? `${pending} pending approval${pending === 1 ? "" : "s"}` : "Notifications: nothing pending"} title={pending ? `${pending} pending approval(s)` : "No pending approvals"}>
            <Bell className="h-4 w-4" aria-hidden />{pending > 0 && <span className="absolute -right-0.5 -top-0.5 min-w-[14px] rounded-full bg-warn px-1 text-center text-[9px] font-semibold leading-[14px] text-white">{pending}</span>}
          </Link>
          <Button variant="ghost" size="sm" aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} title="Toggle theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? <Sun className="h-4 w-4" aria-hidden /> : <Moon className="h-4 w-4" aria-hidden />}
          </Button>
          <div className="relative" ref={menuRef}>
            <Button variant="ghost" size="sm" aria-haspopup="menu" aria-expanded={menu} aria-label="User menu" onClick={() => setMenu((m) => !m)}><User className="h-4 w-4" aria-hidden /></Button>
            {menu && (
              <div role="menu" className="absolute right-0 z-30 mt-1 w-56 rounded-lg border border-line-strong bg-surface p-1 shadow-lg">
                <p className="truncate px-3 py-2 text-xs text-muted">{user?.email}</p>
                <Link role="menuitem" to="/settings" className="flex items-center gap-2 rounded px-3 py-1.5 text-sm hover:bg-surface-2"><Settings className="h-4 w-4" aria-hidden />Settings</Link>
                <button role="menuitem" className="flex w-full items-center gap-2 rounded px-3 py-1.5 text-left text-sm hover:bg-surface-2" onClick={() => void logout()}><LogOut className="h-4 w-4" aria-hidden />Sign out</button>
              </div>
            )}
          </div>
        </header>
        <main id="main" tabIndex={-1} className="min-h-0 flex-1 overflow-y-auto px-4 py-4 md:px-6"><Outlet /></main>
      </div>
    </div>
  );
}
export { Boxes, Gauge };
