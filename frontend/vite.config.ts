/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The app is served by the FastAPI backend under /app (same origin => HttpOnly cookie + CSRF work unchanged).
// In development the Vite server proxies every API path to the backend (default http://127.0.0.1:8000).
const target = process.env.VITE_API_PROXY ?? "http://127.0.0.1:8000";
const apiPaths = ["auth", "projects", "runs", "reports", "approvals", "experiments", "agents", "models", "training", "modellab", "overview", "audit", "health"];
export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: { port: 5173, proxy: Object.fromEntries(apiPaths.map((p) => [`/${p}`, { target, changeOrigin: false }])) },
  build: {
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          query: ["@tanstack/react-query"],
          flow: ["@xyflow/react"],
          markdown: ["react-markdown", "rehype-sanitize"],
        },
      },
    },
  },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/test/setup.ts"], include: ["src/**/*.test.{ts,tsx}"], css: false },
});
