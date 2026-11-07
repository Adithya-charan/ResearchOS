import { defineConfig } from "@playwright/test";

// Runs against a REAL backend (e2e/serve.py: fresh database, embedded worker, built SPA). No browser download:
// it drives the Microsoft Edge that is installed on the machine. Run `npm run build` first.
const PORT = 8793;
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 20_000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${PORT}`, channel: "msedge", trace: "retain-on-failure", screenshot: "only-on-failure" },
  webServer: {
    command: `"../.venv/Scripts/python.exe" e2e/serve.py ${PORT}`,
    url: `http://127.0.0.1:${PORT}/app/`, reuseExistingServer: false, timeout: 60_000,
  },
});
