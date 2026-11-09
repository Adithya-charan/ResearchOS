import { defineConfig } from "@playwright/test";

// Runs against a REAL backend (e2e/serve.py: fresh database, embedded worker, built SPA). No browser download:
// it drives the Microsoft Edge that is installed on the machine. Run `npm run build` first.
// CI overrides: PW_CHANNEL="" uses the Chromium that `npx playwright install chromium` provides; PW_PYTHON points at the Python to run.
const PORT = 8793;
const CHANNEL = process.env.PW_CHANNEL ?? "msedge";
const PYTHON = process.env.PW_PYTHON ?? "../.venv/Scripts/python.exe";
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 20_000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${PORT}`, ...(CHANNEL ? { channel: CHANNEL } : {}), trace: "retain-on-failure", screenshot: "only-on-failure" },
  webServer: {
    command: `"${PYTHON}" e2e/serve.py ${PORT}`,
    url: `http://127.0.0.1:${PORT}/app/`, reuseExistingServer: false, timeout: 60_000,
  },
});
