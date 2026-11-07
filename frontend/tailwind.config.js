/** Design tokens live as CSS variables (src/index.css) so light/dark share one set of utilities. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)", surface: "var(--surface)", "surface-2": "var(--surface-2)", ink: "var(--ink)", muted: "var(--muted)",
        line: "var(--line)", "line-strong": "var(--line-strong)", side: "var(--side)", accent: "var(--accent)", "accent-soft": "var(--accent-soft)", "on-accent": "var(--on-accent)",
        violet: "var(--violet)", ok: "var(--ok)", warn: "var(--warn)", bad: "var(--bad)",
      },
      fontFamily: { sans: ["var(--font-sans)"], serif: ["var(--font-serif)"], mono: ["var(--font-mono)"] },
      fontSize: { xs: ["0.75rem", "1.05rem"], sm: ["0.8125rem", "1.25rem"], base: ["0.875rem", "1.4rem"] },
      borderRadius: { lg: "6px", xl: "8px" },
    },
  },
};
