import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AuthProvider } from "./auth";
import { ToastProvider } from "./components/ui";
import { ApiError } from "./lib/api";
import "./index.css";

const client = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 15_000, refetchOnWindowFocus: false,
      // the API client already retries safe reads once; do not stack another policy on 4xx
      retry: (n, e) => !(e instanceof ApiError && e.status >= 400 && e.status < 500) && n < 1,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter basename="/app"><AuthProvider><ToastProvider><App /></ToastProvider></AuthProvider></BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
