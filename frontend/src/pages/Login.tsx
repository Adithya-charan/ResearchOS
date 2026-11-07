import { zodResolver } from "@hookform/resolvers/zod";
import { Sparkles } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { useAuth } from "../auth";
import { Button, Card, Field, Input, Notice } from "../components/ui";

export const credentialsSchema = z.object({
  email: z.string().trim().min(1, "Enter your email").email("Enter a valid email address"),
  password: z.string().min(1, "Enter your password"),
});
export const registerSchema = credentialsSchema.extend({
  display_name: z.string().trim().max(80, "At most 80 characters").optional(),
  password: z.string().min(10, "Use at least 10 characters").max(256, "At most 256 characters"),
});
type Mode = "login" | "register";

export default function Login({ reason }: { reason?: string }) {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<Mode>("login");
  const [error, setError] = useState<string | null>(null);
  const f = useForm<{ email: string; password: string; display_name?: string }>({ resolver: zodResolver(mode === "login" ? credentialsSchema : registerSchema), mode: "onSubmit" });
  const submit = f.handleSubmit(async (v) => {
    setError(null);
    try { if (mode === "login") await login(v.email, v.password); else await register(v.email, v.password, v.display_name ?? ""); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not sign in."); }
  });
  const e = f.formState.errors;
  return (
    <div className="flex min-h-full items-center justify-center bg-bg p-4">
      <Card className="w-full max-w-md space-y-4 p-6">
        <div className="flex items-center gap-2"><Sparkles className="h-6 w-6 text-violet" aria-hidden /><h1 className="text-xl font-semibold">RESEARCHOS</h1></div>
        <p className="text-sm text-muted">Self-hosted multi-agent research intelligence. {mode === "login" ? "Sign in to continue." : "Create an account on this server."}</p>
        {reason && <Notice>{reason}</Notice>}
        <form onSubmit={submit} className="space-y-3" noValidate aria-label={mode === "login" ? "Sign in" : "Create account"}>
          <Field label="Email" htmlFor="email" error={e.email?.message}>
            <Input id="email" type="email" autoComplete="username" aria-invalid={!!e.email} aria-describedby={e.email ? "email-err" : undefined} {...f.register("email")} />
          </Field>
          {mode === "register" && (
            <Field label="Display name (optional)" htmlFor="display_name" error={e.display_name?.message}>
              <Input id="display_name" autoComplete="name" {...f.register("display_name")} />
            </Field>
          )}
          <Field label="Password" htmlFor="password" error={e.password?.message} hint={mode === "register" ? "At least 10 characters." : undefined}>
            <Input id="password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} aria-invalid={!!e.password} {...f.register("password")} />
          </Field>
          {error && <p role="alert" className="text-sm text-bad">{error}</p>}
          <Button type="submit" variant="primary" className="w-full" busy={f.formState.isSubmitting}>{mode === "login" ? "Sign in" : "Create account"}</Button>
        </form>
        <button className="text-sm text-accent underline" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); f.clearErrors(); }}>
          {mode === "login" ? "Create an account" : "I already have an account"}
        </button>
      </Card>
    </div>
  );
}
