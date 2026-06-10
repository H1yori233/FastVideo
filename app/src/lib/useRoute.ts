import { useCallback, useEffect, useState } from "react";

export interface Route {
  view: "graph" | "results" | "case";
  caseId: string | null;
  runs: string[] | null;
  step: number | null;
  run: string | null;
}

const DEFAULT_ROUTE: Route = { view: "graph", caseId: null, runs: null, step: null, run: null };

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#/, "");
  const [pathPart, queryPart] = raw.split("?");
  const path = (pathPart ?? "").replace(/^\/+|\/+$/g, "");
  const q = new URLSearchParams(queryPart ?? "");

  const runsParam = q.get("runs");
  const stepParam = q.get("step");
  const route: Route = {
    ...DEFAULT_ROUTE,
    runs: runsParam ? runsParam.split(",").filter(Boolean) : null,
    step: stepParam != null && stepParam !== "" ? Number(stepParam) : null,
    run: q.get("run"),
  };

  if (path === "results") {
    route.view = "results";
  } else if (path.startsWith("case/")) {
    route.view = "case";
    route.caseId = path.slice("case/".length) || null;
    if (!route.caseId) route.view = "graph";
  }
  return route;
}

export function buildHash(route: Route): string {
  let path = "/";
  if (route.view === "results") path = "/results";
  else if (route.view === "case" && route.caseId) path = `/case/${route.caseId}`;

  const q = new URLSearchParams();
  if (route.runs?.length) q.set("runs", route.runs.join(","));
  if (route.step != null) q.set("step", String(route.step));
  if (route.run && route.view === "graph") q.set("run", route.run);

  const qs = q.toString().replace(/%2C/gi, ",");
  return `#${path}${qs ? `?${qs}` : ""}`;
}

export function useRoute() {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));

  useEffect(() => {
    const onHash = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = useCallback(
    (partial: Partial<Route>, opts?: { replace?: boolean }) => {
      const current = parseHash(window.location.hash);
      const next: Route = { ...current, ...partial };
      if (partial.view === "graph") next.caseId = null;
      if (partial.view === "results") next.caseId = null;
      const hash = buildHash(next);
      if (hash === window.location.hash) return;
      if (opts?.replace) {
        history.replaceState(null, "", hash);
        setRoute(next);
      } else {
        window.location.hash = hash;
      }
    },
    [],
  );

  return { route, navigate };
}
