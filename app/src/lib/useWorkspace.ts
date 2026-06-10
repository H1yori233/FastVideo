import { useEffect, useState } from "react";
import { loadWorkspace, Workspace } from "./dataStore";

export function useWorkspace() {
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    loadWorkspace()
      .then((w) => alive && setWorkspace(w))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, []);

  return { workspace, error };
}
