import { PropsWithChildren, useEffect } from "react";
import { useDataStore } from "../store/dataStore";

export function AppProviders({ children }: PropsWithChildren) {
  const hydrate = useDataStore((state) => state.hydrate);
  const startPolling = useDataStore((state) => state.startPolling);
  const stopPolling = useDataStore((state) => state.stopPolling);

  useEffect(() => {
    void hydrate();
    // Always poll the live backend for WSD updates every 5 s.
    startPolling(5000);
    return () => stopPolling();
  }, [hydrate, startPolling, stopPolling]);

  return children;
}
