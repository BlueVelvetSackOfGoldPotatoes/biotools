import { useEffect, useRef, useState } from "react";

export function usePolling(fetcher, intervalMs, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [lastSuccessAt, setLastSuccessAt] = useState(null);
  const [nonce, setNonce] = useState(0);
  const timerRef = useRef(null);
  const seqRef = useRef(0);
  const controllerRef = useRef(null);

  useEffect(() => {
    let alive = true;

    setData(null);
    setError(null);
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }

    const tick = async () => {
      const seq = ++seqRef.current;
      const controller = new AbortController();
      controllerRef.current = controller;
      try {
        const next = await fetcher(controller.signal);
        if (alive && seq === seqRef.current) {
          setData(next);
          setError(null);
          setLastSuccessAt(Date.now());
        }
      } catch (err) {
        if (alive && seq === seqRef.current && err?.name !== "AbortError") {
          setError(String(err));
        }
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null;
        if (alive) {
          const hidden = typeof document !== "undefined" && document.visibilityState === "hidden";
          const effectiveInterval = hidden ? Math.max(intervalMs, 15000) : intervalMs;
          timerRef.current = setTimeout(tick, effectiveInterval);
        }
      }
    };

    tick();
    return () => {
      alive = false;
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      if (controllerRef.current) {
        controllerRef.current.abort();
        controllerRef.current = null;
      }
    };
  }, [intervalMs, nonce, ...deps]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    data,
    error,
    isLoading: data === null && error === null,
    lastSuccessAt,
    refresh: () => setNonce((v) => v + 1)
  };
}
