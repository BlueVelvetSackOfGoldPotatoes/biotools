const inFlightJsonRequests = new Map();

export async function fetchJson(url, options = {}) {
  const { signal, dedupe = true } = options;
  if (dedupe && inFlightJsonRequests.has(url)) {
    return inFlightJsonRequests.get(url);
  }

  const request = fetch(url, {
    cache: "no-store",
    signal,
    headers: {
      "Cache-Control": "no-cache",
      Pragma: "no-cache"
    }
  }).then(async (response) => {
    if (!response.ok) {
      let details = "";
      try {
        details = await response.text();
      } catch {
        details = "";
      }
      const suffix = details ? `: ${details.slice(0, 180)}` : "";
      throw new Error(`${response.status} ${response.statusText} for ${url}${suffix}`);
    }
    return response.json();
  }).finally(() => {
    if (dedupe) inFlightJsonRequests.delete(url);
  });

  if (dedupe) inFlightJsonRequests.set(url, request);
  return request;
}
