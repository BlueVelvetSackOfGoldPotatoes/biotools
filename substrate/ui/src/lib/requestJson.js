export async function requestJson(url, { method = "GET", body, signal } = {}) {
  const response = await fetch(url, {
    method,
    cache: "no-store",
    signal,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-cache",
      Pragma: "no-cache"
    },
    body: body == null ? undefined : JSON.stringify(body)
  });
  if (!response.ok) {
    let details = "";
    try {
      details = await response.text();
    } catch {
      details = "";
    }
    throw new Error(`${response.status} ${response.statusText}${details ? `: ${details.slice(0, 200)}` : ""}`);
  }
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}
