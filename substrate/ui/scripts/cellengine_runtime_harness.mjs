import { chromium } from "playwright-core";

const UI_BASE_URL = process.env.CELLENGINE_UI_BASE_URL || "http://127.0.0.1:5173";
const API_BASE_URL = process.env.CELLENGINE_API_BASE_URL || "http://127.0.0.1:8787";
const CHROME_BIN = process.env.CHROME_BIN || "/usr/bin/google-chrome";
const ROUTES = [
  "?tab=cellengine",
  "?tab=cellengine&section=discover",
  "?tab=cellengine&section=replay",
  "?tab=cellengine&section=compare",
  "?tab=cellengine&section=reports"
];

function routeUrl(route) {
  return `${UI_BASE_URL}/${route}`;
}

function shouldFlagConsole(text) {
  return /Unable to preventDefault inside passive event listener invocation|ReferenceError|TypeError|Uncaught/i.test(text);
}

async function fetchJson(pathname, init) {
  const response = await fetch(`${API_BASE_URL}${pathname}`, init);
  const text = await response.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = null;
  }
  return {
    ok: response.ok,
    status: response.status,
    json,
    text
  };
}

function pushIssue(issues, scope, message, detail = null) {
  issues.push({ scope, message, detail });
}

async function runApiChecks(issues) {
  const health = await fetchJson("/api/health");
  if (!health.ok || !health.json?.ok) {
    pushIssue(issues, "api", "health endpoint failed", health);
    return null;
  }

  const overview = await fetchJson("/api/cellengine/overview");
  if (!overview.ok || !overview.json) {
    pushIssue(issues, "api", "overview endpoint failed", overview);
    return null;
  }
  if (!Array.isArray(overview.json.genome_options)) {
    pushIssue(issues, "api", "overview.genome_options missing or invalid", overview.json);
  }
  if (!Array.isArray(overview.json.recent_replays)) {
    pushIssue(issues, "api", "overview.recent_replays missing or invalid", overview.json);
  }

  const artifacts = await fetchJson("/api/cellengine/artifacts");
  if (!artifacts.ok || !artifacts.json) {
    pushIssue(issues, "api", "artifacts endpoint failed", artifacts);
    return { overview: overview.json, artifacts: null };
  }
  if (!Array.isArray(artifacts.json.benchmark_runs)) {
    pushIssue(issues, "api", "artifacts.benchmark_runs missing or invalid", artifacts.json);
  }
  if (!Array.isArray(artifacts.json.discovery_batches)) {
    pushIssue(issues, "api", "artifacts.discovery_batches missing or invalid", artifacts.json);
  }

  if (overview.json.genome_options?.length) {
    const previewProbe = await fetchJson("/api/cellengine/genome-previews", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        items: overview.json.genome_options.slice(0, 2).map((item) => ({
          genome_path: item.genome_path,
          summary_dir: item.summary_dir,
          label: item.label
        }))
      })
    });
    if (!previewProbe.ok || !Array.isArray(previewProbe.json?.items)) {
      pushIssue(issues, "api", "genome-previews probe failed", previewProbe);
    }
  }

  if (overview.json.recent_replays?.length) {
    const replayDir = overview.json.recent_replays[0]?.relative_output_dir;
    if (replayDir) {
      const replayProbe = await fetchJson(`/api/cellengine/replay?dir=${encodeURIComponent(replayDir)}`);
      if (!replayProbe.ok || !Array.isArray(replayProbe.json?.frames)) {
        pushIssue(issues, "api", "replay probe failed", replayProbe);
      }
    }
  }

  if (artifacts.json.discovery_batches?.length) {
    const batchDir = artifacts.json.discovery_batches[0]?.relative_output_dir;
    if (batchDir) {
      const batchProbe = await fetchJson(`/api/cellengine/discover/batch?dir=${encodeURIComponent(batchDir)}`);
      if (!batchProbe.ok || !batchProbe.json?.job || !batchProbe.json?.result) {
        pushIssue(issues, "api", "discover batch probe failed", batchProbe);
      }
    }
  }

  const activeProbe = await fetchJson("/api/cellengine/discover/active");
  if (!activeProbe.ok || !Object.prototype.hasOwnProperty.call(activeProbe.json || {}, "job")) {
    pushIssue(issues, "api", "discover active probe failed", activeProbe);
  }

  return { overview: overview.json, artifacts: artifacts.json };
}

async function withPage(browser, route, handler) {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });
  const messages = [];
  const failedRequests = [];
  page.on("console", (msg) => {
    messages.push({ type: msg.type(), text: msg.text() });
  });
  page.on("pageerror", (err) => {
    messages.push({ type: "pageerror", text: err.message || String(err) });
  });
  page.on("requestfailed", (request) => {
    failedRequests.push({
      url: request.url(),
      method: request.method(),
      failure: request.failure()?.errorText || "request failed"
    });
  });

  await page.goto(routeUrl(route), { waitUntil: "load", timeout: 20000 });
  await page.waitForTimeout(1200);
  try {
    await handler(page, messages, failedRequests);
  } finally {
    await page.close();
  }
  return { messages, failedRequests };
}

async function runUiChecks(issues) {
  const browser = await chromium.launch({
    headless: true,
    executablePath: CHROME_BIN
  });

  try {
    for (const route of ROUTES) {
      const result = await withPage(browser, route, async () => {});
      const badMessages = result.messages.filter((entry) => shouldFlagConsole(entry.text));
      if (badMessages.length) {
        pushIssue(issues, route, "route emitted runtime errors", badMessages);
      }
      if (result.failedRequests.length) {
        pushIssue(issues, route, "route had failed network requests", result.failedRequests);
      }
    }

    const compareResult = await withPage(browser, "?tab=cellengine&section=compare", async (page) => {
      await page.waitForSelector(".cellengine-orbit-surface", { timeout: 15000 });
      const orbit = page.locator(".cellengine-orbit-surface").first();
      const orbitBox = await orbit.boundingBox();
      if (orbitBox) {
        await page.mouse.move(orbitBox.x + orbitBox.width / 2, orbitBox.y + orbitBox.height / 2);
        await page.mouse.wheel(0, 600);
        await page.mouse.wheel(0, -600);
        await page.mouse.down();
        await page.mouse.move(orbitBox.x + orbitBox.width / 2 + 90, orbitBox.y + orbitBox.height / 2 + 50, { steps: 8 });
        await page.mouse.up();
      }

      const surfacePreview = page.locator(".cellengine-surface-preview").first();
      await surfacePreview.click();
      await page.waitForSelector(".plot3d-host", { timeout: 15000 });
      const plot = page.locator(".plot3d-host").first();
      const plotBox = await plot.boundingBox();
      if (plotBox) {
        await page.mouse.move(plotBox.x + plotBox.width / 2, plotBox.y + plotBox.height / 2);
        await page.mouse.wheel(0, 700);
        await page.mouse.wheel(0, -700);
        await page.mouse.down();
        await page.mouse.move(plotBox.x + plotBox.width / 2 + 80, plotBox.y + plotBox.height / 2 + 40, { steps: 8 });
        await page.mouse.up();
      }
      await page.locator(".mini-btn").filter({ hasText: "close" }).first().click();
      await page.waitForTimeout(600);
    });
    const badCompareMessages = compareResult.messages.filter((entry) => shouldFlagConsole(entry.text));
    if (badCompareMessages.length) {
      pushIssue(issues, "compare-interaction", "compare interaction emitted runtime errors", badCompareMessages);
    }
    if (compareResult.failedRequests.length) {
      pushIssue(issues, "compare-interaction", "compare interaction had failed network requests", compareResult.failedRequests);
    }

    const reportsResult = await withPage(browser, "?tab=cellengine&section=reports", async (page) => {
      const evidenceSelector = page.locator("select").first();
      if (await evidenceSelector.count()) {
        const options = await evidenceSelector.locator("option").count();
        if (options > 1) {
          await evidenceSelector.selectOption({ index: 1 });
          await page.waitForTimeout(500);
        }
      }
    });
    const badReportsMessages = reportsResult.messages.filter((entry) => shouldFlagConsole(entry.text));
    if (badReportsMessages.length) {
      pushIssue(issues, "reports-interaction", "reports interaction emitted runtime errors", badReportsMessages);
    }
    if (reportsResult.failedRequests.length) {
      pushIssue(issues, "reports-interaction", "reports interaction had failed network requests", reportsResult.failedRequests);
    }
  } finally {
    await browser.close();
  }
}

async function main() {
  const issues = [];
  await runApiChecks(issues);
  await runUiChecks(issues);
  if (issues.length) {
    console.error(JSON.stringify({ ok: false, issues }, null, 2));
    process.exit(1);
  }
  console.log(JSON.stringify({ ok: true, ui_base_url: UI_BASE_URL, api_base_url: API_BASE_URL }, null, 2));
}

main().catch((err) => {
  console.error(JSON.stringify({ ok: false, fatal: String(err?.message || err) }, null, 2));
  process.exit(1);
});
