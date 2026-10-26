const state = {
  report: null,
  selectedCaseId: null,
  activeTab: "overview",
  search: "",
  kindFilter: "all",
  statusFilter: "all",
};

const elements = {
  loadButton: document.getElementById("loadButton"),
  runButton: document.getElementById("runButton"),
  statusBanner: document.getElementById("statusBanner"),
  loadingState: document.getElementById("loadingState"),
  emptyState: document.getElementById("emptyState"),
  errorState: document.getElementById("errorState"),
  errorMessage: document.getElementById("errorMessage"),
  dashboard: document.getElementById("dashboard"),
  totalCases: document.getElementById("totalCases"),
  passedCases: document.getElementById("passedCases"),
  failedCases: document.getElementById("failedCases"),
  baselineStatus: document.getElementById("baselineStatus"),
  gatewayTerminals: document.getElementById("gatewayTerminals"),
  fallbackSuccesses: document.getElementById("fallbackSuccesses"),
  baselineDetail: document.getElementById("baselineDetail"),
  baselineChanges: document.getElementById("baselineChanges"),
  categoryDeltas: document.getElementById("categoryDeltas"),
  runDetails: document.getElementById("runDetails"),
  coverageContent: document.getElementById("coverageContent"),
  caseSearch: document.getElementById("caseSearch"),
  kindFilter: document.getElementById("kindFilter"),
  statusFilter: document.getElementById("statusFilter"),
  resultsBody: document.getElementById("resultsBody"),
  selectedCaseLabel: document.getElementById("selectedCaseLabel"),
  changedFields: document.getElementById("changedFields"),
  failureList: document.getElementById("failureList"),
  expectedJson: document.getElementById("expectedJson"),
  actualJson: document.getElementById("actualJson"),
  signatureChanges: document.getElementById("signatureChanges"),
  newlyFailing: document.getElementById("newlyFailing"),
  newlyPassing: document.getElementById("newlyPassing"),
  driftChanges: document.getElementById("driftChanges"),
  gwSuccess: document.getElementById("gwSuccess"),
  gwError: document.getElementById("gwError"),
  gwFallback: document.getElementById("gwFallback"),
  gwImmediate: document.getElementById("gwImmediate"),
  gwAttempts: document.getElementById("gwAttempts"),
  gwFallbackAttempted: document.getElementById("gwFallbackAttempted"),
  gatewayCaseLabel: document.getElementById("gatewayCaseLabel"),
  attemptBody: document.getElementById("attemptBody"),
  tabs: [...document.querySelectorAll(".tab")],
  panels: [...document.querySelectorAll(".tab-panel")],
};

function setBanner(message, type = "info") {
  elements.statusBanner.textContent = message;
  elements.statusBanner.className = `banner ${type}`;
}

function showState(name) {
  elements.loadingState.classList.toggle("hidden", name !== "loading");
  elements.loadingState.setAttribute("aria-hidden", name !== "loading");
  elements.emptyState.classList.toggle("hidden", name !== "empty");
  elements.emptyState.setAttribute("aria-hidden", name !== "empty");
  elements.errorState.classList.toggle("hidden", name !== "error");
  elements.errorState.setAttribute("aria-hidden", name !== "error");
  elements.dashboard.classList.toggle("hidden", name !== "dashboard");
}

function activateTab(tabName) {
  state.activeTab = tabName;
  for (const tab of elements.tabs) {
    const active = tab.dataset.tab === tabName;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const panel of elements.panels) {
    const active = panel.id === `panel-${tabName}`;
    panel.classList.toggle("hidden", !active);
    panel.setAttribute("aria-hidden", active ? "false" : "true");
  }
}

function outcomeText(caseItem, side) {
  const data = caseItem[side];
  if (!data) return "unknown";
  if (caseItem.kind === "verify") {
    if (typeof data.ok === "boolean") {
      return data.ok ? "valid" : (data.reason || "invalid");
    }
    return "unknown";
  }
  return data.outcome || "unknown";
}

function changedTopLevelFields(expected, actual) {
  const fields = new Set([...Object.keys(expected || {}), ...Object.keys(actual || {})]);
  const changed = [];
  for (const field of [...fields].sort()) {
    if (JSON.stringify(expected?.[field]) !== JSON.stringify(actual?.[field])) {
      changed.push(field);
    }
  }
  return changed;
}

function attemptChanges(caseItem) {
  const expectedAttempts = caseItem.expected?.attempts || [];
  const actualAttempts = caseItem.actual?.attempts || [];
  const changes = [];
  const count = Math.max(expectedAttempts.length, actualAttempts.length);
  for (let i = 0; i < count; i += 1) {
    const expected = expectedAttempts[i] || {};
    const actual = actualAttempts[i] || {};
    const fields = new Set([...Object.keys(expected), ...Object.keys(actual)]);
    for (const field of [...fields].sort()) {
      if (JSON.stringify(expected[field]) !== JSON.stringify(actual[field])) {
        changes.push(`attempt ${i + 1}: ${field}`);
      }
    }
  }
  if (expectedAttempts.length !== actualAttempts.length) {
    changes.push(`attempt count: ${expectedAttempts.length} expected, ${actualAttempts.length} actual`);
  }
  return changes;
}

function renderCoverage(coverage) {
  const container = elements.coverageContent;
  container.replaceChildren();
  const items = [
    ["Response failure reasons", coverage?.response_failure_reasons],
    ["Error categories", coverage?.error_categories],
    ["Attempt outcomes", coverage?.attempt_outcomes],
  ];
  for (const [label, value] of items) {
    if (!value) continue;
    const chip = document.createElement("span");
    chip.className = `chip ${value.complete ? "good" : "bad"}`;
    chip.textContent = `${label}: ${value.covered?.length || 0} covered, ${value.missing?.length || 0} missing`;
    container.appendChild(chip);
  }
  if (coverage?.error_categories?.documented_exemptions) {
    for (const [category, reason] of Object.entries(coverage.error_categories.documented_exemptions)) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `${category}: ${reason}`;
      container.appendChild(chip);
    }
  }
}

function renderBaseline(baseline) {
  elements.baselineDetail.textContent = baseline.present
    ? `Fingerprint ${baseline.fingerprint_match ? "matches" : "does not match"}.`
    : "No baseline was present.";
  elements.baselineStatus.textContent = baseline.present
    ? (baseline.fingerprint_match ? "match" : "drift")
    : "missing";
  elements.baselineStatus.className = baseline.present && baseline.fingerprint_match ? "good" : "bad";

  const container = elements.baselineChanges;
  container.replaceChildren();
  const changes = [
    ["New cases", baseline.new_cases],
    ["Removed cases", baseline.removed_cases],
    ["Kind changes", baseline.kind_changes],
    ["Newly failing", baseline.newly_failing],
    ["Newly passing", baseline.newly_passing],
    ["Signature changes", baseline.signature_changes],
  ];
  let hasChanges = false;
  for (const [label, values] of changes) {
    if (!values || values.length === 0) continue;
    hasChanges = true;
    const chip = document.createElement("span");
    chip.className = label === "Newly failing" ? "chip bad" : "chip";
    chip.textContent = `${label}: ${values.join(", ")}`;
    container.appendChild(chip);
  }
  if (!hasChanges) {
    const chip = document.createElement("span");
    chip.className = "chip good";
    chip.textContent = "No baseline regressions detected";
    container.appendChild(chip);
  }

  const deltaContainer = elements.categoryDeltas;
  deltaContainer.replaceChildren();
  const deltas = baseline.category_deltas || {};
  for (const [group, values] of Object.entries(deltas)) {
    for (const [key, value] of Object.entries(values || {})) {
      const chip = document.createElement("span");
      chip.className = value > 0 ? "chip bad" : "chip good";
      chip.textContent = `${group}: ${key} ${value > 0 ? "+" : ""}${value}`;
      deltaContainer.appendChild(chip);
    }
  }
}

function renderRunDetails(report) {
  const container = elements.runDetails;
  container.replaceChildren();
  const details = [
    ["Schema version", report.schema_version],
    ["Runner", `${report.runner?.name || "unknown"} v${report.runner?.version ?? "?"}`],
    ["Corpus cases", report.corpus?.case_count],
    ["Verify cases", report.corpus?.verify_count],
    ["Gateway cases", report.corpus?.gateway_count],
    ["Fingerprint", report.corpus?.fingerprint],
    ["Result", report.result?.ok ? "PASS" : "FAIL"],
    ["Exit code", report.result?.exit_code],
  ];
  for (const [key, value] of details) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value ?? "n/a";
    container.appendChild(dt);
    container.appendChild(dd);
  }
}

function filteredCases() {
  if (!state.report?.cases) return [];
  const search = state.search.trim().toLowerCase();
  return state.report.cases.filter((item) => {
    if (state.kindFilter !== "all" && item.kind !== state.kindFilter) return false;
    if (state.statusFilter === "pass" && !item.passed) return false;
    if (state.statusFilter === "fail" && item.passed) return false;
    if (!search) return true;
    const haystack = `${item.id} ${item.kind} ${item.passed ? "pass" : "fail"} ${item.failures?.join(" ") || ""}`.toLowerCase();
    return haystack.includes(search);
  });
}

function renderResults(report) {
  const body = elements.resultsBody;
  body.replaceChildren();
  for (const caseItem of filteredCases()) {
    const row = document.createElement("tr");
    row.dataset.caseId = caseItem.id;
    row.tabIndex = 0;
    if (caseItem.id === state.selectedCaseId) row.classList.add("selected");

    const statusClass = caseItem.passed ? "status-pass" : "status-fail";
    const statusText = caseItem.passed ? "PASS" : "FAIL";
    const cells = [
      caseItem.id,
      caseItem.kind,
      outcomeText(caseItem, "expected"),
      outcomeText(caseItem, "actual"),
      statusText,
    ];
    cells.forEach((value, index) => {
      const cell = document.createElement("td");
      if (index === 4) cell.className = statusClass;
      cell.textContent = value;
      row.appendChild(cell);
    });
    const select = () => selectCase(caseItem.id);
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
    body.appendChild(row);
  }
}

function selectCase(caseId) {
  state.selectedCaseId = caseId;
  const caseItem = state.report?.cases?.find((item) => item.id === caseId);
  if (!caseItem) return;

  elements.selectedCaseLabel.textContent = `${caseItem.id} (${caseItem.kind})`;
  const topChanges = changedTopLevelFields(caseItem.expected, caseItem.actual);
  const changes = [...topChanges.map((field) => `top-level: ${field}`), ...attemptChanges(caseItem)];

  elements.changedFields.replaceChildren();
  if (changes.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No changed fields detected.";
    elements.changedFields.appendChild(item);
  } else {
    for (const change of changes) {
      const item = document.createElement("li");
      item.textContent = change;
      elements.changedFields.appendChild(item);
    }
  }

  elements.failureList.replaceChildren();
  if (!caseItem.failures || caseItem.failures.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No failures.";
    elements.failureList.appendChild(item);
  } else {
    for (const failure of caseItem.failures) {
      const item = document.createElement("li");
      item.textContent = failure;
      elements.failureList.appendChild(item);
    }
  }

  elements.expectedJson.textContent = JSON.stringify(caseItem.expected, null, 2);
  elements.actualJson.textContent = JSON.stringify(caseItem.actual, null, 2);

  for (const row of elements.resultsBody.querySelectorAll("tr")) {
    row.classList.toggle("selected", row.dataset.caseId === caseId);
  }

  if (caseItem.kind === "gateway" && state.activeTab !== "gateway") {
    // keep gateway panel available; user can switch
  }
  renderGatewayAttempts(caseItem);
}

function renderGatewayAttempts(caseItem) {
  const body = elements.attemptBody;
  body.replaceChildren();
  if (!caseItem || caseItem.kind !== "gateway") {
    elements.gatewayCaseLabel.textContent = "Select a gateway case from Test cases.";
    return;
  }
  elements.gatewayCaseLabel.textContent = `${caseItem.id} (${caseItem.actual?.outcome || "unknown"})`;
  const attempts = caseItem.actual?.attempts || [];
  if (attempts.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.textContent = "No attempts recorded.";
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  for (const attempt of attempts) {
    const row = document.createElement("tr");
    const values = [
      attempt.attempt,
      attempt.provider,
      attempt.outcome,
      attempt.status_code ?? "-",
      attempt.error_category ?? "-",
      attempt.fallback_attempted ? "yes" : "no",
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    body.appendChild(row);
  }
}

function renderSignatureChanges(baseline) {
  const container = elements.signatureChanges;
  container.replaceChildren();
  const details = baseline.signature_details || [];
  if (details.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No signature changes detected.";
    container.appendChild(item);
    return;
  }
  for (const detail of details) {
    const item = document.createElement("li");
    const fields = (detail.changed_fields || []).join(", ") || "fields";
    item.textContent = `${detail.id}: ${fields}`;
    const sub = document.createElement("div");
    sub.style.fontSize = "0.78rem";
    sub.style.color = "var(--muted)";
    sub.textContent = `expected ${JSON.stringify(detail.expected)} -> actual ${JSON.stringify(detail.actual)}`;
    item.appendChild(sub);
    container.appendChild(item);
  }
}

function renderChipList(container, values, emptyText) {
  container.replaceChildren();
  if (!values || values.length === 0) {
    const chip = document.createElement("span");
    chip.className = "chip good";
    chip.textContent = emptyText;
    container.appendChild(chip);
    return;
  }
  for (const value of values) {
    const chip = document.createElement("span");
    chip.className = "chip bad";
    chip.textContent = value;
    container.appendChild(chip);
  }
}

function renderReport(report) {
  state.report = report;
  state.selectedCaseId = report.cases?.[0]?.id || null;

  elements.totalCases.textContent = report.summary?.total ?? report.cases?.length ?? 0;
  elements.passedCases.textContent = report.summary?.passed ?? 0;
  elements.failedCases.textContent = report.summary?.failed ?? 0;

  const gateway = report.gateway || {};
  elements.gatewayTerminals.textContent = `${gateway.terminal_results?.success ?? 0} / ${gateway.terminal_results?.error ?? 0}`;
  elements.fallbackSuccesses.textContent = gateway.fallback_success ?? 0;

  renderBaseline(report.baseline || {});
  renderCoverage(report.coverage || {});
  renderResults(report);
  renderRunDetails(report);
  renderSignatureChanges(report.baseline || {});

  renderChipList(elements.newlyFailing, report.baseline?.newly_failing, "No newly failing cases");
  renderChipList(elements.newlyPassing, report.baseline?.newly_passing, "No newly passing cases");

  const drift = [
    ...(report.baseline?.new_cases || []).map((id) => `new: ${id}`),
    ...(report.baseline?.removed_cases || []).map((id) => `removed: ${id}`),
    ...(report.baseline?.kind_changes || []).map((id) => `kind changed: ${id}`),
  ];
  renderChipList(elements.driftChanges, drift, "No baseline drift");

  elements.gwSuccess.textContent = gateway.terminal_results?.success ?? 0;
  elements.gwError.textContent = gateway.terminal_results?.error ?? 0;
  elements.gwFallback.textContent = gateway.fallback_success ?? 0;
  elements.gwImmediate.textContent = gateway.immediate_success ?? 0;
  elements.gwAttempts.textContent = gateway.attempts?.total ?? 0;
  elements.gwFallbackAttempted.textContent = gateway.attempts ? (gateway.fallback_attempted?.true ?? 0) : 0;

  if (state.selectedCaseId) selectCase(state.selectedCaseId);

  showState("dashboard");
  setBanner(`Evaluation complete: ${report.summary?.passed ?? 0}/${report.summary?.total ?? 0} cases passed.`, "success");
}

async function runEvaluation() {
  setBanner("Running evaluation...", "info");
  showState("loading");
  elements.runButton.disabled = true;
  elements.loadButton.disabled = true;
  try {
    const response = await fetch("/api/eval/run", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || payload.message || "Evaluation request failed");
    renderReport(payload);
  } catch (error) {
    showState("error");
    elements.errorMessage.textContent = error.message;
    setBanner("Evaluation failed.", "error");
  } finally {
    elements.runButton.disabled = false;
    elements.loadButton.disabled = false;
  }
}

async function loadReport() {
  setBanner("Loading saved report...", "info");
  showState("loading");
  elements.runButton.disabled = true;
  elements.loadButton.disabled = true;
  try {
    const response = await fetch("/api/eval/report");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || payload.message || "No saved report found");
    renderReport(payload);
  } catch (error) {
    showState("error");
    elements.errorMessage.textContent = error.message;
    setBanner("Could not load saved report.", "error");
  } finally {
    elements.runButton.disabled = false;
    elements.loadButton.disabled = false;
  }
}

elements.runButton.addEventListener("click", runEvaluation);
elements.loadButton.addEventListener("click", loadReport);
elements.caseSearch.addEventListener("input", (event) => {
  state.search = event.target.value;
  renderResults(state.report);
});
elements.kindFilter.addEventListener("change", (event) => {
  state.kindFilter = event.target.value;
  renderResults(state.report);
});
elements.statusFilter.addEventListener("change", (event) => {
  state.statusFilter = event.target.value;
  renderResults(state.report);
});
for (const tab of elements.tabs) {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
  tab.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      activateTab(tab.dataset.tab);
    }
  });
}

showState("empty");
setBanner("Ready. Run the evaluation or load a saved report.", "info");