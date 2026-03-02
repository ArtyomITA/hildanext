import { LogSummary, NormalizedLogEntry } from "../domain/types";

const MAX_VISIBLE = 10_000;

type WorkerRequest =
  | { type: "load"; logs: NormalizedLogEntry[] }
  | { type: "filter"; query: string; levels: string[]; sources: string[] };

type WorkerResponse = {
  type: "snapshot";
  rows: NormalizedLogEntry[];
  summary: LogSummary;
};

let allLogs: NormalizedLogEntry[] = [];

function summarize(rows: NormalizedLogEntry[]): LogSummary {
  const summary: LogSummary = {
    total: allLogs.length,
    visible: rows.length,
    byLevel: {},
    byAction: {},
    byReason: {},
  };

  for (const row of rows) {
    summary.byLevel[row.level] = (summary.byLevel[row.level] ?? 0) + 1;
    if (row.action) {
      summary.byAction[row.action] = (summary.byAction[row.action] ?? 0) + 1;
    }
    if (row.reason) {
      summary.byReason[row.reason] = (summary.byReason[row.reason] ?? 0) + 1;
    }
  }

  return summary;
}

function filterRows(query: string, levels: string[], sources: string[]) {
  const normalizedQuery = query.trim().toLowerCase();
  const levelSet = new Set(levels);
  const sourceSet = new Set(sources);
  const rows: NormalizedLogEntry[] = [];

  for (let index = allLogs.length - 1; index >= 0; index -= 1) {
    const row = allLogs[index];
    if (levelSet.size > 0 && !levelSet.has(row.level)) {
      continue;
    }
    if (sourceSet.size > 0 && !sourceSet.has(row.source)) {
      continue;
    }

    if (
      normalizedQuery &&
      !`${row.message} ${row.action ?? ""} ${row.reason ?? ""} ${row.module ?? ""}`
        .toLowerCase()
        .includes(normalizedQuery)
    ) {
      continue;
    }

    rows.push(row);
    if (rows.length >= MAX_VISIBLE) {
      break;
    }
  }

  rows.reverse();
  return rows;
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  if (event.data.type === "load") {
    allLogs = event.data.logs;
    const rows = allLogs.slice(-MAX_VISIBLE);
    const response: WorkerResponse = {
      type: "snapshot",
      rows,
      summary: summarize(rows),
    };
    self.postMessage(response);
    return;
  }

  const rows = filterRows(event.data.query, event.data.levels, event.data.sources);
  const response: WorkerResponse = {
    type: "snapshot",
    rows,
    summary: summarize(rows),
  };
  self.postMessage(response);
};
