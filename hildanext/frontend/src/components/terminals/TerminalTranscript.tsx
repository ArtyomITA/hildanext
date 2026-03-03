import { useEffect, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { NormalizedLogEntry } from "../../domain/types";
import { formatTimestamp } from "../../domain/formatters";
import styles from "./TerminalTranscript.module.css";

export function TerminalTranscript({
  rows,
  pinnedTag,
}: {
  rows: NormalizedLogEntry[];
  pinnedTag?: string;
}) {
  const parentRef = useRef<HTMLDivElement | null>(null);
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 32,
    overscan: 20,
  });

  // Auto-scroll to bottom when new rows arrive.
  useEffect(() => {
    if (rows.length > 0) {
      rowVirtualizer.scrollToIndex(rows.length - 1, { align: "end" });
    }
  }, [rows.length, rowVirtualizer]);

  return (
    <div className={styles.terminal}>
      <div className={styles.header}>
        <span>PowerShell transcript</span>
        {pinnedTag ? <strong>{pinnedTag}</strong> : null}
      </div>
      <div className={styles.viewport} ref={parentRef}>
        <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, position: "relative" }}>
          {rowVirtualizer.getVirtualItems().map((item) => {
            const row = rows[item.index];
            return (
              <div key={row.id} className={styles.line} style={{ transform: `translateY(${item.start}px)` }}>
                <span>{row.tsUtc ? formatTimestamp(row.tsUtc) : ""}</span>
                <code>{row.message}</code>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
