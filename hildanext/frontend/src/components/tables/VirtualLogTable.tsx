import { useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { NormalizedLogEntry } from "../../domain/types";
import { formatTimestamp } from "../../domain/formatters";
import { SeverityBadge } from "../badges/SeverityBadge";
import styles from "./VirtualLogTable.module.css";

export function VirtualLogTable({
  rows,
  selectedId,
  onSelect,
}: {
  rows: NormalizedLogEntry[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const parentRef = useRef<HTMLDivElement | null>(null);
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 54,
    overscan: 16,
  });

  return (
    <div className={styles.viewport} ref={parentRef}>
      <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, position: "relative" }}>
        {rowVirtualizer.getVirtualItems().map((item) => {
          const row = rows[item.index];
          return (
            <button
              key={row.id}
              className={`${styles.row} ${selectedId === row.id ? styles.selected : ""}`}
              type="button"
              style={{ transform: `translateY(${item.start}px)` }}
              onClick={() => onSelect(row.id)}
            >
              <span className={styles.time}>{formatTimestamp(row.tsUtc)}</span>
              <span className={styles.main}>
                <strong>{row.action ?? row.source}</strong>
                <small>{row.message}</small>
              </span>
              <SeverityBadge level={row.level}>{row.level}</SeverityBadge>
            </button>
          );
        })}
      </div>
    </div>
  );
}
