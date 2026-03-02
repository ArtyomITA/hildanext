import clsx from "clsx";
import styles from "./SeverityBadge.module.css";

export function SeverityBadge({
  level,
  children,
}: {
  level: "info" | "notice" | "warning" | "error";
  children: string;
}) {
  return <span className={clsx(styles.badge, styles[level])}>{children}</span>;
}
