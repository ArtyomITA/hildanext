import { ReactNode } from "react";
import styles from "./MetricHeroCard.module.css";

export function MetricHeroCard({
  label,
  value,
  accent,
  meta,
}: {
  label: string;
  value: string;
  accent: "cyan" | "lime" | "orange" | "red";
  meta?: ReactNode;
}) {
  return (
    <article className={`${styles.card} ${styles[accent]}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      {meta ? <span>{meta}</span> : null}
    </article>
  );
}
