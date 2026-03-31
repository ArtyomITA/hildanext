import clsx from "clsx";
import { NavLink } from "react-router-dom";
import styles from "./TopNav.module.css";

const ITEMS = [
  {
    to: "/chat",
    match: ["/chat", "/inference"],
    label: "Inferenza",
    sublabel: "Chat AR + dLLM",
  },
  {
    to: "/inferenceplus",
    match: ["/inferenceplus"],
    label: "Inferenza+",
    sublabel: "RCD + OTS Search",
  },
  {
    to: "/benchmark",
    match: ["/benchmark"],
    label: "Benchmark",
    sublabel: "Stage 0 Validation",
  },
  {
    to: "/legacy/wsd",
    match: ["/legacy/wsd", "/wsd"],
    label: "WSD Legacy",
    sublabel: "Run logs + diagnostica",
  },
];

export function TopNav({ pathname }: { pathname: string }) {
  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <div className={styles.mark}>
          <span />
          <span />
        </div>
        <div>
          <p className={styles.kicker}>HildaNext Observatory</p>
          <h1 className={styles.title}>
            Studio chat-first per inferenza locale AR + dLLM
          </h1>
        </div>
      </div>
      <nav className={styles.nav} aria-label="Primary">
        {ITEMS.map((item) => (
          <NavLink
            key={item.to}
            className={({ isActive }) =>
              clsx(
                styles.link,
                isActive || item.match.some((prefix) => pathname.startsWith(prefix)) ? styles.active : null,
              )
            }
            to={item.to}
          >
            <strong>{item.label}</strong>
            <span>{item.sublabel}</span>
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
