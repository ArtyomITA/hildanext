import clsx from "clsx";
import { NavLink } from "react-router-dom";
import styles from "./TopNav.module.css";

const ITEMS = [
  { to: "/wsd", label: "WSD", sublabel: "Warmup / Stable / Decay" },
  { to: "/inference", label: "Inference", sublabel: "AR vs diffusion" },
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
            Frontend-only control room for WSD and diffusion inference
          </h1>
        </div>
      </div>
      <nav className={styles.nav} aria-label="Primary">
        {ITEMS.map((item) => (
          <NavLink
            key={item.to}
            className={({ isActive }) =>
              clsx(styles.link, isActive || pathname === item.to ? styles.active : null)
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
