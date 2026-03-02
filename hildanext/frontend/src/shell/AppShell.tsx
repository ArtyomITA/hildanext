import { Outlet, useLocation } from "react-router-dom";
import { TopNav } from "./TopNav";
import styles from "./AppShell.module.css";

export function AppShell() {
  const location = useLocation();

  return (
    <div className={styles.frame}>
      <div className={styles.backdrop} />
      <TopNav pathname={location.pathname} />
      <div className={styles.body}>
        <main className={`${styles.main} fade-in`}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
