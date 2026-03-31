import { Stage0Validation } from "../../features/stage0/Stage0Validation";
import styles from "./BenchmarkPage.module.css";

export function BenchmarkPage() {
  return (
    <div className={styles.page}>
      <Stage0Validation />
    </div>
  );
}
