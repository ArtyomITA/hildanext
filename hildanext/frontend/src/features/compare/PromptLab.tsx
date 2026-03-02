import { ChangeEvent } from "react";
import { Panel } from "../../components/layout/Panel";
import styles from "./PromptLab.module.css";

export interface PromptLabControls {
  prompt: string;
  temperature: number;
  topP: number;
  maxNewTokens: number;
  seed: number;
  mode: "S_MODE" | "Q_MODE";
  effort: "instant" | "low" | "medium" | "high" | "adaptive";
  tauMask: number;
  tauEdit: number;
  scenarioFlavor: "clean" | "degenerate" | "dummy";
}

interface PromptLabProps {
  controls: PromptLabControls;
  onChange: <K extends keyof PromptLabControls>(key: K, value: PromptLabControls[K]) => void;
  onGenerate: () => void;
  onReset: () => void;
  /** When provided a second button "▶ Run AR on Qwen" appears and calls this handler. */
  onGenerateReal?: () => void;
  /** Disables the real-run button and changes its label to "Running…" */
  realRunning?: boolean;
  sourceLabel?: string;
  disabled?: boolean;
}

function onNumericChange(
  event: ChangeEvent<HTMLInputElement>,
  callback: (value: number) => void,
  integer = false,
) {
  const value = integer ? Number.parseInt(event.target.value, 10) : Number.parseFloat(event.target.value);
  callback(Number.isFinite(value) ? value : 0);
}

export function PromptLab({
  controls,
  onChange,
  onGenerate,
  onReset,
  onGenerateReal,
  realRunning,
  sourceLabel,
  disabled,
}: PromptLabProps) {
  return (
    <Panel
      kicker="Prompt lab"
      title="Run inference from your own prompt"
      actions={
        sourceLabel ? (
          <span className={styles.badge}>{sourceLabel}</span>
        ) : undefined
      }
    >
      <div className={styles.grid}>
        <label className={styles.promptBlock}>
          <span>Prompt</span>
          <textarea
            rows={5}
            value={controls.prompt}
            onChange={(event) => onChange("prompt", event.target.value)}
            placeholder="Write the prompt you want to test here."
            disabled={disabled}
          />
        </label>

        <div className={styles.controls}>
          <label>
            <span>Temperature</span>
            <input
              type="range"
              min="0"
              max="1.5"
              step="0.01"
              value={controls.temperature}
              onChange={(event) => onNumericChange(event, (value) => onChange("temperature", value))}
              disabled={disabled}
            />
            <strong>{controls.temperature.toFixed(2)}</strong>
          </label>

          <label>
            <span>Top P</span>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.01"
              value={controls.topP}
              onChange={(event) => onNumericChange(event, (value) => onChange("topP", value))}
              disabled={disabled}
            />
            <strong>{controls.topP.toFixed(2)}</strong>
          </label>

          <label>
            <span>Max new tokens</span>
            <input
              type="number"
              min="8"
              max="512"
              step="1"
              value={controls.maxNewTokens}
              onChange={(event) => onNumericChange(event, (value) => onChange("maxNewTokens", value), true)}
              disabled={disabled}
            />
          </label>

          <label>
            <span>Seed</span>
            <input
              type="number"
              min="0"
              max="999999"
              step="1"
              value={controls.seed}
              onChange={(event) => onNumericChange(event, (value) => onChange("seed", value), true)}
              disabled={disabled}
            />
          </label>

          <label>
            <span>Mode</span>
            <select
              value={controls.mode}
              onChange={(event) => onChange("mode", event.target.value as PromptLabControls["mode"])}
              disabled={disabled}
            >
              <option value="S_MODE">S_MODE</option>
              <option value="Q_MODE">Q_MODE</option>
            </select>
          </label>

          <label>
            <span>Effort</span>
            <select
              value={controls.effort}
              onChange={(event) => onChange("effort", event.target.value as PromptLabControls["effort"])}
              disabled={disabled}
            >
              <option value="instant">instant</option>
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
              <option value="adaptive">adaptive</option>
            </select>
          </label>

          <label>
            <span>Tau mask</span>
            <input
              type="number"
              min="0.01"
              max="1"
              step="0.01"
              value={controls.tauMask}
              onChange={(event) => onNumericChange(event, (value) => onChange("tauMask", value))}
              disabled={disabled}
            />
          </label>

          <label>
            <span>Tau edit</span>
            <input
              type="number"
              min="0.01"
              max="1"
              step="0.01"
              value={controls.tauEdit}
              onChange={(event) => onNumericChange(event, (value) => onChange("tauEdit", value))}
              disabled={disabled}
            />
          </label>

          <label>
            <span>Profile</span>
            <select
              value={controls.scenarioFlavor}
              onChange={(event) =>
                onChange("scenarioFlavor", event.target.value as PromptLabControls["scenarioFlavor"])
              }
              disabled={disabled}
            >
              <option value="clean">clean</option>
              <option value="degenerate">degenerate</option>
              <option value="dummy">dummy</option>
            </select>
          </label>
        </div>
      </div>

      <div className={styles.actions}>
        <button className={styles.primary} type="button" onClick={onGenerate} disabled={disabled || realRunning}>
          Generate mock run
        </button>
        {onGenerateReal && (
          <button
            className={styles.realRun}
            type="button"
            onClick={onGenerateReal}
            disabled={disabled || realRunning}
          >
            {realRunning ? "Running…" : "▶ Run AR on Qwen"}
          </button>
        )}
        <button className={styles.secondary} type="button" onClick={onReset}>
          Reset to scenario
        </button>
      </div>
    </Panel>
  );
}
