import { useEffect, useRef } from "react";
import { TokenFrame } from "../../domain/types";
import styles from "./TokenMaskCanvas.module.css";

const STATE_COLORS = {
  prompt: "#5f7ab0",
  masked: "#1b2a3b",
  new: "#3fe0ff",
  edited: "#ffb648",
  stable: "#c9ff44",
};

export function TokenMaskCanvas({
  frames,
  selectedStep,
}: {
  frames: TokenFrame[];
  selectedStep: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frame = frames.find((item) => item.step === selectedStep) ?? frames[0];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);

    const cellWidth = 88;
    const cellHeight = 32;
    const gap = 8;
    const columns = Math.max(1, Math.floor(width / (cellWidth + gap)));

    frame.tokens.forEach((token, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const x = column * (cellWidth + gap);
      const y = row * (cellHeight + gap);
      context.fillStyle = STATE_COLORS[token.state];
      context.globalAlpha = token.state === "masked" ? 0.35 : 0.88;
      context.fillRect(x, y, cellWidth, cellHeight);
      context.globalAlpha = 1;
      context.fillStyle = "#081018";
      context.font = "12px IBM Plex Mono";
      context.fillText(token.text.slice(0, 10), x + 8, y + 19);
    });
  }, [frame]);

  return (
    <div className={styles.wrap}>
      <div className={styles.legend}>
        {Object.keys(STATE_COLORS).map((key) => (
          <span key={key}>
            <i style={{ background: STATE_COLORS[key as keyof typeof STATE_COLORS] }} />
            {key}
          </span>
        ))}
      </div>
      <canvas className={styles.canvas} ref={canvasRef} />
    </div>
  );
}
