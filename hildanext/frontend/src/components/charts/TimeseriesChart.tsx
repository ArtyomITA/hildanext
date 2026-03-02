import { useEffect, useRef } from "react";
import uPlot from "uplot";
import styles from "./TimeseriesChart.module.css";

interface SeriesConfig {
  label: string;
  stroke: string;
  values: number[];
}

export function TimeseriesChart({
  x,
  series,
  yLabel,
}: {
  x: number[];
  series: SeriesConfig[];
  yLabel: string;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!rootRef.current) {
      return;
    }

    const data = [
      new Float64Array(x),
      ...series.map((item) => new Float64Array(item.values)),
    ] as uPlot.AlignedData;
    const plot = new uPlot(
      {
        width: rootRef.current.clientWidth || 600,
        height: 240,
        legend: { show: false },
        scales: { x: { time: false } },
        axes: [
          { stroke: "rgba(155,180,207,0.45)", grid: { stroke: "rgba(155,180,207,0.08)" } },
          { stroke: "rgba(155,180,207,0.45)", label: yLabel, grid: { stroke: "rgba(155,180,207,0.08)" } },
        ],
        series: [
          {},
          ...series.map((item) => ({
            label: item.label,
            stroke: item.stroke,
            width: 2,
          })),
        ],
      },
      data,
      rootRef.current,
    );

    plotRef.current = plot;

    const resizeObserver = new ResizeObserver(() => {
      requestAnimationFrame(() => {
        if (!rootRef.current || !plotRef.current) {
          return;
        }
        plotRef.current.setSize({ width: rootRef.current.clientWidth, height: 240 });
      });
    });

    resizeObserver.observe(rootRef.current);

    return () => {
      resizeObserver.disconnect();
      plot.destroy();
      plotRef.current = null;
    };
  }, [series, x, yLabel]);

  return <div className={styles.chart} ref={rootRef} />;
}
