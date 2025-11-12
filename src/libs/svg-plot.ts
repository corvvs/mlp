import { writeFileSync } from "fs";
import type { TrainingProgress } from "../types/data.js";
import type { ModelData } from "../types/model.js";

interface PlotConfig {
  width: number;
  height: number;
  marginTop: number;
  marginRight: number;
  marginBottom: number;
  marginLeft: number;
}

const defaultConfig: PlotConfig = {
  width: 800,
  height: 400,
  marginTop: 40,
  marginRight: 120,
  marginBottom: 60,
  marginLeft: 80,
};

/**
 * 学習曲線のSVGを生成する
 */
export function generateTrainingSVG(
  progress: TrainingProgress[],
  outputPath: string,
  modelInfo?: Partial<ModelData>
): void {
  if (progress.length === 0) {
    throw new Error("進捗データが空です");
  }

  // 2つのグラフ（Loss と Accuracy）を横並びで生成
  const svg = createDualPlotSVG(progress, modelInfo);
  writeFileSync(outputPath, svg, "utf-8");
  console.log(`SVGグラフを生成しました: ${outputPath}`);
}

/**
 * Loss と Accuracy の2つのグラフを含むSVGを生成
 */
function createDualPlotSVG(
  progress: TrainingProgress[],
  modelInfo?: Partial<ModelData>
): string {
  const totalWidth = 1000;
  const totalHeight = 920;
  const bestEpoch = modelInfo?.bestEpoch;

  const lossPlot = createMetricsPlot(
    progress,
    "Loss Curves",
    "Loss",
    defaultConfig,
    [
      { name: "Train Loss", value: (p) => p.trainLoss, class: "train-line" },
      { name: "Val Loss", value: (p) => p.valLoss, class: "val-line" },
    ],
    true,
    bestEpoch
  );

  const metricsPlot = createMetricsPlot(
    progress,
    "Metrics Curves",
    "Metrics",
    defaultConfig,
    [
      {
        name: "Train Accuracy",
        value: (p) => p.trainAccuracy,
        class: "train-line",
      },
      {
        name: "Val Accuracy",
        value: (p) => p.valAccuracy,
        class: "metric-accuracy",
      },
      {
        name: "Val Precision",
        value: (p) => p.valPrecision,
        class: "metric-precision",
      },
      {
        name: "Val Recall",
        value: (p) => p.valRecall,
        class: "metric-recall",
      },
      {
        name: "Val Specificity",
        value: (p) => p.valSpecificity,
        class: "metric-specificity",
      },
      {
        name: "Val F1-Score",
        value: (p) => p.valF1Score,
        class: "metric-f1score",
      },
    ],
    false,
    bestEpoch
  );

  const modelInfoText = modelInfo
    ? `Model: Epochs=${modelInfo.bestEpoch}, BatchSize=${modelInfo.batchSize}, Seed=${modelInfo.seed}`
    : "";

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${totalWidth}" height="${totalHeight}" viewBox="0 0 ${totalWidth} ${totalHeight}">
  <defs>
    <style>
      .plot-title { font: bold 18px sans-serif; fill: #333; }
      .axis-label { font: 14px sans-serif; fill: #666; }
      .tick-label { font: 12px sans-serif; fill: #999; }
      .grid-line { stroke: #e0e0e0; stroke-width: 1; }
      .axis-line { stroke: #333; stroke-width: 2; fill: none; }
      .train-line { stroke: #2196F3; stroke-width: 2; fill: none; stroke-dasharray: 5,5; }
      .val-line { stroke: #FF9800; stroke-width: 2; fill: none; }
      .legend-text { font: 12px sans-serif; fill: #333; }
      .info-text { font: 12px sans-serif; fill: #666; }
      
      /* Metrics用の追加スタイル */
      .metric-accuracy { stroke: #4CAF50; stroke-width: 1.5; fill: none; }
      .metric-precision { stroke: #2196F3; stroke-width: 1.5; fill: none; }
      .metric-recall { stroke: #FF9800; stroke-width: 1.5; fill: none; }
      .metric-specificity { stroke: #9C27B0; stroke-width: 1.5; fill: none; }
      .metric-f1score { stroke: #F44336; stroke-width: 1.5; fill: none; }
    </style>
  </defs>
  
  <!-- タイトル -->
  <text x="${
    totalWidth / 2
  }" y="25" class="plot-title" text-anchor="middle">Training Progress</text>
  ${
    modelInfoText
      ? `<text x="${
          totalWidth / 2
        }" y="45" class="info-text" text-anchor="middle">${modelInfoText}</text>`
      : ""
  }
  
  <!-- Loss Plot -->
  <g transform="translate(100, 70)">
    ${lossPlot}
  </g>
  
  <!-- Metrics Plot -->
  <g transform="translate(100, 490)">
    ${metricsPlot}
  </g>
</svg>`;
}

/**
 * 目盛り値を生成（線形スケール）
 */
function generateTicks(min: number, max: number, count: number): number[] {
  const range = max - min;
  const step = range / (count - 1);
  return Array.from({ length: count }, (_, i) => min + step * i);
}

/**
 * 複数のメトリクスを表示するプロットを生成
 */
function createMetricsPlot(
  progress: TrainingProgress[],
  title: string,
  yAxisName: string,
  config: PlotConfig,
  metricsDef: {
    name: string;
    class: string;
    value: (p: TrainingProgress) => number;
  }[],
  useLogScale: boolean,
  bestEpoch?: number
): string {
  const { width, height, marginTop, marginRight, marginBottom, marginLeft } =
    config;

  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;

  // データ抽出
  const epochs = progress.map((p) => p.epoch);

  // 各メトリクスのデータ
  const metrics = metricsDef.map((metricDef) => {
    const values = progress.map((p) => metricDef.value(p));
    return { name: metricDef.name, class: metricDef.class, values };
  });

  // スケール計算（全メトリクスの最小・最大を取得）
  const xMin = Math.min(...epochs);
  const xMax = Math.max(...epochs);
  const allValues = metrics.flatMap((m) => m.values);
  const yMin = Math.min(...allValues);
  const yMax = Math.max(...allValues);

  // Y軸の範囲を少し広げる（線形スケール）
  const yPadding = (yMax - yMin) * 0.1;
  let yMinPadded = Math.max(0, yMin - yPadding);
  let yMaxPadded = Math.min(1, yMax + yPadding);

  // スケール関数
  const scaleX = (x: number) =>
    marginLeft + ((x - xMin) / (xMax - xMin)) * plotWidth;

  let scaleY: (y: number) => number;

  if (useLogScale) {
    // 対数スケールの場合
    // 0やマイナスの値を避けるため、最小値は1e-10とする
    const logYMin = Math.log10(Math.max(yMin, 1e-10));
    const logYMax = Math.log10(Math.max(yMax, 1e-10));
    const logPadding = (logYMax - logYMin) * 0.1;
    const logYMinPadded = logYMin - logPadding;
    const logYMaxPadded = logYMax + logPadding;

    yMinPadded = Math.pow(10, logYMinPadded);
    yMaxPadded = Math.pow(10, logYMaxPadded);

    scaleY = (y: number) => {
      const logY = Math.log10(Math.max(y, 1e-10));
      return (
        marginTop +
        plotHeight -
        ((logY - logYMinPadded) / (logYMaxPadded - logYMinPadded)) * plotHeight
      );
    };
  } else {
    // 線形スケールの場合
    const yPadding = (yMax - yMin) * 0.1;
    yMinPadded = Math.max(0, yMin - yPadding);
    yMaxPadded = yMax + yPadding;

    scaleY = (y: number) =>
      marginTop +
      plotHeight -
      ((y - yMinPadded) / (yMaxPadded - yMinPadded)) * plotHeight;
  }

  // 各メトリクスのパス生成
  const metricPaths = metrics.map((metric) => {
    const path = epochs
      .map((epoch, i) => {
        const x = scaleX(epoch);
        const y = scaleY(metric.values[i]);
        return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
      })
      .join(" ");
    return { name: metric.name, path, class: metric.class };
  });

  // 凡例の生成
  const legendItems = metricPaths
    .map((metric, i) => {
      const y = marginTop + 20 + i * 20;
      return `
    <line x1="${marginLeft + plotWidth + 20}" y1="${y}" x2="${
        marginLeft + plotWidth + 60
      }" y2="${y}" class="${metric.class}"/>
    <text x="${marginLeft + plotWidth + 70}" y="${y + 5}" class="legend-text">${
        metric.name
      }</text>`;
    })
    .join("\n");

  return generateMetricsPlot({
    width,
    height,
    marginTop,
    marginRight,
    marginBottom,
    marginLeft,
    legendItems,
    metricPaths,
    title,
    xAxisName: "Epoch",
    yAxisName,
    bestEpoch,
    xMin,
    xMax,
    yMinPadded,
    yMaxPadded,
    scaleY,
  });
}

function generateMetricsPlot(
  props: PlotConfig & {
    legendItems: string;
    metricPaths: { path: string; class: string }[];
    title: string;
    xAxisName: string;
    yAxisName: string;
    bestEpoch: number | undefined;
    xMin: number;
    xMax: number;
    yMinPadded: number;
    yMaxPadded: number;
    scaleY: (y: number) => number;
  }
) {
  const {
    width,
    height,
    marginLeft,
    marginTop,
    marginBottom,
    marginRight,
    metricPaths,
    legendItems,
    title,
    xAxisName,
    yAxisName,
    bestEpoch,
    xMin,
    xMax,
    yMinPadded,
    yMaxPadded,
    scaleY,
  } = props;
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;
  const scaleX = (x: number) =>
    marginLeft + ((x - xMin) / (xMax - xMin)) * plotWidth;

  // X軸の目盛り（エポック数）
  const xTicks = generateTicks(xMin, xMax, 5);
  const xAxisLabels = xTicks
    .map((tick) => {
      const x = scaleX(tick);
      return `<text x="${x}" y="${
        marginTop + plotHeight + 35
      }" class="tick-label" text-anchor="middle">${Math.round(tick)}</text>`;
    })
    .join("\n    ");

  // グリッド線とY軸の目盛り
  const yTicks = generateTicks(yMinPadded, yMaxPadded, 5);
  const gridLines = yTicks
    .map((tick) => {
      const y = scaleY(tick);
      return `<line x1="${marginLeft}" y1="${y}" x2="${
        marginLeft + plotWidth
      }" y2="${y}" class="grid-line"/>`;
    })
    .join("\n    ");

  const yAxisLabels = yTicks
    .map((tick) => {
      const y = scaleY(tick);
      return `<text x="${marginLeft - 10}" y="${
        y + 4
      }" class="tick-label" text-anchor="end">${tick.toFixed(2)}</text>`;
    })
    .join("\n    ");

  return `
    <!-- Plot Area -->
    <rect x="${marginLeft}" y="${marginTop}" width="${plotWidth}" height="${plotHeight}" fill="white" stroke="#ddd"/>
    
    <!-- Grid Lines -->
    ${gridLines}
    
    <!-- Axes -->
    <line x1="${marginLeft}" y1="${marginTop}" x2="${marginLeft}" y2="${
    marginTop + plotHeight
  }" class="axis-line"/>
    <line x1="${marginLeft}" y1="${marginTop + plotHeight}" x2="${
    marginLeft + plotWidth
  }" y2="${marginTop + plotHeight}" class="axis-line"/>
    
    <!-- Data Lines -->
    ${metricPaths
      .map((metric) => `<path d="${metric.path}" class="${metric.class}"/>`)
      .join("\n    ")}
    
    <!-- Axis Labels -->
    <text x="${marginLeft + plotWidth / 2}" y="${
    marginTop + plotHeight + 60
  }" class="axis-label" text-anchor="middle">${xAxisName}</text>
    <text x="${marginLeft - 60}" y="${
    marginTop + plotHeight / 2
  }" class="axis-label" text-anchor="middle" transform="rotate(-90, ${
    marginLeft - 60
  }, ${marginTop + plotHeight / 2})">${yAxisName}</text>
    
    <!-- Y Axis Ticks -->
    ${yAxisLabels}
    
    <!-- X Axis Ticks -->
    ${xAxisLabels}
    
    <!-- Title -->
    <text x="${marginLeft + plotWidth / 2}" y="${
    marginTop - 10
  }" class="plot-title" text-anchor="middle">${title}</text>
    
    <!-- Best Epoch Line -->
    ${
      bestEpoch
        ? `<line x1="${scaleX(bestEpoch)}" y1="${marginTop}" x2="${scaleX(
            bestEpoch
          )}" y2="${
            marginTop + plotHeight
          }" stroke="#E91E63" stroke-width="2" stroke-dasharray="3,3"/>
    <rect x="${scaleX(bestEpoch) - 30}" y="${
            marginTop + plotHeight + 5
          }" width="60" height="16" fill="white" stroke="#E91E63" stroke-width="1" rx="3"/>
    <text x="${scaleX(bestEpoch)}" y="${
            marginTop + plotHeight + 16
          }" class="info-text" text-anchor="middle" fill="#E91E63" font-weight="bold">Best: ${bestEpoch}</text>`
        : ""
    }
    
    <!-- Legend -->
    ${legendItems}
  `;
}
