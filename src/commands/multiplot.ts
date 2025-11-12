import { readFileSync } from "fs";
import type { ModelData } from "../types/model.js";
import { generateMultiModelSVG } from "../libs/svg-plot.js";
import { openInBrowser } from "../libs/browser.js";
import { readJSONFile } from "../libs/io.js";

export type MultiplotCommandOptions = {
  modelFilePaths: string[];
  outputFilePath: string;
  noPlot?: boolean;
  metrics: string[]; // 表示するメトリクスのリスト (例: ['loss', 'accuracy'])
};

export function command(options: MultiplotCommandOptions): void {
  const { modelFilePaths, outputFilePath, metrics } = options;

  if (modelFilePaths.length === 0) {
    throw new Error("少なくとも1つのモデルファイルを指定してください");
  }

  // メトリクスの検証
  const validMetrics = [
    "loss",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1score",
  ];
  const normalizedMetrics = metrics.map((m) => m.toLowerCase());

  for (const metric of normalizedMetrics) {
    if (!validMetrics.includes(metric)) {
      throw new Error(
        `無効なメトリクス: ${metric}。有効なメトリクス: ${validMetrics.join(
          ", "
        )}`
      );
    }
  }

  console.log(`${modelFilePaths.length}個のモデルファイルを読み込みます...`);

  // 各モデルファイルを読み込み
  const models: Array<{ path: string; data: ModelData }> = [];
  for (const filePath of modelFilePaths) {
    try {
      const data: ModelData = readJSONFile(filePath);
      models.push({ path: filePath, data });
    } catch (err) {
      console.error(`${filePath} の読み込みに失敗しました:`, err);
      throw err;
    }
  }

  generateMultiModelSVG(models, outputFilePath, normalizedMetrics);

  if (!options.noPlot) {
    openInBrowser(outputFilePath);
  } else {
    console.log("\nグラフの自動表示がスキップされました (--no-plot)");
  }
}
