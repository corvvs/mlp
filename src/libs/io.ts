import { readFileSync, writeFileSync } from "fs";
import type { TrainingProgress } from "../types/data.js";

/**
 * 前処理対象のCSVファイルを読み, 行の配列として返す
 * @param path
 * @returns
 */
export function readCSVFile(path: string): string[][] {
  const data = readFileSync(path, "utf-8");
  //   console.log(`データファイル ${path ?? "stdin"} を読み込みました`);

  // dataを見出しなしのCSVと考え, パースする
  const rows = data
    .split("\n")
    .map((line) => line.split(","))
    .filter((row) => row.length > 1); // 空行を除去

  // サイズをチェック
  if (rows.length === 0) {
    throw new Error("データファイルが空です");
  }
  const numCols = rows[0].length;
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].length !== numCols) {
      throw new Error(
        `データファイルの各行の列数が一致しません at ${i}, expected ${numCols}, got ${rows[i].length}`
      );
    }
  }

  return rows;
}

export function writeCSVFile(path: string, rows: string[][]) {
  const data = rows.map((row) => row.join(",") + "\n").join("");
  writeFileSync(path, data, "utf-8");
}

export function writeJSONFile(path: string, obj: any) {
  const data = JSON.stringify(obj, null, 2);
  writeFileSync(path, data, "utf-8");
}

export function writeGNUPlotFile(path: string, data: TrainingProgress[]) {
  const lines = data
    .map((row) => {
      return `${row.epoch} ${row.trainLoss} ${row.valLoss} ${row.trainAccuracy} ${row.valAccuracy}\n`;
    })
    .join("");
  writeFileSync(path, lines, "utf-8");
}

export function readJSONFile<T>(path: string): T {
  const data = readFileSync(path, "utf-8");
  return JSON.parse(data) as T;
}
