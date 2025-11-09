import type { ScaleFactor } from "../../types/layer.js";
import { sumNumArray } from "../arithmetics.js";

export function standardizeData(data: string[][]): {
  rows: number[][];
  scaleFactors: (ScaleFactor | null)[];
} {
  const width = data[0].length;
  const n = data.length;
  const rows: number[][] = [];
  const means1: number[] = new Array(width).fill(0);
  const means2: number[] = new Array(width).fill(0);
  const means1arr: number[][] = means1.map(() => []);
  const means2arr: number[][] = means1.map(() => []);

  // string -> number への変換と, 平均(一乗, 二乗)の計算
  data.forEach((rawRow, i) => {
    const row: number[] = [];
    for (let j = 0; j < width; j++) {
      const rawVal = rawRow[j];
      const val = parseFloat(rawVal);
      if (!isFinite(val)) {
        throw new Error(`数値変換エラー at (${i}, ${j}): ${rawVal}`);
      }

      switch (j) {
        case 0: // Answer列: 標準化しない
          row.push(val);
          break;
        default:
          row.push(val);
          means1arr[j].push(val);
          means2arr[j].push(val * val);
          break;
      }
    }
    rows.push(row);
  });

  // 平均化, 標準偏差の計算
  const stddevs: number[] = new Array(width).fill(0);
  for (let j = 0; j < width; j++) {
    const mean = sumNumArray(means1arr[j]) / n;
    means1[j] = mean;
    means2[j] = sumNumArray(means2arr[j]) / n;
    stddevs[j] = Math.sqrt(means2[j] - mean * mean);
  }

  // 標準化の実行
  rows.forEach((row, i) => {
    for (let j = 0; j < width; j++) {
      switch (j) {
        case 0: // Answer列: 標準化しない
          break;
        default:
          if (stddevs[j] > 0) {
            row[j] = (row[j] - means1[j]) / stddevs[j];
          } else {
            // 標準偏差が0の場合は標準化しない
            row[j] = row[j] - means1[j];
          }
          break;
      }
    }
  });
  return {
    rows: rows,
    scaleFactors: stddevs.map((stddev, j) =>
      j === 0 ? null : { mean: means1[j], stddev: stddev }
    ),
  };
}

export function applyStandardization(
  data: string[][],
  scaleFactors: (ScaleFactor | null)[]
): number[][] {
  const width = data[0].length;
  return data.map((rawRow, i) => {
    const row: number[] = [];
    for (let j = 0; j < width; j++) {
      const rawVal = rawRow[j];
      const val = parseFloat(rawVal);
      if (!isFinite(val)) {
        throw new Error(`数値変換エラー at (${i}, ${j}): ${rawVal}`);
      }
      const sf = scaleFactors[j];
      if (sf === null) {
        // 標準化しない列
        row.push(val);
        continue;
      }
      const { mean, stddev } = sf;
      if (stddev > 0) {
        row.push((val - mean) / stddev);
      } else {
        row.push(val - mean);
      }
    }
    return row;
  });
}
