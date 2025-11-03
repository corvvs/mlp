import { sprintf } from "sprintf-js";
import { readCSVFile, writeCSVFile, writeJSONFile } from "../libs/io.js";
import type { LayerInfo, ScaleFactor } from "../types/layer.js";
import type { ModelData } from "../types/model.js";
import { initializeParams } from "../libs/train/initialization.js";
import type { Init } from "v8";
import type { InitializationMethod } from "../types/initialization.js";

function standardizeData(data: string[][]): {
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
    const mean = means1arr[j].sort().reduce((a, b) => a + b, 0) / n;
    means1[j] = mean;
    means2[j] = means2arr[j].sort().reduce((a, b) => a + b, 0) / n;
    // const variance =
    //   means1arr[j]
    //     .map((v) => (v - mean) ** 2)
    //     .sort()
    //     .reduce((a, b) => a + b, 0) / n;
    stddevs[j] = Math.sqrt(means2[j] - mean * mean);
    console.log(`列 ${j}: mean=${means1[j]}, stddev=${stddevs[j]}`);
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

function buildModelData(props: {
  scaleFactors: (ScaleFactor | null)[];
}): ModelData {
  const initialization: InitializationMethod = {
    method: "Xavier",
    dist: "uniform",
  };
  const layers: LayerInfo[] = [
    {
      // 入力層
      size: props.scaleFactors.length,
      scaleFactors: props.scaleFactors,
    },
    {
      // 出力層
      size: 2,
      activationFunction: {
        method: "softmax",
      },
    },
  ];
  const parameters = initializeParams({
    initialization,
    layers,
  });

  return {
    version: "1.0.0",

    batchSize: 0,
    layerNumber: 2,
    layers,
    initialization,

    lossFunction: {
      method: "BCE",
      eps: 1e-9,
    },

    optimization: {
      method: "SGD",
      learningRate: 0.01,
    },

    parameters,
  };
}

export function command(props: {
  dataFilePath: string;
  modelOutFilePath: string;
}) {
  console.log("訓練コマンドが呼び出されました");

  // データファイルの読み取り
  const dataFilePath = props.dataFilePath;
  const csvRows = readCSVFile(dataFilePath);

  // 前処理: Answer列以外の標準化
  console.log("データの標準化を行います...");
  const standardizedResult = standardizeData(csvRows);
  console.log("データの標準化が完了しました");
  console.log(standardizedResult.rows.length, standardizedResult.scaleFactors);

  // デバッグ用: 標準化後のデータを再度標準化してみる
  // const standardizedResult2 = standardizeData(
  //   standardizedResult.rows.map((row) => row.map((v) => sprintf("%1.4f", v)))
  // );
  // console.log(
  //   standardizedResult2.rows.length,
  //   standardizedResult2.scaleFactors
  // );

  // デバッグ用: 標準化後のデータをファイルに書き出す
  writeCSVFile(
    "debug.csv",
    standardizedResult.rows.map((row) => row.map((v) => sprintf("%1.4f", v)))
  );

  // 初期モデルの構築
  const model = buildModelData({
    scaleFactors: standardizedResult.scaleFactors,
  });
  console.log("初期モデルの構築が完了しました");
  model.parameters.forEach((param, i) => {
    console.log(
      `層 ${i}: weights=${param.weights.length}x${param.weights[0].length}, biases=${param.biases.length}`
    );
    console.log(param.weights);
    console.log(param.biases);
  });

  // とりあえず書き出してみる
  writeJSONFile("debug.json", model);
}
