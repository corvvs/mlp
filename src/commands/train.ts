import { sprintf } from "sprintf-js";
import { readCSVFile, writeCSVFile, writeJSONFile } from "../libs/io.js";
import type { LayerInfo, ScaleFactor } from "../types/layer.js";
import type { ModelData } from "../types/model.js";
import { initializeParams } from "../libs/train/initialization.js";
import type { Init } from "v8";
import type { InitializationMethod } from "../types/initialization.js";
import {
  getActivationFunctionActual,
  getDerivativeActivationFunctionActual,
  softmax,
} from "../libs/train/af.js";
import { getLossFunctionActual } from "../libs/train/loss.js";
import {
  hadamardVector,
  mulTMatMat,
  mulTMatVec,
  subVector,
} from "../libs/arithmetics.js";
import {
  getOptimizationFunctionActual,
  makeOptimizationParam,
} from "../libs/train/optimization.js";

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

  const actualInputSize = props.scaleFactors.length - 1; // Answer列を除く

  const layers: LayerInfo[] = [
    {
      // 入力層
      layerType: "input",
      size: actualInputSize,
      scaleFactors: props.scaleFactors,
    },
    // 隠れ層
    {
      layerType: "hidden",
      size: 100,
      activationFunction: {
        method: "ReLU",
      },
    },
    {
      layerType: "hidden",
      size: 10,
      activationFunction: {
        method: "ReLU",
      },
    },
    {
      // 出力層
      layerType: "output",
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
    layerNumber: layers.length,
    layers,
    initialization,

    lossFunction: {
      method: "BCE",
      eps: 1e-9,
    },

    optimization: makeOptimizationParam({
      method: "Adam",
    }),

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
    // console.log(param.weights);
    // console.log(param.biases);
  });

  // とりあえず書き出してみる
  writeJSONFile("debug.json", model);

  const inputVectors = standardizedResult.rows;
  const actualLossFunction = getLossFunctionActual(model.lossFunction);
  const B = inputVectors.length;
  const actualOptimizationFunction = getOptimizationFunctionActual(
    model.optimization,
    model.layers
  );

  // とりあえず1エポック, バッチサイズ0(=全データ)でやってみる
  const maxEpochs = 1000;
  for (let epoch = 0; epoch < maxEpochs; epoch++) {
    console.log(`エポック ${epoch + 1} / ${maxEpochs} を開始します`);

    // 順伝播
    let aMat: number[][] = inputVectors.map((row) => row.slice(1));
    const aMats: number[][][] = [aMat];
    const zMats: number[][][] = [[]];
    for (let k = 1; k < model.layers.length; k++) {
      const prevLayer = model.layers[k - 1];
      const currLayer = model.layers[k];
      // console.log(
      //   `層 ${k} の順伝播を開始します: ${prevLayer.size} -> ${currLayer.size}`
      // );
      const w = model.parameters[k - 1].weights;
      const b = model.parameters[k - 1].biases;

      // 内部チェック
      const ww = w[0].length;
      const ah = aMat[0].length;
      if (ww !== ah) {
        throw new Error(
          `重み行列のサイズと前層の出力サイズが一致しません: weights=${ww}, a=${ah}`
        );
      }
      const wh = w.length;
      const bh = b.length;
      if (wh !== bh) {
        throw new Error(
          `重み行列のサイズとバイアスのサイズが一致しません: weights=${wh}, biases=${bh}`
        );
      }

      // 活性化前出力の計算
      const zMat = aMat.map((a, k) =>
        w.map(
          (wRow, i) =>
            wRow
              .map((wij, j) => wij * a[j])
              .sort()
              .reduce((sum, val) => sum + val, 0) + b[i]
        )
      );
      zMats.push(zMat);
      // console.log(`層 ${k} の活性化前出力が計算されました`);

      // 活性化後出力の計算
      switch (currLayer.layerType) {
        case "output": {
          // 出力層: softmax
          const aMatNext = zMat.map(softmax);
          aMat = aMatNext;
          break;
        }
        case "hidden": {
          // 隠れ層: 指定された活性化関数
          const afActual = getActivationFunctionActual(
            currLayer.activationFunction
          );
          const aMatNext = zMat.map((zRow, k) => zRow.map(afActual));
          aMat = aMatNext;
          break;
        }
        default:
          throw new Error(`未知の層タイプ: ${(currLayer as any).layerType}`);
      }
      // console.log(`層 ${k} の順伝播が完了しました`);
      aMats.push(aMat);
    }

    // 学習誤差の計算
    let meanLoss = 0;
    inputVectors.map((inputVector, k) => {
      const outVector = aMats[aMats.length - 1][k];
      const yTrue = inputVector[0];
      const yPred = outVector[0]; // 正解ラベル1に対応する出力

      // 損失の計算
      const loss = actualLossFunction(yTrue, yPred);
      meanLoss += loss;
    });
    meanLoss /= B;
    console.log(`学習誤差 (平均損失): ${meanLoss}`);
    // console.log("aMats:", aMats.length);

    // 逆伝播
    const dMats: number[][][] = model.layers.map(() => null as any); // NOTE: dMats[0] は使わない
    for (let k = model.layers.length - 1; k >= 1; k--) {
      const currLayer = model.layers[k];
      // console.log(`層 ${k} の逆伝播を開始します`);
      // zの誤差ベクトルの計算
      // console.log(`zの誤差ベクトルを計算します: ${k}`);
      switch (currLayer.layerType) {
        case "output": {
          // 出力層
          const aMat = aMats[k];
          aMat.map((aVec, l) => {
            const y00 = inputVectors[l][0];
            const y01 = 1 - y00;
            const y = [y00, y01];
            const dVec = subVector(aVec, y);
            dMats[k] = dMats[k] ?? [];
            dMats[k].push(dVec);
          });
          break;
        }
        case "hidden": {
          // 隠れ層
          const zMat = zMats[k];
          const dMatNext = dMats[k + 1];
          const wNext = model.parameters[k].weights;
          const actualGDash = getDerivativeActivationFunctionActual(
            currLayer.activationFunction
          );
          zMat.map((zVec, j) => {
            const dVec = hadamardVector(
              zVec.map(actualGDash),
              mulTMatVec(wNext, dMatNext[j])
            );
            dMats[k] = dMats[k] ?? [];
            dMats[k].push(dVec);
          });
          break;
        }
        default:
          throw new Error(`未知の層タイプ: ${(currLayer as any).layerType}`);
      }
      // console.log(`層 ${k} のzの誤差ベクトルが計算されました`);

      // パラメータの誤差ベクトルを計算
      // console.log(`パラメータの誤差ベクトルを計算します: ${k}`);
      const prevLayer = model.layers[k - 1];
      const dW: number[][] = Array.from({ length: currLayer.size }, () =>
        Array(prevLayer.size).fill(0)
      );
      const dB: number[] = new Array(currLayer.size).fill(0);
      const aMatPrev = aMats[k - 1];
      const dMat = dMats[k];
      dMat.map((dVec, l) => {
        // 重みの誤差行列
        const dWMat = mulTMatMat([dVec], [aMatPrev[l]]);
        // console.log(
        //   `dVec: (${dVec.length}), dWMat: (${dWMat.length}, ${dWMat[0].length})`
        // );
        const dBMat = dVec;
        for (let i = 0; i < currLayer.size; i++) {
          for (let j = 0; j < prevLayer.size; j++) {
            dW[i][j] += dWMat[i][j];
          }
          dB[i] += dBMat[i];
        }
      });

      for (let i = 0; i < currLayer.size; i++) {
        for (let j = 0; j < prevLayer.size; j++) {
          dW[i][j] /= B;
        }
        dB[i] /= B;
      }

      const w = model.parameters[k - 1].weights;
      const b = model.parameters[k - 1].biases;
      // console.log(`層 ${k} のパラメータ更新を行います`);
      actualOptimizationFunction(w, b, dW, dB, k - 1);
      // console.log(`層 ${k} の逆伝播が完了しました`);
    }
  }

  console.log("訓練が完了しました");
  writeJSONFile("trained.json", model);
}
