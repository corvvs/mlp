import { sprintf } from "sprintf-js";
import { readCSVFile, writeGNUPlotFile, writeJSONFile } from "../libs/io.js";
import { getLoss, getLossFunctionActual } from "../libs/train/loss.js";
import { getOptimizationFunctionActual } from "../libs/train/optimization.js";
import { printModel } from "../libs/print/model.js";
import { applyStandardization, standardizeData } from "../libs/train/data.js";
import { buildModelData } from "../libs/train/model.js";
import { forwardPass } from "../libs/train/forward.js";
import { backwardPass } from "../libs/train/backward.js";
import { splitData, splitDataBatch } from "../libs/split.js";
import {
  getRegularizationFunctionActual,
  getRegularizationGradientFunctionActual,
} from "../libs/train/regularization.js";
import type { ModelData } from "../types/model.js";
import type { TrainingProgress } from "../types/data.js";

export function command(props: {
  dataFilePath: string;
  modelOutFilePath: string;
}) {
  console.log("[Train]");

  // データファイルの読み取り
  const dataFilePath = props.dataFilePath;
  const csvRows = readCSVFile(dataFilePath);
  const { a: trainDataRaw, b: valDataRaw } = splitData(csvRows, 0.8);

  // 前処理: Answer列以外の標準化
  console.log("Standardizing Input...");
  const standardizedResult = standardizeData(trainDataRaw);
  const trainData = standardizedResult.rows;
  const valData = applyStandardization(
    valDataRaw,
    standardizedResult.scaleFactors
  );

  // // デバッグ用: 標準化後のデータをファイルに書き出す
  // writeCSVFile(
  //   "debug.csv",
  //   trainData.map((row) => row.map((v) => sprintf("%1.4f", v)))
  // );

  // 初期モデルの構築
  const model = buildModelData({
    seed: 1234,
    scaleFactors: standardizedResult.scaleFactors,
  });
  console.log("Initialized Model:");
  printModel(model);
  console.log();

  // とりあえず書き出してみる
  // writeJSONFile("debug.json", model);

  const actualLossFunction = getLossFunctionActual(model.lossFunction);
  const actualOptimizationFunction = getOptimizationFunctionActual(
    model.optimization,
    model.layers
  );
  const actualRegularizationFunction = getRegularizationFunctionActual(
    model.regularization
  );
  const actualRegularizationGradientFunction =
    getRegularizationGradientFunctionActual(model.regularization);
  const batchedData = splitDataBatch(
    trainData,
    model.batchSize || trainData.length
  );
  const fullSize = trainData.length;

  const maxEpochs = 5000;

  const progress: TrainingProgress[] = [];
  let lastTrainLoss: number = Infinity;
  let lastValLoss: number = Infinity;
  let valLossIncreaseCount = 0;
  let latestGoodModel: {
    epoch: number;
    loss: number;
    model: ModelData | null;
  } = {
    epoch: 0,
    loss: Infinity,
    model: null,
  };

  for (let epoch = 0; epoch < maxEpochs; epoch++) {
    // 学習
    let meanTrainLoss = 0;
    let trainAccuracy = 0;

    // ミニバッチ
    for (let b = 0; b < batchedData.length; b++) {
      // 順伝播
      const trainData = batchedData[b];
      const B = trainData.length;
      const { aMats: aMatsTrain, zMats: zMatsTrain } = forwardPass({
        inputVectors: trainData,
        model,
      });

      // 学習誤差の計算
      const { meanLoss: trainLoss, corrects: trainCorrects } = getLoss({
        inputVectors: trainData,
        outputMats: aMatsTrain,
        wMats: model.parameters.map((p) => p.weights),
        lossFunction: actualLossFunction,
        regularizationFunction: actualRegularizationFunction,
      });

      meanTrainLoss += trainLoss * B;
      trainAccuracy += trainCorrects;

      // 逆伝播
      backwardPass({
        inputVectors: trainData,
        model,
        B,
        aMats: aMatsTrain,
        zMats: zMatsTrain,
        actualOptimizationFunction,
        actualRegularizationGradientFunction,
      });
    }

    // 評価
    const { aMats: aMatsVal } = forwardPass({
      inputVectors: valData,
      model,
    });
    const { meanLoss: valLoss, corrects: valCorrects } = getLoss({
      inputVectors: valData,
      outputMats: aMatsVal,
      wMats: [],
      lossFunction: actualLossFunction,
      regularizationFunction: null,
    });

    const trainLoss = meanTrainLoss / fullSize;
    const trainLossDiff = trainLoss - lastTrainLoss;
    lastTrainLoss = trainLoss;
    const valLossDiff = valLoss - lastValLoss;
    lastValLoss = valLoss;
    trainAccuracy /= fullSize;
    const valAccuracy = valCorrects / valData.length;

    console.log(
      sprintf(
        "Epoch %4d: TrLoss = %1.6f(Diff = %+1.6f), ValLoss = %1.6f(Diff = %+1.6f), TrAcc = %1.2f, ValAcc = %1.2f",
        epoch + 1,
        trainLoss,
        trainLossDiff,
        valLoss,
        valLossDiff,
        trainAccuracy,
        valAccuracy
      )
    );
    progress.push({
      epoch: epoch + 1,
      trainLoss,
      valLoss,
      trainAccuracy,
      valAccuracy,
    });

    if (valLoss < latestGoodModel.loss) {
      latestGoodModel.loss = valLoss;
      latestGoodModel.model = JSON.parse(JSON.stringify(model));
      latestGoodModel.epoch = epoch + 1;
    }
    if (valLossDiff > -0.000001) {
      valLossIncreaseCount++;
      if (
        valLossIncreaseCount >= 10 ||
        valLoss > 2 * latestGoodModel.loss ||
        latestGoodModel.epoch < epoch - 100
      ) {
        console.log("Stopping.");
        break;
      }
    } else {
      valLossIncreaseCount = 0;
    }
  }

  console.log(`訓練が完了しました: Best Val Loss: ${latestGoodModel.loss}`);
  writeJSONFile("trained.json", latestGoodModel.model);
  writeGNUPlotFile("training_progress.dat", progress);
}
