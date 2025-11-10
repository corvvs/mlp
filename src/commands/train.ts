import { sprintf } from "sprintf-js";
import { readCSVFile, writeGNUPlotFile, writeJSONFile } from "../libs/io.js";
import {
  getLoss,
  getLossFunctionActual,
  getMetrics,
  getMetricsImprovement,
} from "../libs/train/loss.js";
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
import type { EpochMetrics } from "../types/loss.js";
import type { ActivationFunctionSingleArgumentMethod } from "../types/af.js";
import { defaultModelFilePath } from "../constants.js";
import { shuffleArray } from "../libs/random.js";
import type { RegularizationMethod } from "../types/regularization.js";

export function command(props: {
  dataFilePath: string;
  modelOutFilePath: string;
  epochs: number;
  seed?: number;
  batchSize: number;
  defaultActivationFunction?: ActivationFunctionSingleArgumentMethod;
  hiddenLayerSizes: number[];
  regularization: RegularizationMethod | null;
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
  const { seed, batchSize } = props;
  const model = buildModelData({
    maxEpochs: props.epochs,
    seed: seed ?? null,
    batchSize: batchSize ?? null,
    scaleFactors: standardizedResult.scaleFactors,
    defaultActivationFunction: props.defaultActivationFunction ?? "ReLU",
    hiddenLayerSizes: props.hiddenLayerSizes,
    regularization: props.regularization,
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
  const fullSize = trainData.length;

  const progress: TrainingProgress[] = [];
  let lastValMetrics: EpochMetrics = {
    loss: Infinity,
    accuracy: Infinity,
    precision: Infinity,
    recall: Infinity,
    specificity: Infinity,
    f1Score: Infinity,
  };
  let valLossIncreaseCount = 0;
  let latestGoodModel: {
    epoch: number;
    score: number;
    model: ModelData | null;
  } = {
    epoch: 0,
    score: Infinity,
    model: null,
  };

  console.log(
    sprintf(
      "%5s %7s %7s %7s %7s %7s %7s %7s",
      "Epoch",
      "TrLoss",
      "TrAcc",
      "ValLoss",
      "ValAcc",
      "ValSpec",
      "ValRec",
      "ValF1"
    )
  );
  for (let epoch = 0; epoch < model.maxEpochs; epoch++) {
    // 学習
    let meanTrainLoss = 0;
    let trainAccuracy = 0;
    model.bestEpoch = epoch + 1;
    const trainTfpn = {
      tp: 0,
      tn: 0,
      fp: 0,
      fn: 0,
    };

    // ミニバッチ
    const batchedData = splitDataBatch(
      shuffleArray(trainData),
      model.batchSize || trainData.length
    );
    for (let b = 0; b < batchedData.length; b++) {
      // 順伝播
      const trainData = batchedData[b];
      const answerVector = trainData.map((row) => row[0]);
      const B = trainData.length;
      const { aMats: aMatsTrain, zMats: zMatsTrain } = forwardPass({
        inputVectors: trainData,
        model,
      });

      // 学習誤差の計算
      const {
        meanLoss: trainLoss,
        corrects: trainCorrects,
        ...batchTfpn
      } = getLoss({
        inputVectors: trainData,
        outputMats: aMatsTrain,
        wMats: model.parameters.map((p) => p.weights),
        lossFunction: actualLossFunction,
        regularizationFunction: actualRegularizationFunction,
      });
      trainTfpn.tp += batchTfpn.tp;
      trainTfpn.tn += batchTfpn.tn;
      trainTfpn.fp += batchTfpn.fp;
      trainTfpn.fn += batchTfpn.fn;

      meanTrainLoss += trainLoss * B;
      trainAccuracy += trainCorrects;

      // 逆伝播
      backwardPass({
        answer: answerVector,
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
    const {
      meanLoss: valLoss,
      corrects: valCorrects,
      ...valTfpn
    } = getLoss({
      inputVectors: valData,
      outputMats: aMatsVal,
      wMats: [],
      lossFunction: actualLossFunction,
      regularizationFunction: null,
    });

    const trainLoss = meanTrainLoss / fullSize;
    trainAccuracy /= fullSize;
    const trainMetrics = getMetrics({ loss: trainLoss, ...trainTfpn });
    const valMetrics = getMetrics({ loss: valLoss, ...valTfpn });
    model.trainMetrics.push(trainMetrics);
    model.valMetrics.push(valMetrics);

    const valAccuracy = valCorrects / valData.length;

    const valMetricsImprovement = getMetricsImprovement(
      lastValMetrics,
      valMetrics
    );
    lastValMetrics = valMetrics;

    console.log(
      sprintf(
        "%5d %1.5f %1.5f %1.5f %1.5f %1.5f %1.5f %1.5f",
        epoch + 1,
        trainLoss,
        trainAccuracy,
        valLoss,
        valAccuracy,
        valMetrics.specificity,
        valMetrics.recall,
        valMetrics.f1Score
      )
    );
    progress.push({
      epoch: epoch + 1,
      trainLoss,
      valLoss,
      trainAccuracy,
      valAccuracy,
    });

    const scoreToMinimize = valMetrics.loss;
    if (scoreToMinimize < latestGoodModel.score) {
      latestGoodModel.score = scoreToMinimize;
      latestGoodModel.model = JSON.parse(JSON.stringify(model));
      latestGoodModel.epoch = epoch + 1;
    }
    if (valMetricsImprovement.loss > -0.0001) {
      valLossIncreaseCount++;
      if (
        valLossIncreaseCount >= 10 ||
        scoreToMinimize > 2 * latestGoodModel.score ||
        latestGoodModel.epoch < epoch - 100
      ) {
        console.log("Stopping.");
        break;
      }
    } else {
      valLossIncreaseCount = 0;
    }
  }

  console.log(
    `訓練が完了しました: Best Score: ${latestGoodModel.score} at epoch ${latestGoodModel.epoch}`
  );
  writeJSONFile(defaultModelFilePath, {
    ...latestGoodModel.model,
    trainMetrics: model.trainMetrics,
    valMetrics: model.valMetrics,
  });
  writeGNUPlotFile("training_progress.dat", progress);
}
