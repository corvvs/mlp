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
import type { EpochMetrics, LossFunction } from "../types/loss.js";
import type { ActivationFunctionSingleArgument } from "../types/af.js";
import { defaultModelFilePath } from "../constants.js";
import { shuffleArray } from "../libs/random.js";
import type { RegularizationMethod } from "../types/regularization.js";
import type { OptimizationMethod } from "../types/optimization.js";
import type { InitializationMethod } from "../types/initialization.js";
import type { EarlyStopping } from "../types/es.js";
import { getEarlyStoppingActual } from "../libs/train/es.js";

export function command(props: {
  dataFilePath: string;
  modelOutFilePath: string;
  epochs: number;
  splitRatio: number;
  seed?: number;
  batchSize: number;
  initialization: InitializationMethod;
  defaultActivationFunction: ActivationFunctionSingleArgument;
  hiddenLayerSizes: number[];
  lossFunction: LossFunction;
  regularization: RegularizationMethod | null;
  optimization: OptimizationMethod;
  earlyStopping: EarlyStopping | null;
}) {
  console.log("[Train]");

  // データファイルの読み取り
  const dataFilePath = props.dataFilePath;
  const csvRows = readCSVFile(dataFilePath);
  const { a: trainDataRaw, b: valDataRaw } = splitData(
    csvRows,
    props.splitRatio
  );

  // 前処理: Answer列以外の標準化
  console.log("Standardizing Input...");
  const standardizedResult = standardizeData(trainDataRaw);
  const trainData = standardizedResult.rows;
  const valData = applyStandardization(
    valDataRaw,
    standardizedResult.scaleFactors
  );

  // 初期モデルの構築
  const { seed, batchSize } = props;
  const model = buildModelData({
    maxEpochs: props.epochs,
    seed: seed ?? null,
    batchSize: batchSize ?? null,
    splitRatio: props.splitRatio,
    scaleFactors: standardizedResult.scaleFactors,
    initialization: props.initialization,
    defaultActivationFunction: props.defaultActivationFunction,
    hiddenLayerSizes: props.hiddenLayerSizes,
    lossFunction: props.lossFunction,
    regularization: props.regularization,
    optimization: props.optimization,
    earlyStopping: props.earlyStopping,
  });
  console.log("Initialized Model:");
  printModel(model);
  console.log();

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
  const actualEarlyStoppingFunction = getEarlyStoppingActual(
    model.earlyStopping ?? null
  );
  const fullSize = trainData.length;

  const progress: TrainingProgress[] = [];
  let lastValMetrics: EpochMetrics = {
    loss: Infinity,
    accuracy: -Infinity,
    precision: -Infinity,
    recall: -Infinity,
    specificity: -Infinity,
    f1Score: -Infinity,
  };
  let valLossIncreaseCount = 0;
  let latestGoodModel: {
    epoch: number;
    score: number;
    model: ModelData | null;
  } = {
    epoch: 0,
    score: Infinity,
    model: JSON.parse(JSON.stringify(model)),
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
        trainMetrics.accuracy,
        valLoss,
        valMetrics.accuracy,
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

    // [早期終了]
    // 「早期終了指標」を1つ指定し, その改善を監視する.
    //
    // 指定可能な早期終了指標(case-insensitive):
    // - 評価損失     loss
    // - 評価精度     accuracy
    // - 評価適合率   precision
    // - 評価再現率   recall
    // - 評価特異度   specificity
    // - 評価F1スコア f1Score
    //
    // 終了条件:
    // - 「早期終了指標」が最良値より改善しないエポックが一定以上続いたとき
    // - 「早期終了指標」が最良値の2倍以上に悪化したとき
    //
    // ハイパーパラメータとして指定可能な要素:
    // - 早期終了指標             early-stopping-metric
    // - 改善しないエポック数の閾値 early-stopping-patience
    if (actualEarlyStoppingFunction) {
      const esMessage = actualEarlyStoppingFunction(
        model,
        latestGoodModel,
        epoch + 1,
        valMetrics
      );
      if (esMessage) {
        console.log(esMessage);
        break;
      }
    }

    // const scoreToMinimize = valMetrics.loss;
    // if (scoreToMinimize < latestGoodModel.score) {
    //   latestGoodModel.score = scoreToMinimize;
    //   latestGoodModel.model = JSON.parse(JSON.stringify(model));
    //   latestGoodModel.epoch = epoch + 1;
    // }
    // if (valMetricsImprovement.loss > -0.00001) {
    //   valLossIncreaseCount++;
    //   if (
    //     valLossIncreaseCount >= 10 ||
    //     scoreToMinimize > 2 * latestGoodModel.score ||
    //     latestGoodModel.epoch < epoch - 100
    //   ) {
    //     console.log("Stopping.");
    //     break;
    //   }
    // } else {
    //   valLossIncreaseCount = 0;
    // }
  }

  // 学習終了
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
