import { sprintf } from "sprintf-js";
import { readCSVFile, writeCSVFile, writeJSONFile } from "../libs/io.js";
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

export function command(props: {
  dataFilePath: string;
  modelOutFilePath: string;
}) {
  console.log("[Train]");

  // データファイルの読み取り
  const dataFilePath = props.dataFilePath;
  const csvRows = readCSVFile(dataFilePath);
  const { trainData: trainDataRaw, testData: testDataRaw } = splitData(
    csvRows,
    0.8
  );

  // 前処理: Answer列以外の標準化
  console.log("Standardizing Input...");
  const standardizedResult = standardizeData(trainDataRaw);
  const trainData = standardizedResult.rows;
  const testData = applyStandardization(
    testDataRaw,
    standardizedResult.scaleFactors
  );

  // デバッグ用: 標準化後のデータをファイルに書き出す
  writeCSVFile(
    "debug.csv",
    trainData.map((row) => row.map((v) => sprintf("%1.4f", v)))
  );

  // 初期モデルの構築
  const model = buildModelData({
    seed: 111221111111,
    scaleFactors: standardizedResult.scaleFactors,
  });
  console.log("Initialized Model:");
  printModel(model);
  console.log();

  // とりあえず書き出してみる
  writeJSONFile("debug.json", model);

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

  // とりあえず1エポック, バッチサイズ0(=全データ)でやってみる
  const maxEpochs = 5000;
  let lastTrainLoss: number = Infinity;
  let lastTestLoss: number = Infinity;
  let testLossIncreaseCount = 0;
  for (let epoch = 0; epoch < maxEpochs; epoch++) {
    // 学習
    // 順伝播
    for (let b = 0; b < batchedData.length; b++) {
      const trainData = batchedData[b];
      const B = trainData.length;
      const { aMats: aMatsTrain, zMats: zMatsTrain } = forwardPass({
        inputVectors: trainData,
        model,
      });

      // 学習誤差の計算
      const trainLoss = getLoss({
        inputVectors: trainData,
        outputMat: aMatsTrain[aMatsTrain.length - 1],
        wMats: model.parameters.map((p) => p.weights),
        lossFunction: actualLossFunction,
        regularizationFunction: actualRegularizationFunction,
      });

      // const trainLossDiff = trainLoss - lastTrainLoss;
      // lastTrainLoss = trainLoss;

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
    const { aMats: aMatsTest } = forwardPass({
      inputVectors: testData,
      model,
    });
    const testLoss = getLoss({
      inputVectors: testData,
      outputMat: aMatsTest[aMatsTest.length - 1],
      wMats: [],
      lossFunction: actualLossFunction,
      regularizationFunction: null,
    });
    const testLossDiff = testLoss - lastTestLoss;
    lastTestLoss = testLoss;

    console.log(
      sprintf(
        "Epoch %4d / %4d: Test Loss = %1.6f(Diff = %+1.6f)",
        epoch + 1,
        maxEpochs,
        testLoss,
        testLossDiff
      )
    );

    if (testLossDiff > -0.0001) {
      testLossIncreaseCount++;
      if (testLossIncreaseCount >= 10) {
        console.log("Test loss increased for 10 consecutive epochs. Stopping.");
        break;
      }
    } else {
      testLossIncreaseCount = 0;
    }
  }

  console.log("訓練が完了しました");
  writeJSONFile("trained.json", model);
}
