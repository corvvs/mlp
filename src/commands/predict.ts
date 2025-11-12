import { sprintf } from "sprintf-js";
import { readCSVFile, readJSONFile } from "../libs/io.js";
import { applyStandardization } from "../libs/train/data.js";
import { forwardPass } from "../libs/train/forward.js";
import type { ModelData } from "../types/model.js";
import { printModel } from "../libs/print/model.js";
import {
  getLoss,
  getLossFunctionActual,
  getMetrics,
} from "../libs/train/loss.js";

export function command(props: {
  modelFilePath: string;
  dataFilePath: string;
}) {
  console.log("[Predict]");

  // データファイルの読み取り
  const dataFilePath = props.dataFilePath;
  const csvRows = readCSVFile(dataFilePath);
  console.log(`データファイル ${dataFilePath} を読み込みました`);

  // モデルファイルの読み取り
  const modelFilePath = props.modelFilePath;
  const model = readJSONFile<ModelData>(modelFilePath);
  console.log(`モデルファイル ${modelFilePath} を読み込みました`);
  printModel(model);
  console.log();
  const actualLossFunction = getLossFunctionActual(model.lossFunction);

  const inputLayer = model.layers[0];
  if (inputLayer.layerType !== "input") {
    throw new Error("モデルの最初の層が入力層ではありません");
  }
  const testData = applyStandardization(csvRows, inputLayer.scaleFactors);
  console.log("データに標準化を適用しました");

  console.log("予測を開始します");
  const { aMats } = forwardPass({
    inputVectors: testData,
    model,
  });

  const {
    meanLoss: testLoss,
    tp,
    fp,
    tn,
    fn,
  } = getLoss({
    inputVectors: testData,
    outputMats: aMats,
    wMats: [],
    lossFunction: actualLossFunction,
    regularizationFunction: null,
  });

  console.log(
    sprintf(
      "%4s %4s %4s %4s %5s %5s %5s",
      "Result",
      "ID",
      "Ans",
      "Pred",
      "Class",
      "P(M)",
      "P(B)"
    )
  );
  const predictions = aMats[aMats.length - 1];
  let correctCount = 0;
  for (let i = 0; i < predictions.length; i++) {
    const inputRow = csvRows[i];
    const predRow = predictions[i];
    const yAnswer = inputRow[0];
    const yPredPos = predRow[0];
    const yPredNeg = predRow[1];
    const answerLabel = yAnswer === "1" ? "M" : "B";
    const predLabel = yPredPos >= 0.5 ? "M" : "B";
    const isCorrect = predLabel === answerLabel;
    const classification = isCorrect
      ? predLabel === "M"
        ? "TP"
        : "TN"
      : predLabel === "M"
      ? "FP"
      : "FN";
    if (isCorrect) {
      correctCount++;
      continue;
    }
    console.log(
      sprintf(
        "[%s]   %4d %4s %4s %5s %1.3f %1.3f",
        isCorrect ? "ok" : "KO",
        i + 1,
        answerLabel,
        predLabel,
        classification,
        yPredPos,
        yPredNeg
      )
    );
  }
  const testMetrics = getMetrics({ loss: testLoss, tp, tn, fp, fn });
  console.log(
    sprintf(
      "Loss: %1.4f, Accuracy: %d / %d = %1.2f%%, Precision: %02.2f%%, Recall: %02.2f%%, Specificity: %01.2f%%",
      testLoss,
      correctCount,
      predictions.length,
      (correctCount / predictions.length) * 100,
      testMetrics.precision * 100,
      testMetrics.recall * 100,
      testMetrics.specificity * 100
    )
  );
}
