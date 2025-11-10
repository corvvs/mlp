import { sprintf } from "sprintf-js";
import { readCSVFile, readJSONFile } from "../libs/io.js";
import { applyStandardization } from "../libs/train/data.js";
import { forwardPass } from "../libs/train/forward.js";
import type { ModelData } from "../types/model.js";
import { printModel } from "../libs/print/model.js";
import { getLoss, getLossFunctionActual } from "../libs/train/loss.js";

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

  const { meanLoss: testLoss } = getLoss({
    inputVectors: testData,
    outputMats: aMats,
    wMats: [],
    lossFunction: actualLossFunction,
    regularizationFunction: null,
  });

  const predictions = aMats[aMats.length - 1];
  let correctCount = 0;
  for (let i = 0; i < predictions.length; i++) {
    const inputRow = csvRows[i];
    const predRow = predictions[i];
    const yAnswer = inputRow[0];
    const yPred = predRow[0];
    const predLabel = yPred >= 0.5 ? "M" : "B";
    const isCorrect = predLabel === (yAnswer === "1" ? "M" : "B");
    if (isCorrect) {
      correctCount++;
      continue;
    }
    console.log(
      sprintf(
        "[%s] %4d: %s prob_M: %1.4f%%",
        isCorrect ? "ok" : "KO",
        i + 1,
        predLabel,
        yPred * 100
      )
    );
  }
  console.log(
    sprintf(
      "Loss: %1.4f, Accuracy: %d / %d = %1.2f%%",
      testLoss,
      correctCount,
      predictions.length,
      (correctCount / predictions.length) * 100
    )
  );
}
