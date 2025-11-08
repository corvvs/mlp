import { sprintf } from "sprintf-js";
import { readCSVFile, writeCSVFile, writeJSONFile } from "../libs/io.js";
import { getLossFunctionActual } from "../libs/train/loss.js";
import { getOptimizationFunctionActual } from "../libs/train/optimization.js";
import { printModel } from "../libs/print/model.js";
import { standardizeData } from "../libs/train/data.js";
import { buildModelData } from "../libs/train/model.js";
import { forwardPass } from "../libs/train/forward.js";
import { backwardPass } from "../libs/train/backward.js";

export function command(props: {
  dataFilePath: string;
  modelOutFilePath: string;
}) {
  console.log("[Train]");

  // データファイルの読み取り
  const dataFilePath = props.dataFilePath;
  const csvRows = readCSVFile(dataFilePath);

  // 前処理: Answer列以外の標準化
  console.log("Standardizing Input...");
  const standardizedResult = standardizeData(csvRows);

  // デバッグ用: 標準化後のデータをファイルに書き出す
  writeCSVFile(
    "debug.csv",
    standardizedResult.rows.map((row) => row.map((v) => sprintf("%1.4f", v)))
  );

  // 初期モデルの構築
  const model = buildModelData({
    scaleFactors: standardizedResult.scaleFactors,
  });
  console.log("Initialized Model:");
  printModel(model);
  console.log();

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
  let last_loss: number = Infinity;
  for (let epoch = 0; epoch < maxEpochs; epoch++) {
    // 順伝播
    const { aMats, zMats } = forwardPass({ inputVectors, model });

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

    const lossDiff = Math.abs(meanLoss - last_loss);
    last_loss = meanLoss;
    console.log(
      sprintf(
        "Epoch %4d / %4d: Loss = %1.6f (Diff = %1.6f)",
        epoch + 1,
        maxEpochs,
        meanLoss,
        lossDiff
      )
    );

    // 逆伝播
    backwardPass({
      inputVectors,
      model,
      B,
      aMats,
      zMats,
      actualOptimizationFunction,
    });
  }

  console.log("訓練が完了しました");
  writeJSONFile("trained.json", model);
}
