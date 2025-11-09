import type { LossFunction, LossFunctionActual } from "../../types/loss.js";
import type { RegularizationFunction } from "../../types/regularization.js";

export function getLossFunctionActual(
  lossFunction: LossFunction
): LossFunctionActual {
  switch (lossFunction.method) {
    case "BCE":
      // ** ベクトルサイズ2を仮定し, スカラー値の組が与えられるとする **
      const eps = lossFunction.eps;
      return (yAnswer, yPred) => {
        const yPredC = Math.min(Math.max(yPred, eps), 1 - eps); // Math.log(0) を防ぐためのクランピング
        return -(
          yAnswer * Math.log(yPredC) +
          (1 - yAnswer) * Math.log(1 - yPredC)
        );
      };
    default:
      throw new Error(`未知の損失関数: ${lossFunction}`);
  }
}

export function getLoss(props: {
  inputVectors: number[][];
  outputMat: number[][];
  wMats: number[][][];
  lossFunction: LossFunctionActual;
  regularizationFunction: RegularizationFunction | null;
}) {
  let meanLoss = 0;
  const {
    inputVectors,
    outputMat,
    lossFunction,
    wMats,
    regularizationFunction,
  } = props;
  const regularizationContribution = regularizationFunction
    ? regularizationFunction(wMats)
    : 0;
  let corrects = 0;
  inputVectors.forEach((inputVector, k) => {
    const outVector = outputMat[k];
    const yAnswer = inputVector[0];
    const yPred = outVector[0]; // 正解ラベル1に対応する出力
    const isCorrect = (yPred >= 0.5 ? 1 : 0) === yAnswer ? 1 : 0;

    // 損失の計算
    const loss = lossFunction(yAnswer, yPred) + regularizationContribution;
    meanLoss += loss;
    corrects += isCorrect;
  });
  meanLoss /= inputVectors.length;
  return { meanLoss, corrects };
}
