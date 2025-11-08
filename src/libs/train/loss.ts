import type { LossFunction, LossFunctionActual } from "../../types/loss.js";

export function getLossFunctionActual(
  lossFunction: LossFunction
): LossFunctionActual {
  switch (lossFunction.method) {
    case "BCE":
      // ** ベクトルサイズ2を仮定し, スカラー値の組が与えられるとする **
      const eps = lossFunction.eps;
      return (yTrue, yPred) => {
        const yPredC = Math.min(Math.max(yPred, eps), 1 - eps); // Math.log(0) を防ぐためのクランピング
        return -(yTrue * Math.log(yPredC) + (1 - yTrue) * Math.log(1 - yPredC));
      };
    default:
      throw new Error(`未知の損失関数: ${lossFunction}`);
  }
}

export function getLoss(props: {
  inputVectors: number[][];
  outputMat: number[][];
  lossFunction: LossFunctionActual;
}) {
  let meanLoss = 0;
  const { inputVectors, outputMat, lossFunction } = props;
  inputVectors.map((inputVector, k) => {
    const outVector = outputMat[k];
    const yTrue = inputVector[0];
    const yPred = outVector[0]; // 正解ラベル1に対応する出力

    // 損失の計算
    const loss = lossFunction(yTrue, yPred);
    meanLoss += loss;
  });
  meanLoss /= inputVectors.length;
  return meanLoss;
}
