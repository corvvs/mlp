import type { LossFunction } from "../../types/loss.js";

export function getLossFunctionActual(
  lossFunction: LossFunction
): (yTrue: number, yPred: number) => number {
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
