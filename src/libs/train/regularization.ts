import type {
  RegularizationFunction,
  RegularizationGradientFunction,
  RegularizationMethod,
} from "../../types/regularization.js";
import { mulMatrixByScalar } from "../arithmetics.js";

export function getRegularizationFunctionActual(
  regularization?: RegularizationMethod
): RegularizationFunction | null {
  if (!regularization) {
    return null;
  }
  switch (regularization.method) {
    case "L2":
      return (wMats: number[][][]) => {
        let sumSq = 0;
        for (let k = 0; k < wMats.length; k++) {
          const wMat = wMats[k];
          for (let i = 0; i < wMat.length; i++) {
            for (let j = 0; j < wMat[i].length; j++) {
              sumSq += wMat[i][j] * wMat[i][j];
            }
          }
        }
        return (regularization.lambda / 2) * sumSq;
      };
    default:
      throw new Error(`未知の正則化タイプ: ${regularization.method}`);
  }
}

export function getRegularizationGradientFunctionActual(
  regularization?: RegularizationMethod
): RegularizationGradientFunction | null {
  if (!regularization) {
    return null;
  }
  switch (regularization.method) {
    case "L2":
      return (wMat: number[][], B: number) =>
        mulMatrixByScalar(wMat, regularization.lambda / B);
    default:
      throw new Error(`未知の正則化タイプ: ${regularization.method}`);
  }
}
