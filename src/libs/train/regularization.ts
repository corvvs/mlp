import type {
  RegularizationFunction,
  RegularizationGradientFunction,
  RegularizationMethod,
} from "../../types/regularization.js";
import { mulMatrixByScalar } from "../arithmetics.js";

export function parseRegularizationMethod(
  str: string | null
): RegularizationMethod | null {
  if (!str) {
    return null;
  }
  const parts = str.split(",");
  const method = parts[0];
  switch (method) {
    case "L2": {
      const f: RegularizationMethod = {
        method: "L2",
        lambda: 0.01,
      };
      if (parts.length >= 2) {
        const lambda = parseFloat(parts[1]);
        if (isNaN(lambda) || lambda < 0) {
          throw new Error(`不正なL2正則化パラメータ: ${parts[1]}`);
        }
        f.lambda = lambda;
      }
      return f;
    }
    default:
      throw new Error(`未知の正則化方式: ${method}`);
  }
}

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
            const wMati = wMat[i];
            for (let j = 0; j < wMati.length; j++) {
              sumSq += wMati[j] * wMati[j];
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
      return (wMat: number[][]) =>
        mulMatrixByScalar(wMat, regularization.lambda);
    default:
      throw new Error(`未知の正則化タイプ: ${regularization.method}`);
  }
}
