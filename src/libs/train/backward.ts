import type { ModelData } from "../../types/model.js";
import type { RegularizationGradientFunction } from "../../types/regularization.js";
import {
  addMatX,
  hadamardVector,
  mulTMatMat,
  mulTMatVec,
  subVector,
} from "../arithmetics.js";
import { getDerivativeActivationFunctionActual } from "./af.js";
import { getDerivativeLossFunctionActual } from "./loss.js";
import type { OptimizationFunction } from "./optimization.js";

export function backwardPass(props: {
  answer: number[];
  model: ModelData;
  B: number;
  aMats: number[][][];
  zMats: number[][][];
  actualOptimizationFunction: OptimizationFunction;
  actualRegularizationGradientFunction: RegularizationGradientFunction | null;
}) {
  const {
    answer,
    model,
    B,
    aMats,
    zMats,
    actualOptimizationFunction,
    actualRegularizationGradientFunction,
  } = props;
  // 逆伝播
  const dMats: number[][][] = model.layers.map(() => null as any); // NOTE: dMats[0] は使わない
  for (let k = model.layers.length - 1; k >= 1; k--) {
    const currLayer = model.layers[k];
    // console.log(`層 ${k} の逆伝播を開始します`);
    // zの誤差ベクトルの計算
    // console.log(`zの誤差ベクトルを計算します: ${k}`);
    switch (currLayer.layerType) {
      case "output": {
        // 出力層
        const aMat = aMats[k];
        const actualDerivativeLossFunction = getDerivativeLossFunctionActual(
          model.lossFunction
        );
        aMat.map((aVec, l) => {
          const y00 = answer[l]; // 悪性(M)なら1, 良性(B)なら0
          const y01 = 1 - y00;
          const dVec = actualDerivativeLossFunction(aVec, [y00, y01]);
          dMats[k] = dMats[k] ?? [];
          dMats[k].push(dVec);
        });
        break;
      }
      case "hidden": {
        // 隠れ層
        const zMat = zMats[k];
        const dMatNext = dMats[k + 1];
        const wNext = model.parameters[k].weights;
        const actualGDash = getDerivativeActivationFunctionActual(
          currLayer.activationFunction
        );
        zMat.map((zVec, j) => {
          const dVec = hadamardVector(
            zVec.map(actualGDash),
            mulTMatVec(wNext, dMatNext[j])
          );
          dMats[k] = dMats[k] ?? [];
          dMats[k].push(dVec);
        });
        break;
      }
      default:
        throw new Error(`未知の層タイプ: ${(currLayer as any).layerType}`);
    }
    // console.log(`層 ${k} のzの誤差ベクトルが計算されました`);

    // パラメータの誤差ベクトルを計算
    // console.log(`パラメータの誤差ベクトルを計算します: ${k}`);
    const w = model.parameters[k - 1].weights;
    const prevLayer = model.layers[k - 1];
    let dW: number[][] = Array.from({ length: currLayer.size }, () =>
      Array(prevLayer.size).fill(0)
    );
    let dB: number[] = new Array(currLayer.size).fill(0);
    const aMatPrev = aMats[k - 1];
    const dMat = dMats[k];
    dMat.map((dVec, l) => {
      // 重みの誤差行列
      let dWMat = mulTMatMat([dVec], [aMatPrev[l]]);
      if (actualRegularizationGradientFunction) {
        addMatX(dWMat, actualRegularizationGradientFunction(w));
      }

      const dBMat = dVec;
      for (let i = 0; i < currLayer.size; i++) {
        const dWi = dW[i];
        const dWMati = dWMat[i];
        for (let j = 0; j < prevLayer.size; j++) {
          dWi[j] += dWMati[j];
        }
        dB[i] += dBMat[i];
      }
    });

    for (let i = 0; i < currLayer.size; i++) {
      for (let j = 0; j < prevLayer.size; j++) {
        dW[i][j] /= B;
      }
      dB[i] /= B;
    }

    const b = model.parameters[k - 1].biases;
    // console.log(`層 ${k} のパラメータ更新を行います`);

    const gradientNorm = Math.sqrt(
      dW.flat().reduce((sum, x) => sum + x * x, 0) +
        dB.reduce((sum, x) => sum + x * x, 0)
    );
    const maxNorm = 5.0; // 閾値
    if (gradientNorm > maxNorm) {
      const scale = maxNorm / gradientNorm;
      dW = dW.map((row) => row.map((x) => x * scale));
      dB = dB.map((x) => x * scale);
    }

    actualOptimizationFunction(w, b, dW, dB, k - 1);
  }
}
