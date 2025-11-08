import type { ModelData } from "../../types/model.js";
import {
  hadamardVector,
  mulTMatMat,
  mulTMatVec,
  subVector,
} from "../arithmetics.js";
import { getDerivativeActivationFunctionActual } from "./af.js";
import type { OpzimizationFunction } from "./optimization.js";

export function backwardPass(props: {
  inputVectors: number[][];
  model: ModelData;
  B: number;
  aMats: number[][][];
  zMats: number[][][];
  actualOptimizationFunction: OpzimizationFunction;
}) {
  const { inputVectors, model, B, aMats, zMats, actualOptimizationFunction } =
    props;
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
        aMat.map((aVec, l) => {
          const y00 = inputVectors[l][0];
          const y01 = 1 - y00;
          const y = [y00, y01];
          const dVec = subVector(aVec, y);
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
    const prevLayer = model.layers[k - 1];
    const dW: number[][] = Array.from({ length: currLayer.size }, () =>
      Array(prevLayer.size).fill(0)
    );
    const dB: number[] = new Array(currLayer.size).fill(0);
    const aMatPrev = aMats[k - 1];
    const dMat = dMats[k];
    dMat.map((dVec, l) => {
      // 重みの誤差行列
      const dWMat = mulTMatMat([dVec], [aMatPrev[l]]);
      // console.log(
      //   `dVec: (${dVec.length}), dWMat: (${dWMat.length}, ${dWMat[0].length})`
      // );
      const dBMat = dVec;
      for (let i = 0; i < currLayer.size; i++) {
        for (let j = 0; j < prevLayer.size; j++) {
          dW[i][j] += dWMat[i][j];
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

    const w = model.parameters[k - 1].weights;
    const b = model.parameters[k - 1].biases;
    // console.log(`層 ${k} のパラメータ更新を行います`);
    actualOptimizationFunction(w, b, dW, dB, k - 1);
  }
}
