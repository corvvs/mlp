import type { ModelData } from "../../types/model.js";
import { sumNumArray } from "../arithmetics.js";
import { getActivationFunctionActual, softmax } from "./af.js";

export function forwardPass(props: {
  inputVectors: number[][];
  model: ModelData;
}) {
  const { inputVectors, model } = props;
  let aMat: number[][] = inputVectors.map((row) => row.slice(1));
  const aMats: number[][][] = [aMat];
  const zMats: number[][][] = [[]];
  for (let k = 1; k < model.layers.length; k++) {
    const currLayer = model.layers[k];
    const w = model.parameters[k - 1].weights;
    const b = model.parameters[k - 1].biases;

    // 内部チェック
    const ww = w[0].length;
    const ah = aMat[0].length;
    if (ww !== ah) {
      throw new Error(
        `重み行列のサイズと前層の出力サイズが一致しません: weights=${ww}, a=${ah}`
      );
    }
    const wh = w.length;
    const bh = b.length;
    if (wh !== bh) {
      throw new Error(
        `重み行列のサイズとバイアスのサイズが一致しません: weights=${wh}, biases=${bh}`
      );
    }

    // 活性化前出力の計算
    const zMat = aMat.map((a, k) =>
      w.map((wRow, i) => sumNumArray(wRow.map((wij, j) => wij * a[j])) + b[i])
    );
    zMats.push(zMat);

    // 活性化後出力の計算
    switch (currLayer.layerType) {
      case "output": {
        // 出力層: softmax
        const aMatNext = zMat.map(softmax);
        aMat = aMatNext;
        break;
      }
      case "hidden": {
        // 隠れ層: 指定された活性化関数
        const afActual = getActivationFunctionActual(
          currLayer.activationFunction
        );
        const aMatNext = zMat.map((zRow, k) => zRow.map(afActual));
        aMat = aMatNext;
        break;
      }
      default:
        throw new Error(`未知の層タイプ: ${(currLayer as any).layerType}`);
    }

    aMats.push(aMat);
  }
  return {
    aMats,
    zMats,
  };
}
