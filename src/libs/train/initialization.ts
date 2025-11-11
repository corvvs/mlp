import type { InitializationMethod } from "../../types/initialization.js";
import type { LayerInfo } from "../../types/layer.js";
import type { LayerParameter } from "../../types/model.js";
import { localRandom, normalRandom } from "../random.js";

export function parseInitializationMethod(str: string): InitializationMethod {
  const parts = str.split(",");
  const m = parts[0];
  const dist = parts[1] ?? "uniform";
  switch (m.toLowerCase()) {
    case "he": {
      if (dist !== "uniform" && dist !== "normal") {
        throw new Error(`不正なHe初期化分布: ${dist}`);
      }
      return {
        method: "He",
        dist,
      };
    }
    case "xavier": {
      if (dist !== "uniform" && dist !== "normal") {
        throw new Error(`不正なXavier初期化分布: ${dist}`);
      }
      return {
        method: "Xavier",
        dist,
      };
    }
    default:
      throw new Error(`未知の初期化手法: ${m}`);
  }
}

function getRandFunc(
  initialization: InitializationMethod
): (fanIn: number, fanOut: number) => number {
  switch (initialization.method) {
    case "He":
      switch (initialization.dist) {
        case "uniform":
          return (fanIn: number, fanOut: number) => {
            const limit = Math.sqrt(6 / fanIn);
            return localRandom() * 2 * limit - limit;
          };
        case "normal":
          return (fanIn: number, fanOut: number) => {
            const stddev = Math.sqrt(2 / fanIn);
            return normalRandom(0, stddev);
          };
        default:
          throw new Error(`Unknown He distribution: ${initialization.dist}`);
      }
    case "Xavier":
      switch (initialization.dist) {
        case "uniform":
          return (fanIn: number, fanOut: number) => {
            const limit = Math.sqrt(6 / (fanIn + fanOut));
            return localRandom() * 2 * limit - limit;
          };
        case "normal":
          return (fanIn: number, fanOut: number) => {
            const stddev = Math.sqrt(2 / (fanIn + fanOut));
            return normalRandom(0, stddev);
          };
        default:
          throw new Error(
            `Unknown Xavier distribution: ${initialization.dist}`
          );
      }
    case "Uniform":
      return (fanIn: number, fanOut: number) => localRandom() - 0.5;
    default:
      throw new Error(`Unknown initialization method: ${initialization}`);
  }
}

function initParams(
  fanIn: number,
  fanOut: number,
  initialization: InitializationMethod
) {
  const initializer = getRandFunc(initialization);

  const weights: number[][] = [];
  const biases: number[] = [];
  for (let i = 0; i < fanOut; i++) {
    const weightRow: number[] = [];
    for (let j = 0; j < fanIn; j++) {
      weightRow.push(initializer(fanIn, fanOut));
    }
    weights.push(weightRow);
    biases.push(0); // バイアスは0で初期化
  }
  return { weights, biases };
}

export function initializeParams(props: {
  layers: LayerInfo[];
  initialization: InitializationMethod;
}) {
  console.log(
    `Initializing parameters with method: ${props.initialization.method}...`
  );
  const params: LayerParameter[] = [];
  for (let i = 1; i < props.layers.length; i++) {
    const layerPrev = props.layers[i - 1];
    const layerCurr = props.layers[i];
    const fanIn = layerPrev.size;
    const fanOut = layerCurr.size;
    const param = initParams(fanIn, fanOut, props.initialization);
    params.push(param);
  }
  return params;
}
