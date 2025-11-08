import type { InitializationMethod } from "../../types/initialization.js";
import type { LayerInfo, ScaleFactor } from "../../types/layer.js";
import type { ModelData } from "../../types/model.js";
import { initializeParams } from "./initialization.js";
import { makeOptimizationParam } from "./optimization.js";

export function buildModelData(props: {
  scaleFactors: (ScaleFactor | null)[];
}): ModelData {
  const initialization: InitializationMethod = {
    method: "Xavier",
    dist: "uniform",
  };

  const actualInputSize = props.scaleFactors.length - 1; // Answer列を除く

  const layers: LayerInfo[] = [
    {
      // 入力層
      layerType: "input",
      size: actualInputSize,
      scaleFactors: props.scaleFactors,
    },
    // 隠れ層
    {
      layerType: "hidden",
      size: 24,
      activationFunction: {
        method: "ReLU",
      },
    },
    {
      layerType: "hidden",
      size: 24,
      activationFunction: {
        method: "ReLU",
      },
    },
    {
      layerType: "hidden",
      size: 24,
      activationFunction: {
        method: "ReLU",
      },
    },
    {
      // 出力層
      layerType: "output",
      size: 2,
      activationFunction: {
        method: "softmax",
      },
    },
  ];
  const parameters = initializeParams({
    initialization,
    layers,
  });

  return {
    version: "1.0.0",

    batchSize: 0,
    layerNumber: layers.length,
    layers,
    initialization,

    lossFunction: {
      method: "BCE",
      eps: 1e-9,
    },

    optimization: makeOptimizationParam({
      method: "RMSProp",
    }),

    parameters,
  };
}
