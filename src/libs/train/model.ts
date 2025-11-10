import type {
  ActivationFunctionSingleArgument,
  ActivationFunctionSingleArgumentMethod,
} from "../../types/af.js";
import type { InitializationMethod } from "../../types/initialization.js";
import type {
  HiddenLayerInfo,
  LayerInfo,
  ScaleFactor,
} from "../../types/layer.js";
import type { ModelData } from "../../types/model.js";
import type { OptimizationMethod } from "../../types/optimization.js";
import type { RegularizationMethod } from "../../types/regularization.js";
import { localSrand } from "../random.js";
import { initializeParams } from "./initialization.js";
import { makeOptimizationParam } from "./optimization.js";

export function buildModelData(props: {
  scaleFactors: (ScaleFactor | null)[];
  maxEpochs: number;
  seed: number | null;
  batchSize?: number;
  defaultActivationFunction: ActivationFunctionSingleArgument;
  regularization: RegularizationMethod | null;
  hiddenLayerSizes: number[];
  optimization: OptimizationMethod;
}): ModelData {
  const hiddenLayerSizes = props.hiddenLayerSizes ?? [24, 24];

  const hiddenLayers: HiddenLayerInfo[] = hiddenLayerSizes.map((size) => ({
    layerType: "hidden",
    size,
    activationFunction: props.defaultActivationFunction,
  }));

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
    ...hiddenLayers,

    {
      // 出力層
      layerType: "output",
      size: 2,
      activationFunction: {
        method: "softmax",
      },
    },
  ];
  const seed = props.seed ?? 123;
  localSrand(seed);
  const parameters = initializeParams({
    initialization,
    layers,
  });

  return {
    version: "1.0.0",
    maxEpochs: props.maxEpochs,
    seed,

    batchSize: props.batchSize ?? 8,
    layerNumber: layers.length,
    layers,
    initialization,

    lossFunction: {
      method: "CCE",
      eps: 1e-9,
    },

    optimization: props.optimization,

    ...(props.regularization ? { regularization: props.regularization } : {}),

    bestEpoch: 0,
    trainMetrics: [],
    valMetrics: [],
    parameters,
  };
}
