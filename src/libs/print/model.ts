import type { ActivationFunction } from "../../types/af.js";
import type { EarlyStopping } from "../../types/es.js";
import type { InitializationMethod } from "../../types/initialization.js";
import type { LayerInfo } from "../../types/layer.js";
import type { ModelData } from "../../types/model.js";
import type { OptimizationMethod } from "../../types/optimization.js";
import type { RegularizationMethod } from "../../types/regularization.js";

export function printModel(model: ModelData): void {
  console.log(
    "B (Batch Size):",
    model.batchSize === 0 ? "ALL" : model.batchSize
  );
  console.log("Layers:", model.layerNumber);
  model.layers.forEach((layer, i) => {
    console.log(` Layer ${i}: ${outLayer(layer)}`);
  });
  console.log(
    "Parameters Initialization Method:",
    outInitializationMethod(model.initialization)
  );
  console.log("Loss Function:", model.lossFunction.method);
  if (model.regularization) {
    console.log(
      "Regularization Method:",
      outRegularizationMethod(model.regularization)
    );
  }
  console.log(
    "Optimization Method:",
    outOptimizationMethod(model.optimization)
  );
  console.log("Early Stopping:", outEarlyStopping(model.earlyStopping ?? null));
}

function outLayer(layer: LayerInfo): string {
  switch (layer.layerType) {
    case "input":
      return `(Input, ${layer.size})`;
    case "hidden":
      return `(Hidden, ${layer.size}, ${outAF(layer.activationFunction)})`;
    case "output":
      return `(Output, ${layer.size}, ${outAF(layer.activationFunction)})`;
  }
}

function outAF(af: ActivationFunction): string {
  switch (af.method) {
    case "ReLU":
      return `ReLU`;
    case "linear":
      return `Linear`;
    case "sigmoid":
      return `Sigmoid`;
    case "tanh":
      return `Tanh`;
    case "LeakyReLU":
      return `LeakyReLU(alpha=${af.alpha})`;
    case "softmax":
      return `Softmax`;
    default:
      throw new Error(`未知の活性化関数: ${af}`);
  }
}

function outInitializationMethod(method: InitializationMethod): string {
  switch (method.method) {
    case "Xavier":
      return `Xavier (dist=${method.dist})`;
    case "He":
      return `He (dist=${method.dist})`;
    default:
      throw new Error(`未知の初期化方法: ${method}`);
  }
}

function outRegularizationMethod(method: RegularizationMethod): string {
  switch (method.method) {
    case "L2":
      return `L2 (lambda=${method.lambda})`;
    default:
      throw new Error(`未知の正則化方法: ${method}`);
  }
}

function outOptimizationMethod(method: OptimizationMethod): string {
  switch (method.method) {
    case "SGD":
      return `SGD (lr=${method.learningRate})`;
    case "MomentumSGD":
      return `MomentumSGD (lr=${method.learningRate}, alpha=${method.alpha})`;
    case "AdaGrad":
      return `AdaGrad (lr=${method.learningRate}, eps=${method.eps})`;
    case "RMSProp":
      return `RMSProp (lr=${method.learningRate}, rho=${method.rho}, eps=${method.eps})`;
    case "Adam":
      return `Adam (lr=${method.learningRate}, beta1=${method.beta1}, beta2=${method.beta2}, eps=${method.eps})`;
    case "AdamW":
      return `AdamW (lr=${method.learningRate}, beta1=${method.beta1}, beta2=${method.beta2}, eps=${method.eps}, weightDecay=${method.weightDecay})`;
    default:
      throw new Error(`未知の最適化方法: ${method}`);
  }
}

export function outEarlyStopping(args: EarlyStopping | null): string {
  if (!args) {
    return "Disabled";
  }
  return `Enabled (metric=${args.metric}, patience=${args.patience})`;
}
