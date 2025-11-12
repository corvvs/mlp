import type { LayerInfo } from "../../types/layer.js";
import type {
  OptAdaGrad,
  OptAdam,
  OptAdamW,
  OptimizationMethod,
  OptimizationMethodParam,
  OptMomentumSGD,
  OptRMSProp,
  OptSGD,
  PartialOptAdaGrad,
  PartialOptAdam,
  PartialOptAdamW,
  PartialOptMomentumSGD,
  PartialOptRMSProp,
  PartialOptSGD,
} from "../../types/optimization.js";
import {
  addFactorMatX,
  addFactorVecX,
  addMatX,
  addVecX,
  mulMatX,
  mulVecX,
  zeroMat,
  zeroVec,
} from "../arithmetics.js";

export function parseOptimization(optimizationStr: string): OptimizationMethod {
  const parts = optimizationStr.split(",");
  const method = parts[0];
  switch (method.toLowerCase()) {
    case "sgd": {
      const opt: PartialOptSGD = { method: "SGD" };
      if (parts.length >= 2) {
        const learningRate = parseFloat(parts[1]);
        if (isNaN(learningRate) || learningRate <= 0) {
          throw new Error(`不正なSGD学習率: ${parts[1]}`);
        }
        opt.learningRate = learningRate;
      }
      return makeSGDParam(opt);
    }
    case "msgd":
    case "momentumsgd": {
      const opt: PartialOptMomentumSGD = { method: "MomentumSGD" };
      if (parts.length >= 2) {
        const learningRate = parseFloat(parts[1]);
        if (isNaN(learningRate) || learningRate <= 0) {
          throw new Error(`不正なMomentumSGD学習率: ${parts[1]}`);
        }
        if (parts.length >= 3) {
          const alpha = parseFloat(parts[2]);
          if (isNaN(alpha) || alpha < 0 || alpha >= 1) {
            throw new Error(`不正なMomentumSGDアルファ値: ${parts[2]}`);
          }
          opt.alpha = alpha;
        }
        opt.learningRate = learningRate;
      }
      return makeMomentumSGDParam(opt);
    }
    case "adagrad": {
      const opt: PartialOptAdaGrad = { method: "AdaGrad" };
      if (parts.length >= 2) {
        const learningRate = parseFloat(parts[1]);
        if (isNaN(learningRate) || learningRate <= 0) {
          throw new Error(`不正なAdaGrad学習率: ${parts[1]}`);
        }
        if (parts.length >= 3) {
          const eps = parseFloat(parts[2]);
          if (isNaN(eps) || eps <= 0) {
            throw new Error(`不正なAdaGradイプシロン値: ${parts[2]}`);
          }
          opt.eps = eps;
        }
        opt.learningRate = learningRate;
      }
      return makeAdaGradParam(opt);
    }
    case "rmsprop": {
      const opt: PartialOptRMSProp = { method: "RMSProp" };
      if (parts.length >= 2) {
        const rho = parseFloat(parts[1]);
        if (isNaN(rho) || rho < 0 || rho >= 1) {
          throw new Error(`不正なRMSPropρ値: ${parts[1]}`);
        }
        if (parts.length >= 3) {
          const learningRate = parseFloat(parts[2]);
          if (isNaN(learningRate) || learningRate <= 0) {
            throw new Error(`不正なRMSProp学習率: ${parts[2]}`);
          }
          opt.learningRate = learningRate;
        }
        if (parts.length >= 4) {
          const eps = parseFloat(parts[3]);
          if (isNaN(eps) || eps <= 0) {
            throw new Error(`不正なRMSPropイプシロン値: ${parts[3]}`);
          }
          opt.eps = eps;
        }
        opt.rho = rho;
      }
      return makeRMSPropParam(opt);
    }
    case "adam": {
      const opt: PartialOptAdam = { method: "Adam" };
      if (parts.length >= 2) {
        const learningRate = parseFloat(parts[1]);
        if (isNaN(learningRate) || learningRate <= 0) {
          throw new Error(`不正なAdam学習率: ${parts[1]}`);
        }
        if (parts.length >= 3) {
          const beta1 = parseFloat(parts[2]);
          if (isNaN(beta1) || beta1 < 0 || beta1 >= 1) {
            throw new Error(`不正なAdamβ1値: ${parts[2]}`);
          }
          opt.beta1 = beta1;
        }
        if (parts.length >= 4) {
          const beta2 = parseFloat(parts[3]);
          if (isNaN(beta2) || beta2 < 0 || beta2 >= 1) {
            throw new Error(`不正なAdamβ2値: ${parts[3]}`);
          }
          opt.beta2 = beta2;
        }
        if (parts.length >= 5) {
          const eps = parseFloat(parts[4]);
          if (isNaN(eps) || eps <= 0) {
            throw new Error(`不正なAdamイプシロン値: ${parts[4]}`);
          }
          opt.eps = eps;
        }
        opt.learningRate = learningRate;
      }
      return makeAdamParam(opt);
    }
    case "adamw": {
      const opt: PartialOptAdamW = { method: "AdamW" };
      if (parts.length >= 2) {
        const weightDecay = parseFloat(parts[1]);
        if (isNaN(weightDecay) || weightDecay < 0) {
          throw new Error(`不正なAdamW Weight Decay値: ${parts[1]}`);
        }
        opt.weightDecay = weightDecay;
      }
      if (parts.length >= 3) {
        const learningRate = parseFloat(parts[2]);
        if (isNaN(learningRate) || learningRate <= 0) {
          throw new Error(`不正なAdamW学習率: ${parts[2]}`);
        }
        opt.learningRate = learningRate;
      }
      if (parts.length >= 4) {
        const beta1 = parseFloat(parts[3]);
        if (isNaN(beta1) || beta1 < 0 || beta1 >= 1) {
          throw new Error(`不正なAdamWβ1値: ${parts[3]}`);
        }
        opt.beta1 = beta1;
      }
      if (parts.length >= 5) {
        const beta2 = parseFloat(parts[4]);
        if (isNaN(beta2) || beta2 < 0 || beta2 >= 1) {
          throw new Error(`不正なAdamWβ2値: ${parts[4]}`);
        }
        opt.beta2 = beta2;
      }
      if (parts.length >= 6) {
        const eps = parseFloat(parts[5]);
        if (isNaN(eps) || eps <= 0) {
          throw new Error(`不正なAdamWイプシロン値: ${parts[5]}`);
        }
        opt.eps = eps;
      }
      return makeAdamWParam(opt);
    }
    default:
      throw new Error(`未知の最適化方法: ${method}`);
  }
}

export type OptimizationFunction = (
  W: number[][],
  b: number[],
  dW: number[][],
  db: number[],
  k: number
) => void;

function makeSGDParam(optimization: PartialOptSGD): OptSGD {
  return {
    ...optimization,
    learningRate: optimization.learningRate ?? 0.01,
  };
}

function makeMomentumSGDParam(
  optimization: PartialOptMomentumSGD
): OptMomentumSGD {
  return {
    ...optimization,
    learningRate: optimization.learningRate ?? 0.01,
    alpha: optimization.alpha ?? 0.9,
  };
}

function makeAdaGradParam(optimization: PartialOptAdaGrad): OptAdaGrad {
  return {
    ...optimization,
    learningRate: optimization.learningRate ?? 0.01,
    eps: optimization.eps ?? 1e-8,
  };
}

function makeRMSPropParam(optimization: PartialOptRMSProp): OptRMSProp {
  return {
    ...optimization,
    rho: optimization.rho ?? 0.9,
    learningRate: optimization.learningRate ?? 0.001,
    eps: optimization.eps ?? 1e-8,
  };
}

function makeAdamParam(optimization: PartialOptAdam): OptAdam {
  return {
    ...optimization,
    beta1: optimization.beta1 ?? 0.9,
    beta2: optimization.beta2 ?? 0.999,
    learningRate: optimization.learningRate ?? 0.001,
    eps: optimization.eps ?? 1e-8,
  };
}

function makeAdamWParam(optimization: PartialOptAdamW): OptAdamW {
  return {
    ...optimization,
    beta1: optimization.beta1 ?? 0.9,
    beta2: optimization.beta2 ?? 0.999,
    learningRate: optimization.learningRate ?? 0.001,
    eps: optimization.eps ?? 1e-8,
    weightDecay: optimization.weightDecay ?? 1e-4,
  };
}

export function makeOptimizationParam(
  optimization: OptimizationMethodParam
): OptimizationMethod {
  switch (optimization.method) {
    case "SGD":
      return makeSGDParam(optimization);
    case "MomentumSGD":
      return makeMomentumSGDParam(optimization);
    case "AdaGrad":
      return makeAdaGradParam(optimization);
    case "RMSProp":
      return makeRMSPropParam(optimization);
    case "Adam":
      return makeAdamParam(optimization);
    case "AdamW":
      return makeAdamWParam(optimization);
    default:
      throw new Error(`未知の最適化方法: ${optimization}`);
  }
}

function getActualSGD(optimization: OptSGD): OptimizationFunction {
  const learningRate = optimization.learningRate;
  return (
    W: number[][],
    b: number[],
    dW: number[][],
    db: number[],
    _: number
  ) => {
    // console.log("SGDによるパラメータ更新");
    addFactorMatX(W, -learningRate, dW);
    addFactorVecX(b, -learningRate, db);
  };
}

function getActualMomentumSGD(
  optimization: OptMomentumSGD,
  layers: LayerInfo[]
): OptimizationFunction {
  const learningRate = optimization.learningRate;
  const alpha = optimization.alpha;

  const vws: number[][][] = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroMat(layers[k + 1].size, layers[k].size)
  );
  const vbs: number[][] = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroVec(layers[k + 1].size)
  );

  return (
    W: number[][],
    b: number[],
    dW: number[][],
    db: number[],
    k: number
  ) => {
    const vw = vws[k];
    const vb = vbs[k];

    mulMatX(vw, alpha);
    mulVecX(vb, alpha);
    addFactorMatX(vw, -learningRate, dW);
    addFactorVecX(vb, -learningRate, db);
    addMatX(W, vw);
    addVecX(b, vb);
  };
}

// NOTE: NAG は特殊すぎるな・・・

function getActualAdaGrad(
  optimization: OptAdaGrad,
  layers: LayerInfo[]
): OptimizationFunction {
  const learningRate = optimization.learningRate;
  const eps = optimization.eps;

  const Gws: number[][][] = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroMat(layers[k + 1].size, layers[k].size)
  );
  const Gbs: number[][] = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroVec(layers[k + 1].size)
  );

  return (
    W: number[][],
    b: number[],
    dW: number[][],
    db: number[],
    k: number
  ) => {
    const Gw = Gws[k];
    const Gb = Gbs[k];

    // 勾配の二乗和を更新
    for (let i = 0; i < Gw.length; i++) {
      const Gwi = Gw[i];
      const dWi = dW[i];
      for (let j = 0; j < Gw[0].length; j++) {
        Gwi[j] += dWi[j] * dWi[j];
      }
      Gb[i] += db[i] * db[i];
    }

    // パラメータ更新
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[0].length; j++) {
        W[i][j] -= (learningRate / Math.sqrt(Gw[i][j] + eps)) * dW[i][j];
      }
      b[i] -= (learningRate / Math.sqrt(Gb[i] + eps)) * db[i];
    }
  };
}

export function getActualRMSProp(
  optimization: OptRMSProp,
  layers: LayerInfo[]
): OptimizationFunction {
  const rho = optimization.rho;
  const learningRate = optimization.learningRate;
  const eps = optimization.eps;

  const rws: number[][][] = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroMat(layers[k + 1].size, layers[k].size)
  );
  const rbs: number[][] = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroVec(layers[k + 1].size)
  );

  return (
    W: number[][],
    b: number[],
    dW: number[][],
    db: number[],
    k: number
  ) => {
    const rw = rws[k];
    const rb = rbs[k];
    for (let i = 0; i < rw.length; i++) {
      for (let j = 0; j < rw[0].length; j++) {
        rw[i][j] = rw[i][j] * rho + (1 - rho) * dW[i][j] * dW[i][j];
        W[i][j] -= (learningRate / Math.sqrt(rw[i][j] + eps)) * dW[i][j];
      }
      rb[i] = rb[i] * rho + (1 - rho) * db[i] * db[i];
      b[i] -= (learningRate / Math.sqrt(rb[i] + eps)) * db[i];
    }
  };
}

export function getActualAdam(
  optimization: OptAdam,
  layers: LayerInfo[]
): OptimizationFunction {
  const mws = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroMat(layers[k + 1].size, layers[k].size)
  );
  const vws = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroMat(layers[k + 1].size, layers[k].size)
  );
  const mbs = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroVec(layers[k + 1].size)
  );
  const vbs = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroVec(layers[k + 1].size)
  );
  const ts = zeroVec(layers.length - 1);
  const beta1 = optimization.beta1;
  const beta2 = optimization.beta2;
  const learningRate = optimization.learningRate;
  const eps = optimization.eps;

  return (
    W: number[][],
    b: number[],
    dW: number[][],
    db: number[],
    k: number
  ) => {
    const mw = mws[k];
    const vw = vws[k];
    const mb = mbs[k];
    const vb = vbs[k];
    ts[k] += 1;
    const t = ts[k];
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[0].length; j++) {
        mw[i][j] = beta1 * mw[i][j] + (1 - beta1) * dW[i][j];
        vw[i][j] = beta2 * vw[i][j] + (1 - beta2) * dW[i][j] * dW[i][j];
        const mwhat = mw[i][j] / (1 - Math.pow(beta1, t));
        const vwhat = vw[i][j] / (1 - Math.pow(beta2, t));
        W[i][j] -= (learningRate * mwhat) / (Math.sqrt(vwhat) + eps);
      }
      mb[i] = beta1 * mb[i] + (1 - beta1) * db[i];
      vb[i] = beta2 * vb[i] + (1 - beta2) * db[i] * db[i];
      const mbhat = mb[i] / (1 - Math.pow(beta1, t));
      const vbhat = vb[i] / (1 - Math.pow(beta2, t));
      b[i] -= (learningRate * mbhat) / (Math.sqrt(vbhat) + eps);
    }
  };
}

export function getActualAdamW(
  optimization: OptAdamW,
  layers: LayerInfo[]
): OptimizationFunction {
  const mws = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroMat(layers[k + 1].size, layers[k].size)
  );
  const vws = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroMat(layers[k + 1].size, layers[k].size)
  );
  const mbs = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroVec(layers[k + 1].size)
  );
  const vbs = Array.from({ length: layers.length - 1 }, (_, k) =>
    zeroVec(layers[k + 1].size)
  );
  const ts = zeroVec(layers.length - 1);
  const beta1 = optimization.beta1;
  const beta2 = optimization.beta2;
  const learningRate = optimization.learningRate;
  const eps = optimization.eps;
  const weightDecay = optimization.weightDecay;

  return (
    W: number[][],
    b: number[],
    dW: number[][],
    db: number[],
    k: number
  ) => {
    const mw = mws[k];
    const vw = vws[k];
    const mb = mbs[k];
    const vb = vbs[k];
    ts[k] += 1;
    const t = ts[k];
    for (let i = 0; i < W.length; i++) {
      const Wi = W[i];
      const dWi = dW[i];
      const mwi = mw[i];
      const vwi = vw[i];
      for (let j = 0; j < W[0].length; j++) {
        mwi[j] = beta1 * mwi[j] + (1 - beta1) * dWi[j];
        vwi[j] = beta2 * vwi[j] + (1 - beta2) * dWi[j] * dWi[j];
        const mwhat = mwi[j] / (1 - Math.pow(beta1, t));
        const vwhat = vwi[j] / (1 - Math.pow(beta2, t));
        Wi[j] -=
          (learningRate * mwhat) / (Math.sqrt(vwhat) + eps) +
          learningRate * weightDecay * Wi[j];
      }
      mb[i] = beta1 * mb[i] + (1 - beta1) * db[i];
      vb[i] = beta2 * vb[i] + (1 - beta2) * db[i] * db[i];
      const mbhat = mb[i] / (1 - Math.pow(beta1, t));
      const vbhat = vb[i] / (1 - Math.pow(beta2, t));
      b[i] -= (learningRate * mbhat) / (Math.sqrt(vbhat) + eps);
    }
  };
}

export function getOptimizationFunctionActual(
  optimization: OptimizationMethod,
  layers: LayerInfo[]
): OptimizationFunction {
  switch (optimization.method) {
    case "SGD":
      return getActualSGD(optimization);
    case "MomentumSGD":
      return getActualMomentumSGD(optimization, layers);
    case "AdaGrad":
      return getActualAdaGrad(optimization, layers);
    case "RMSProp":
      return getActualRMSProp(optimization, layers);
    case "Adam":
      return getActualAdam(optimization, layers);
    case "AdamW":
      return getActualAdamW(optimization, layers);
    default:
      throw new Error(`未知の最適化方法: ${optimization}`);
  }
}
