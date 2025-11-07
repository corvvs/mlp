import type { LayerInfo } from "../../types/layer.js";
import type {
  OptAdaGrad,
  OptAdam,
  OptimizationMethod,
  OptimizationMethodParam,
  OptMomentumSGD,
  OptRMSProp,
  OptSGD,
  PartialOptAdaGrad,
  PartialOptAdam,
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

type OpzimizationFunction = (
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
    default:
      throw new Error(`未知の最適化方法: ${optimization}`);
  }
}

function getActualSGD(optimization: OptSGD): OpzimizationFunction {
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
): OpzimizationFunction {
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
    console.log(`Momentum SGDによるパラメータ更新: ${k}`);
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
): OpzimizationFunction {
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
      for (let j = 0; j < Gw[0].length; j++) {
        Gw[i][j] += dW[i][j] * dW[i][j];
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
): OpzimizationFunction {
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
): OpzimizationFunction {
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

export function getOptimizationFunctionActual(
  optimization: OptimizationMethod,
  layers: LayerInfo[]
): OpzimizationFunction {
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
    default:
      throw new Error(`未知の最適化方法: ${optimization}`);
  }
}
