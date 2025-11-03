// 最適化方法の情報
export type OptSGD = {
  method: "SGD";
  learningRate: number;
};

export type OptMomentumSGD = {
  method: "MomentumSGD";
  learningRate: number;
  alpha: number;
};

export type OptNAG = {
  method: "NAG";
  learningRate: number;
  alpha: number;
};

export type OptAdaGrad = {
  method: "AdaGrad";
  learningRate: number;
  eps: number;
};

export type OptRMSProp = {
  method: "RMSProp";
  rho: number;
  learningRate: number;
  eps: number;
};

export type OptAdam = {
  method: "Adam";
  beta1: number;
  beta2: number;
  learningRate: number;
  eps: number;
};

export type OptimizationMethod =
  | OptSGD
  | OptMomentumSGD
  | OptNAG
  | OptAdaGrad
  | OptRMSProp
  | OptAdam;
