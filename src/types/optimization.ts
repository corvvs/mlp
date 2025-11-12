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

export type OptAdamW = {
  method: "AdamW";
  beta1: number;
  beta2: number;
  learningRate: number;
  eps: number;
  weightDecay: number;
};

export type OptimizationMethod =
  | OptSGD
  | OptMomentumSGD
  | OptAdaGrad
  | OptRMSProp
  | OptAdam
  | OptAdamW;

export type PartialOptSGD = Pick<OptSGD, "method"> &
  Partial<Omit<OptSGD, "method">>;
export type PartialOptMomentumSGD = Pick<OptMomentumSGD, "method"> &
  Partial<Omit<OptMomentumSGD, "method">>;
export type PartialOptAdaGrad = Pick<OptAdaGrad, "method"> &
  Partial<Omit<OptAdaGrad, "method">>;
export type PartialOptRMSProp = Pick<OptRMSProp, "method"> &
  Partial<Omit<OptRMSProp, "method">>;
export type PartialOptAdam = Pick<OptAdam, "method"> &
  Partial<Omit<OptAdam, "method">>;
export type PartialOptAdamW = Pick<OptAdamW, "method"> &
  Partial<Omit<OptAdamW, "method">>;

export type OptimizationMethodParam =
  | PartialOptSGD
  | PartialOptMomentumSGD
  | PartialOptAdaGrad
  | PartialOptRMSProp
  | PartialOptAdam
  | PartialOptAdamW;
