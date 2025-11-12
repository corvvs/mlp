// 活性化関数の情報
export type AFSoftmax = {
  method: "softmax";
};

export type AFLinear = {
  method: "linear";
};

export type AFSigmoid = {
  method: "sigmoid";
};

export type AFTanH = {
  method: "tanh";
};

export type AFReLU = {
  method: "ReLU";
};

export type AFLeakyReLU = {
  method: "LeakyReLU";
  alpha: number;
};

export type ActivationFunctionSingleArgument =
  | AFLinear
  | AFSigmoid
  | AFTanH
  | AFReLU
  | AFLeakyReLU;

export type ActivationFunctionSingleArgumentMethod =
  ActivationFunctionSingleArgument["method"];

export type ActivationFunction = AFSoftmax | ActivationFunctionSingleArgument;

export type ActivatationFunctionActual = (x: number) => number;
