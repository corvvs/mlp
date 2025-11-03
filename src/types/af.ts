// 活性化関数の情報
export type AFSoftmax = {
  method: "softmax";
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

export type ActivationFunction =
  | AFSoftmax
  | AFSigmoid
  | AFTanH
  | AFReLU
  | AFLeakyReLU;
