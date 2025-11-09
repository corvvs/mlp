// 正則化手法の情報
export type RegL2 = {
  method: "L2";
  lambda: number;
};

export type RegularizationMethod = RegL2;

export type RegularizationFunction = (wMats: number[][][]) => number;

export type RegularizationGradientFunction = (
  wMat: number[][],
  B: number
) => number[][];
