// 損失関数の情報
export type LossBCE = {
  method: "BCE";
  eps: number; // log-sum-exp のパラメータ
};

export type LossFunction = LossBCE;
