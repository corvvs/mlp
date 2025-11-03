// 初期化手法の情報
export type InitUniform = {
  method: "Uniform";
};

export type InitHe = {
  method: "He";
  dist: "uniform" | "normal";
};

export type InitXavier = {
  method: "Xavier";
  dist: "uniform" | "normal";
};

export type InitializationMethod = InitUniform | InitHe | InitXavier;
