import * as _ from "lodash"

// データ型などを定義する

/**
 * 生徒データ
 * Rawじゃないやつは今後登場するのだろうか・・・
 */
export type StudentRaw = {
  index: number;
  hogwarts_house: string;
  first_name: string;
  last_name: string;
  birthday: string;
  best_hand: string;
  scores: {
    [K in string]: number;
  };
  corrected?: boolean;
  raw_splitted: {
    [name: string]: string;
  };
}

export type Item = {
  id: string;
  actual: string;
  raw_data: string[];
  scores: {
    [K: string]: number;
  };
  corrected?: boolean;
};

/**
 * ある特徴量に関する統計データ
 */
export type FeatureStats = {
  name: string;
  // 総数
  count: number;
  // 平均値
  mean: number;
  // 標準偏差
  std: number;
  // 最小値(0パーセンタイル値)
  p0: number;
  // 25パーセンタイル値
  p25: number;
  // 50パーセンタイル値
  p50: number;
  // 75パーセンタイル値
  p75: number;
  // 最大値(100パーセンタイル値)
  p100: number;
  // 欠損データ数
  defects: number;
};

export type FeatureCorrelation = {
  f1: string;
  f2: string;
  variance: number;
  cor: number;
};

/**
 * 階級値データ
 */
export type Histogram = {
  feature: string,
  stat: FeatureStats;
  bins: number;
  counts: number[];
  layered_counts: { [house: string]: number }[];
  max_count: number;
  max_layered_count: number;
};

export type Standardizer = Pick<FeatureStats, "mean" | "std">;

export type Standardizers = {
  [feature: string]: Standardizer;
};

/**
 * パラメータデータ
 */
export type TrainedParameters = {
  weights: {
    [house: string]: {
      [feature: string]: number;
    };
  };
  standardizers: Standardizers;
};

/**
 * 2次元の点
 */
export type Vector2D = {
  x: number;
  y: number;
};

export type Affine2D = {
  xx: number; xy: number; xt: number;
  yx: number; yy: number; yt: number;
};

/**
 * 矩形領域
 */
export type Box = {
  p1: Vector2D;
  p2: Vector2D;
};

export type Dimension = {
  width: number;
  height: number;
};

/**
 * インセット, すなわちある矩形領域の境界から内部方向への「余白」
 */
export type Inset = {
  top: number;
  bottom: number;
  left: number;
  right: number;
};

export type Vector2DPlus = Vector2D & DrawOption;

export type PairedData = {
  feature_x: FeatureStats;
  feature_y: FeatureStats;
  count: number;
  pairs: Vector2DPlus[];
  box: Box;
};

type Color = string | "none";
type Size = string | number;

export type ShapeOption = {
  stroke?: Color;
  "stroke-width"?: Size;
  fill?: Color;
};

export type TextOption = {
  font?: string;
  "font-weight"?: "bold" | string;
};

export type DrawOption = ShapeOption & {
  r?: number;
};

export const ActualToKey = {
  M: "is_m",
  B: "is_b",
};

export const keyToActual = _(ActualToKey)
  .mapValues((k, h) => ({ k, h }))
  .mapKeys((v, h) => v.k)
  .mapValues((v, k) => v.h).value();

export const HouseToKey = {
  Ravenclaw: "is_r",
  Slytherin: "is_s",
  Gryffindor: "is_g",
  Hufflepuff: "is_h",
};

export const KeyToHouse = _(HouseToKey)
  .mapValues((k, h) => ({ k, h }))
  .mapKeys((v, h) => v.k)
  .mapValues((v, k) => v.h).value();

