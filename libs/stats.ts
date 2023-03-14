import * as _ from "lodash"
import { sprintf } from "sprintf-js";
import { StudentRaw, FeatureStats, Histogram, Vector2D, Box, PairedData, Vector2DPlus, FeatureCorrelation, Item, ActualToKey } from "./definitions"
import { Utils } from "./utils";

// 統計量をあつかう

export namespace Stats {
  /**
   * 生徒データ`students`をもとに、特徴量`feature`の統計量を計算する.
   * @param feature 
   * @param students 
   * @returns 
   */
  export function derive_feature_stats(feature: string, items: Item[]): FeatureStats {
    let count = 0;
    let defects = 0;
    let sum = 0;
    const values: number[] = [];
    items.forEach(r => {
      const val = r.scores[feature];
      if (!_.isFinite(val)) {
        defects += 1;
        return;
      }
      count += 1;
      sum += val;
      values.push(val);
    });
    const mean = sum / count;
    let sqsum = 0;
    items.forEach(r => {
      const val = r.scores[feature];
      if (!_.isFinite(val)) { return; }
      sqsum += (val - mean) ** 2;
    });
    let std = Math.sqrt(sqsum / count);
    const sorted_values = _.sortBy(values, v => v);
    const i0 = 0;
    const i25 = Math.floor(count * 0.25);
    const i50 = Math.floor(count * 0.5);
    const i75 = Math.floor(count * 0.75);
    const i100 = count - 1;
    const [p0, p25, p50, p75, p100] = [i0, i25, i50, i75, i100].map(i => sorted_values[i]);
    return {
      name: feature,
      count,
      mean,
      std,
      p0,
      p25,
      p50,
      p75,
      p100,
      defects,
    };
  }

  export function derive_features_covariances(feature_stats: FeatureStats[], items: Item[]) {
    return feature_stats.map((f1, i) => {
      return feature_stats.map((f2, j) => {
        let n = 0;
        const variance = (i == j) ? f1.std ** 2 : items.reduce((sum, s) => {
          const v1 = s.scores[f1.name];
          const v2 = s.scores[f2.name];
          if (_.isFinite(v1) && _.isFinite(v2)) {
            n += 1;
            return sum + (v1 - f1.mean) * (v2 - f2.mean);
          } else {
            return sum;
          }
        }, 0) / n;
        return {
          f1: f1.name,
          f2: f2.name,
          variance,
          cor: variance / f1.std / f2.std,
        };
      });
    });
  }

  /**
   * FeatureStats に含まれる各統計量を1行ずつにまとめて表示する.
   * @param feature_stats 
   */
  export function print_stats_for_features(
    feature_stats: FeatureStats[],
    correlation_matrix?: FeatureCorrelation[][]
  ) {
    // フィールド長の算出
    // [見出し行]
    let line = "";
    line += sprintf("%10s", "");
    feature_stats.forEach(s => {
      const w = Math.max(s.name.length, 10);
      line += sprintf(` %${w}s`, s.name);
    });
    console.log(line);

    // [統計量]
    const name_map = {
      count: "Count",
      mean: "Mean",
      std: "Std",
      p0: "Min",
      p25: "25%",
      p50: "50%",
      p75: "75%",
      p100: "Max",
    };
    (["count", "mean", "std", "p0", "p25", "p50", "p75", "p100"] as const).forEach(key => {
      const display_name = name_map[key];
      let line = "";
      line += sprintf("%-10s", display_name);
      feature_stats.forEach(s => {
        const w = Math.max(s.name.length, 10);
        line += sprintf(` %${w}.4f`, s[key]);
      });
      console.log(line);
    });
    if (!correlation_matrix) { return; }
    // [相関係数行列の表示]
    console.log("[correlation coefficients]");
    correlation_matrix.forEach(row => {
      const row_name = row[0].f1;
      let line = "";
      line += sprintf("%-10s", Utils.abbreviate(row_name, 10, "..."));
      row.forEach(item => {
        const w = Math.max(item.f2.length, 10);
        if (item.f1 === item.f2) {
          line += sprintf(` %${w}s`, "-");
        } else {
          line += sprintf(` %${w}.4f`, item.cor);
        }
      });
      console.log(line);
    });
  }

  /**
   * 生徒データから, 特定の特徴量を選んで階級化する.
   * 対象の特徴量は引数`stats`で指定する.
   * @param items 生徒データ
   * @param stat 階級化する特徴量の統計データ
   * @param bins 階級数; 1以上であること. 整数でない場合は切り捨てる.
   */
  export function students_to_bins(items: Item[], stat: FeatureStats, bins: number): Histogram {
    const n_bins = Math.floor(bins);
    if (n_bins < 1) {
      throw new Error("binsが1以上ではありません");
    }
    
    const feature = stat.name;
    const [min, max] = [stat.p0, stat.p100];
    const size = max - min;
    const counts = _.range(bins).map(s => 0);
    const layered_counts = _.range(bins).map(s => _.mapValues(ActualToKey, () => 0));
    items.forEach(s => {
      const val = s.scores[feature];
      if (!_.isFinite(val)){ return; } // おかしな値(nan, infty, null, ...)をはじく
      const i = Math.min(Math.floor((val - min) / size * bins), bins - 1);
      counts[i] += 1;
      const actual = s.actual;
      if (_.isFinite(layered_counts[i][actual])) {
        layered_counts[i][actual] += 1;
      }
    });
    const max_layered_count = _(layered_counts).map(lc => _(lc).values().max()).max()!;
    return {
      feature,
      stat,
      bins,
      counts,
      layered_counts,
      max_count: Math.max(...counts),
      max_layered_count,
    };
  }

  export function make_pair(
    feature_x: FeatureStats,
    feature_y: FeatureStats,
    items: Item[]
  ): PairedData {
    let count = 0;
    const pairs: Vector2DPlus[] = _(items).map(s => {
      const x = s.scores[feature_x.name];
      const y = s.scores[feature_y.name];
      if (!_.isFinite(x) || !_.isFinite(y)) { return null; }
      count += 1;
      return { x, y, fill: colorForItem(s), r: typeof s.corrected === "boolean" ? (s.corrected ? 4 : 16) : null };
    }).compact().value();
    const box: Box = {
      p1: { x: feature_x.p0, y: feature_y.p0 },
      p2: { x: feature_x.p100, y: feature_y.p100 },
    };
    return {
      feature_x,
      feature_y,
      count,
      pairs,
      box,
    };
  }

  function colorForStudent(student: StudentRaw) {
    if (student.scores.is_r) {
      return "royalblue";
    }
    if (student.scores.is_s) {
      return "lightgreen";
    }
    if (student.scores.is_g) {
      return "red";
    }
    if (student.scores.is_h) {
      return "gold";
    }
    return undefined;
  }

  export function colorForItem(item: Item) {
    if (item.actual === "M") {
      return "royalblue";
    }
    if (item.actual === "B") {
      return "gold";
    }
    return undefined;
  }

  export function colorForActual(actual: string) {
    switch (actual) {
      case "M":
        return "royalblue";
      case "B":
        return "gold";
    }
    return undefined;
  }
};
