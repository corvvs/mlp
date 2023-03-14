import * as _ from "lodash"
import { FeatureStats, Item, KeyToHouse, Standardizer, StudentRaw } from "./definitions";
import { Stats } from "./stats";

/**
 * データの前(とは限らない)処理
 */
export namespace Preprocessor {

  /**
   * (破壊的変更)
   * 生徒データのうち選択した`feature`を標準化する.
   * すなわち平均0, 分散1となるように全体を線形に変換する.
   * @param feature 
   * @param students 
   */
  export function standardize(stats: FeatureStats, students: StudentRaw[]) {
    students.forEach(s => {
      const v = s.scores[stats.name];
      if (!_.isFinite(v)) { return; }
      s.scores[stats.name] = (v - stats.mean) / stats.std;
    });
  }

  /**
   * (破壊的変更)
   * 生徒データのうち選択した`feature`を標準化する.
   * 標準化の際は`standardizer`の中身(mean, std)を使う.
   * @param feature 
   * @param standardizer 
   * @param students 
   */
  export function standardize_by_ex(feature: string, standardizer: Standardizer, students: StudentRaw[]) {
    students.forEach(s => {
      const v = s.scores[feature];
      if (!_.isFinite(v)) { return; }
      s.scores[feature] = (v - standardizer.mean) / standardizer.std;
    });
  }

  /**
   * (破壊的変更)
   * データを正規化する
   * @param stats 
   * @param students 
   */
  export function normalize(stats: FeatureStats, students: StudentRaw[]) {
    const w = (stats.p100 - stats.p0);
    students.forEach(s => {
      const v = s.scores[stats.name];
      if (!_.isFinite(v)) { return; }
      s.scores[stats.name] = (v - stats.p0) / w;
    });
  }

  export function shuffle<T>(data: T[]) {
    for (let i = 0; i < data.length; ++i) {
      const j = Math.floor((data.length - i) * Math.random()) + i;
      [data[i], data[j]] = [data[j], data[i]];
    }
  }

  export function reduce_by_corelation(stats: FeatureStats[], items: Item[]) {
    const useless_keys: { [K: string]: string } = {};
    Stats.derive_features_covariances(stats, items).forEach((row, i) => {
      row.forEach((r, j) => {
        if (i < j && r.f1 !== r.f2 && Math.abs(r.cor) > 0.95) {
          console.log("reduce:", r.f1, r.f2, r.variance, r.cor);
          useless_keys[r.f2] = r.f2;
        }
      });
    });
    const reduced_stats = stats.filter(s => !useless_keys[s.name]);
    return reduced_stats;
  }
}

export namespace Probability {
  /**
   * ロジスティック回帰におけるある生徒の確率
   */
  export function logreg(student: StudentRaw, features: string[], weights: number[]) {
    const r = features.map((f, i) => weights[i] * student.scores[f]).reduce((s, n) => s + n, 0);
    return 1.0 / (1 + Math.exp(-r));
  };
}

/**
 * 勾配ベクトルのある特徴量の成分のうち, ある生徒による寄与
 */
function partial_difference(feature: string, actual: number, student: StudentRaw, likelyhood: number) {
  const yi = likelyhood;
  const ti = actual;
  const xik = student.scores[feature];
  const v = (yi - ti) * xik;
  return v;
}

/**
 * 学習
 */
export namespace Trainer {

  /**
   * 勾配降下法によりパラメータベクトルを算出する
   * @param target 
   * @param features 
   * @param students 
   * @param params 
   * @returns 
   */
  export function gradient_descent(target: string, features: string[], students: StudentRaw[], params: {
    // イテレーション終了時のパラメータベクトル変化量
    delta_limit: number;
    // 学習率初期値
    initial_learning_rate: number;
    // 学習率減衰率
    learning_rate_decay: number;
  } = {
    delta_limit: 0.001,
    initial_learning_rate: 0.5,
    learning_rate_decay: 0.99998,
  }) {
    // [パラメータベクトルの初期化]
    let weights = features.map(v => (Math.random() * 2 - 1) * 10);
    const {
      delta_limit,
      initial_learning_rate,
      learning_rate_decay,
    } = params;

    // 学習率
    let learning_rate = initial_learning_rate;
    let epoch = 0;
    // 変化量
    while (true) {
      let d = 0;
      const delta_weights = features.map(f => 0);
      students.forEach(s => {
        const likelihood = Probability.logreg(s, features, weights);
        const actual = s.scores[target];
        features.forEach((f, k) => {
          const dv = partial_difference(f, actual, s, likelihood);
          delta_weights[k] += dv;
        });
      });
      
      delta_weights.forEach((dw, k) => {
        const v = dw / students.length;
        d += (v ** 2);
        weights[k] += -v * learning_rate;
      });
      d = Math.sqrt(d * learning_rate);
      if (d < delta_limit) {
        break;
      }
      learning_rate *= learning_rate_decay;
      epoch += 1;
    }
    return weights;
  }

  /**
   * 確率的勾配降下法でパラメータベクトルを計算する
   * @param target 
   * @param features 
   * @param data 
   * @param params 
   * @returns 
   */
  export function stochastic_gradient_descent(target: string, features: string[], students: StudentRaw[], params: {
    // イテレーション終了時のパラメータベクトル変化量
    delta_limit: number;
    // 学習率初期値
    initial_learning_rate: number;
    // 学習率減衰率
    learning_rate_decay: number;
    // バッチサイズ
    batch_size: number;
  } = {
    delta_limit: 0.00001,
    initial_learning_rate: 0.1,
    learning_rate_decay: 0.0001,
    batch_size: 32,
  }) {
    const data = [...students];
    // [パラメータベクトルの初期化]
    let weights = features.map(v => (Math.random() * 2 - 1) * 10);    
    const {
      delta_limit,
      initial_learning_rate,
      learning_rate_decay,
      batch_size,
    } = params;
    // 学習率
    let learning_rate = initial_learning_rate;
    let epoch = 1;
    let i = 0;
    Preprocessor.shuffle(data);
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-10;
    let previous_delta_weights = features.map(f => 0);
    let previous_delta_rms = 0;
    while (true) {
      let d = 0;
      const delta_weights = features.map(f => 0);
      let n = 0;
      for (; i < data.length && n < batch_size; ++i, ++n) {
        const s = data[i];
        const likelihood = Probability.logreg(s, features, weights);
        const actual = s.scores[target];
        features.forEach((f, k) => {
          const dv = partial_difference(f, actual, s, likelihood);
          delta_weights[k] += dv;
        });
      }
      delta_weights.forEach((dw, k) => delta_weights[k] = dw / n);
      const delta_rms = delta_weights.reduce((s, g) => s + g ** 2, 0);
      const vt = (beta2 * previous_delta_rms + (1 - beta2) * delta_rms) / (1 - Math.pow(beta2, epoch));
      delta_weights.forEach((dw, k) => {
        const mt = (beta1 * previous_delta_weights[k] + (1 - beta1) * dw) / (1 - Math.pow(beta1, epoch));
        const x = mt / Math.sqrt(vt + epsilon);
        d += (x ** 2);
        weights[k] += -x * learning_rate;
      });
      d = Math.sqrt(d * learning_rate);
      if (d < delta_limit) {
        console.log("epoch:", epoch, "i:", i, "delta:", d, "delta_rms:", delta_rms);
        break;
      }
      previous_delta_weights = delta_weights;
      previous_delta_rms = delta_rms;
      if (i === data.length) {
        i = 0;
        learning_rate = learning_rate / (1 + learning_rate_decay * epoch);
        Preprocessor.shuffle(data);
        epoch += 1;
      }
    }
    return weights;
  }
}

/**
 * 評価
 */
export namespace Validator {
  export function validate_weights(
    weights: { [key in string]: number[] },
    students: StudentRaw[],
    probability: (student: StudentRaw, weights: number[]) => number
  ) {
    let ok = 0;
    let no = 0;
    for (let i = 0; i < students.length; ++i) {
      const student = students[i];
      const probabilities = _.mapValues(weights, (ws) => probability(student, ws));
      Object.assign(student.scores, _.mapKeys(probabilities, (v, key) => `predicted_${key}`));
      const predicted = _.maxBy(_.keys(probabilities), (key) => probabilities[key])!;
      const is_ok = (student.scores[predicted] === 1);
      student.corrected = is_ok;
      if (is_ok) {
        ok += 1;
      } else {
        // console.log(is_ok ? "[ok]" : "[KO]", i, student.first_name, student.last_name, student.hogwarts_house, predicted, probabilities);
        no += 1;
      }
    }
    return { ok, no };
  }

  /**
   * 重みデータをもとに、生徒1人の所属寮を予測して返す.
   * @param weights 
   * @param features 
   * @param student 
   * @returns 
   */
  export function predict_house(
    weights: { [key in string]: number[] },
    student: StudentRaw,
    probability: (student: StudentRaw, weights: number[]) => number
  ) {
    const probabilities = _.mapValues(weights, (ws) => probability(student, ws));
    const predicted = _.maxBy(_.keys(probabilities), (key) => probabilities[key])!;
    const house = KeyToHouse[predicted];
    // const score = probabilities[predicted] / _.reduce(probabilities, (s, p) => s + p, 0);
    return house;
  }
}
