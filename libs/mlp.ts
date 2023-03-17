import * as _ from "lodash"
import { sprintf } from "sprintf-js";

type Matrix = {
  /**
   * 行数
   */
  h: number;
  /**
   * 列数
   */
  w: number;
  /**
   * 本体
   */
  values: number[][];
};

/**
 * 余ベクトルを取り余ベクトルを返す形式の活性化関数
 */
type VectorActivationFunc = (covector: Matrix) => Matrix;

/**
 * 余ベクトルを2つ取り余ベクトルを返す形式の損失関数
 */
type VectorLossFunc = (actual: Matrix, predicted: Matrix) => Matrix;


/**
 * レイヤー
 */
type Layer = {
  /**
   * このレイヤーの入力ベクトルの次数
   */
  n_in: number;
  /**
   * このレイヤーの出力ベクトルの次数
   */
  n_out: number;
  /**
   * n_in x n_out 行列
   */
  weights: Matrix;
  /**
   * 1 x n_out 行列
   */
  bias: Matrix;
  /**
   * 活性化関数
   */
  activator: {
    /**
     * 活性化関数
     */
    f: VectorActivationFunc;
    /**
     * 活性化関数の導関数
     */
    df: VectorActivationFunc;
  };
};

/**
 * 多層パーセプトロン
 */
type MultiLayeredPerceptron = {
  /**
   * レイヤー数
   * 入力層/出力層を含む
   */
  layer_count: number;
  /**
   * 入力ベクトルの次数
   */
  n_input: number;
  /**
   * 出力ベクトルの次数
   */
  n_output: number;
  /**
   * **入力層以外の**レイヤー
   */
  layers: Layer[];
};

// ベクトルは (h, w) = (1, w)
// 余ベクトルは (h, w) = (h, 1)
// として表現する

/**
 * 線形代数
 */
export namespace LA {

  /**
   * 指定したサイズの単位行列を生成する
   */
  export function unit(h: number, w: number) {
    if (h !== w) {
      throw new Error("not square");
    }
    const z = zero(h, h);
    _.range(h).forEach((x, i) => z[i][i] = 1);
    return z;
  }

  /**
   * 指定したサイズのゼロ行列を生成する
   */
  export function zero(h: number, w: number) {
    const zeroes = _.range(w).map(x => 0);
    const values = _.map(_.range(h), x => [...zeroes]);
    return {
      h,
      w,
      values,
    };
  }

  /**
   * 指定したサイズのランダム行列を生成する
   */
  export function random(h: number, w: number) {
    const z = zero(h, w);
    for (let i = 0; i < z.h; ++i) {
      for (let j = 0; j < z.w; ++j) {
        z.values[i][j] = Math.random() * 2 - 1;
      }
    }
    return z;
  }

  /**
   * 指定した行列を転置した行列を生成する
   */
  export function transpose(m: Matrix): Matrix {
    const zeroes = _.range(m.h).map(x => 0);
    const values = _.map(_.range(m.w), x => [...zeroes]);
    const z = zero(m.w, m.h);
    m.values.forEach((row, i) => {
      row.forEach((v, j) => {
        z.values[j][i] = v;
      });
    });
    return z;
  }

  /**
   * 2つの行列m1, m2の和を生成する
   */
  export function add(m1: Matrix, m2: Matrix): Matrix {
    if (m1.h !== m2.h) {
      throw new Error("m1.h and m2.h do not match!!");
    }
    if (m1.w !== m2.w) {
      throw new Error("m1.w and m2.w do not match!!");
    }
    const z = zero(m1.h, m2.w);
    for (let i = 0; i < z.h; ++i) {
      for (let j = 0; j < z.w; ++j) {
        z.values[i][j] = m1.values[i][j] + m2.values[i][j];
      }
    }
    return z;
  }

  /**
   * 2つの行列m1, m2の積を生成する
   * @returns 
   */
  export function prod(m1: Matrix, m2: Matrix): Matrix {
    if (m1.w !== m2.h) {
      throw new Error("m1.w and m2.h do not match!!");
    }
    const z = zero(m1.h, m2.w);
    for (let i = 0; i < z.h; ++i) {
      for (let j = 0; j < z.w; ++j) {
        for (let k = 0; k < m1.w; ++k) {
          z.values[i][j] += m1.values[i][k] * m2.values[k][j];
        }
      }
    }
    return z;
  }

  export function make_vector(vs: number[]): Matrix {
    return {
      h: 1,
      w: vs.length,
      values: [[...vs]],
    }
  }

  export function print_matrix(m: Matrix) {
    console.log(`(${m.h} x ${m.w})`);
    m.values.forEach(row => {
      console.log("[", row.map(r => sprintf("%+1.2f", r)).join(" "), "]");
    });
  }
};

/**
 * 多層パーセプトロン
 */
export namespace MLP {
  /**
   * 多層パーセプトロンオブジェクトを生成する
   */
  export function make(n_input: number, layer_count: number, n_output: number) {
    if (n_input < 1) {
      throw new Error("n_input must be >= 1");
    }
    if (layer_count < 2) {
      throw new Error("layer_count must be >= 2");
    }
    if (n_output < 1) {
      throw new Error("n_output must be >= 1");
    }
    // TODO: n_mid をハイパーパラメータとする
    const n_mid = 10; // 固定
    const mlp: MultiLayeredPerceptron = {
      layer_count,
      n_input,
      n_output,
      layers: _.range(layer_count - 1).map(i => {
        const is_first = i === 0;
        const is_last = i + 1 === layer_count - 1;
        const n_in = is_first ? n_input : n_mid;
        const n_out = is_last ? n_output : n_mid;
        return {
          n_in,
          n_out,
          weights: LA.random(n_out, n_in),
          bias: LA.random(n_out, 1),
          activator: is_last ? Activator.softmax() : Activator.tanh(),
        };
      }),
    };
    return mlp;
  }

  /**
   * 入力ベクトル`v_input`を多層パーセプトロン`perceptron`に流した時の各層の出力ベクトルを返す.
   * なお結果は入力ベクトルを含む.
   * また入力以外の結果は活性化関数を通す前のものになる.
   */
  export function forward(perceptron: MultiLayeredPerceptron, v_input: Matrix): Matrix[] {
    let v = LA.transpose(v_input);
    const results: Matrix[] = [v];
    for (let i = 1; i < perceptron.layer_count; ++i) {
      const layer = perceptron.layers[i - 1];
      // 層を進める
      // z = Wz + b
      const x = LA.prod(layer.weights, v);
      v = LA.add(x, layer.bias);
      results.push(v);
      // 活性化関数を適用
      v = layer.activator.f(v);
    }
    return results;
  }

  /**
   * 各層の出力リストと損失関数から, 各層の重みとバイアスに対する勾配を計算する
   * @param perceptron パーセプトロン
   * @param results 各層の出力
   * @param actual 教師データ
   * @param loss (1データ)損失関数
   */
  export function backward(
    perceptron: MultiLayeredPerceptron,
    results: Matrix[],
    actual: Matrix,
    loss: VectorLossFunc
  ) {
    const layer = perceptron.layers[perceptron.layer_count - 2];
    const r = _.last(results)!;
    const predicted = layer.activator.f(r);
    console.log("[r]");
    LA.print_matrix(r);
    console.log("[actual]");
    LA.print_matrix(actual);
    console.log("[predicted]");
    LA.print_matrix(predicted);
    console.log("[loss]");
    LA.print_matrix(loss(actual, predicted));
  }
};

/**
 * 活性化関数
 */
export namespace Activator {

  function f_tanh(vs: Matrix) {
    if (vs.w !== 1) {
      throw new Error("vs is not co-vector");
    }
    return {
      h: vs.h,
      w: vs.w,
      values: _.map(_.range(vs.h), i => {
        const v = vs.values[i][0];
        return [(1 - Math.exp(-2 * v)) / (1 + Math.exp(-2 * v))];
      }),
    };
  }

  function df_tanh(vs: Matrix) {
    if (vs.w !== 1) {
      throw new Error("vs is not co-vector");
    }
    const values = _.range(vs.h).map(l => {
      return _.range(vs.h).map(n => {
        if (l !== n) { return 0; }
        const v = vs.values[n][0];
        return 4 / Math.pow(Math.exp(v) + Math.exp(-v), 2);
      });
    });
    return {
      h: vs.h,
      w: vs.h,
      values,
    };
  }

  // Matrix は 余ベクトルであること
  export function tanh() {
    return {
      f: f_tanh,
      df: df_tanh,
    };
  }

  function f_softmax(vs: Matrix) {
    if (vs.w !== 1) {
      throw new Error("vs is not co-vector");
    }
    const z = _.range(vs.h).reduce((s,i) => s + Math.exp(vs.values[i][0]), 0);
    return {
      h: vs.h,
      w: vs.w,
      values: _.map(_.range(vs.h), i => {
        const v = vs.values[i][0];
        return [Math.exp(v) / z];
      }),
    };
  }

  function df_softmax(vs: Matrix) {
    if (vs.w !== 1) {
      throw new Error("vs is not co-vector");
    }
    const fv = f_softmax(vs);
    const values = _.range(vs.h).map(l => {
      return _.range(vs.h).map(n => {
        const delta = l === n ? 1 : 0;
        return fv[l] * (delta - fv[n]);
      });
    });
    return {
      h: vs.h,
      w: vs.h,
      values,
    };
  }

  export function softmax() {
    // f はベクトル, df は行列
    return {
      f: f_softmax,
      df: df_softmax,
    };
  }
};

/**
 * データ1つに対する損失関数
 * 入力はすべて余ベクトル
 */
export namespace Loss {
  /**
   * 交差エントロピー
   */
  export function cross_entropy(actual: Matrix, predicted: Matrix): Matrix {
    if (actual.h !== predicted.h) {
      throw new Error("actual.h and predicted.h do not match");
    }
    if (actual.w !== 1) {
      throw new Error("actual is not co-vector");
    }
    if (predicted.w !== 1) {
      throw new Error("predicted is not co-vector");
    }
    const values = _.range(actual.h).map(i => {
      const y = actual.values[i][0];
      const p = predicted.values[i][0];
      const l = [y * Math.log(p) + (1 - y) * Math.log(1 - p)];
      return l;
    });
    return {
      h: actual.h,
      w: actual.w,
      values,
    };
  }
};

