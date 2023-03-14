import * as _ from "lodash"
import * as fs from 'fs';
import { Parser } from './libs/parse';
import { Stats } from "./libs/stats";
import { Graph } from "./libs/graph";
import { Box, FeatureCorrelation, Item, Standardizers, StudentRaw } from "./libs/definitions";
import { Geometric } from "./libs/geometric";
import { Preprocessor, Probability, Trainer, Validator } from "./libs/train";
import { IO } from "./libs/io";
import { Flow } from "./libs/flow";
import { Utils } from "./libs/utils";
import { sprintf } from "sprintf-js";
import { Spider } from "./libs/spider";
import { exec } from "child_process";
import { LA, MLP } from "./libs/mlp";

/**
 * 便宜上のメイン関数
 * @returns 
 */
function main() {
  // [treat ARGV]
  const [dataset_path] = process.argv.slice(2);
  if (!dataset_path) {
    Flow.exit_with_error(
      `usage: ${Utils.basename(process.argv[0])} ${Utils.basename(process.argv[1])} [dataset path]`
    );
  }
  // [データセット読み取り]
  const data = fs.readFileSync(dataset_path, 'utf-8');

  // [スキーマ処理]
  const rows = _.compact(data.split("\n"));
  const splitted_rows = rows.map(row => row.split(","));
  const float_features: string[] = _.range(splitted_rows[0].length - 2).map(i => `p${i}`);
  const schema = ["id", "raw_actual", ...float_features];

  // [データの生成]
  const items: Item[] = splitted_rows.map(s => {
    const raw_data = s;
    const id = s[0];
    const actual = s[1];
    const scores: any = {};
    _.slice(s, 2).forEach((v, i) => {
      scores[schema[i + 2]] = parseFloat(v);
    });
    return {
      id, actual, scores, raw_data,
    };
  });

  // [統計データの計算]
  const feature_stats = float_features.map(feature => Stats.derive_feature_stats(feature, items));
  const correlations: FeatureCorrelation[][] = Stats.derive_features_covariances(feature_stats, items);

  // // [統計量の出力]
  // Stats.print_stats_for_features(feature_stats_0, correlations);

  // // [欠損データの補完]
  // for (let i = 0; i < raw_students.length; ++i) {
  //   const student = raw_students[i];
  //   feature_stats_0.forEach(stats => {
  //     const v = student.scores[stats.name];
  //     if (!_.isFinite(v)) {
  //       student.scores[stats.name] = stats.p50;
  //     }
  //   })
  // }

  // // [統計データの再計算]
  // // 標準化するとデータが変更されるので, 復元用にとっておく
  // const feature_stats = (() => {
  //   const feature_stats = float_features.map(feature => Stats.derive_feature_stats(feature, raw_students));
  //   return Preprocessor.reduce_by_corelation(feature_stats, raw_students);
  // })();

  // [前処理:標準化]
  feature_stats.forEach(stats => Preprocessor.standardize(stats, items));
  const using_features = feature_stats.map(f => f.name);
  const standardized_stats = float_features.map(feature => Stats.derive_feature_stats(feature, items));

  const item_vectors = items.map(item => LA.make_vector(using_features.map(f => item.scores[f])));
  const perceptron = MLP.make(using_features.length, 3, 2);
  console.log(perceptron);
  const rs = MLP.forward(perceptron, item_vectors[0]);
  rs.forEach((r, i) => {
    console.log(i);
    LA.print_matrix(LA.transpose(r));
    if (0 < i) {
      const layer = perceptron.layers[i - 1];
      const act = layer.activator.f(r);
      console.log(i);
      LA.print_matrix(act);
    }
  });

  // // [ペアプロット]
  // {
  //   // [SVGの作成]
  //   // [SVGの初期化]
  //   const width = 600;
  //   const height = 600;
  //   const box: Box = {
  //     p1: { x: 0, y: 0 },
  //     p2: { x: width * float_features.length, y: height * float_features.length },
  //   };
  //   const dimension = Geometric.formDimensionByBox(box);
  //   const svg = new Spider(dimension);
  //   Graph.drawPairPlot(svg, { width, height }, items, standardized_stats);
  //   const pair_plot_svg = svg.render();
  
  //   // [ファイルに書き出す]
  //   const out_path = "pair_plot.svg";
  //   IO.save(out_path, pair_plot_svg);
  //   exec(`open ${out_path}`, (error, strout, strerr) => {
  //     if (error) {
  //       console.error(strerr);
  //     }
  //   });
  // }

  // // [定数項の付加]
  // raw_students.forEach(s => s.scores["constant"] = 1);
  // using_features.push("constant");

  // // [学習]
  // Preprocessor.shuffle(raw_students);
  // const cv_test_rate = 0.25;
  // const n_test = Math.floor(raw_students.length * cv_test_rate);
  // const students_test = raw_students.slice(0, n_test);
  // const students_rest = raw_students.slice(n_test);
  // const cv_validation_division = 8;
  // const house_key = ["is_g", "is_r", "is_h", "is_s"];
  // const results = _(_.range(cv_validation_division)).map(i => {
  //   const from = Math.floor(students_rest.length / cv_validation_division * i);
  //   const to = Math.floor(students_rest.length / cv_validation_division * (i + 1));
  //   const students_validate = students_rest.slice(from, to);
  //   const students_training = students_rest.filter((s, i) => i < from || to <= i);
  //   if (students_validate.length === 0 || students_training.length === 0) { return null; }
  //   console.log(i, students_training.length, from, to, students_validate.length);
  //   const ws = _(house_key).keyBy(key => key).mapValues(key => {
  //     const t0 = Date.now();
  //     const ws = Trainer.stochastic_gradient_descent(key, using_features, students_training);
  //     const t1 = Date.now();
  //     console.log(`${key}: ${t1 - t0}ms`);
  //     return ws;
  //   }).value();

  //   // [評価]
  //   const f_probability = (student: StudentRaw, weights: number[]) => Probability.logreg(student, using_features, weights);
  //   const { ok, no } = Validator.validate_weights(ws, students_validate, f_probability);
  //   const precision = ok / (ok + no);
  //   console.log(i, precision, `= ${ok} / (${ok} + ${no})`);
  //   _.each(ws, (ws, key) => {
  //     console.log(key, ":", "[", ws.map(w => sprintf("%+1.2f", w)).join(", "), "]");
  //   });
  //   return { i, ws, precision };
  // }).compact().value();

  // const ws = _.mapValues(results[0].ws, (ws, key) => {
  //   return Utils.average_vectors(results.map(r => r.ws[key]))
  // })

  // // console.log("wins:", champion.i, champion.precision);
  // // _.each(champion.ws, (ws, key) => {
  // //   console.log(key, ":", "[", ws.map(w => sprintf("%+1.2f", w)).join(", "), "]");
  // // });
  // console.log("averaged:");
  // _.each(ws, (ws, key) => {
  //   console.log(key, ":", "[", ws.map(w => sprintf("%+1.2f", w)).join(", "), "]");
  // });
  // {
  //   const f_probability = (student: StudentRaw, weights: number[]) => Probability.logreg(student, using_features, weights);
  //   const { ok, no } = Validator.validate_weights(ws, students_test, f_probability);
  //   console.log("test: ", ok / (ok + no), `= ${ok} / (${ok} + ${no})`);
  // }
  // {
  //   const f_probability = (student: StudentRaw, weights: number[]) => Probability.logreg(student, using_features, weights);
  //   const { ok, no } = Validator.validate_weights(ws, raw_students, f_probability);
  //   console.log("whole:", ok / (ok + no), `= ${ok} / (${ok} + ${no})`);
  // }


  // // [パラメータファイル出力]
  // {
  //   const standardizers: Standardizers = _(feature_stats).keyBy(s => s.name).mapValues((s) => ({
  //     mean: s.mean,
  //     std: s.std,
  //   })).value();
  //   const weights = _.mapValues(ws, (weights, key) => {
  //     const d: { [key: string]: number } = {};
  //     weights.forEach((w, i) => d[using_features[i]] = w);
  //     return d;
  //   });
  //   IO.save("parameters.json", JSON.stringify({
  //     weights, standardizers,
  //   }, null, 2));
  // }

  // // // [外したデータをでっかくしてペアプロット]
  // {
  //   house_key.forEach(key => using_features.push(`predicted_${key}`));
  //   const sorted_students = _.sortBy(raw_students, s => s.corrected ? 0 : 1);
  //   const out_features = using_features.filter(f => f !== "constant");
  //   const out_stats = out_features.map(feature => Stats.derive_feature_stats(feature, sorted_students));
  //   const width = 600;
  //   const height = 600;
  //   const box: Box = {
  //     p1: { x: 0, y: 0 },
  //     p2: { x: width * out_features.length, y: height * out_features.length },
  //   };
  //   const dimension = Geometric.formDimensionByBox(box);
  //   const svg = new Spider(dimension);

  //   // [SVGの作成]
  //   Graph.drawPairPlot(svg, { width, height }, sorted_students, out_stats);
  //   const pair_plot_svg = svg.render();

  //   IO.save("training.svg", pair_plot_svg);
  // }
}

try {
  main();
} catch (e) {
  // if (e instanceof Error) {
  //   console.error(`[${e.name}]`, e.message);
  // } else {
    console.error(e);
  // }
}
