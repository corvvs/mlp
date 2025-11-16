import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import {
  defaultModelFilePath,
  defaultTestDataFilePath,
  defaultTrainDataFilePath,
} from "./constants.js";
import type { ActivationFunctionSingleArgument } from "./types/af.js";
import { parseRegularizationMethod } from "./libs/train/regularization.js";
import { parseActivationFunction } from "./libs/train/af.js";
import { parseOptimization } from "./libs/train/optimization.js";
import { parseLossFunction } from "./libs/train/loss.js";
import { parseInitializationMethod } from "./libs/train/initialization.js";
import { parseEarlyStopping } from "./libs/train/es.js";

const yargsInstance = yargs(hideBin(process.argv))
  .scriptName("mlp")
  .option("data", {
    type: "string",
    description: "データファイルのパス",
  })
  .option("out", {
    type: "string",
    description: "出力ファイルのパス",
    requiresArg: false,
  })
  // split 向けオプション
  .option("split-ratio", {
    type: "number",
    description: "分割比率 (train: split-ratio, test:1 - split-ratio)",
    default: 0.8,
  })
  .option("out-train", {
    type: "string",
    description: "訓練データ出力ファイルのパス",
    default: defaultTrainDataFilePath,
    requiresArg: false,
  })
  .option("out-test", {
    type: "string",
    description: "テストデータ出力ファイルのパス",
    default: defaultTestDataFilePath,
    requiresArg: false,
  })
  // train 向けオプション
  .option("out-model", {
    type: "string",
    description: "モデル出力ファイルのパス",
    default: defaultModelFilePath,
    requiresArg: false,
  })
  .option("epochs", {
    type: "number",
    description: "エポック数",
    default: 5000,
  })
  .option("batch-size", {
    type: "number",
    description: "バッチサイズ",
    default: 32,
  })
  .option("seed", {
    type: "number",
    description: "乱数シード",
    default: 1234,
  })
  .option("hidden-layers", {
    type: "string",
    description: "隠れ層の構成(カンマ区切りのユニット数)",
    default: "24,24",
  })
  .option("initialization", {
    type: "string",
    description:
      'パラメータ初期化手法; "手法名,分布" の形式で指定; 手法名は Xavier, He から選択, 分布は uniform, normal から選択',
    default: "he,normal",
  })
  .option("default-activation", {
    type: "string",
    description:
      'デフォルトの活性化関数; "関数名,パラメータ..." の形式で指定; 活性化関数名は ReLU, LeakyReLU, Sigmoid, Tanh, Linear から選択',
    default: "ReLU",
  })
  .option("loss", {
    type: "string",
    description:
      '損失関数; "関数名,パラメータ..." の形式で指定; 関数名は CCE, WeightedCCE から選択',
    default: "CCE,1e-9",
  })
  .option("regularization", {
    type: "string",
    description:
      '正則化; "正則化方式,パラメータ..." の形式で指定; 現在は L2 のみ対応',
    example: "L2,0.001",
  })
  .option("optimization", {
    type: "string",
    description:
      '最適化手法; "手法名,パラメータ..." の形式で指定; 手法は SGD, MomentumSGD, AdaGrad, RMSProp, Adam, AdamW から選択',
    example: "Adam,0.001,0.9,0.999",
    default: "SGD,0.01",
  })
  .option("early-stopping-metric", {
    type: "string",
    description:
      "Early Stopping の評価指標; accuracy, loss, precision, recall, f1-score から選択",
    default: "loss",
  })
  .option("early-stopping-patience", {
    type: "number",
    description: "Early Stopping の patience (非負整数)",
    default: 10,
  })
  .option("no-plot", {
    type: "boolean",
    description: "訓練後にグラフをブラウザで自動的に開かない",
  })
  // predict 向けオプション
  .option("model", {
    type: "string",
    description: "モデルファイルのパス",
    default: defaultModelFilePath,
    requiresArg: false,
  })
  // multiplot 向けオプション
  .option("models", {
    type: "string",
    description: "モデルファイルのパス（カンマ区切り）",
    requiresArg: true,
  })
  .option("metrics", {
    type: "string",
    description:
      "表示するメトリクス（カンマ区切り）: loss, accuracy, precision, recall, specificity, f1score",
    default: "loss,accuracy",
    requiresArg: false,
  })
  .option("output", {
    type: "string",
    description: "出力SVGファイルのパス",
    default: "multiplot.svg",
    requiresArg: false,
  })
  // grid-search 向けオプション
  .option("config", {
    type: "string",
    description: "グリッドサーチ設定ファイルのパス",
    requiresArg: true,
  })
  .option("max-parallel", {
    type: "number",
    description: "並行実行数の上限",
    requiresArg: false,
  })
  // コマンド定義
  .command(
    "help",
    "Show help message",
    () => {},
    (argv) => {
      console.log("Hello", argv);
    }
  )
  .command(
    "preproc",
    "前処理を実行します",
    () => {},
    async (argv) => {
      const { command } = await import("./commands/preproc.js");
      try {
        command({
          dataFilePath: argv.data,
          outFilePath: argv.out,
        });
      } catch (err) {
        console.error("前処理中にエラーが発生しました:", err);
        process.exit(1);
      }
    }
  )
  .command(
    "split",
    "データを分割します",
    () => {},
    async (argv) => {
      const { command } = await import("./commands/split.js");
      try {
        command({
          dataFilePath: argv.data,
          ratio: argv.splitRatio,
          outTrainDataFilePath: argv.outTrain,
          outTestDataFilePath: argv.outTest,
        });
      } catch (err) {
        console.error("データ分割中にエラーが発生しました:", err);
        process.exit(1);
      }
    }
  )
  .command(
    "train",
    "モデルの訓練を実行します",
    () => {},
    async (argv) => {
      const { command } = await import("./commands/train.js");
      try {
        const initialization = parseInitializationMethod(argv.initialization);
        const hiddenLayerSizes = ((arg: string) => {
          const a = arg.trim();
          if (a.length === 0) {
            return [];
          }
          const m = a.match(/^(\d+)(,\d+)*$/);
          if (!m) {
            throw new Error(
              `不正な隠れ層の指定: ${arg} (カンマ区切りの正の整数で指定してください)`
            );
          }
          return a.split(",").map((s) => parseInt(s, 10));
        })(argv.hiddenLayers);
        const defaultActivationFunction: ActivationFunctionSingleArgument =
          parseActivationFunction(argv.defaultActivation);
        const lossFunction = parseLossFunction(argv.loss);
        const regularization = parseRegularizationMethod(
          argv.regularization ?? null
        );
        const optimization = parseOptimization(argv.optimization);
        const earlyStopping = parseEarlyStopping({
          metric: argv.earlyStoppingMetric,
          patience: argv.earlyStoppingPatience,
        });

        command({
          dataFilePath: argv.data ?? defaultTrainDataFilePath,
          splitRatio: argv.splitRatio,
          modelOutFilePath: argv.outModel,
          epochs: argv.epochs,
          seed: argv.seed,
          batchSize: argv.batchSize,
          initialization,
          defaultActivationFunction,
          hiddenLayerSizes,
          lossFunction,
          regularization,
          optimization,
          earlyStopping,
          noPlot: argv["no-plot"] !== undefined,
        });
      } catch (err) {
        console.error("モデル訓練中にエラーが発生しました:", err);
        process.exit(1);
      }
    }
  )
  .command(
    "predict",
    "予測を実行します",
    () => {},
    async (argv) => {
      const { command } = await import("./commands/predict.js");
      try {
        command({
          dataFilePath: argv.data ?? defaultTestDataFilePath,
          modelFilePath: argv.model ?? defaultModelFilePath,
        });
      } catch (err) {
        console.error("予測中にエラーが発生しました:", err);
        process.exit(1);
      }
    }
  )
  .command(
    "multiplot",
    "複数モデルの検証メトリクスを比較するSVGを生成します",
    () => {},
    async (argv) => {
      const { command } = await import("./commands/multiplot.js");
      try {
        if (!argv.models) {
          throw new Error(
            "--models オプションでモデルファイルのパスを指定してください（カンマ区切り）"
          );
        }
        const modelFilePaths = argv.models
          .split(",")
          .map((path) => path.trim());
        const metrics = argv.metrics
          .split(",")
          .map((metric: string) => metric.trim());
        command({
          modelFilePaths,
          outputFilePath: argv.output,
          noPlot: argv["no-plot"] !== undefined,
          metrics,
        });
      } catch (err) {
        console.error("マルチプロット生成中にエラーが発生しました:", err);
        process.exit(1);
      }
    }
  )
  .command(
    "grid-search",
    "グリッドサーチを実行します",
    () => {},
    async (argv) => {
      const { command } = await import("./commands/grid-search.js");
      try {
        if (!argv.config) {
          throw new Error(
            "--config オプションで設定ファイルのパスを指定してください"
          );
        }
        await command({
          configFile: argv.config,
          ...(argv.maxParallel !== undefined
            ? { maxParallel: argv.maxParallel }
            : {}),
        });
      } catch (err) {
        console.error("グリッドサーチ中にエラーが発生しました:", err);
        process.exit(1);
      }
    }
  )
  .command(
    "*",
    false,
    () => {},
    (argv) => {
      if (argv._[0]) {
        console.error(`未知のコマンド: ${argv._[0]}`);
      }
      yargsInstance.showHelp();
      process.exit(1);
    }
  )
  .demandCommand(1, "コマンドが必要です。使用可能なコマンド: help")
  .help(false)
  .fail((msg, err, yargs) => {
    yargs.showHelp();
    process.exit(1);
  });

await yargsInstance.parse();
