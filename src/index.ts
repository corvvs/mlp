import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import {
  defaultModelFilePath,
  defaultTestDataFilePath,
  defaultTrainDataFilePath,
} from "./constants.js";
import type { ActivationFunctionSingleArgumentMethod } from "./types/af.js";
import { parseRegularizationMethod } from "./libs/train/regularization.js";

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
  .option("ratio", {
    type: "number",
    description: "分割比率 (train:ratio, test:1-ratio)",
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
  .option("default-activation", {
    type: "string",
    enums: ["ReLU", "sigmoid", "tanh", "leakyReLU"],
    description: "デフォルトの活性化関数",
    default: "ReLU",
  })
  .option("regularization", {
    type: "string",
    description: '正則化; "正則化方式,パラメータ..." の形式で指定',
    example: "L2,0.001",
  })
  // predict 向けオプション
  .option("model", {
    type: "string",
    description: "モデルファイルのパス",
    default: defaultModelFilePath,
    requiresArg: false,
  })
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
      console.log(argv);
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
          ratio: argv.ratio,
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
        const hiddenLayerSizes = ((arg: string) => {
          const a = arg.trim();
          const m = a.match(/^(\d+)(,\d+)*$/);
          if (!m) {
            throw new Error(
              `不正な隠れ層の指定: ${arg} (カンマ区切りの正の整数で指定してください)`
            );
          }
          return a.split(",").map((s) => parseInt(s, 10));
        })(argv.hiddenLayers);
        const defaultActivationFunction: ActivationFunctionSingleArgumentMethod =
          ((arg: string) => {
            switch (arg.toLowerCase()) {
              case "relu":
                return "ReLU";
              case "tanh":
                return "tanh";
              case "leakyrelu":
                return "LeakyReLU";
              case "sigmoid":
                return "sigmoid";
              default:
                throw new Error(
                  `不明な活性化関数: ${arg} (ReLU, sigmoid, tanh, LeakyReLU のいずれかを指定してください)`
                );
            }
          })(argv.defaultActivation);
        const regularization = parseRegularizationMethod(
          argv.regularization ?? null
        );

        command({
          dataFilePath: argv.data ?? defaultTrainDataFilePath,
          modelOutFilePath: argv.outModel,
          epochs: argv.epochs,
          seed: argv.seed,
          batchSize: argv.batchSize,
          defaultActivationFunction: defaultActivationFunction,
          hiddenLayerSizes,
          regularization,
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
