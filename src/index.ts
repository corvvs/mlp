import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import {
  defaultModelFilePath,
  defaultTestDataFilePath,
  defaultTrainDataFilePath,
} from "./constants.js";

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
        command({
          dataFilePath: argv.data ?? defaultTrainDataFilePath,
          modelOutFilePath: argv.outModel,
        });
      } catch (err) {
        console.error("モデル訓練中にエラーが発生しました:", err);
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
