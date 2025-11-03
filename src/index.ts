import yargs from "yargs";
import { hideBin } from "yargs/helpers";

const yargsInstance = yargs(hideBin(process.argv))
  .scriptName("mlp")
  .option("data", {
    type: "string",
    description: "データファイルのパス",
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
      const { command } = await import("./commands/preproc.js");
      command({
        dataFilePath: argv.data,
      });
    }
  )
  .command(
    "split",
    "データを分割します",
    () => {},
    async (argv) => {
      const { command } = await import("./commands/split.js");
      command();
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
