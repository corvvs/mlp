import { spawn } from "child_process";
import { cpus } from "os";
import { readJSONFile, writeJSONFile } from "../libs/io.js";
import type {
  GridSearchConfig,
  GridCombination,
  GridSearchMetadata,
  GridCombinationResult,
} from "../types/grid-search.js";
import { sprintf } from "sprintf-js";

/**
 * グリッドサーチコマンドのメイン関数
 */
export async function command(props: {
  configFile: string;
  maxParallel?: number;
}) {
  console.log("[Grid Search]");
  console.log(`設定ファイル: ${props.configFile}`);

  // 設定ファイルの読み込み
  const config = readJSONFile<GridSearchConfig>(props.configFile);

  // グリッドの組み合わせを生成
  const combinations = generateCombinations(config);
  console.log(`生成された組み合わせ数: ${combinations.length}`);

  // 並行実行数の決定
  const maxParallel = props.maxParallel ?? config.maxParallel ?? cpus().length;
  console.log(`並行実行数: ${maxParallel}`);

  // メタデータの初期化
  const metadata: GridSearchMetadata = {
    configFile: props.configFile,
    startTime: new Date().toISOString(),
    results: [],
    multiplotFile: config.output.multiplotFile,
  };

  console.log("\n=== 訓練開始 ===\n");

  // 並行実行
  const results = await runParallel(
    combinations,
    maxParallel,
    async (combo) => {
      return await runTraining(combo);
    }
  );

  metadata.results = results;
  metadata.endTime = new Date().toISOString();

  // メタデータの保存
  const metadataFile = config.output.metadataFile ?? "grid-search-results.json";
  writeJSONFile(metadataFile, metadata);
  console.log(`\nメタデータを ${metadataFile} に保存しました`);

  // 成功した結果のみをフィルタリング
  const successfulResults = results.filter((r) => r.success);
  console.log(
    `\n成功: ${successfulResults.length} / ${results.length} 組み合わせ`
  );

  if (successfulResults.length === 0) {
    console.error("成功した訓練がないため、multiplotをスキップします");
    return;
  }

  // multiplotの実行
  console.log("\n=== Multiplot生成 ===\n");
  await runMultiplot(
    successfulResults.map((r) => r.modelFile),
    config.output.metrics,
    config.output.multiplotFile
  );

  console.log(`\nMultiplotを ${config.output.multiplotFile} に保存しました`);
  console.log("\n=== Grid Search完了 ===");
}

/**
 * グリッドの全組み合わせを生成
 */
function generateCombinations(config: GridSearchConfig): GridCombination[] {
  const gridKeys = Object.keys(config.grid);
  const gridValues = gridKeys.map((key) => config.grid[key]);

  // 直積を計算
  const cartesianProduct = <T>(arrays: T[][]): T[][] => {
    if (arrays.length === 0) return [[]];
    const [first, ...rest] = arrays;
    const restProduct = cartesianProduct(rest);
    return first.flatMap((item) =>
      restProduct.map((items) => [item, ...items])
    );
  };

  const products = cartesianProduct(gridValues);

  return products.map((product, index) => {
    const params: Record<string, string | number | boolean> = {
      ...config.baseOptions,
    };

    // グリッドパラメータで上書き
    gridKeys.forEach((key, i) => {
      params[key] = product[i];
    });

    // モデルファイル名を生成
    const modelFile = `${config.output.modelPrefix}${index}.json`;

    return {
      index,
      params,
      modelFile,
    };
  });
}

/**
 * 並行実行制御
 */
async function runParallel<T, R>(
  items: T[],
  maxParallel: number,
  fn: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = [];
  const queue = [...items];
  let activeCount = 0;

  return new Promise((resolve, reject) => {
    const runNext = () => {
      if (queue.length === 0 && activeCount === 0) {
        resolve(results);
        return;
      }

      while (queue.length > 0 && activeCount < maxParallel) {
        const item = queue.shift()!;
        activeCount++;

        fn(item)
          .then((result) => {
            results.push(result);
          })
          .catch((error) => {
            reject(error);
          })
          .finally(() => {
            activeCount--;
            runNext();
          });
      }
    };

    runNext();
  });
}

/**
 * 1つの訓練を実行
 */
async function runTraining(
  combo: GridCombination
): Promise<GridCombinationResult> {
  const startTime = Date.now();

  // コマンドライン引数を構築
  const args = buildTrainArgs(combo.params, combo.modelFile);

  console.log(
    sprintf(
      "[%3d] 訓練開始: %s",
      combo.index,
      Object.entries(combo.params)
        .map(([k, v]) => `${k}=${v}`)
        .join(", ")
    )
  );

  try {
    await runCommand("node", [
      "dist/index.js",
      "train",
      ...args,
      "--no-plot=1",
    ]);
    const executionTime = (Date.now() - startTime) / 1000;
    console.log(sprintf("[%3d] 訓練完了 (%.1f秒)", combo.index, executionTime));

    return {
      index: combo.index,
      params: combo.params,
      modelFile: combo.modelFile,
      success: true,
      executionTime,
    };
  } catch (error) {
    const executionTime = (Date.now() - startTime) / 1000;
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(
      sprintf(
        "[%3d] 訓練失敗 (%.1f秒): %s",
        combo.index,
        executionTime,
        errorMessage
      )
    );

    return {
      index: combo.index,
      params: combo.params,
      modelFile: combo.modelFile,
      success: false,
      error: errorMessage,
      executionTime,
    };
  }
}

/**
 * trainコマンドの引数を構築
 */
function buildTrainArgs(
  params: Record<string, string | number | boolean>,
  modelFile: string
): string[] {
  const args: string[] = [];

  // out-modelは必須
  args.push(`--out-model=${modelFile}`);

  // その他のパラメータ
  for (const [key, value] of Object.entries(params)) {
    if (typeof value === "boolean") {
      if (value) {
        args.push(`--${key}`);
      }
    } else {
      args.push(`--${key}=${value}`);
    }
  }

  return args;
}

/**
 * multiplotを実行
 */
async function runMultiplot(
  modelFiles: string[],
  metrics: string[],
  outputFile: string
): Promise<void> {
  const args = [
    `--models=${modelFiles.join(",")}`,
    `--metrics=${metrics.join(",")}`,
    `--output=${outputFile}`,
  ];

  await runCommand("node", ["dist/index.js", "multiplot", ...args]);
}

/**
 * コマンドを実行してPromiseを返す
 */
function runCommand(
  command: string,
  args: string[],
  options: { cwd?: string } = {}
): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd ?? process.cwd(),
      stdio: ["inherit", "ignore", "inherit"],
    });

    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed with exit code ${code}`));
      }
    });

    child.on("error", (error) => {
      reject(error);
    });
  });
}
