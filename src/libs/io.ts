import { readFileSync, writeFileSync } from "fs";

/**
 * 前処理対象のCSVファイルを読み, 行の配列として返す
 * @param path
 * @returns
 */
export function readCSVFile(path: string | undefined) {
  // パスが指定されている場合はそこから, 指定されていない場合はstdinから読み込む
  const data = readFileSync(path ?? "dev/stdin", "utf-8");
  console.log(`データファイル ${path ?? "stdin"} を読み込みました`);

  // dataを見出しなしのCSVと考え, パースする
  const rows = data
    .split("\n")
    .map((line) => line.split(","))
    .filter((row) => row.length > 1); // 空行を除去

  // サイズをチェック
  if (rows.length === 0) {
    throw new Error("データファイルが空です");
  }
  console.log(`データファイルの行数: ${rows.length}`);
  const numCols = rows[0].length;
  console.log(`データファイルの列数: ${numCols}`);
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].length !== numCols) {
      throw new Error(
        `データファイルの各行の列数が一致しません at ${i}, expected ${numCols}, got ${rows[i].length}`
      );
    }
  }

  return rows;
}

export function writeCSVFile(path: string, rows: string[][]) {
  const data = rows.map((row) => row.join(",")).join("\n");
  writeFileSync(path, data, "utf-8");
}
