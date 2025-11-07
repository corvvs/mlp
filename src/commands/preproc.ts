import { defaultPreprocDataFilePath } from "../constants.js";
import { readCSVFile, writeCSVFile } from "../libs/io.js";

const colIndexID = 0; // ID列のインデックス
const colIndexAnswer = 1; // 正解ラベル列のインデックス

/**
 * 前処理を実行する
 * @param rows
 * @returns
 */
function execPreproc(rows: string[][]) {
  return rows.map((row, i) => {
    const newRow: string[] = [];
    for (let j = 0; j < row.length; j++) {
      const value = row[j];
      switch (j) {
        case colIndexID:
          // ID列: 無視
          break;
        case colIndexAnswer:
          // 正解ラベル列: "M" -> 1, "B" -> 0 に変換
          switch (value.trim()) {
            case "M":
              newRow.push("1");
              break;
            case "B":
              newRow.push("0");
              break;
            default:
              throw new Error(`不明な正解ラベル at(${i}, ${j}): ${value}`);
          }
          break;
        default:
          // その他の列: 前後の空白を削除
          newRow.push(row[j].trim());
          break;
      }
    }
    return newRow;
  });
  // .filter((row, i) => {
  //   // 欠損値を含む行を削除 -> 消えすぎるのでやらない
  //   for (let j = 0; j < row.length; j++) {
  //     const floatVal = parseFloat(row[j]);
  //     if (!isFinite(Number(row[j])) || floatVal === 0) {
  //       console.log(`欠損値を含む行を削除 at row ${i}`);
  //       return false;
  //     }
  //   }
  //   return true;
  // });
}

export function command(props: {
  dataFilePath: string | undefined;
  outFilePath: string | undefined;
}) {
  console.log("前処理を開始しました");
  const dataFilePath = props.dataFilePath ?? "/dev/stdin";
  const csvRows = readCSVFile(dataFilePath);
  console.log(`データファイル ${dataFilePath} を読み込みました`);
  const preprocRows = execPreproc(csvRows);
  console.log("前処理を完了しました", preprocRows.length);

  const outFilePath = props.outFilePath ?? defaultPreprocDataFilePath;
  writeCSVFile(outFilePath, preprocRows);
  console.log(`前処理済みデータを ${outFilePath} に出力しました`);
}
