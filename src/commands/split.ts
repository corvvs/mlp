import { sprintf } from "sprintf-js";
import { readCSVFile, writeCSVFile } from "../libs/io.js";
import { getShuffledPermutation } from "../libs/random.js";
import { defaultPreprocDataFilePath } from "../constants.js";

export function command(props: {
  dataFilePath: string | undefined;
  ratio: number;
  outTrainDataFilePath: string;
  outTestDataFilePath: string;
}) {
  const actualDataFilePath = props.dataFilePath ?? defaultPreprocDataFilePath;
  const rows = readCSVFile(actualDataFilePath);
  console.log(`データファイル ${actualDataFilePath} を読み込みました`);

  // rows を train と test に ratio : (1 - ratio) の比率で分割する
  const numRows = rows.length;
  const shuffledIndices = getShuffledPermutation(numRows);
  const numTrain = Math.floor(numRows * props.ratio);
  const trainRows: string[][] = [];
  const testRows: string[][] = [];
  for (let i = 0; i < numRows; i++) {
    if (i < numTrain) {
      trainRows.push(rows[shuffledIndices[i]]);
    } else {
      testRows.push(rows[shuffledIndices[i]]);
    }
  }

  console.log(
    sprintf(
      `データの分割率: %1.2f (訓練データ), %1.2f (テストデータ)`,
      props.ratio,
      1 - props.ratio
    )
  );
  console.log(`訓練データの行数: ${trainRows.length}`);
  console.log(`テストデータの行数: ${testRows.length}`);

  // 分割したデータをファイルに書き出す
  writeCSVFile(props.outTrainDataFilePath, trainRows);
  console.log(`訓練データを ${props.outTrainDataFilePath} に書き出しました`);
  writeCSVFile(props.outTestDataFilePath, testRows);
  console.log(`テストデータを ${props.outTestDataFilePath} に書き出しました`);
}
