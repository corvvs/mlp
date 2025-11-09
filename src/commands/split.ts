import { sprintf } from "sprintf-js";
import { readCSVFile, writeCSVFile } from "../libs/io.js";
import { getShuffledPermutation } from "../libs/random.js";
import { defaultPreprocDataFilePath } from "../constants.js";
import { splitData } from "../libs/split.js";

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
  const { a: trainData, b: testData } = splitData(rows, props.ratio);

  console.log(
    sprintf(
      `データの分割率: %1.2f (訓練データ), %1.2f (テストデータ)`,
      props.ratio,
      1 - props.ratio
    )
  );
  console.log(`訓練データの行数: ${trainData.length}`);
  console.log(`テストデータの行数: ${testData.length}`);

  // 分割したデータをファイルに書き出す
  writeCSVFile(props.outTrainDataFilePath, trainData);
  console.log(`訓練データを ${props.outTrainDataFilePath} に書き出しました`);
  writeCSVFile(props.outTestDataFilePath, testData);
  console.log(`テストデータを ${props.outTestDataFilePath} に書き出しました`);
}
