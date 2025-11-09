import { getShuffledPermutation } from "./random.js";

export function splitData<T>(
  data: T[],
  ratio: number
): { trainData: T[]; testData: T[] } {
  const numRows = data.length;
  const shuffledIndices = getShuffledPermutation(numRows);
  const numTrain = Math.floor(numRows * ratio);
  const trainData: T[] = [];
  const testData: T[] = [];
  for (let i = 0; i < numRows; i++) {
    if (i < numTrain) {
      trainData.push(data[shuffledIndices[i]]);
    } else {
      testData.push(data[shuffledIndices[i]]);
    }
  }
  return { trainData, testData };
}

export function splitDataBatch<T>(data: T[], batchSize: number): T[][] {
  const batches: T[][] = [];
  for (let i = 0; i < data.length; i += batchSize) {
    batches.push(data.slice(i, i + batchSize));
  }
  return batches;
}
