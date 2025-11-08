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
