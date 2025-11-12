import { getShuffledPermutation } from "./random.js";

export function stratifiedSplitData<T>(
  data: T[],
  getClass: (item: T) => string,
  ratio: number
): { a: T[]; b: T[] } {
  const classMap: { [key: string]: T[] } = {};
  for (const item of data) {
    const cls = getClass(item);
    if (!(cls in classMap)) {
      classMap[cls] = [];
    }
    classMap[cls].push(item);
  }
  const a: T[] = [];
  const b: T[] = [];
  for (const cls in classMap) {
    const items = classMap[cls];
    const { a: aItems, b: bItems } = splitData(items, ratio);
    a.push(...aItems);
    b.push(...bItems);
  }
  return { a, b };
}

export function splitData<T>(data: T[], ratio: number): { a: T[]; b: T[] } {
  const numRows = data.length;
  const shuffledIndices = getShuffledPermutation(numRows);
  const numTrain = Math.floor(numRows * ratio);
  const a: T[] = [];
  const b: T[] = [];
  for (let i = 0; i < numRows; i++) {
    if (i < numTrain) {
      a.push(data[shuffledIndices[i]]);
    } else {
      b.push(data[shuffledIndices[i]]);
    }
  }
  return { a, b };
}

export function stratifiedSpllitData<T>(
  data: T[],
  getClass: (item: T) => string,
  batchSize: number
): T[][] {
  const classMap: { [key: string]: T[] } = {};
  for (const item of data) {
    const cls = getClass(item);
    if (!(cls in classMap)) {
      classMap[cls] = [];
    }
    classMap[cls].push(item);
  }

  const classBatches: { [key: string]: T[][] } = {};
  for (const cls in classMap) {
    const items = classMap[cls];
    classBatches[cls] = splitDataBatch(items, batchSize);
  }

  const batches: T[][] = [];
  for (let i = 0; i < batchSize; i++) {
    const bs: T[] = [];
    Object.keys(classBatches).forEach((cls) => {
      const clsBatches = classBatches[cls];
      const cb = clsBatches.splice(0, 1);
      if (!cb) {
        return;
      }
      bs.push(...cb[0]);
    });
    batches.push(bs);
  }
  return batches;
}

export function splitDataBatch<T>(data: T[], batchSize: number): T[][] {
  const batches: T[][] = [];
  getShuffledPermutation;
  for (let i = 0; i < data.length; i += batchSize) {
    batches.push(data.slice(i, i + batchSize));
  }
  return batches;
}
