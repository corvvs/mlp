// ベクトル算術
export function addVector(a: number[], b: number[]): number[] {
  if (a.length !== b.length) {
    throw new Error("ベクトルの長さが異なります");
  }
  return a.map((val, idx) => val + b[idx]);
}

export function subVector(a: number[], b: number[]): number[] {
  if (a.length !== b.length) {
    throw new Error("ベクトルの長さが異なります");
  }
  return a.map((val, idx) => val - b[idx]);
}

export function mulVectorByScalar(vec: number[], scalar: number): number[] {
  return vec.map((val) => val * scalar);
}

export function hadamardVector(a: number[], b: number[]): number[] {
  if (a.length !== b.length) {
    throw new Error("ベクトルの長さが異なります");
  }
  return a.map((val, idx) => val * b[idx]);
}

// 行列算術
export function addMatrix(a: number[][], b: number[][]): number[][] {
  if (a.length !== b.length || a[0].length !== b[0].length) {
    throw new Error("行列のサイズが異なります");
  }
  return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}

export function subMatrix(a: number[][], b: number[][]): number[][] {
  if (a.length !== b.length || a[0].length !== b[0].length) {
    throw new Error("行列のサイズが異なります");
  }
  return a.map((row, i) => row.map((val, j) => val - b[i][j]));
}

export function mulMatrixByScalar(mat: number[][], scalar: number): number[][] {
  return mat.map((row) => row.map((val) => val * scalar));
}

export function mulMatrix(a: number[][], b: number[][]): number[][] {
  if (a[0].length !== b.length) {
    throw new Error("行列のサイズが乗算に適合しません");
  }
  const result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    const row: number[] = [];
    const ai = a[i];
    for (let j = 0; j < b[0].length; j++) {
      const sum = sumNumArray(ai.map((aik, k) => aik * b[k][j]));
      row.push(sum);
    }
    result.push(row);
  }
  return result;
}

export function mulMatVec(mat: number[][], vec: number[]): number[] {
  if (mat[0].length !== vec.length) {
    throw new Error("行列とベクトルのサイズが乗算に適合しません");
  }
  const result: number[] = [];
  for (let i = 0; i < mat.length; i++) {
    const row = mat[i];
    const sum = sumNumArray(row.map((rowj, j) => rowj * vec[j]));
    result.push(sum);
  }
  return result;
}

export function mulTMatVec(mat: number[][], vec: number[]): number[] {
  if (mat.length !== vec.length) {
    throw new Error("転置行列とベクトルのサイズが乗算に適合しません");
  }
  const result: number[] = [];
  for (let j = 0; j < mat[0].length; j++) {
    const sum = sumNumArray(mat.map((mati, i) => mati[j] * vec[i]));
    result.push(sum);
  }
  return result;
}

export function mulMatTMat(a: number[][], b: number[][]): number[][] {
  if (a[0].length !== b[0].length) {
    throw new Error("行列と転置行列のサイズが乗算に適合しません");
  }
  const result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    const row: number[] = [];
    for (let j = 0; j < b.length; j++) {
      const ai = a[i];
      const sum = sumNumArray(ai.map((aik, k) => aik * b[j][k]));
      row.push(sum);
    }
    result.push(row);
  }
  return result;
}

export function mulTMatMat(a: number[][], b: number[][]): number[][] {
  if (a.length !== b.length) {
    throw new Error("転置行列と行列のサイズが乗算に適合しません");
  }
  const result: number[][] = [];
  for (let i = 0; i < a[0].length; i++) {
    const row: number[] = [];
    for (let j = 0; j < b[0].length; j++) {
      const sum = sumNumArray(a.map((ak, k) => ak[i] * b[k][j]));
      row.push(sum);
    }
    result.push(row);
  }
  return result;
}

// mat1 += mat2
export function addMatX(mat1: number[][], mat2: number[][]) {
  if (mat1.length !== mat2.length || mat1[0].length !== mat2[0].length) {
    throw new Error("行列のサイズが異なります");
  }
  for (let i = 0; i < mat1.length; i++) {
    const mat1i = mat1[i];
    const mat2i = mat2[i];
    for (let j = 0; j < mat1[0].length; j++) {
      mat1i[j] += mat2i[j];
    }
  }
}

// vec1 += vec2
export function addVecX(vec1: number[], vec2: number[]) {
  if (vec1.length !== vec2.length) {
    throw new Error("ベクトルの長さが異なります");
  }
  for (let i = 0; i < vec1.length; i++) {
    vec1[i] += vec2[i];
  }
}

// mat1 += factor * mat2
export function addFactorMatX(
  mat1: number[][],
  factor: number,
  mat2: number[][]
) {
  if (mat1.length !== mat2.length || mat1[0].length !== mat2[0].length) {
    throw new Error("行列のサイズが異なります");
  }
  for (let i = 0; i < mat1.length; i++) {
    const mat1i = mat1[i];
    const mat2i = mat2[i];
    for (let j = 0; j < mat1[0].length; j++) {
      mat1i[j] += factor * mat2i[j];
    }
  }
}

// vec1 += factor * vec2
export function addFactorVecX(vec1: number[], factor: number, vec2: number[]) {
  if (vec1.length !== vec2.length) {
    throw new Error("ベクトルの長さが異なります");
  }
  for (let i = 0; i < vec1.length; i++) {
    vec1[i] += factor * vec2[i];
  }
}

// mat *= factor
export function mulMatX(mat: number[][], factor: number) {
  for (let i = 0; i < mat.length; i++) {
    for (let j = 0; j < mat[0].length; j++) {
      mat[i][j] *= factor;
    }
  }
}

// vec *= factor
export function mulVecX(vec: number[], factor: number) {
  for (let i = 0; i < vec.length; i++) {
    vec[i] *= factor;
  }
}

export function zeroMat(n: number, m: number): number[][] {
  return Array.from({ length: n }, () => new Array(m).fill(0));
}

export function zeroVec(n: number): number[] {
  return new Array(n).fill(0);
}

export function sumNumArray(arr: number[]): number {
  let c = 0;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const y = arr[i] - c;
    const t = sum + y;
    c = t - sum - y;
    sum = t;
  }
  return sum;
}
