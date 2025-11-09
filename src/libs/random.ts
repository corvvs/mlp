let localRandomState = 2463534242;
export function localRandom(): number {
  let x = localRandomState;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  localRandomState = x >>> 0; // 符号なし32bit化
  return localRandomState / 0xffffffff; // 0〜1の範囲に正規化
}

export function localSrand(seed: number) {
  localRandomState = seed || 2463534242;
}

let normalState = 0;
let z0: number;
let z1: number;
export function normalRandom(mean: number, stddev: number): number {
  // Box-Muller法
  if (normalState === 0) {
    const u1 = localRandom();
    const u2 = localRandom();
    z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
    normalState = 1;
    return z0 * stddev + mean;
  } else {
    normalState = 0;
    return z1 * stddev + mean;
  }
}

/**
 * 0からn-1までの整数のシャッフルされた順列を生成する
 * @param n
 * @returns
 */
export function getShuffledPermutation(n: number): number[] {
  // Fisher-Yatesアルゴリズムでシャッフル
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(localRandom() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}
