export function localRandom(): number {
  return Math.random();
}

export function normalRandom(mean: number, stddev: number): number {
  // Box-Muller法
  const u1 = localRandom();
  const u2 = localRandom();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stddev + mean;
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
