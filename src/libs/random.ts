/**
 * 0からn-1までの整数のシャッフルされた順列を生成する
 * @param n
 * @returns
 */
export function getShuffledPermutation(n: number): number[] {
  // Fisher-Yatesアルゴリズムでシャッフル
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}
