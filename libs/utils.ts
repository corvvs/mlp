import * as _ from "lodash"

export namespace Utils {
  export function basename(path: string) {
    return _.last(path.split(/\/+/)) || "";
  }

  /**
   * 文字列`str`を, 文字数が`max_len`を超えないように短縮して返す.
   * 文字列`abbrev`が指定されている場合, 短縮した`str`に`abbrev`を結合して返すが,
   * その際結合後の長さが`max_len`を超えないようにする.
   * @param str
   * @param max_len 
   * @param abbrev 
   */
  export function abbreviate(str: string, max_len: number, abbrev?: string) {
    const a = abbrev || "";
    if (str.length <= max_len) {
      return str;
    }
    const max_str_len = Math.min(str.length, max_len - a.length);
    if (max_str_len + a.length <= max_len) {
      return str.substring(0, max_str_len) + a;
    } else {
      return str.substring(0, max_str_len);
    }
  }

  export function average_vectors(vectors: number[][]): number[] {
    return vectors
      .reduce((vs, v) => vs.map((x, i) => x + v[i]), vectors[0].map(x => 0))
      .map(x => x / vectors.length);
  }
}
