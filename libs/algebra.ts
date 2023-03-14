import * as _ from "lodash"
import { Affine2D, Box, DrawOption, Vector2D } from './definitions';

export namespace Algebra {
  /**
   * 矩形領域`from`を矩形領域`to`に写すようなアフィン変換を返す\
   * ただし変換は平行移動と拡大縮小のみで表現される
   * @param from 
   * @param to 
   */
  export function affine_box_to_box(from: Box, to: Box): Affine2D {
    const mx = (to.p2.x - to.p1.x) / (from.p2.x - from.p1.x);
    const my = (to.p2.y - to.p1.y) / (from.p2.y - from.p1.y);
    return {
      xx: mx, xy: 0, xt: -from.p1.x * mx + to.p1.x,
      yx: 0, yy: my, yt: -from.p1.y * my + to.p1.y,
    };
  }

  /**
   * ベクトル`v`にアフィン変換`t`を適用した結果を返す
   * @param t 
   * @param v 
   * @returns 
   */
  export function apply_affine(t: Affine2D, v: Vector2D): Vector2D {
    return {
      ...v,
      x: t.xx * v.x + t.xy * v.y + t.xt,
      y: t.yx * v.x + t.yy * v.y + t.yt,
    };
  }
}

export namespace Draw {
  export function box(svg: any, box: Box, option: DrawOption = {}) {
    const x = Math.min(box.p1.x, box.p2.x);
    const y = Math.min(box.p1.y, box.p2.y);
    const width = Math.abs(box.p2.x - box.p1.x);
    const height = Math.abs(box.p2.y - box.p1.y);
    svg.rect({
      x,
      y,
      width,
      height,
      ...option,
    });    
  }
}
