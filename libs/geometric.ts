import { Box, Dimension, Inset, Vector2D } from "./definitions";

export namespace Geometric {

  export function vector_add(p1: Vector2D, p2: Vector2D): Vector2D {
    return {
      x: p1.x + p2.x,
      y: p1.y + p2.y,
    };
  }

  export function vector_sub(p1: Vector2D, p2: Vector2D): Vector2D {
    return {
      x: p2.x - p1.x,
      y: p2.y - p1.y,
    };
  }

  export function formBoxByDimension(dimension: Dimension): Box {
    return {
      p1: { x: 0, y: 0 },
      p2: { x: dimension.width, y: dimension.height },
    };
  }

  export function formDimensionByBox(box: Box): Dimension {
    return {
      width: box.p2.x - box.p1.x,
      height: box.p2.y - box.p1.y,
    };
  }

  export function formBox(p1: Vector2D, p2: Vector2D): Box {
    return { p1, p2 };
  }

  export function formBoxByInset(box: Box, inset: Inset): Box {
    return {
      p1: { x: box.p1.x + inset.left, y: box.p1.y + inset.top },
      p2: { x: box.p2.x - inset.right, y: box.p2.y - inset.bottom },
    };
  }
};