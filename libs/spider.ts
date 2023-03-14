import * as _ from "lodash"
import { Dimension } from "./definitions";

type Element = {
  element: string;
  params: any,
  content?: string;
};

export class Spider {

  width: number;
  height: number;
  elements: Element[];

  constructor(dimension: Dimension) {
    this.width = dimension.width;
    this.height = dimension.height;
    this.elements = [];
  }

  text(params: any, content: string) {
    this.elements.push({ params, element: "text", content });
  }

  line(params: any) {
    this.elements.push({ params, element: "line" });
  }

  rect(params: any) {
    this.elements.push({ params, element: "rect" });

  }

  circle(params: any) {
    this.elements.push({ params, element: "circle" });
  }

  render() {
    const start = `<svg height="${this.height}" width="${this.width}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">`;
    const end = "</svg>";
    return [
      start,
      ...this.elements.map(s => {
        switch (s.element) {
          case "text": {
            return `<${s.element} ${_.map(s.params, (v, k) => `${k}="${v}"`).join(" ")}>${s.content}</${s.element}>`
          }
          default: {
            return `<${s.element} ${_.map(s.params, (v, k) => `${k}="${v}"`).join(" ")} />`
          }
        }
      }),
      end,
    ].join("\n");
  }
}
