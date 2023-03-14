import * as _ from "lodash"
import { sprintf } from "sprintf-js";
import { Algebra, Draw } from "./algebra";
import { ActualToKey, Box, Dimension, FeatureStats, Histogram, Inset, Item, PairedData, StudentRaw } from './definitions';
import { Geometric } from './geometric';
import { Stats } from "./stats";

const SvgParameter = {
  margin: {
    top: 50,
    bottom: 20,
    left: 100,
    right: 20,
  },
  // 縦軸詳細メモリの幅
  xFineScalerSize: 16,
  // 横軸詳細メモリの高さ
  yFineScalerSize: 8,
};

/**
 * 各種グラフ描画を行う
 */
export namespace Graph {

  /**
   * 指定領域内にヒストグラムを描画する
   * @param svg 
   * @param box 
   * @param histo 
   * @param option 
   */
  export function drawHistogram(svg: any, box: Box, histo: Histogram, option: {
  } = {}) {
    // [SVGを生成する]
    const figureInset: Inset = {
      top: 25,
      bottom: 85,
      left: 85,
      right: 25,
    };

    const HistogramInset: Inset = {
      top: 10,
      bottom: 10,
      left: 10,
      right: 10,
    };

    // [細かいサイズパラメータの定義]
    const figureOutBox = Geometric.formBoxByInset(box, figureInset);
    const figureInBox = Geometric.formBoxByInset(figureOutBox, HistogramInset);
    const { width: figureWidth } = Geometric.formDimensionByBox(figureOutBox);
    const figureInDimension = Geometric.formDimensionByBox(figureInBox);
    // ヒストグラムの最大階級の高さ
    const maxLevelHeight = figureInDimension.height;
    // ヒストグラム部分の幅
    const levelsWidth = figureInDimension.width;
    // 1カウント当たりの高さ
    // const heightPerCount = maxLevelHeight / histo.max_count;
    const heightPerCount = maxLevelHeight / histo.max_layered_count;

    // [キャプション]
    {
      const x_center = figureOutBox.p1.x + figureWidth / 2;
      const y_top = figureOutBox.p2.y;
      svg.text({
        x: x_center,
        y: y_top,
        "font-size": 32,
        "text-anchor": "middle",
        "font-weight": "bold",
        dy: 48,
      }, histo.feature);
    }

    // [枠線]
    Draw.box(svg, figureOutBox, {
      stroke: "#000",
      "stroke-width": "2",
      fill: "none",
    });

    // [目盛: 横軸]
    {
      _.range(histo.bins + 1).forEach(i => {
        const x = figureInBox.p1.x + levelsWidth / histo.bins * i;
        const y_top = figureInBox.p2.y;
        const y_bottom = y_top + SvgParameter.yFineScalerSize;
        svg.line({
          x1: x,
          y1: y_top,
          x2: x,
          y2: y_bottom,
          stroke: "#000",
          stroke_width: "1",
          fill: "none",
        });
      });
    }

    // [目盛: 縦軸]
    {
      const xScalerUnit = 25;
      const x_right = figureOutBox.p1.x;
      const x_left = x_right - SvgParameter.xFineScalerSize;
      for (let i = 0; i * xScalerUnit < histo.max_layered_count; ++i) {
        const y = figureOutBox.p2.y - i * xScalerUnit * heightPerCount;
        // 細かい目盛
        svg.line({
          x1: x_left,
          y1: y,
          x2: x_right,
          y2: y,
          stroke: "#000",
          stroke_width: "1",
          fill: "none",
        });
        // ラベル
        const x_text_right = x_left - 5;
        svg.text({
          x: x_text_right,
          y: y,
          "font-size": 16,
          "text-anchor": "end",
          dy: 6,
        }, `${i * xScalerUnit}`);
        // 点線
        const x_dots_left = figureOutBox.p1.x;
        const x_dots_right = figureOutBox.p2.x;
        svg.line({
          x1: x_dots_left,
          y1: y,
          x2: x_dots_right,
          y2: y,
          stroke: "#000",
          stroke_width: "1",
          fill: "none",
          "stroke-dasharray": [5,2],
        });
      }
    }

    // [ヒストグラム本体]
    histo.layered_counts.forEach((lc, i) => {
      _(lc).keys().sortBy(k => -lc[k]).each(actual => {
        const n = lc[actual];
        const y_bottom = figureOutBox.p2.y;
        const y_top = y_bottom - heightPerCount * n;
        const x_left = figureInBox.p1.x + levelsWidth / histo.bins * i;
        const x_right = figureInBox.p1.x + levelsWidth / histo.bins * (i + 1);
        svg.rect({
          x: x_left,
          y: y_top,
          width: x_right - x_left,
          height: y_bottom - y_top,
          fill: Stats.colorForActual(actual),
          "fill-opacity": 0.80,
        });
      });
    });
  }

  /**
   * 指定領域内に散布図を描画する
   * @param svg
   * @param box 
   * @param paired_data 
   * @param option 
   */
  export function drawScatter(svg: any, box: Box, paired_data: PairedData, option: {
    xLabel?: boolean;
    yLabel?: boolean;
  } = {}) {
    // [SVGを生成する]
    const figureInset = {
      top: 25,
      bottom: 85,
      left: 85,
      right: 25,
    };

    const ScatterInset = {
      top: 10,
      bottom: 10,
      left: 10,
      right: 10,
    };

    const scalerSize = 8;

    // [細かいサイズパラメータの定義]
    const figureOutBox = Geometric.formBoxByInset(box, figureInset);
    const figureInBox = Geometric.formBoxByInset(figureOutBox, ScatterInset);
    const { width: figureWidth } = Geometric.formDimensionByBox(figureOutBox);

    // データ座標系における, figureInBox にマップされる矩形領域
    // y が逆転していることに注意.
    const dataBox: Box = {
      p1: { x: paired_data.feature_x.p0, y: paired_data.feature_y.p100 },
      p2: { x: paired_data.feature_x.p100, y: paired_data.feature_y.p0 },
    };
    const affineDataToFigure = Algebra.affine_box_to_box(dataBox, figureInBox);
    const affineFigureToDate = Algebra.affine_box_to_box(figureInBox, dataBox);

    // [ラベル]
    {
      if (option.yLabel) {
        const x_center = box.p1.x + 15;
        const y_center = (figureOutBox.p1.y + figureOutBox.p2.y) / 2;
        svg.text({
          x: x_center,
          y: y_center,
          "font-size": 20,
          "text-anchor": "middle",
          "font-weight": "bold",
          transform: `rotate(90, ${x_center}, ${y_center})`,
        }, paired_data.feature_y.name);
      }
      if (option.xLabel) {
        const x_center = (figureOutBox.p1.x + figureOutBox.p2.x) / 2;
        const y_center = box.p2.y - 15;
        svg.text({
          x: x_center,
          y: y_center,
          "font-size": 20,
          "text-anchor": "middle",
          "font-weight": "bold",
        }, paired_data.feature_x.name);
      }
    }

    // [枠線]
    Draw.box(svg, figureOutBox, {
      stroke: "#000",
      "stroke-width": "2",
      fill: "none",
    });

    // [目盛:x]
    {
      const division = 5;
      const y_top = figureOutBox.p1.y;
      const y_mid = figureOutBox.p2.y;
      const y_bottom = figureOutBox.p2.y + scalerSize;
      _.range(division + 1).map(i => {
        // 目盛線
        const x = figureOutBox.p1.x + i / division * figureWidth;
        const v = Algebra.apply_affine(affineFigureToDate, { x, y: 0 }).x;
        svg.line({
          x1: x,
          y1: y_mid,
          x2: x,
          y2: y_bottom,
          stroke: "#000",
          fill: "none",
        });
        // 点線
        svg.line({
          x1: x,
          y1: y_top,
          x2: x,
          y2: y_mid,
          stroke: "#000",
          stroke_width: "1",
          fill: "none",
          "stroke-dasharray": [5,2],
        });
        const dy = 15;
        // ラベル
        svg.text({
          x,
          y: y_bottom,
          dy,
          "font-size": 10,
          "text-anchor": "middle",
          "font-weight": "bold",
        }, sprintf("%1.2f", v));
      });
    }

    // [目盛:y]
    {
      const division = 5;
      const x_top = figureOutBox.p2.x;
      const x_mid = figureOutBox.p1.x;
      const x_bottom = figureOutBox.p1.x - scalerSize;
      _.range(division + 1).map(i => {
        // 目盛線
        const y = figureOutBox.p1.y + i / division * figureWidth;
        const v = Algebra.apply_affine(affineFigureToDate, { x: 0, y }).y;
        svg.line({
          x1: x_mid,
          y1: y,
          x2: x_bottom,
          y2: y,
          stroke: "#000",
          fill: "none",
        });
        // 点線
        svg.line({
          x1: x_top,
          y1: y,
          x2: x_mid,
          y2: y,
          stroke: "#000",
          stroke_width: "1",
          fill: "none",
          "stroke-dasharray": [5,2],
        });
        const dx = -8;
        const dy = 5;
        // ラベル
        svg.text({
          x: x_bottom,
          dx,
          y,
          dy,
          "font-size": 10,
          "text-anchor": "end",
          "font-weight": "bold",
        }, sprintf("%1.2f", v));
      });
    }

    // [散布図本体]
    {
      paired_data.pairs.forEach((p, i) => {
        const q = Algebra.apply_affine(affineDataToFigure, p);
        svg.circle({
          r: p.r || 4,
          cx: q.x,
          cy: q.y,
          fill: p.fill || "#fff",
          stroke: "#000",
        });
        if (p.r && p.r > 5) {
          svg.text({
            x: q.x,
            y: q.y,
            "font-weight": "bold",
          }, `#${i}`);
        }
      });
    }
  }

  export function drawPairPlot(svg: any, dimension: Dimension, items: Item[], feature_stats: FeatureStats[]) {
    for (let i = 0; i < feature_stats.length; ++i) {
      for (let j = 0; j < feature_stats.length; ++j) {
        const stat_x = feature_stats[i];
        const stat_y = feature_stats[j];
        const subbox = {
          p1: { x: dimension.width * j, y: dimension.height * i },
          p2: { x: dimension.width * (j + 1), y: dimension.height * (i + 1) },
        };
        if (i === j) {
          const histo = Stats.students_to_bins(items, stat_x, 40);
          Graph.drawHistogram(svg, subbox, histo);
        } else {
          const show_x_label = i === feature_stats.length - 1;
          const show_y_label = j === 0;
          const paired_data = Stats.make_pair(stat_y, stat_x, items);
          Graph.drawScatter(svg, subbox, paired_data, {
            xLabel: show_x_label,
            yLabel: show_y_label,
          });
        }
      }
    }
  }
}
