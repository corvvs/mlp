// Grid Search設定の型定義

/**
 * グリッドサーチの設定ファイルの型
 */
export interface GridSearchConfig {
  /**
   * ベースとなるオプション
   * プレースホルダーを含むことができる
   */
  baseOptions: Record<string, string | number | boolean>;

  /**
   * グリッドパラメータ
   * キーはオプション名、値は試行する値のリスト
   */
  grid: Record<string, (string | number | boolean)[]>;

  /**
   * 出力設定
   */
  output: {
    /**
     * モデルファイルのプレフィックス
     * 例: "grid_model_" → grid_model_0.json, grid_model_1.json, ...
     */
    modelPrefix: string;

    /**
     * multiplotで生成するSVGファイルのパス
     */
    multiplotFile: string;

    /**
     * multiplotで表示するメトリクス
     */
    metrics: string[];

    /**
     * 実行結果のメタデータを保存するJSONファイル
     */
    metadataFile?: string;
  };

  /**
   * 並行実行の最大数
   * デフォルトはCPU数
   */
  maxParallel?: number;
}

/**
 * グリッドの1つの組み合わせ
 */
export interface GridCombination {
  /**
   * 組み合わせのインデックス
   */
  index: number;

  /**
   * パラメータの組み合わせ
   */
  params: Record<string, string | number | boolean>;

  /**
   * 出力モデルファイルのパス
   */
  modelFile: string;
}

/**
 * グリッドサーチの実行結果メタデータ
 */
export interface GridSearchMetadata {
  /**
   * 設定ファイルのパス
   */
  configFile: string;

  /**
   * 実行開始時刻
   */
  startTime: string;

  /**
   * 実行終了時刻
   */
  endTime?: string;

  /**
   * 各組み合わせの結果
   */
  results: GridCombinationResult[];

  /**
   * multiplotで生成されたSVGファイル
   */
  multiplotFile: string;
}

/**
 * 1つの組み合わせの実行結果
 */
export interface GridCombinationResult {
  /**
   * 組み合わせのインデックス
   */
  index: number;

  /**
   * パラメータの組み合わせ
   */
  params: Record<string, string | number | boolean>;

  /**
   * 出力モデルファイル
   */
  modelFile: string;

  /**
   * 実行成功したかどうか
   */
  success: boolean;

  /**
   * エラーメッセージ（失敗時）
   */
  error?: string;

  /**
   * 実行時間（秒）
   */
  executionTime?: number;
}
