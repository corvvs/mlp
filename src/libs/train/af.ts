import type {
  ActivatationFunctionActual,
  ActivationFunction,
  ActivationFunctionSingleArgument,
  AFLeakyReLU,
} from "../../types/af.js";

export function parseActivationFunction(
  str: string
): ActivationFunctionSingleArgument {
  const parts = str.split(",");
  const method = parts[0];
  switch (method.toLowerCase()) {
    case "linear":
      return { method: "linear" };
    case "sigmoid":
      return { method: "sigmoid" };
    case "tanh":
      return { method: "tanh" };
    case "relu":
      return { method: "ReLU" };
    case "leakyrelu": {
      const f: AFLeakyReLU = {
        method: "LeakyReLU",
        alpha: 0.01,
      };
      if (parts.length >= 2) {
        const alpha = parseFloat(parts[1]);
        if (isNaN(alpha)) {
          throw new Error(`不正なLeakyReLUパラメータ: ${parts[1]}`);
        }
        f.alpha = alpha;
      }
      return f;
    }
    default:
      throw new Error(`未知の活性化関数: ${method}`);
  }
}

// 活性化関数の実装を取得する; ただし, softmax はここでは扱わない
export function getActivationFunctionActual(
  af: ActivationFunction
): ActivatationFunctionActual {
  switch (af.method) {
    case "softmax":
      throw new Error("Softmax の実装はここでは扱いません");
    case "linear":
      return (x: number) => x;
    case "sigmoid":
      return (x: number) => 1 / (1 + Math.exp(-x));
    case "tanh":
      return (x: number) => Math.tanh(x);
    case "ReLU":
      return (x: number) => Math.max(0, x);
    case "LeakyReLU":
      return (x: number) => (x >= 0 ? x : af.alpha * x);
    default:
      throw new Error(`Unknown activation function: ${(af as any).method}`);
  }
}

// 活性化関数の導関数の実装を取得する; ただし, softmax はここでは扱わない
export function getDerivativeActivationFunctionActual(
  af: ActivationFunction
): ActivatationFunctionActual {
  switch (af.method) {
    case "softmax":
      throw new Error("Softmax の実装はここでは扱いません");
    case "linear":
      return (_x: number) => 1;
    case "sigmoid":
      return (x: number) => {
        const fx = 1 / (1 + Math.exp(-x));
        return fx * (1 - fx);
      };
    case "tanh":
      return (x: number) => {
        const fx = Math.tanh(x);
        return 1 - fx * fx;
      };
    case "ReLU":
      return (x: number) => (x >= 0 ? 1 : 0);
    case "LeakyReLU":
      return (x: number) => (x >= 0 ? 1 : af.alpha);
    default:
      throw new Error(`Unknown activation function: ${(af as any).method}`);
  }
}

export function softmax(xs: number[]): number[] {
  const maxX = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - maxX));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((exp) => exp / sumExps);
}

// softmax の導関数は不要...なはず
