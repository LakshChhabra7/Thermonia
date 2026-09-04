// Multiple linear regression (ordinary least squares) solved via the normal
// equations with Gauss-Jordan elimination and ridge regularisation for stability.
// Mirrors what a Python script using numpy/sklearn would do, in pure TypeScript.

export type Matrix = number[][];

function transpose(a: Matrix): Matrix {
  const rows = a.length;
  const cols = a[0]!.length;
  const out: Matrix = Array.from({ length: cols }, () => new Array<number>(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j]![i] = a[i]![j]!;
  }
  return out;
}

function multiply(a: Matrix, b: Matrix): Matrix {
  const n = a.length;
  const k = b.length;
  const m = b[0]!.length;
  const out: Matrix = Array.from({ length: n }, () => new Array<number>(m).fill(0));
  for (let i = 0; i < n; i++) {
    const ai = a[i]!;
    const oi = out[i]!;
    for (let p = 0; p < k; p++) {
      const av = ai[p]!;
      if (av === 0) continue;
      const bp = b[p]!;
      for (let j = 0; j < m; j++) oi[j] = oi[j]! + av * bp[j]!;
    }
  }
  return out;
}

/** Solves A x = B for x, where B may have several columns (multi-output). */
function solve(A: Matrix, B: Matrix): Matrix {
  const n = A.length;
  const m = B[0]!.length;
  const M: Matrix = A.map((row, i) => [...row, ...B[i]!]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r]![col]!) > Math.abs(M[pivot]![col]!)) pivot = r;
    }
    if (Math.abs(M[pivot]![col]!) < 1e-12) continue;
    const tmp = M[col]!;
    M[col] = M[pivot]!;
    M[pivot] = tmp;
    const pivotRow = M[col]!;
    const d = pivotRow[col]!;
    for (let j = col; j < n + m; j++) pivotRow[j] = pivotRow[j]! / d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const row = M[r]!;
      const f = row[col]!;
      if (f === 0) continue;
      for (let j = col; j < n + m; j++) row[j] = row[j]! - f * pivotRow[j]!;
    }
  }

  return M.map((row) => row.slice(n));
}

export interface FittedModel {
  /** coefficients[featureIndex + 1][targetIndex]; row 0 is the intercept. */
  coefficients: Matrix;
  featureNames: string[];
  targetNames: string[];
  /** Coefficient of determination per target. */
  r2: number[];
  sampleCount: number;
}

export function fit(
  X: Matrix,
  Y: Matrix,
  featureNames: string[],
  targetNames: string[],
  lambda = 1e-6,
): FittedModel {
  const Xd = X.map((row) => [1, ...row]);
  const Xt = transpose(Xd);
  const XtX = multiply(Xt, Xd);
  for (let i = 1; i < XtX.length; i++) XtX[i]![i] = XtX[i]![i]! + lambda;
  const XtY = multiply(Xt, Y);
  const coefficients = solve(XtX, XtY);

  const predictions = multiply(Xd, coefficients);
  const r2 = targetNames.map((_, t) => {
    const mean = Y.reduce((s, row) => s + row[t]!, 0) / Y.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < Y.length; i++) {
      ssRes += (Y[i]![t]! - predictions[i]![t]!) ** 2;
      ssTot += (Y[i]![t]! - mean) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  });

  return { coefficients, featureNames, targetNames, r2, sampleCount: X.length };
}

export function predict(model: FittedModel, features: number[]): number[] {
  const row = [1, ...features];
  return model.targetNames.map((_, t) =>
    row.reduce((sum, v, i) => sum + v * model.coefficients[i]![t]!, 0),
  );
}
