export type Dict = Record<string, unknown>;

export type IntelligenceReport = {
  metadata: Dict;
  strategic_profiles: Record<string, StrategicProfile>;
  tactical_signal: TacticalSignal;
  layer_interaction: Record<string, LayerFusion>;
  strategic_diagnostics: StrategicDiagnostics;
  backtest: BacktestPayload | { error?: string };
  persona_guidance?: Record<string, string>;
};

export type StrategicProfile = {
  profile: string;
  weights: Record<string, number>;
  expected_return: number;
  expected_volatility: number;
  sharpe: number;
  confidence: number;
  rebalancing_note: string;
  rebalance_frequency: string;
  next_rebalance_date: string;
  risk_contributions: Record<string, number>;
  diversification_ratio: number;
  risk_metrics: Dict;
  monte_carlo: Dict;
  optimization_method: string;
};

export type TacticalSignal = {
  signal: number;
  confidence: number;
  regime: string;
  risk_off_alert: boolean;
  rebalance_window_active: boolean;
  rationale: string[];
  suggested_position_size: number;
  target_volatility: number;
  realized_volatility: number;
  model_evaluation: Dict;
  /** When present, Signals tab can re-apply alignment for custom mixes (engine ≥ replay snapshot). */
  replay_snapshot?: Dict;
};

export type LayerFusion = {
  strategic_bias: string;
  tactical_signal: number;
  confidence: number;
  action: string;
  note: string;
  suggested_tilt: Record<string, number>;
};

export type CloudBundle = {
  returns: number[];
  volatilities: number[];
  sharpes?: number[];
};

export type StrategicDiagnostics = {
  assets?: string[];
  correlation_matrix?: Record<string, Record<string, number>>;
  frontier_line?: CloudBundle;
  random_cloud?: CloudBundle;
  reference_portfolios?: Record<string, RefPortfolio>;
  annual_mean_returns?: Record<string, number>;
  annual_volatility?: Record<string, number>;
  /** Row/column order matches `assets` */
  annual_covariance?: number[][];
};

export type RefPortfolio = {
  weights: Record<string, number>;
  expected_return: number;
  volatility: number;
  sharpe: number;
  objective?: string;
};

export type BacktestPayload = {
  dates?: string[];
  evaluation_window?: string[] | { start?: string; end?: string };
  daily_dates?: string[];
  cumulative_returns?: Record<string, number[]>;
  daily_cumulative_returns?: Record<string, number[]>;
  daily_drawdowns?: Record<string, number[]>;
  /** Column order for `daily_asset_returns` rows (same length as `daily_dates`). */
  daily_asset_return_assets?: string[];
  daily_asset_returns?: Record<string, number[]>;
  rebalance_dates?: string[];
  transaction_cost_bps?: number;
  benchmarks?: Dict;
  portfolios?: Dict;
  walk_forward_oos?: Dict;
};

export function isBacktestPayload(b: unknown): b is BacktestPayload {
  return typeof b === "object" && b !== null && !("error" in b && Object.keys(b).length === 1);
}

export function sampleIndices(length: number, maxPoints: number): number[] {
  if (length <= maxPoints) return Array.from({ length }, (_, i) => i);
  const step = length / maxPoints;
  const out: number[] = [];
  for (let i = 0; i < maxPoints; i++) out.push(Math.min(length - 1, Math.floor(i * step)));
  return out;
}

/** Client replay of current slider weights on `daily_asset_returns` (performance + drawdown charts). */
export const WEIGHT_REPLAY_SERIES_KEY = "Replay (your weights)";

export const ASSET_COLORS: Record<string, string> = {
  EGX30: "#2563eb",
  EGX100: "#06b6d4",
  Gold: "#f59e0b",
  TBills: "#22c55e",
  EgyptiansRealEstateFund: "#dc2626",
  SP500: "#7c3aed",
  EGX_TBills_60_40: "#64748b",
  Optimal: "#0f4c9e",
  [WEIGHT_REPLAY_SERIES_KEY]: "#9333ea",
  "Custom mix": "#9333ea",
};
