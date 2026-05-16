"use client";

import type { Dict } from "@/lib/intelligence";

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null;
}

export function TacticalModelEvaluation({ modelEvaluation }: { modelEvaluation: Dict }) {
  if (!modelEvaluation || !Object.keys(modelEvaluation).length) {
    return <p className="text-sm text-[#5c6570]">No classifier evaluation payload in this run.</p>;
  }

  const acc = modelEvaluation.accuracy;
  const f1 = modelEvaluation.f1_weighted;
  const prec = modelEvaluation.precision_weighted;
  const rec = modelEvaluation.recall_weighted;
  const foldStd = modelEvaluation.fold_f1_std_max;
  const disagree = modelEvaluation.ensemble_disagreement;
  const calib = modelEvaluation.confidence_calibration_error;
  const status = modelEvaluation.status;
  const holdout = modelEvaluation.holdout_size;
  const cm = modelEvaluation.confusion_matrix;

  const statusStr = status != null ? String(status) : "";
  const statusTone =
    statusStr === "ok" || statusStr === "success"
      ? "border-[#bbf7d0] bg-[#f0fdf4] text-[#166534]"
      : statusStr.includes("insufficient") || statusStr.includes("imbalance")
        ? "border-[#fde68a] bg-[#fffbeb] text-[#92400e]"
        : "border-[#e8ecf1] bg-[#f7f9fc] text-[#3d454d]";

  return (
    <div className="space-y-5 text-sm">
      <div className="rounded-lg border border-[#dbeafe] bg-[#f0f7ff] px-4 py-3 text-xs text-[#1e3a5f]">
        <p className="font-semibold text-[#0f4c9e]">Where ML shows up</p>
        <p className="mt-1.5 leading-relaxed">
          The <strong>tactical layer</strong> blends logistic regression, SVM, and (when available) gradient boosting on
          lagged returns and technical features. Holdout metrics below inform classifier confidence; regime and momentum
          then shape the discrete signal you see under <strong>Signals &amp; fusion</strong>. The{" "}
          <strong>strategic</strong> layer is mean–variance and simulation — not this classifier stack.
        </p>
      </div>

      {status != null ? (
        <div className={`flex flex-wrap items-center gap-2 rounded-lg border px-3 py-2 text-xs ${statusTone}`}>
          <span className="font-semibold uppercase tracking-wide">Evaluation</span>
          <span className="rounded-full bg-white/70 px-2 py-0.5 font-mono text-[11px]">{statusStr}</span>
          {holdout != null ? (
            <span className="text-[#5c6570]">
              Holdout: <span className="tabular-nums font-medium text-[#1a1a1a]">{String(holdout)}</span>
            </span>
          ) : null}
        </div>
      ) : null}

      <div>
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-[#5c6570]">Headline metrics</p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Accuracy", acc],
          ["F1 (weighted)", f1],
          ["Precision (w.)", prec],
          ["Recall (w.)", rec],
        ].map(([label, v]) => (
          <div key={String(label)} className="rounded-lg border border-[#e8ecf1] bg-white px-3 py-2.5 shadow-sm">
            <p className="text-[11px] text-[#5c6570]">{String(label)}</p>
            <p className="text-lg font-semibold tabular-nums text-[#1a1a1a]">
              {typeof v === "number" ? (v <= 1 ? `${(v * 100).toFixed(1)}%` : v.toFixed(4)) : "—"}
            </p>
          </div>
        ))}
        </div>
      </div>

      <div>
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-[#5c6570]">Stability &amp; calibration</p>
        <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-[#e8ecf1] bg-white px-3 py-2 shadow-sm">
          <p className="text-[11px] text-[#5c6570]">Fold F1 stability (max std)</p>
          <p className="text-base font-semibold tabular-nums text-[#1a1a1a]">
            {typeof foldStd === "number" ? foldStd.toFixed(4) : "—"}
          </p>
        </div>
        <div className="rounded-lg border border-[#e8ecf1] bg-white px-3 py-2 shadow-sm">
          <p className="text-[11px] text-[#5c6570]">Ensemble disagreement</p>
          <p className="text-base font-semibold tabular-nums text-[#1a1a1a]">
            {typeof disagree === "number" ? disagree.toFixed(4) : "—"}
          </p>
        </div>
        <div className="rounded-lg border border-[#e8ecf1] bg-white px-3 py-2 shadow-sm">
          <p className="text-[11px] text-[#5c6570]">Calibration error</p>
          <p className="text-base font-semibold tabular-nums text-[#1a1a1a]">
            {typeof calib === "number" ? calib.toFixed(4) : "—"}
          </p>
        </div>
        </div>
      </div>

      {Array.isArray(cm) && cm.length > 0 ? (
        <div>
          <p className="mb-2 text-xs font-semibold text-[#5c6570]">Confusion matrix (holdout)</p>
          <div className="overflow-x-auto rounded-lg border border-[#e8ecf1]">
            <table className="min-w-full border-collapse text-xs">
              <tbody>
                {cm.map((row: unknown, i: number) => (
                  <tr key={i} className="border-t border-[#f0f2f5]">
                    {Array.isArray(row) ? (
                      row.map((cell: unknown, j: number) => (
                        <td
                          key={j}
                          className="min-w-[2.75rem] px-2 py-2 text-center tabular-nums text-[#1a1a1a] sm:px-3"
                        >
                          {typeof cell === "number" ? cell : String(cell)}
                        </td>
                      ))
                    ) : (
                      <td className="px-3 py-2 break-words">{String(row)}</td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {isRecord(modelEvaluation.per_class) ? (
        <details className="rounded border border-[#e8ecf1] bg-[#f7f9fc] px-3 py-2 text-xs">
          <summary className="cursor-pointer font-medium text-[#3d454d]">Per-class metrics</summary>
          <pre className="mt-2 max-h-48 overflow-auto text-[11px] text-[#1a1a1a]">
            {JSON.stringify(modelEvaluation.per_class, null, 2)}
          </pre>
        </details>
      ) : null}
    </div>
  );
}
