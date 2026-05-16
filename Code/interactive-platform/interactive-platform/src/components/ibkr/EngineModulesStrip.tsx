"use client";

/** Maps the six engine capabilities to tabs + optional scroll anchors in `PlatformDashboard`. */
export type PlatformTabId = "dashboard" | "signals" | "models" | "diagnostics";

const MODULES: {
  id: string;
  title: string;
  blurb: string;
  tab: PlatformTabId;
  anchor?: string;
}[] = [
  {
    id: "data",
    title: "Market data",
    blurb: "Universe, panel window, quality",
    tab: "dashboard",
    anchor: "module-data",
  },
  {
    id: "strategic",
    title: "Strategic",
    blurb: "Allocation, frontier inputs, mix",
    tab: "dashboard",
    anchor: "module-strategic",
  },
  {
    id: "tactical",
    title: "Tactical",
    blurb: "Signal, regime, confidence",
    tab: "signals",
    anchor: "module-tactical",
  },
  {
    id: "fusion",
    title: "Fusion",
    blurb: "Bias, tilt, inflation overlay",
    tab: "signals",
    anchor: "module-fusion",
  },
  {
    id: "backtest",
    title: "Backtest",
    blurb: "Performance, drawdown, replay",
    tab: "dashboard",
    anchor: "module-backtest",
  },
  {
    id: "ml",
    title: "ML & validation",
    blurb: "Classifiers, holdout metrics",
    tab: "models",
    anchor: "module-ml",
  },
];

export function EngineModulesStrip({
  activeTab,
  onSelect,
  disabled,
}: {
  activeTab: PlatformTabId;
  onSelect: (tab: PlatformTabId, anchor?: string) => void;
  disabled: boolean;
}) {
  return (
    <section
      className="rounded-xl border border-[#d8dee6] bg-white p-4 shadow-[0_1px_3px_rgba(0,0,0,0.04)]"
      aria-label="Engine modules"
    >
      <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wide text-[#5c6570]">Six engine modules</h2>
          <p className="mt-0.5 text-[11px] text-[#7a8490]">
            Jump to each layer of the Egypt multi-asset pipeline. Active tab is highlighted.
          </p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {MODULES.map((m) => {
          const isHere = activeTab === m.tab;
          return (
            <button
              key={m.id}
              type="button"
              disabled={disabled}
              onClick={() => {
                onSelect(m.tab, m.anchor);
              }}
              className={`flex min-h-[4.5rem] flex-col rounded-lg border px-3 py-2.5 text-left transition ${
                disabled
                  ? "cursor-not-allowed border-[#eef1f5] bg-[#f7f9fc] opacity-60"
                  : isHere
                    ? "border-[#0f4c9e] bg-[#f0f7ff] shadow-sm"
                    : "border-[#e8ecf1] bg-[#fafbfc] hover:border-[#0f4c9e]/35 hover:bg-white"
              }`}
            >
              <span className={`text-[11px] font-semibold leading-tight ${isHere ? "text-[#0f4c9e]" : "text-[#1a1a1a]"}`}>
                {m.title}
              </span>
              <span className="mt-1 line-clamp-2 text-[10px] leading-snug text-[#5c6570]">{m.blurb}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
