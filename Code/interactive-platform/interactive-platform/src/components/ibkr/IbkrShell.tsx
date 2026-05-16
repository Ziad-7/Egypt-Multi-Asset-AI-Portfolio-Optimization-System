"use client";

const tabs = [
  { id: "dashboard" as const, label: "Dashboard" },
  { id: "signals" as const, label: "Signals & fusion" },
  { id: "models" as const, label: "Tactical ML" },
  { id: "diagnostics" as const, label: "Diagnostics" },
];

export function IbkrShell({
  activeTab,
  onTabChange,
  profileCount,
  apiConnected,
  filterBar,
  children,
}: {
  activeTab: (typeof tabs)[number]["id"];
  onTabChange: (id: (typeof tabs)[number]["id"]) => void;
  profileCount: number;
  apiConnected: boolean | null;
  filterBar: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-full bg-[#eef1f5] font-sans text-[#1a1a1a] antialiased">
      <header className="border-b border-[#d8dee6] bg-white">
        <div className="mx-auto flex max-w-[1400px] flex-col gap-3 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-md bg-[#0f4c9e] text-sm font-bold text-white">
              PA
            </div>
            <div>
              <h1 className="text-lg font-semibold leading-tight text-[#1a1a1a]">Portfolio Analyst</h1>
              <p className="text-xs text-[#5c6570]">Egypt multi-asset engine · strategic + tactical</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <span className="rounded border border-[#d8dee6] bg-[#f7f9fc] px-3 py-1.5 text-[#3d454d]">
              Profiles: <strong className="text-[#1a1a1a]">{profileCount}</strong>
            </span>
            <span
              className={`rounded px-3 py-1.5 text-xs font-medium ${
                apiConnected === null
                  ? "bg-[#f0f2f5] text-[#5c6570]"
                  : apiConnected
                    ? "bg-[#e6f4ea] text-[#0d7a3e]"
                    : "bg-[#fce8e6] text-[#c62828]"
              }`}
            >
              API {apiConnected === null ? "…" : apiConnected ? "connected" : "offline"}
            </span>
            <button type="button" className="rounded p-2 text-[#5c6570] hover:bg-[#f0f2f5]" aria-label="Settings">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
          </div>
        </div>
        <nav
          className="mx-auto max-w-[1400px] border-t border-[#eef1f5] px-2 sm:px-4"
          aria-label="Primary sections"
        >
          <div className="-mx-1 flex gap-0.5 overflow-x-auto pb-px">
            {tabs.map((t) => {
              const active = activeTab === t.id;
              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => onTabChange(t.id)}
                  className={`relative shrink-0 whitespace-nowrap px-3 py-3 text-sm font-medium transition sm:px-4 ${
                    active ? "text-[#0f4c9e]" : "text-[#5c6570] hover:text-[#1a1a1a]"
                  }`}
                >
                  {t.label}
                  {active ? (
                    <span className="absolute bottom-0 left-2 right-2 h-0.5 rounded-full bg-[#0f4c9e]" />
                  ) : null}
                </button>
              );
            })}
          </div>
        </nav>
        <div className="border-t border-[#d8dee6] bg-[#f7f9fc]">
          <div className="mx-auto max-w-[1400px] px-4 py-3">{filterBar}</div>
        </div>
      </header>
      <main className="mx-auto max-w-[1400px] px-4 py-6">{children}</main>
      <footer className="border-t border-[#d8dee6] bg-white py-4 text-center text-[11px] text-[#7a8490]">
        Research &amp; transparency only — not investment advice. Methodology per Egypt engine documentation.
      </footer>
    </div>
  );
}
