import Link from "next/link";

const nav = [
  { href: "#overview", label: "Overview" },
  { href: "#layers", label: "Two layers" },
  { href: "#strategic", label: "Strategic" },
  { href: "#tactical", label: "Tactical" },
  { href: "#fusion", label: "Fusion" },
  { href: "#diagnostics", label: "Diagnostics" },
  { href: "#backtest", label: "Backtest" },
];

export function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-full flex flex-col bg-[#070b12] text-slate-100">
      <header className="sticky top-0 z-30 border-b border-white/10 bg-[#070b12]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.2em] text-teal-300/90">
              Interactive intelligence
            </p>
            <h1 className="text-lg font-semibold text-white sm:text-xl">
              Egypt multi-asset portfolio engine
            </h1>
          </div>
          <nav className="flex flex-wrap gap-2 text-sm">
            {nav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-slate-200 transition hover:border-teal-400/40 hover:bg-teal-500/10 hover:text-white"
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-10">{children}</main>
      <footer className="border-t border-white/10 py-6 text-center text-xs text-slate-500">
        Data and models are provided for research and operational transparency. Not investment advice.
      </footer>
    </div>
  );
}
