"use client";

export function DashboardCard({
  title,
  subtitle,
  right,
  children,
  className = "",
  id,
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <div
      id={id}
      className={`flex flex-col overflow-hidden rounded-lg border border-[#d8dee6] bg-white shadow-[0_1px_3px_rgba(0,0,0,0.06)] ${id ? "scroll-mt-28" : ""} ${className}`}
    >
      <div className="flex items-start justify-between gap-3 border-b border-[#e8ecf1] px-4 py-3">
        <div className="min-w-0 flex-1">
          <h3 className="text-[15px] font-semibold leading-snug text-[#1a1a1a]">{title}</h3>
          {subtitle ? <p className="mt-0.5 text-xs leading-relaxed text-[#5c6570]">{subtitle}</p> : null}
        </div>
        {right ? <div className="flex shrink-0 items-center gap-2">{right}</div> : null}
      </div>
      <div className="flex-1 p-4">{children}</div>
    </div>
  );
}
