"use client";

import { ASSET_COLORS } from "@/lib/intelligence";

export function UniverseStrip({ assets }: { assets: string[] }) {
  if (!assets.length) return null;
  return (
    <div className="flex flex-wrap gap-2">
      {assets.map((a) => (
        <span
          key={a}
          className="inline-flex items-center gap-2 rounded-full border border-[#d8dee6] bg-[#f7f9fc] px-3 py-1 text-xs font-medium text-[#1a1a1a]"
        >
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: ASSET_COLORS[a] ?? "#64748b" }} />
          {a}
        </span>
      ))}
    </div>
  );
}
