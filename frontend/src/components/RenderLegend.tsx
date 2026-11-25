import React from 'react';

interface RenderLegendProps {
  activeMode: string;
  heatmapBounds: { min: number; max: number } | null;
}

export const RenderLegend: React.FC<RenderLegendProps> = ({ activeMode, heatmapBounds }) => {
  if (activeMode === 'LAND COVER') {
    return (
      <div className="bg-[#1A1816]/90 backdrop-blur-md border border-[#5A564C]/50 rounded-sm p-4 text-[#D9D2C2] font-mono shadow-2xl pointer-events-auto">
        <div className="text-[10px] tracking-[0.2em] text-[#8B8478] mb-3 uppercase flex items-center gap-2">
          LAND COVER <span className="text-[#E0433C]">//</span>
        </div>
        <div className="flex flex-col gap-2 text-xs tracking-wider">
          <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full bg-[#22C55E] shadow-[0_0_8px_rgba(34,197,94,0.4)]" />
            <span>VEGETATION</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full bg-[#38BDF8] shadow-[0_0_8px_rgba(56,189,248,0.4)]" />
            <span>WATER</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full bg-[#A8A29E] shadow-[0_0_8px_rgba(168,162,158,0.4)]" />
            <span>BUILT-UP</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full bg-[#292524] shadow-[0_0_8px_rgba(41,37,36,0.4)] border border-[#5A564C]/30" />
            <span>SHADOW</span>
          </div>
        </div>
      </div>
    );
  }

  if (activeMode === 'HEATMAP') {
    const minVal = heatmapBounds ? heatmapBounds.min.toFixed(1) : '0.0';
    const maxVal = heatmapBounds ? heatmapBounds.max.toFixed(1) : '90.0';

    return (
      <div className="bg-[#1A1816]/90 backdrop-blur-md border border-[#5A564C]/50 rounded-sm p-4 text-[#D9D2C2] font-mono shadow-2xl pointer-events-auto min-w-[240px]">
        <div className="text-[10px] tracking-[0.2em] text-[#8B8478] mb-3 uppercase flex items-center gap-2">
          SLOPE HEATMAP <span className="text-[#E0433C]">//</span>
        </div>
        <div className="flex justify-between text-xs tracking-wider mb-2">
          <span>{minVal}°</span>
          <span>{maxVal}°</span>
        </div>
        <div className="w-full h-3 rounded-sm mb-2" style={{
          background: 'linear-gradient(to right, #2563EB, #06B6D4, #22C55E, #FACC15, #F97316, #EF4444)'
        }} />
        <div className="flex justify-between text-[9px] text-[#8B8478] tracking-widest uppercase">
          <span>LOW</span>
          <span>HIGH</span>
        </div>
      </div>
    );
  }

  return null;
};
