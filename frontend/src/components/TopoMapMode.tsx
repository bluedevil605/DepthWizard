import React from 'react';
import { Compass } from 'lucide-react';

export const TopoMapMode: React.FC<{ meta?: any }> = ({ meta }) => {
  const origin = meta?.elevation_origin ?? 0;
  const minElev = meta?.min_elevation ?? 0;
  const maxElev = meta?.max_elevation ?? 0;
  
  const absMin = origin + minElev;
  const absMax = origin + maxElev;
  const span = Math.max(1, absMax - absMin);
  
  // Calculate a reasonable contour interval based on the span
  const contourInterval = (span / 14).toFixed(1);

  return (
    <section id="topo-map" className="relative py-24 font-mono">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <div className="inline-flex items-center gap-2 text-[10px] text-[#8B8478] tracking-[0.25em] uppercase mb-3">
            <span className="w-1.5 h-1.5 bg-[#E0433C]" />
            <span>ORTHOGRAPHIC RECON MAP</span>
          </div>
          <h2 className="font-heading text-3xl sm:text-4xl font-bold text-[#D9D2C2] uppercase tracking-tight">
            Topographic Field Map
          </h2>
          <p className="mt-3 text-xs text-[#B7AF9F]">
            High-precision orthographic contour map with coordinate grid & tactical waypoints.
          </p>
        </div>

        {/* Topo Map metadata overlay — contours render in the unified background canvas */}
        <div className="max-w-3xl mx-auto">
          <div className="relative p-8 bg-[#1A1816]/75 backdrop-blur-sm border border-[#5A564C]/70 min-h-[400px] flex items-center justify-center">
            {/* Corner metadata */}
            <div className="absolute top-4 left-4 text-[10px] text-[#8B8478] space-y-1">
              <div>GRID: 37.412° N / 76.219° E</div>
              <div>DATUM: WGS-84 STONE SLATE</div>
              <div>CONTOUR INTERVAL: {contourInterval}m</div>
            </div>

            <div className="absolute top-4 right-4 text-[10px] text-[#8B8478] text-right space-y-1">
              <div>ELEVATION RANGE: {absMin.toFixed(0)}m - {absMax.toFixed(0)}m</div>
              <div>BANDS: 14 ISO-LEVELS</div>
              <div>METHOD: MARCHING SQUARES</div>
            </div>

            {/* Center indicator */}
            <div className="text-center space-y-4">
              <div className="w-16 h-16 border border-[#5A564C] bg-[#211E1B]/80 flex items-center justify-center mx-auto shadow-[0_0_20px_rgba(224,67,60,0.15)]">
                <Compass className="w-8 h-8 text-[#E0433C]" />
              </div>
              <div className="text-xs text-[#D9D2C2] tracking-widest uppercase">
                TOP-DOWN CONTOUR VIEW ACTIVE IN BACKGROUND
              </div>
              <div className="text-[10px] text-[#8B8478]">
                Camera transitions to orthographic top-down as you scroll through this section
              </div>
            </div>

            {/* Bottom metadata */}
            <div className="absolute bottom-4 right-4 flex items-center gap-2 text-[10px] text-[#B7AF9F] bg-[#211E1B] px-3 py-1.5 border border-[#5A564C]">
              <Compass className="w-3.5 h-3.5 text-[#E0433C]" />
              <span>NORTH UP • LIVE CONTOUR RENDER</span>
            </div>

            <div className="absolute bottom-4 left-4 flex items-center gap-2 text-[10px] text-[#8B8478]">
              <div className="w-2 h-2 bg-[#E0433C] rounded-full animate-pulse shadow-[0_0_8px_rgba(224,67,60,0.5)]" />
              <span>MARCHING SQUARES ENGINE ACTIVE</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
