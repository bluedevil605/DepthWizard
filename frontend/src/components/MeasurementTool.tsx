import React, { useState } from 'react';
import { Crosshair, Ruler, ArrowUpRight, RefreshCcw } from 'lucide-react';

export const MeasurementTool: React.FC = () => {
  const [pointA, setPointA] = useState({ x: -2.4, y: 0.8, z: -1.2 });
  const [pointB, setPointB] = useState({ x: 3.1, y: 3.4, z: 2.8 });

  // Calculate distance, elevation diff, and slope
  const dx = pointB.x - pointA.x;
  const dy = pointB.y - pointA.y;
  const dz = pointB.z - pointA.z;

  const distanceHorizontal = Math.sqrt(dx * dx + dz * dz);
  const distance3D = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const elevationDiff = dy;
  const slopeAngle = Math.atan2(Math.abs(dy), distanceHorizontal) * (180 / Math.PI);

  const handleResetPoints = () => {
    setPointA({
      x: parseFloat(((Math.random() - 0.5) * 6).toFixed(1)),
      y: parseFloat((Math.random() * 2).toFixed(1)),
      z: parseFloat(((Math.random() - 0.5) * 6).toFixed(1))
    });
    setPointB({
      x: parseFloat(((Math.random() - 0.5) * 8).toFixed(1)),
      y: parseFloat((Math.random() * 4 + 1).toFixed(1)),
      z: parseFloat(((Math.random() - 0.5) * 8).toFixed(1))
    });
  };

  return (
    <section id="measurement-tool" className="relative py-16 bg-[#211E1B]/75 backdrop-blur-md border-t border-[#5A564C]/50 font-mono">
      <div className="max-w-6xl mx-auto px-4">
        <div className="mono-panel p-6 border border-[#5A564C] relative bg-[#1A1816]/90 backdrop-blur-md">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6 border-b border-[#5A564C] pb-6 mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 border border-[#5A564C] bg-[#211E1B] flex items-center justify-center text-[#E0433C]">
                <Ruler className="w-5 h-5" />
              </div>
              <div>
                <div className="text-xs text-[#8B8478] tracking-widest uppercase">TACTICAL MEASUREMENT TOOL</div>
                <h3 className="font-heading text-lg font-bold text-[#D9D2C2] uppercase">POINT-TO-POINT SURVEY READOUT</h3>
              </div>
            </div>

            <button
              onClick={handleResetPoints}
              className="px-4 py-2 bg-[#211E1B] border border-[#5A564C] text-xs text-[#B7AF9F] hover:border-[#8B8478] hover:text-[#D9D2C2] transition-all flex items-center gap-2"
            >
              <RefreshCcw className="w-3.5 h-3.5 text-[#E0433C]" />
              <span>[ SAMPLE NEW FIELD POINTS ]</span>
            </button>
          </div>

          {/* Point Metrics Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-left">
            <div className="bg-[#211E1B]/90 p-4 border border-[#5A564C]">
              <div className="text-[10px] text-[#8B8478] uppercase tracking-widest flex items-center justify-between">
                <span>3D SPATIAL DISTANCE</span>
                <Crosshair className="w-3 h-3 text-[#E0433C]" />
              </div>
              <div className="text-2xl font-bold text-[#D9D2C2] mt-2">
                {(distance3D * 120).toFixed(1)}m
              </div>
              <div className="text-[10px] text-[#8B8478] mt-1">
                HORIZONTAL: {(distanceHorizontal * 120).toFixed(1)}m
              </div>
            </div>

            <div className="bg-[#211E1B]/90 p-4 border border-[#5A564C]">
              <div className="text-[10px] text-[#8B8478] uppercase tracking-widest flex items-center justify-between">
                <span>ELEVATION DIFFERENCE (Δ)</span>
                <ArrowUpRight className="w-3 h-3 text-[#E0433C]" />
              </div>
              <div className="text-2xl font-bold text-[#D9D2C2] mt-2">
                {elevationDiff >= 0 ? `+${(elevationDiff * 120).toFixed(1)}m` : `${(elevationDiff * 120).toFixed(1)}m`}
              </div>
              <div className="text-[10px] text-[#8B8478] mt-1">
                VERTICAL DELTA
              </div>
            </div>

            <div className="bg-[#211E1B]/90 p-4 border border-[#5A564C]">
              <div className="text-[10px] text-[#8B8478] uppercase tracking-widest flex items-center justify-between">
                <span>SLOPE GRADIENT ANGLE</span>
                <span className="text-xs text-[#E0433C]">∠</span>
              </div>
              <div className="text-2xl font-bold text-[#D9D2C2] mt-2">
                {slopeAngle.toFixed(1)}°
              </div>
              <div className="text-[10px] text-[#8B8478] mt-1">
                INCLINE GRADIENT
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
