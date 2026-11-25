// @ts-nocheck
import React from 'react';
import { CheckCircle2, WifiOff, AlertTriangle } from 'lucide-react';

interface TechBlueprintProps {
  activeModel: any;
  stats: any;
  validationMetrics: any;
  benchmarkData: any;
  calibrationMode: string;
  activeDatasetName: string;
}

export const TechBlueprint: React.FC<TechBlueprintProps> = ({
  activeModel,
  stats,
  validationMetrics,
  benchmarkData,
  calibrationMode,
  activeDatasetName
}) => {
  const isRelative = calibrationMode === 'relative';
  
  // Format numbers nicely
  const formatNum = (num: any, decimals = 2) => {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return Number(num).toFixed(decimals);
  };

  const pearsonStr = isRelative || validationMetrics?.skipped 
    ? 'N/A' 
    : `${formatNum((validationMetrics?.pearson_r || 0) * 100, 2)}%`;
    
  const r2Str = isRelative || validationMetrics?.skipped 
    ? 'N/A' 
    : `${formatNum((validationMetrics?.r2 || 0) * 100, 2)}%`;
    
  const rmseStr = isRelative || validationMetrics?.skipped 
    ? 'N/A' 
    : `${formatNum(validationMetrics?.rmse)} m`;
    
  const maeStr = isRelative || validationMetrics?.skipped 
    ? 'N/A' 
    : `${formatNum(validationMetrics?.mae)} m`;

  const specs = [
    { label: 'DATASET', value: activeDatasetName || 'NO DATASET LOADED' },
    { label: 'CALIBRATION MODE', value: calibrationMode.toUpperCase() },
    { label: 'PEARSON CORRELATION', value: pearsonStr },
    { label: 'R² SCORE', value: r2Str },
    { label: 'RMSE (ROOT MEAN SQUARE ERROR)', value: rmseStr },
    { label: 'MAE (MEAN ABSOLUTE ERROR)', value: maeStr }
  ];

  return (
    <section id="architecture" className="relative py-24 bg-[#211E1B]/75 backdrop-blur-md border-t border-[#5A564C]/50 font-mono">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <div className="text-xs text-[#8B8478] tracking-[0.2em] uppercase mb-2">
            [ FLAT 2D TECHNICAL BLUEPRINT ]
          </div>
          <h2 className="font-heading text-3xl sm:text-4xl font-bold text-[#D9D2C2] uppercase tracking-tight">
            TELEMETRY & VALIDATION
          </h2>
          <p className="mt-2 text-xs text-[#B7AF9F]">
            High-contrast 2D architectural summary for field-deployable disaster spatial mapping.
          </p>
        </div>

        {/* Specs Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {specs.map((item, idx) => (
            <div key={idx} className="mono-panel p-5 border border-[#5A564C] flex items-center justify-between bg-[#1A1816]/90 backdrop-blur-md">
              <div>
                <div className="text-[10px] text-[#8B8478] tracking-widest">{item.label}</div>
                <div className="text-sm font-bold text-[#D9D2C2] mt-1">{item.value}</div>
              </div>
              {item.value !== 'N/A' ? (
                <CheckCircle2 className="w-4 h-4 text-[#E0433C] flex-shrink-0" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-[#8B8478] flex-shrink-0" />
              )}
            </div>
          ))}
        </div>

        {/* Security & Offline Compliance Box */}
        <div className="mono-panel p-6 border border-[#5A564C] bg-[#1A1816]/90 backdrop-blur-md flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 border border-[#5A564C] bg-[#211E1B] flex items-center justify-center text-[#E0433C]">
              <WifiOff className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-bold text-[#D9D2C2] uppercase">OFFLINE TACTICAL DESIGN</div>
              <div className="text-xs text-[#8B8478]">No internet connection, satellite uplinks, or cloud GPUs required in remote disaster zones.</div>
            </div>
          </div>

          <div className="px-4 py-2 border border-[#5A564C] bg-[#211E1B] text-xs text-[#D9D2C2] font-bold tracking-widest uppercase">
            SIH COMPLIANT ARCHITECTURE
          </div>
        </div>
      </div>
    </section>
  );
};
