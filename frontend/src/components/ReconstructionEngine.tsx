import React from 'react';
import { RefreshCw, CheckCircle2, AlertTriangle } from 'lucide-react';

interface ReconstructionEngineProps {
  stage: string;
  message: string;
}

export const ReconstructionEngine: React.FC<ReconstructionEngineProps> = ({ stage, message }) => {
  const isComplete = stage === 'COMPLETE';
  const isError = stage === 'ERROR';

  // Map stages to a pseudo-progress percentage for visual effect
  let progress = 10;
  if (stage === 'ANALYZING IMAGE') progress = 25;
  if (stage === 'DEPTH ESTIMATION') progress = 50;
  if (stage === 'CALIBRATION') progress = 75;
  if (stage === 'DSM GENERATION') progress = 85;
  if (stage === 'LOADING TERRAIN') progress = 95;
  if (isComplete) progress = 100;
  if (isError) progress = 100;

  return (
    <section className="relative py-20 bg-[#211E1B]/75 backdrop-blur-md border-t border-[#5A564C]/50 font-mono">
      <div className="max-w-5xl mx-auto px-4 text-center">
        {/* Progress Header */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 border border-[#5A564C] bg-[#1A1816] text-xs text-[#B7AF9F] uppercase tracking-widest mb-6">
          {!isComplete && !isError && <RefreshCw className="w-3.5 h-3.5 text-[#E0433C] animate-spin" />}
          {isComplete && <CheckCircle2 className="w-3.5 h-3.5 text-[#10B981]" />}
          {isError && <AlertTriangle className="w-3.5 h-3.5 text-[#ff3366]" />}
          <span>{isComplete ? 'RECONSTRUCTION COMPLETE' : (isError ? 'RECONSTRUCTION FAILED' : 'RECONSTRUCTION ENGINE IN PROGRESS')}</span>
        </div>

        <h2 className="font-heading text-3xl sm:text-5xl font-bold text-[#D9D2C2] uppercase tracking-tight mb-2">
          {stage}
        </h2>
        <div className="text-xs text-[#8B8478] tracking-widest uppercase mb-10 min-h-[1rem]">
          {message}
        </div>

        {/* Technical Readout Dashboard Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-10 text-left">
          <div className="mono-panel p-4 border border-[#5A564C]">
            <div className="text-[10px] text-[#8B8478] uppercase tracking-widest">PIPELINE STAGE</div>
            <div className="text-xl font-bold text-[#D9D2C2] mt-1 truncate">
              {stage}
            </div>
          </div>

          <div className="mono-panel p-4 border border-[#5A564C]">
            <div className="text-[10px] text-[#8B8478] uppercase tracking-widest">DEPTH ENGINE</div>
            <div className="text-xl font-bold text-[#D9D2C2] mt-1">
              Depth Anything V2
            </div>
          </div>

          <div className="mono-panel p-4 border border-[#5A564C]">
            <div className="text-[10px] text-[#8B8478] uppercase tracking-widest">CALIBRATION</div>
            <div className="text-xl font-bold text-[#D9D2C2] mt-1">
              Active
            </div>
          </div>

          <div className="mono-panel p-4 border border-[#5A564C]">
            <div className="text-[10px] text-[#8B8478] uppercase tracking-widest">OUTPUT FORMAT</div>
            <div className="text-xl font-bold text-[#D9D2C2] mt-1">
              3D GLB Mesh
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-[#1A1816] border border-[#5A564C] h-3 p-0.5 overflow-hidden mb-6">
          <div
            className={`h-full shadow-[0_0_12px_rgba(224,67,60,0.4)] transition-all duration-300 ${isError ? 'bg-[#ff3366]' : 'bg-[#E0433C]'}`}
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="flex items-center justify-between text-xs text-[#8B8478] tracking-widest uppercase">
          <span>PROCESSING ARCHITECTURE: FastAPI Backend</span>
          <span>{isComplete ? 'READY FOR ANALYSIS' : (isError ? 'SYSTEM HALTED' : 'EXTRACTING DENSE POINT MATRIX...')}</span>
        </div>
      </div>
    </section>
  );
};
