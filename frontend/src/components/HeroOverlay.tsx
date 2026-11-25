import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Eye, RotateCcw, Crosshair, Radio, ArrowLeft } from 'lucide-react';
import { playScanPulseSound, playModeWhipSound } from '../utils/soundEffects';

interface HeroOverlayProps {
  scrollProgress: number;
  onStartReconstruction: () => void;
  onEnterTerrain: () => void;
  isDronePOV?: boolean;
  setIsDronePOV?: (val: boolean) => void;
  isBreathing?: boolean;
  setIsBreathing?: (val: boolean) => void;
  isDiving?: boolean;
  setIsDiving?: (val: boolean) => void;
  terrainSource?: 'demo' | 'processed';
}

export const HeroOverlay: React.FC<HeroOverlayProps> = ({
  scrollProgress,
  onStartReconstruction,
  onEnterTerrain,
  isDronePOV: extDronePOV,
  setIsDronePOV: setExtDronePOV,
  isBreathing: extBreathing,
  setIsBreathing: setExtBreathing,
  isDiving: extDiving,
  setIsDiving: setExtDiving,
  terrainSource = 'demo',
}) => {
  const [activeStage, setActiveStage] = useState(1);
  const [localDronePOV, setLocalDronePOV] = useState(false);
  const [localBreathing, setLocalBreathing] = useState(true);
  const [localDiving, setLocalDiving] = useState(false);

  const isDronePOV = extDronePOV !== undefined ? extDronePOV : localDronePOV;
  const setIsDronePOV = setExtDronePOV || setLocalDronePOV;

  const isBreathing = extBreathing !== undefined ? extBreathing : localBreathing;
  const setIsBreathing = setExtBreathing || setLocalBreathing;

  const isDiving = extDiving !== undefined ? extDiving : localDiving;
  const setIsDiving = setExtDiving || setLocalDiving;

  // Intro stage sequence
  useEffect(() => {
    const timer1 = setTimeout(() => setActiveStage(2), 1200);
    const timer2 = setTimeout(() => {
      setActiveStage(3);
      playScanPulseSound();
    }, 2800);
    const timer3 = setTimeout(() => setActiveStage(4), 4500);
    const timer4 = setTimeout(() => setActiveStage(5), 6000);
    const timer5 = setTimeout(() => setActiveStage(6), 7500);
    const timer6 = setTimeout(() => setActiveStage(7), 8800);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
      clearTimeout(timer5);
      clearTimeout(timer6);
    };
  }, []);

  const stageNames = [
    '01. SILENCE',
    '02. CRAFT ENTRY',
    '03. SCAN PULSE',
    '04. GRID SURFACE',
    '05. RIVERS OF LIGHT',
    '06. PULL BACK',
    '07. TERRAIN DATASET READY'
  ];

  // Fade hero out as user scrolls past it
  const heroOpacity = Math.max(0, 1 - scrollProgress * 5);
  const heroVisible = scrollProgress < 0.22;

  const handleDiveClick = () => {
    setIsDiving(!isDiving);
    if (!isDiving) {
      onEnterTerrain();
    }
  };

  const handleResetView = () => {
    playModeWhipSound();
    setIsDiving(false);
    setIsDronePOV(false);
  };

  if (!heroVisible) return null;

  return (
    <div
      className="fixed inset-0 z-20 pointer-events-none flex flex-col items-center justify-center"
      style={{ opacity: heroOpacity, willChange: 'opacity' }}
    >
      {/* Sits between the canvas and the text layer, only across the hero text band */}
      {terrainSource === 'demo' && (
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center -z-10">
          <div
            className="w-full max-w-3xl h-[520px] pointer-events-none rounded-3xl"
            style={{
              background: 'linear-gradient(to bottom, rgba(33,30,27,0) 0%, rgba(33,30,27,0.75) 25%, rgba(33,30,27,0.75) 75%, rgba(33,30,27,0.95) 100%)',
              backdropFilter: 'blur(8px)',
              WebkitBackdropFilter: 'blur(8px)',
            }}
          />
        </div>
      )}

      {/* Sequence HUD Stage Readout (Top Left) */}
      <div className="absolute top-20 left-6 hidden sm:block pointer-events-auto">
        <div className="mono-panel p-3 rounded-none border border-[#5A564C] text-[11px] text-[#B7AF9F] space-y-1 font-mono">
          <div className="flex items-center gap-2 text-[#D9D2C2] font-semibold">
            <Radio className="w-3.5 h-3.5 text-[#E0433C] animate-pulse" />
            <span>RECON PASS SEQUENCE</span>
          </div>
          <div className="text-[#8B8478]">{isDiving ? 'ENTERED TERRAIN DIVE NODE' : stageNames[activeStage - 1]}</div>
          <div className="w-32 h-1 bg-[#5A564C] overflow-hidden mt-1">
            <div
              className="h-full bg-[#E0433C] transition-all duration-300"
              style={{ width: `${isDiving ? 100 : (activeStage / 7) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Telemetry HUD Strip (Top Right) */}
      <div className="absolute top-20 right-6 hidden md:block pointer-events-auto">
        <div className="mono-panel p-3.5 rounded-none border border-[#5A564C] text-right font-mono space-y-2">
          <div className="text-xl font-bold text-[#D9D2C2] tracking-widest">
            {isDiving ? '0210m' : '1842m'} &nbsp; {isDiving ? '12.1°' : '37.4°'} &nbsp; 94.6%
          </div>
          <div className="text-[10px] text-[#8B8478] tracking-widest uppercase flex justify-end gap-6">
            <span>ALTITUDE</span>
            <span>SLOPE</span>
            <span>CONFIDENCE</span>
          </div>
        </div>
      </div>

      {/* Hero Central Title Overlay */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 text-center flex flex-col items-center pointer-events-auto">
        <AnimatePresence>
          {activeStage >= 6 && !isDiving && terrainSource === 'demo' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              style={{ willChange: 'transform, opacity' }}
              className="flex flex-col items-center"
            >
              <div className="text-xs font-mono tracking-[0.3em] text-[#B7AF9F] uppercase mb-4 border-b border-[#5A564C] pb-1">
                DEPTH WIZARD &nbsp;•&nbsp; FIELD NODE 01
              </div>

              <h1 className="font-heading text-5xl sm:text-7xl md:text-8xl font-bold text-[#D9D2C2] tracking-tight uppercase leading-[0.95] mb-6">
                ONE PICTURE.<br />
                <span className="text-[#8B8478]">ONE TERRAIN.</span>
              </h1>

              <p className="text-sm sm:text-base text-[#B7AF9F] max-w-xl font-mono leading-relaxed mb-8 drop-shadow-md">
                Monocular depth estimation & volumetric field reconstruction. Instant explorable 3D topography from a single uncalibrated video pass.
              </p>

              <div className="flex flex-wrap items-center justify-center gap-4">
                <button
                  onClick={onStartReconstruction}
                  className="px-6 py-3.5 bg-[#E0433C] text-[#FFFFFF] font-mono text-xs font-bold uppercase tracking-wider hover:bg-[#C4272B] shadow-[0_0_20px_rgba(224,67,60,0.35)] transition-all flex items-center gap-2 border border-[#E0433C]"
                >
                  <Play className="w-4 h-4 fill-[#FFFFFF]" />
                  <span>[ START RECONSTRUCTION ]</span>
                </button>

                <button
                  onClick={handleDiveClick}
                  className="px-6 py-3.5 bg-[#1A1816] text-[#D9D2C2] font-mono text-xs font-bold uppercase tracking-wider border border-[#5A564C] hover:border-[#8B8478] hover:bg-[#211E1B] transition-all flex items-center gap-2"
                >
                  <Crosshair className="w-4 h-4 text-[#E0433C]" />
                  <span>[ ENTER TERRAIN DIVE ]</span>
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Exit Dive button */}
        {isDiving && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-8"
          >
            <button
              onClick={() => setIsDiving(false)}
              className="px-6 py-3 bg-[#E0433C] text-[#FFFFFF] font-mono text-xs font-bold uppercase tracking-widest hover:bg-[#C4272B] shadow-[0_0_20px_rgba(224,67,60,0.35)] transition-all flex items-center gap-2 border border-[#E0433C]"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>[ EXIT TERRAIN DIVE ]</span>
            </button>
          </motion.div>
        )}
      </div>

      {/* 3D Controls Bar */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 max-w-full px-4 pointer-events-auto">
        <div className="mono-panel p-2 rounded-none border border-[#5A564C] flex flex-wrap items-center justify-center gap-3 text-xs font-mono">
          <button
            onClick={() => setIsDronePOV(!isDronePOV)}
            className={`px-3 py-1.5 border transition-all flex items-center gap-1.5 ${
              isDronePOV
                ? 'bg-[#E0433C] text-[#FFFFFF] border-[#E0433C] font-bold shadow-[0_0_12px_rgba(224,67,60,0.35)]'
                : 'bg-[#1A1816] text-[#B7AF9F] border-[#5A564C] hover:border-[#8B8478]'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>DRONE POV: {isDronePOV ? 'ON' : 'OFF'}</span>
          </button>

          <button
            onClick={handleResetView}
            className="px-3 py-1.5 bg-[#1A1816] text-[#B7AF9F] border border-[#5A564C] hover:border-[#8B8478] hover:text-[#D9D2C2] transition-all flex items-center gap-1.5"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>[ RESET VIEW ]</span>
          </button>

          <button
            onClick={() => setIsBreathing(!isBreathing)}
            className={`px-3 py-1.5 border transition-all ${
              isBreathing
                ? 'bg-[#211E1B] text-[#D9D2C2] border-[#8B8478]'
                : 'bg-[#1A1816] text-[#8B8478] border-[#5A564C]'
            }`}
          >
            BREATHING: {isBreathing ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>
    </div>
  );
};
