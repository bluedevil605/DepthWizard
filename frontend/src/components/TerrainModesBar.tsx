import React from 'react';
import { Sliders } from 'lucide-react';
import { playModeWhipSound } from '../utils/soundEffects';

interface TerrainModesBarProps {
  activeMode: string;
  onModeChange: (mode: string) => void;
}

export const TerrainModesBar: React.FC<TerrainModesBarProps> = ({ activeMode, onModeChange }) => {
  const modes = [
    { id: 'ELEVATION', name: 'ELEVATION' },
    { id: 'DEPTH', name: 'DEPTH' },
    { id: 'SLOPE', name: 'SLOPE' },
    { id: 'HEATMAP', name: 'HEATMAP', isSim: true },
    { id: 'LAND COVER', name: 'LAND COVER' },
    { id: 'OBJECTS', name: 'OBJECTS' },
    { id: 'ERROR', name: 'ERROR', isSim: true },
    { id: 'CONFIDENCE', name: 'CONFIDENCE' },
    { id: 'OPTICAL', name: 'OPTICAL' }
  ];

  const handleSelectMode = (id: string) => {
    playModeWhipSound();
    onModeChange(id);
  };

  return (
    <div className="w-full bg-[#211E1B]/85 backdrop-blur-md border-b border-[#5A564C]/70 py-2.5 font-mono relative z-20">
      <div className="max-w-6xl mx-auto px-4 flex items-center justify-between gap-4">
        {/* Label */}
        <div className="hidden sm:flex items-center gap-2 text-[10px] text-[#8B8478] tracking-[0.22em] uppercase whitespace-nowrap">
          <Sliders className="w-3.5 h-3.5 text-[#E0433C]" />
          <span>RENDER MODE</span>
          <span className="text-[#5A564C]">//</span>
        </div>

        {/* Mode selector — responsive grid/flex */}
        <div className="w-full sm:w-auto mx-auto sm:mx-0 flex flex-wrap justify-center sm:justify-start items-stretch border-t border-l border-[#5A564C]/70 bg-[#1A1816]/90 shadow-[0_2px_12px_rgba(0,0,0,0.35)]">
          {modes.map((mode) => {
            const isActive = activeMode === mode.id;
            return (
              <button
                key={mode.id}
                onClick={() => handleSelectMode(mode.id)}
                aria-pressed={isActive}
                className={`relative flex-grow sm:flex-grow-0 min-w-0 whitespace-nowrap px-3 sm:px-4 py-1.5 text-[10px] font-bold tracking-[0.14em] sm:tracking-[0.18em] uppercase transition-colors duration-150 border-r border-b border-[#5A564C]/70 ${
                  isActive
                    ? 'bg-[#E0433C] text-[#FFFFFF] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.25)]'
                    : 'text-[#8B8478] hover:text-[#D9D2C2] hover:bg-[#211E1B]'
                }`}
              >
                {mode.isSim && (
                  <span
                    aria-hidden="true"
                    className={`absolute top-0.5 right-0.5 w-1 h-1 rounded-full transition-colors duration-150 ${
                      isActive ? 'bg-[#FFFFFF]' : 'bg-[#E0433C] animate-pulse'
                    }`}
                  />
                )}
                <span>{mode.name}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
