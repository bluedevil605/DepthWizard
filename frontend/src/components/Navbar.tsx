import React, { useState, useEffect } from 'react';
import { Terminal, Radio, Volume2, VolumeX } from 'lucide-react';

interface NavbarProps {
  isMuted: boolean;
  onToggleMute: () => void;
  activeSection: string;
  onNavChange: (section: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ isMuted, onToggleMute, activeSection, onNavChange }) => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 font-mono ${
        scrolled
          ? 'bg-[#211E1B]/95 backdrop-blur-md border-b border-[#5A564C] py-3'
          : 'bg-transparent py-5'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between">
        {/* Logo */}
        <a href="#" className="flex items-center gap-3 group">
          <div className="w-8 h-8 border border-[#5A564C] bg-[#1A1816] flex items-center justify-center text-[#D9D2C2] group-hover:border-[#8B8478] transition-colors">
            <Terminal className="w-4 h-4 text-[#E0433C]" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-bold tracking-[0.25em] text-[#D9D2C2] uppercase">
              DEPTH<span className="text-[#E0433C]"> WIZARD</span>
            </span>
            <span className="text-[9px] text-[#8B8478] tracking-widest uppercase">
              FIELD RECON NODE 01
            </span>
          </div>
        </a>

        {/* Navigation Links */}
        <nav className="hidden lg:flex items-center gap-6 text-xs uppercase tracking-wider whitespace-nowrap">
          <button 
            onClick={() => onNavChange('overview')}
            className={`transition-colors ${activeSection === 'overview' ? 'text-[#D9D2C2] font-bold' : 'text-[#B7AF9F] hover:text-[#D9D2C2]'}`}
          >
            [ OVERVIEW ]
          </button>
          <button 
            onClick={() => onNavChange('chamber')}
            className={`transition-colors ${activeSection === 'chamber' ? 'text-[#D9D2C2] font-bold' : 'text-[#B7AF9F] hover:text-[#D9D2C2]'}`}
          >
            [ CHAMBER ]
          </button>
          <button 
            onClick={() => onNavChange('exploded')}
            className={`transition-colors ${activeSection === 'exploded' ? 'text-[#D9D2C2] font-bold' : 'text-[#B7AF9F] hover:text-[#D9D2C2]'}`}
          >
            [ EXPLODED STACK ]
          </button>
          <button 
            onClick={() => onNavChange('topo')}
            className={`transition-colors ${activeSection === 'topo' ? 'text-[#D9D2C2] font-bold' : 'text-[#B7AF9F] hover:text-[#D9D2C2]'}`}
          >
            [ TOPO MAP ]
          </button>
          <button 
            onClick={() => onNavChange('architecture')}
            className={`transition-colors ${activeSection === 'architecture' ? 'text-[#D9D2C2] font-bold' : 'text-[#B7AF9F] hover:text-[#D9D2C2]'}`}
          >
            [ ARCHITECTURE ]
          </button>
        </nav>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleMute}
            className="p-2 border border-[#5A564C] bg-[#1A1816] text-[#B7AF9F] hover:text-[#D9D2C2] hover:border-[#8B8478] transition-all"
            aria-label="Toggle Audio Synth"
            title="Toggle Web Audio Synth"
          >
            {isMuted ? <VolumeX className="w-4 h-4 text-[#8B8478]" /> : <Volume2 className="w-4 h-4 text-[#D9D2C2]" />}
          </button>

          <button
            onClick={() => onNavChange('chamber')}
            className="px-3.5 py-1.5 bg-[#E0433C] text-[#FFFFFF] text-xs font-bold uppercase tracking-widest hover:bg-[#C4272B] border border-[#E0433C] shadow-[0_0_15px_rgba(224,67,60,0.3)] transition-all hidden sm:inline-flex items-center gap-1.5"
          >
            <Radio className="w-3.5 h-3.5 animate-pulse text-[#FFFFFF]" />
            <span>INGEST VIDEO</span>
          </button>
        </div>
      </div>
    </header>
  );
};
