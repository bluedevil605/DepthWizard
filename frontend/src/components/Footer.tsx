import React from 'react';
import { Terminal, ShieldCheck } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="relative bg-[#211E1B]/90 backdrop-blur-md border-t border-[#5A564C]/50 py-12 text-[#8B8478] font-mono text-xs z-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 pb-10 border-b border-[#5A564C]">
          {/* Brand & Purpose */}
          <div className="md:col-span-2 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 border border-[#5A564C] bg-[#1A1816] flex items-center justify-center text-[#E0433C]">
                <Terminal className="w-3.5 h-3.5" />
              </div>
              <span className="font-heading text-base font-bold text-[#D9D2C2] tracking-widest uppercase">
                DEPTH <span className="text-[#E0433C]">WIZARD</span>
              </span>
            </div>
            <p className="text-xs text-[#8B8478] max-w-md leading-relaxed">
              Lightweight, offline field-mapping tool converting single video passes into explorable 3D terrain models in minutes. Designed for emergency response & field survey teams.
            </p>
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#1A1816] border border-[#5A564C] text-[10px] text-[#B7AF9F]">
              <ShieldCheck className="w-3.5 h-3.5 text-[#E0433C]" />
              <span>OFFLINE FIELD NODE • NO CLOUD REQUIREMENT</span>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-[#D9D2C2] font-bold uppercase tracking-wider mb-3">SECTIONS</h4>
            <ul className="space-y-2 text-xs">
              <li><a href="#hero" className="hover:text-[#D9D2C2] transition-colors">[ 01. THE PASS ]</a></li>
              <li><a href="#upload-chamber" className="hover:text-[#D9D2C2] transition-colors">[ 02. SCAN CHAMBER ]</a></li>
              <li><a href="#explode-view" className="hover:text-[#D9D2C2] transition-colors">[ 03. EXPLODED STACK ]</a></li>
              <li><a href="#topo-map" className="hover:text-[#D9D2C2] transition-colors">[ 04. TOPO MAP ]</a></li>
              <li><a href="#architecture" className="hover:text-[#D9D2C2] transition-colors">[ 05. BLUEPRINT ]</a></li>
            </ul>
          </div>

          {/* Technical Info */}
          <div>
            <h4 className="text-[#D9D2C2] font-bold uppercase tracking-wider mb-3">SPECIFICATION</h4>
            <p className="text-xs text-[#8B8478] mb-3">
              Japanese Landscape Palette (Crimson & Stone Aesthetic).
            </p>
            <div className="text-[10px] text-[#8B8478] border-t border-[#5A564C] pt-2">
              BUILD: WebGL2 / THREE.JS / REACTION STACK
            </div>
          </div>
        </div>

        {/* Copyright */}
        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-[10px] text-[#8B8478]">
          <div>
            © {new Date().getFullYear()} DEPTH WIZARD • ALL RIGHTS RESERVED.
          </div>
          <div className="tracking-wider">
            CHARCOAL SHADOWS. STONE RIDGES. VERMILION SUN.
          </div>
        </div>
      </div>
    </footer>
  );
};
