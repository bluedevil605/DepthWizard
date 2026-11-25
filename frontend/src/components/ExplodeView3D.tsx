import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, ArrowDown, Layers } from 'lucide-react';

interface LayerSpec {
  num: string;
  label: string;
  desc: string;
  isSource?: boolean;
  isFinal?: boolean;
}

/** Reconstruction pipeline order: 01 source → 05 final vectors */
const LAYERS: LayerSpec[] = [
  { num: '01', label: 'ORIGINAL CAPTURE', desc: 'Raw RGB video frame input', isSource: true },
  { num: '02', label: 'DEPTH MATRIX', desc: 'Monocular depth per-pixel grid' },
  { num: '03', label: 'POINT CLOUD', desc: 'Dense vertex scatter' },
  { num: '04', label: 'SURFACE MESH', desc: 'Delaunay wireframe skin' },
  { num: '05', label: 'CONTOUR FIELD', desc: 'Marching-squares iso-curves', isFinal: true },
];

const EASE = [0.16, 1, 0.3, 1] as const;

export const ExplodeView3D: React.FC = () => {
  return (
    <section id="explode-view" className="relative py-24 font-mono">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        {/* Heading — same transparent pattern as Topographic Field Map (no scrim box) */}
        <div className="text-center max-w-2xl mx-auto mb-12">
          <div className="inline-flex items-center gap-2 text-[10px] text-[#8B8478] tracking-[0.25em] uppercase mb-3">
            <span className="w-1.5 h-1.5 bg-[#E0433C]" />
            <span>DECOMPOSED SPATIAL STACK</span>
          </div>
          <h2 className="font-heading text-3xl sm:text-4xl font-bold text-[#D9D2C2] uppercase tracking-tight">
            Exploded Layer View
          </h2>
          <p className="mt-3 text-xs text-[#B7AF9F]">
            Deconstruct the terrain dataset into five independent spatial strata — from raw RGB frame to contour vectors.
          </p>
        </div>

        {/* Single wide HUD panel — mirrors the Topo Map frame pattern */}
        <div className="max-w-4xl mx-auto">
          <div className="relative p-6 sm:p-8 bg-[#1A1816]/75 backdrop-blur-sm border border-[#5A564C]/70">
            {/* Corner metadata — top left */}
            <div className="absolute top-4 left-4 text-[10px] text-[#8B8478] space-y-1">
              <div>STACK: RGB FRAME → ISO-VECTORS</div>
              <div>STRATA: 5 INDEPENDENT LAYERS</div>
              <div>ORDER: 01 → 05</div>
            </div>

            {/* Corner metadata — top right */}
            <div className="absolute top-4 right-4 text-[10px] text-[#8B8478] text-right space-y-1">
              <div>LINKAGE: DECOUPLED</div>
              <div>DEPTH: MONOCULAR ESTIMATE</div>
              <div>REBUILD: ADDITIVE</div>
            </div>

            {/* Center — reconstruction pipeline, 01 → 05 */}
            <div className="py-16 sm:py-20">
              <div className="flex flex-col lg:flex-row items-center justify-center">
                {LAYERS.map((layer, i) => (
                  <React.Fragment key={layer.num}>
                    {i > 0 && (
                      <>
                        {/* Horizontal connector (desktop) */}
                        <div className="hidden lg:flex items-center self-center px-1.5">
                          <span className="w-6 h-px bg-[#5A564C]" />
                          <ArrowRight className="w-3 h-3 text-[#8B8478]" />
                        </div>
                        {/* Vertical connector (mobile) */}
                        <div className="lg:hidden flex flex-col items-center py-1">
                          <span className="w-px h-5 bg-[#5A564C]" />
                          <ArrowDown className="w-3 h-3 text-[#8B8478]" />
                        </div>
                      </>
                    )}

                    <motion.div
                      initial={{ opacity: 0, y: 14 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true, margin: '-40px' }}
                      transition={{ duration: 0.45, delay: 0.1 + i * 0.09, ease: EASE }}
                      className="flex lg:flex-1 flex-col items-center text-center gap-3 min-w-0 max-w-[220px]"
                    >
                      <div
                        className="w-11 h-11 border flex items-center justify-center text-[11px] font-bold tracking-wider shadow-[0_0_18px_rgba(0,0,0,0.4)]"
                        style={{
                          borderColor: layer.isSource ? '#E0433C' : layer.isFinal ? '#D9D2C2' : '#5A564C',
                          color: layer.isSource ? '#E0433C' : layer.isFinal ? '#D9D2C2' : '#B7AF9F',
                          backgroundColor: layer.isSource ? 'rgba(92,19,21,0.5)' : '#211E1B',
                        }}
                      >
                        {layer.num}
                      </div>
                      <div>
                        <div
                          className="text-[10px] font-bold tracking-[0.18em] uppercase"
                          style={{ color: layer.isSource ? '#E0433C' : '#D9D2C2' }}
                        >
                          {layer.label}
                        </div>
                        <div className="text-[9px] text-[#8B8478] mt-1.5 leading-snug max-w-[150px] mx-auto">
                          {layer.desc}
                        </div>
                      </div>
                    </motion.div>
                  </React.Fragment>
                ))}
              </div>
            </div>

            {/* Bottom status — left */}
            <div className="absolute bottom-4 left-4 flex items-center gap-2 text-[10px] text-[#8B8478]">
              <div className="w-2 h-2 bg-[#E0433C] rounded-full animate-pulse shadow-[0_0_8px_rgba(224,67,60,0.5)]" />
              <span>SPATIAL STRATA DECOUPLED</span>
            </div>

            {/* Bottom status — right */}
            <div className="absolute bottom-4 right-4 flex items-center gap-2 text-[10px] text-[#B7AF9F] bg-[#211E1B] px-3 py-1.5 border border-[#5A564C]">
              <Layers className="w-3.5 h-3.5 text-[#E0433C]" />
              <span>5-LAYER RECONSTRUCTION</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
