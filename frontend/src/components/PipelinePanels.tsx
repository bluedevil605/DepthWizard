import React, { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { ScrollControls } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Activity, CheckCircle2, ArrowRight } from 'lucide-react';
import { SolvingFragments } from './SolvingFragments';

export const PipelinePanels: React.FC = () => {
  const captions = [
    {
      step: '01',
      title: 'Input 2D Drone Image',
      tag: 'Raw Aerial Photo',
      desc: 'Uncalibrated RGB photo captured by commercial drone or search responder.'
    },
    {
      step: '02',
      title: 'Depth Estimation Model',
      tag: 'Neural Spatial Inference',
      desc: 'Deep transformer computes continuous per-pixel depth map in <90ms.'
    },
    {
      step: '03',
      title: '3D Mesh Reconstruction',
      tag: 'Volumetric Displaced Mesh',
      desc: 'Height matrix extrudes 3D terrain surface with smoothed vertex normals.'
    },
    {
      step: '04',
      title: 'Autonomous 3D Flythrough',
      tag: '60FPS WebGL Engine',
      desc: 'Navigable drone camera path for emergency flood and landslide response.'
    }
  ];

  const [activeCaptionIdx, setActiveCaptionIdx] = useState<number>(0);
  const [lockedCount, setLockedCount] = useState<number>(0);
  const [scrollProgress, setScrollProgress] = useState<number>(0);

  const handleFragmentLock = (index: number) => {
    setLockedCount((prev) => Math.max(prev, index + 1));
    const captionIndex = Math.min(
      Math.floor((index / 24) * captions.length),
      captions.length - 1
    );
    setActiveCaptionIdx(captionIndex);
  };

  const currentCaption = captions[activeCaptionIdx];

  return (
    <section id="how-it-works" className="relative py-28 bg-[#0d0d18] border-y border-violet-500/10 overflow-hidden">
      {/* Background Spotlight Glows */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[850px] h-[450px] bg-gradient-neurona-soft blur-[140px] rounded-full pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-900/80 border border-violet-500/30 text-violet-300 text-xs font-mono mb-4">
            <Activity className="w-3.5 h-3.5 text-magenta-pink animate-pulse" />
            <span>Scroll-Scrubbed 3D Solving Engine</span>
          </div>
          <h2 className="font-heading text-3xl sm:text-5xl font-extrabold text-white tracking-tight uppercase">
            Assemble The <span className="gradient-text-neurona">Depth Landscape</span>
          </h2>
          <p className="mt-4 text-slate-300 text-base sm:text-lg">
            Scroll to scrub through 24 scrambled spatial fragments as they align and lock into a cohesive 3D disaster terrain mesh.
          </p>
        </div>

        {/* 3D Solving Canvas Stage */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left / Center 3D Stage (8 cols) */}
          <div className="lg:col-span-8 h-[500px] w-full rounded-3xl border border-violet-500/20 bg-slate-950/60 backdrop-blur-xl relative overflow-hidden shadow-[0_0_35px_rgba(139,92,246,0.15)]">
            <Canvas camera={{ position: [0, 0, 9], fov: 45 }}>
              <ambientLight intensity={0.6} />
              <directionalLight position={[6, 8, 6]} intensity={1.5} color="#4d6dff" />
              <pointLight position={[-6, -4, -4]} intensity={1.2} color="#e952d1" />

              <ScrollControls pages={2} damping={0.25}>
                <SolvingFragments
                  onFragmentLock={handleFragmentLock}
                  groupRotationSpeed={0.06}
                  scrollProgressOverride={scrollProgress}
                />
              </ScrollControls>
            </Canvas>

            {/* Manual Slider Fallback Control inside stage */}
            <div className="absolute bottom-4 left-4 right-4 bg-slate-900/90 p-3 rounded-xl border border-violet-500/30 flex items-center justify-between gap-4 z-20 backdrop-blur-md">
              <span className="text-xs font-mono text-violet-300 font-semibold flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-magenta-pink" /> Scrub Assembly:
              </span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={scrollProgress}
                onChange={(e) => setScrollProgress(parseFloat(e.target.value))}
                className="w-full accent-magenta-pink cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
              <span className="text-xs font-mono text-white font-bold min-w-[45px] text-right">
                {Math.round(scrollProgress * 100)}%
              </span>
            </div>
          </div>

          {/* Right Sequence Caption Overlay Card (4 cols) */}
          <div className="lg:col-span-4 space-y-6">
            <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-violet-500/30 shadow-xl relative overflow-hidden">
              {/* Top Accent Gradient Line */}
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-neurona" />

              <div className="flex items-center justify-between mb-6">
                <span className="text-xs font-mono px-3 py-1 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/40">
                  Step {currentCaption.step} of 04
                </span>
                <span className="text-xs font-mono text-pink-400 font-bold">
                  {lockedCount}/24 Fragments
                </span>
              </div>

              <AnimatePresence mode="wait">
                <motion.div
                  key={currentCaption.step}
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -15 }}
                  transition={{ duration: 0.3 }}
                >
                  <span className="text-xs font-mono text-magenta-pink uppercase tracking-wider block mb-1">
                    {currentCaption.tag}
                  </span>
                  <h3 className="text-2xl font-bold font-heading text-white mb-3">
                    {currentCaption.title}
                  </h3>
                  <p className="text-sm text-slate-300 leading-relaxed mb-6">
                    {currentCaption.desc}
                  </p>
                </motion.div>
              </AnimatePresence>

              {/* Progress Steps List */}
              <div className="border-t border-slate-800 pt-4 space-y-2.5 text-xs font-mono">
                {captions.map((cap, idx) => (
                  <div
                    key={cap.step}
                    className={`flex items-center justify-between p-2 rounded-lg transition-all ${
                      idx === activeCaptionIdx
                        ? 'bg-violet-500/20 text-white font-bold border border-violet-500/40'
                        : 'text-slate-500'
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <CheckCircle2 className={`w-3.5 h-3.5 ${idx <= activeCaptionIdx ? 'text-teal-400' : 'text-slate-700'}`} />
                      {cap.step}. {cap.title}
                    </span>
                    {idx === activeCaptionIdx && (
                      <ArrowRight className="w-3.5 h-3.5 text-magenta-pink" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
