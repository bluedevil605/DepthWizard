import React from 'react';
import { Cpu, ArrowRight, CheckCircle2, Terminal, Shield } from 'lucide-react';

export const Architecture: React.FC = () => {
  const pipelineNodes = [
    {
      step: '01',
      title: 'Uncalibrated 2D Photo',
      tech: 'Input Layer',
      desc: 'Single drone image (RGB, JPEG/PNG), zero camera matrix calibration required.'
    },
    {
      step: '02',
      title: 'Depth Anything / MiDaS',
      tech: 'ONNX WebGL Model',
      desc: 'Monocular depth transformer infers continuous per-pixel depth map in <90ms.'
    },
    {
      step: '03',
      title: 'Point Cloud & Heightmap',
      tech: 'Mesh Processing',
      desc: 'Luminance matrix converted to displaced plane geometry with vertex normal smoothing.'
    },
    {
      step: '04',
      title: 'Flythrough Renderer',
      tech: 'React Three Fiber',
      desc: 'Autonomous camera drone path, cinematic lighting, and custom GLSL shader effects.'
    }
  ];

  const techStack = [
    { name: 'Depth Anything v2 / MiDaS', category: 'Depth AI Inference Engine', status: 'Active (<90ms)' },
    { name: 'React Three Fiber & Drei', category: '3D WebGL Canvas Framework', status: '60 FPS' },
    { name: 'Custom GLSL Displacement', category: 'Vertex & Fragment Shaders', status: 'Hardware Accelerated' },
    { name: '@react-three/postprocessing', category: 'Bloom & Depth of Field', status: 'Cinematic Visuals' },
    { name: 'Tailwind CSS & Framer Motion', category: 'UI Layout & Animations', status: 'Responsive' }
  ];

  return (
    <section id="architecture" className="relative py-28 bg-[#0a0a12] border-t border-violet-500/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-900/80 border border-violet-500/30 text-violet-300 text-xs font-mono mb-4">
            <Cpu className="w-3.5 h-3.5 text-magenta-pink" />
            <span>Flat 2D Technical Contrast Section</span>
          </div>
          <h2 className="font-heading text-3xl sm:text-5xl font-extrabold text-white tracking-tight uppercase">
            End-to-End <span className="gradient-text-neurona">AI Pipeline</span>
          </h2>
          <p className="mt-4 text-slate-300 text-base sm:text-lg">
            High-performance modular stack designed for instant low-latency execution directly in-browser or edge servers.
          </p>
        </div>

        {/* 2D Horizontal Pipeline Flow Diagram */}
        <div className="glass-panel p-8 sm:p-10 rounded-3xl border border-violet-500/20 mb-16 shadow-[0_0_30px_rgba(139,92,246,0.08)]">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
            {pipelineNodes.map((node, idx) => (
              <div key={idx} className="relative flex flex-col justify-between bg-slate-950/80 p-6 rounded-2xl border border-slate-800">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-xs font-mono px-2.5 py-1 rounded-lg bg-violet-500/20 text-violet-300 border border-violet-500/40">
                      {node.tech}
                    </span>
                    <span className="font-mono text-xs text-slate-600 font-bold">
                      {node.step}
                    </span>
                  </div>

                  <h3 className="text-lg font-bold font-heading text-white mb-2">
                    {node.title}
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {node.desc}
                  </p>
                </div>

                {/* Arrow Connector between steps */}
                {idx < pipelineNodes.length - 1 && (
                  <div className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 z-10">
                    <div className="w-6 h-6 rounded-full bg-violet-500/20 border border-violet-500/40 flex items-center justify-center text-magenta-pink">
                      <ArrowRight className="w-3.5 h-3.5" />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Tech Stack Specs Table */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-7 glass-panel p-6 sm:p-8 rounded-3xl border border-violet-500/20">
            <h3 className="text-lg font-bold font-heading text-white mb-6 flex items-center gap-2">
              <Terminal className="w-5 h-5 text-violet-400" /> Core Tech Specifications
            </h3>

            <div className="space-y-4">
              {techStack.map((tech, idx) => (
                <div key={idx} className="flex items-center justify-between p-3.5 rounded-2xl bg-slate-950/70 border border-slate-800 text-xs font-mono">
                  <div>
                    <div className="font-bold text-white text-sm">{tech.name}</div>
                    <div className="text-slate-400 text-[11px] mt-0.5">{tech.category}</div>
                  </div>
                  <span className="px-3 py-1 rounded-lg bg-magenta-pink/15 text-magenta-pink border border-magenta-pink/30 font-semibold">
                    {tech.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="lg:col-span-5 glass-panel p-6 sm:p-8 rounded-3xl border border-violet-500/20 flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-2xl bg-violet-500/20 border border-violet-500/40 flex items-center justify-center text-magenta-pink mb-4">
                <Shield className="w-5 h-5" />
              </div>
              <h3 className="text-xl font-bold font-heading text-white mb-3">
                Smart India Hackathon Compliance
              </h3>
              <p className="text-sm text-slate-300 leading-relaxed mb-6">
                Engineered specifically to solve SIH26175 problem statement requirements for single-view 3D spatial reconstruction in tactical disaster management.
              </p>
            </div>

            <div className="space-y-3 border-t border-slate-800/80 pt-4">
              <div className="flex items-center gap-2 text-xs text-slate-300">
                <CheckCircle2 className="w-4 h-4 text-teal-400 flex-shrink-0" />
                <span>Zero server dependency for local inference preview</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-300">
                <CheckCircle2 className="w-4 h-4 text-teal-400 flex-shrink-0" />
                <span>Sub-meter height resolution calibration</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-300">
                <CheckCircle2 className="w-4 h-4 text-teal-400 flex-shrink-0" />
                <span>Full WebGL hardware acceleration</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
