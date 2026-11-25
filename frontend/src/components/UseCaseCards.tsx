import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { motion } from 'framer-motion';
import { Waves, Mountain, Building2, Compass, ShieldAlert, ArrowUpRight, TrendingUp } from 'lucide-react';
import { createTerrainTexture } from '../utils/depthMapGenerator';
import type { TerrainPreset } from '../utils/depthMapGenerator';

interface MiniTerrainCanvasProps {
  preset: TerrainPreset;
  color: string;
}

const MiniTerrainCanvas: React.FC<MiniTerrainCanvasProps> = ({ preset, color }) => {
  const texture = useMemo(() => createTerrainTexture(preset, 128), [preset]);

  return (
    <Canvas camera={{ position: [0, 2.5, 4], fov: 45 }}>
      <ambientLight intensity={0.8} />
      <directionalLight position={[3, 5, 3]} intensity={1.2} color={color} />
      <mesh rotation={[-Math.PI / 3, 0, 0]}>
        <planeGeometry args={[4, 4, 40, 40]} />
        <meshStandardMaterial
          displacementMap={texture}
          displacementScale={0.8}
          color={color}
          wireframe={true}
          roughness={0.3}
        />
      </mesh>
    </Canvas>
  );
};

export const UseCaseCards: React.FC = () => {
  const useCases = [
    {
      id: 'flood',
      preset: 'flood' as TerrainPreset,
      color: '#4d6dff',
      icon: Waves,
      title: 'Flood Inundation & Basin Mapping',
      subtitle: 'Hydrological Emergency Response',
      desc: 'Rapid elevation profiling identifies submerged road networks, standing water depth contours, and safe evacuation corridors during severe monsoon flash floods.',
      impact: '84% faster inundation analysis',
      metric: '0.15m Depth Precision'
    },
    {
      id: 'landslide',
      preset: 'landslide' as TerrainPreset,
      color: '#8b5cf6',
      icon: Mountain,
      title: 'Landslide Debris Volumetrics',
      subtitle: 'Geological Hazard Mitigation',
      desc: '3D volume estimation of collapsed hillside slopes enables heavy machinery deployment teams to accurately gauge earth clearing requirements.',
      impact: '3.2k m³ rubble calculated in <2 mins',
      metric: '94.8% Volumetric Accuracy'
    },
    {
      id: 'collapse',
      preset: 'collapse' as TerrainPreset,
      color: '#e952d1',
      icon: Building2,
      title: 'Structural Void & Rubble Detection',
      subtitle: 'Urban Search & Rescue (USAR)',
      desc: 'Post-earthquake aerial imagery turns collapsed multi-story structures into 3D height maps to locate structural void pockets for trapped survivor extraction.',
      impact: 'Detects void cavities <50cm wide',
      metric: 'Sub-second Void Location'
    },
    {
      id: 'relief',
      preset: 'mountain' as TerrainPreset,
      color: '#38bdf8',
      icon: Compass,
      title: 'Autonomous UAV Relief Pathfinding',
      subtitle: 'Tactical Drone Logistics',
      desc: 'Generates 3D spatial obstacle maps so autonomous medical payload drones can safely navigate complex mountain valleys and cut-off disaster sectors.',
      impact: 'Zero-collision flight corridor generator',
      metric: '100% Autonomous Waypoints'
    }
  ];

  return (
    <section id="use-cases" className="relative py-28 bg-[#0d0d18]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-900/80 border border-violet-500/30 text-violet-300 text-xs font-mono mb-4">
            <ShieldAlert className="w-3.5 h-3.5 text-magenta-pink" />
            <span>Disaster Response Applications</span>
          </div>
          <h2 className="font-heading text-3xl sm:text-5xl font-extrabold text-white tracking-tight uppercase">
            Designed For <span className="gradient-text-neurona">Field Operations</span>
          </h2>
          <p className="mt-4 text-slate-300 text-base sm:text-lg">
            Empowering NDRF teams, state disaster management authorities, and UAV pilots with instant 3D tactical terrain intelligence.
          </p>
        </div>

        {/* 4 Card Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {useCases.map((item, idx) => {
            const Icon = item.icon;
            return (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: idx * 0.1 }}
                whileHover={{ y: -6 }}
                className="glass-panel p-6 sm:p-8 rounded-3xl border border-violet-500/20 glass-panel-hover flex flex-col justify-between overflow-hidden relative group"
              >
                {/* Accent Top Gradient Line */}
                <div
                  className="absolute top-0 left-0 right-0 h-1 transition-all group-hover:h-1.5"
                  style={{ backgroundColor: item.color }}
                />

                {/* Card Top Header */}
                <div>
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-10 h-10 rounded-2xl flex items-center justify-center border"
                        style={{
                          backgroundColor: `${item.color}18`,
                          borderColor: `${item.color}40`,
                          color: item.color
                        }}
                      >
                        <Icon className="w-5 h-5" />
                      </div>
                      <div>
                        <span className="text-[10px] font-mono uppercase tracking-wider block" style={{ color: item.color }}>
                          {item.subtitle}
                        </span>
                        <h3 className="text-xl font-bold font-heading text-white group-hover:text-violet-300 transition-colors">
                          {item.title}
                        </h3>
                      </div>
                    </div>

                    <ArrowUpRight className="w-5 h-5 text-slate-600 group-hover:text-white transition-colors" />
                  </div>

                  <p className="text-sm text-slate-300 leading-relaxed mb-6">
                    {item.desc}
                  </p>
                </div>

                {/* Embedded 3D Micro-Canvas */}
                <div className="h-44 w-full rounded-2xl bg-slate-950/80 border border-slate-800 overflow-hidden relative my-4">
                  <MiniTerrainCanvas preset={item.preset} color={item.color} />
                  <div className="absolute top-2 left-2 px-2.5 py-1 rounded-lg bg-slate-900/90 text-[10px] font-mono text-slate-300 border border-slate-800">
                    Live 3D Geometry
                  </div>
                </div>

                {/* Metrics Footer */}
                <div className="flex items-center justify-between border-t border-slate-800/80 pt-4 text-xs font-mono">
                  <div className="flex items-center gap-1.5 text-magenta-pink font-semibold">
                    <TrendingUp className="w-3.5 h-3.5" />
                    <span>{item.impact}</span>
                  </div>
                  <div className="text-slate-400 bg-slate-900 px-3 py-1 rounded-lg border border-slate-800">
                    {item.metric}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
