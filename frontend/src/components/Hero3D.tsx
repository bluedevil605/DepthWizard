import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { EffectComposer, Bloom, DepthOfField } from '@react-three/postprocessing';
import * as THREE from 'three';
import { motion } from 'framer-motion';
import { Play, Layers, Cpu, Sparkles } from 'lucide-react';
import { createTerrainTexture } from '../utils/depthMapGenerator';
import type { TerrainPreset } from '../utils/depthMapGenerator';
import { isWebGLAvailable } from '../utils/webglDetect';
import { WebGLFallback } from './WebGLFallback';

// -------------------------------------------------------------
// 3D Terrain Mesh Component
// -------------------------------------------------------------
interface TerrainMeshProps {
  preset: TerrainPreset;
  wireframe: boolean;
}

const TerrainMesh: React.FC<TerrainMeshProps> = ({ preset, wireframe }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  // Generate heightmap texture dynamically on preset change
  const texture = useMemo(() => {
    return createTerrainTexture(preset, 256);
  }, [preset]);

  // Subtle wave animation over time
  useFrame(({ clock }) => {
    if (meshRef.current) {
      const elapsedTime = clock.getElapsedTime();
      meshRef.current.rotation.z = Math.sin(elapsedTime * 0.1) * 0.05;
    }
  });

  return (
    <group rotation={[-Math.PI / 2.8, 0, 0]} position={[0, -2, -2]}>
      {/* Primary Displaced Mesh */}
      <mesh ref={meshRef} receiveShadow castShadow>
        <planeGeometry args={[20, 20, 160, 160]} />
        <meshStandardMaterial
          displacementMap={texture}
          displacementScale={3.5}
          roughness={0.2}
          metalness={0.8}
          color="#06283d"
          wireframe={wireframe}
          emissive="#005b96"
          emissiveIntensity={0.2}
        />
      </mesh>

      {/* Glowing Wireframe Contour Grid */}
      <mesh position={[0, 0, 0.05]}>
        <planeGeometry args={[20.2, 20.2, 80, 80]} />
        <meshStandardMaterial
          displacementMap={texture}
          displacementScale={3.52}
          color="#00d4ff"
          wireframe={true}
          transparent={true}
          opacity={0.35}
          emissive="#00d4ff"
          emissiveIntensity={0.4}
        />
      </mesh>
    </group>
  );
};

// -------------------------------------------------------------
// Autonomous Drone Flythrough Camera Controller
// -------------------------------------------------------------
const DroneCameraController: React.FC = () => {
  const { camera } = useThree();
  const scrollRef = useRef(0);

  useEffect(() => {
    const handleScroll = () => {
      scrollRef.current = window.scrollY;
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useFrame(({ clock, mouse }) => {
    const time = clock.getElapsedTime();
    const scrollFactor = Math.min(scrollRef.current / 800, 1);

    // Combine autonomous drone pan/tilt with mouse position & scroll parallax
    const targetX = Math.sin(time * 0.2) * 2 + mouse.x * 1.5;
    const targetY = 4 + Math.cos(time * 0.15) * 0.8 - scrollFactor * 3 + mouse.y * 1.2;
    const targetZ = 10 - scrollFactor * 4;

    camera.position.x += (targetX - camera.position.x) * 0.03;
    camera.position.y += (targetY - camera.position.y) * 0.03;
    camera.position.z += (targetZ - camera.position.z) * 0.03;

    camera.lookAt(0, -1, -2);
  });

  return null;
};

// -------------------------------------------------------------
// Main Hero Component
// -------------------------------------------------------------
export const Hero3D: React.FC = () => {
  const [hasWebGL, setHasWebGL] = useState(true);
  const [preset, setPreset] = useState<TerrainPreset>('flood');
  const [wireframe, setWireframe] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    setHasWebGL(isWebGLAvailable());
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return (
    <section id="hero" className="relative min-h-screen w-full flex items-center justify-center overflow-hidden pt-20 pb-16">
      {/* Background Gradients & Glow */}
      <div className="absolute inset-0 bg-[#06090e] bg-grid-pattern opacity-60 pointer-events-none" />
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full glow-spotlight pointer-events-none" />

      {/* 3D WebGL Canvas Layer or Mobile / 2D Fallback */}
      <div className="absolute inset-0 z-0">
        {hasWebGL && !isMobile ? (
          <Canvas
            shadows
            camera={{ position: [0, 5, 10], fov: 45 }}
            gl={{ antialias: true, alpha: true }}
          >
            <fog attach="fog" args={['#06090e', 8, 26]} />
            <ambientLight intensity={0.4} />
            <directionalLight position={[10, 20, 15]} intensity={1.5} color="#00d4ff" castShadow />
            <directionalLight position={[-10, -10, -5]} intensity={0.8} color="#2dd4bf" />
            <pointLight position={[0, 2, 2]} intensity={2} color="#00d4ff" distance={15} />

            <TerrainMesh preset={preset} wireframe={wireframe} />
            <DroneCameraController />

            <EffectComposer>
              <Bloom intensity={1.1} luminanceThreshold={0.2} luminanceSmoothing={0.9} height={300} />
              <DepthOfField focusDistance={0.01} focalLength={0.02} bokehScale={2.5} height={480} />
            </EffectComposer>
          </Canvas>
        ) : (
          <WebGLFallback />
        )}
      </div>

      {/* Content Overlay */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 text-center flex flex-col items-center">
        {/* Hackathon Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-300 text-xs font-mono tracking-wider uppercase backdrop-blur-md mb-6 shadow-[0_0_20px_rgba(0,212,255,0.2)]"
        >
          <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
          <span>Smart India Hackathon • AI Disaster Management</span>
        </motion.div>

        {/* Hero Main Heading */}
        <motion.h1
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="font-heading text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight text-white max-w-4xl leading-[1.1]"
        >
          Single 2D Photo into <br />
          <span className="text-gradient-cyan">Navigable 3D Depth Landscape</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mt-6 text-lg sm:text-xl text-slate-300 max-w-2xl font-normal leading-relaxed"
        >
          <strong className="text-cyan-400 font-semibold">DepthWizard</strong> applies real-time neural height estimation to aerial drone photos, generating instant 3D terrain flythroughs for rapid emergency flood & landslide response.
        </motion.p>

        {/* CTA Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-10 flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto"
        >
          <a
            href="#live-demo"
            className="w-full sm:w-auto px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-400 text-black font-semibold text-sm uppercase tracking-wider shadow-[0_0_30px_rgba(0,212,255,0.4)] hover:shadow-[0_0_45px_rgba(0,212,255,0.6)] hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-2 group"
          >
            <Play className="w-4 h-4 fill-black group-hover:translate-x-0.5 transition-transform" />
            <span>Try Live 3D Engine</span>
          </a>

          <a
            href="#architecture"
            className="w-full sm:w-auto px-8 py-4 rounded-xl border border-slate-700/80 bg-slate-900/60 text-slate-200 font-semibold text-sm tracking-wider hover:border-cyan-500/40 hover:bg-slate-800/80 backdrop-blur-md transition-all flex items-center justify-center gap-2"
          >
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span>View Architecture</span>
          </a>
        </motion.div>

        {/* 3D Terrain Controller Bar (Interactive Preset Switcher) */}
        {hasWebGL && !isMobile && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="mt-14 glass-panel p-2 rounded-2xl flex flex-wrap items-center justify-center gap-2 border border-cyan-500/20"
          >
            <span className="text-xs font-mono text-slate-400 px-3 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              Simulated Terrain:
            </span>
            {(['flood', 'landslide', 'collapse', 'mountain'] as TerrainPreset[]).map((p) => (
              <button
                key={p}
                onClick={() => setPreset(p)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition-all ${
                  preset === p
                    ? 'bg-cyan-500 text-black font-semibold shadow-[0_0_15px_rgba(0,212,255,0.4)]'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                {p} Zone
              </button>
            ))}
            <div className="h-4 w-[1px] bg-slate-700 mx-1 hidden sm:block" />
            <button
              onClick={() => setWireframe(!wireframe)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                wireframe
                  ? 'bg-teal-500/20 text-teal-300 border border-teal-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Wireframe: {wireframe ? 'ON' : 'OFF'}
            </button>
          </motion.div>
        )}
      </div>

      {/* Scroll Down Indicator */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 opacity-60 hover:opacity-100 transition-opacity pointer-events-none">
        <span className="text-[10px] font-mono text-cyan-400 tracking-widest uppercase">Scroll To Explore</span>
        <div className="w-5 h-9 rounded-full border-2 border-cyan-500/30 flex items-start justify-center p-1">
          <div className="w-1 h-2 rounded-full bg-cyan-400 animate-bounce" />
        </div>
      </div>
    </section>
  );
};
