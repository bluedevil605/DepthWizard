import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { motion } from 'framer-motion';
import { Sparkles, Play, Cpu, Activity, ShieldCheck } from 'lucide-react';
import { createTerrainTexture } from '../utils/depthMapGenerator';
import { isWebGLAvailable } from '../utils/webglDetect';
import { WebGLFallback } from './WebGLFallback';

// -------------------------------------------------------------
// Ford Mustang Reveal: Draped Cloth & Photo-to-3D Extrusion Mesh
// -------------------------------------------------------------
const ClothAndExtrudedTerrain: React.FC<{ scrollProgress: number }> = ({ scrollProgress }) => {
  const clothRef = useRef<THREE.Mesh>(null);
  const terrainRef = useRef<THREE.Mesh>(null);

  const texture = useMemo(() => createTerrainTexture('flood', 256), []);

  useFrame(({ clock }) => {
    const time = clock.getElapsedTime();
    // Cloth lifting/wind animation driven by scroll progress + time
    const liftFactor = Math.min(scrollProgress * 2.2, 1);

    if (clothRef.current) {
      clothRef.current.position.y = 1.2 + liftFactor * 6;
      clothRef.current.position.z = -1 - liftFactor * 4;
      clothRef.current.rotation.x = -Math.PI / 3 - liftFactor * 0.8;
      clothRef.current.rotation.z = Math.sin(time * 2) * 0.1 * (1 - liftFactor);
    }

    if (terrainRef.current) {
      // Photo lifts off pedestal & extrudes heightmap as cloth lifts!
      const extrusionScale = liftFactor * 3.2;
      (terrainRef.current.material as THREE.MeshStandardMaterial).displacementScale = extrusionScale;
      terrainRef.current.rotation.z = Math.sin(time * 0.15) * 0.04;
    }
  });

  return (
    <group position={[0, -1.2, 0]}>
      {/* Pedestal Base */}
      <mesh position={[0, -0.6, 0]} receiveShadow>
        <cylinderGeometry args={[5, 5.5, 0.4, 32]} />
        <meshStandardMaterial color="#0d0d18" roughness={0.6} metalness={0.8} />
      </mesh>

      {/* Extruded 3D Disaster Terrain (Flat Photo -> 3D Landscape) */}
      <mesh ref={terrainRef} rotation={[-Math.PI / 2.8, 0, 0]} position={[0, 0, 0]} castShadow receiveShadow>
        <planeGeometry args={[12, 12, 128, 128]} />
        <meshStandardMaterial
          displacementMap={texture}
          displacementScale={0.1}
          roughness={0.2}
          metalness={0.7}
          color="#121829"
          emissive="#4d6dff"
          emissiveIntensity={0.25}
        />
      </mesh>

      {/* Contour Wireframe Elevation Highlight Grid */}
      <mesh rotation={[-Math.PI / 2.8, 0, 0]} position={[0, 0.05, 0]}>
        <planeGeometry args={[12.1, 12.1, 64, 64]} />
        <meshStandardMaterial
          displacementMap={texture}
          displacementScale={3.25}
          color="#e952d1"
          wireframe={true}
          transparent={true}
          opacity={0.35}
          emissive="#8b5cf6"
          emissiveIntensity={0.5}
        />
      </mesh>

      {/* Draped Wind-Blown Cloth (Lift Reveal) */}
      <mesh ref={clothRef} rotation={[-Math.PI / 3, 0, 0]} position={[0, 1.2, -1]} castShadow>
        <planeGeometry args={[14, 14, 32, 32]} />
        <meshStandardMaterial
          color="#1e1b4b"
          roughness={0.8}
          metalness={0.2}
          side={THREE.DoubleSide}
          transparent={true}
          opacity={0.85}
          emissive="#4d6dff"
          emissiveIntensity={0.1}
        />
      </mesh>
    </group>
  );
};

// -------------------------------------------------------------
// Dust Particles floating in diagonal spotlight beam
// -------------------------------------------------------------
const SpotlightDustParticles: React.FC = () => {
  const particlesRef = useRef<THREE.Points>(null);

  const [positions, colors] = useMemo(() => {
    const count = 180;
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    const c1 = new THREE.Color('#4d6dff');
    const c2 = new THREE.Color('#8b5cf6');
    const c3 = new THREE.Color('#e952d1');

    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 16;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 12 + 2;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 16;

      const rand = Math.random();
      const chosenColor = rand < 0.33 ? c1 : rand < 0.66 ? c2 : c3;
      col[i * 3] = chosenColor.r;
      col[i * 3 + 1] = chosenColor.g;
      col[i * 3 + 2] = chosenColor.b;
    }
    return [pos, col];
  }, []);

  useFrame(({ clock }) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y = clock.getElapsedTime() * 0.03;
    }
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.08} vertexColors={true} transparent={true} opacity={0.7} />
    </points>
  );
};

// -------------------------------------------------------------
// Neurona Live Stat Counter Card (Bottom-Right)
// -------------------------------------------------------------
const StatCounterCard: React.FC = () => {
  const [count, setCount] = useState(14892);

  useEffect(() => {
    const interval = setInterval(() => {
      setCount((prev) => prev + Math.floor(Math.random() * 3 + 1));
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass-panel p-4 rounded-2xl border border-violet-500/30 shadow-[0_0_25px_rgba(139,92,246,0.25)] flex flex-col gap-2 w-64 backdrop-blur-xl">
      <div className="flex items-center justify-between text-xs font-mono text-slate-400">
        <span className="flex items-center gap-1.5 text-violet-400">
          <Activity className="w-3.5 h-3.5 animate-pulse" /> Live Telemetry
        </span>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-300 border border-violet-500/40">
          GLOBAL
        </span>
      </div>

      <div className="flex items-baseline justify-between">
        <span className="text-2xl font-extrabold font-heading text-white tracking-tight">
          {count.toLocaleString()}
        </span>
        <span className="text-xs font-mono text-emerald-400 font-semibold flex items-center gap-0.5">
          +18.4%
        </span>
      </div>
      <div className="text-[11px] font-mono text-slate-400 -mt-1">
        3D Models Reconstructed
      </div>

      {/* Live Animated SVG Sparkline */}
      <div className="h-8 w-full mt-1">
        <svg className="w-full h-full overflow-visible" viewBox="0 0 100 24">
          <defs>
            <linearGradient id="sparklineGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#4d6dff" />
              <stop offset="50%" stopColor="#8b5cf6" />
              <stop offset="100%" stopColor="#e952d1" />
            </linearGradient>
          </defs>
          <path
            d="M 0 18 Q 15 8, 30 14 T 60 6 T 90 10 L 100 4"
            fill="none"
            stroke="url(#sparklineGrad)"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
        </svg>
      </div>
    </div>
  );
};

// -------------------------------------------------------------
// Main Cloth Reveal & Hero Component
// -------------------------------------------------------------
export const ClothReveal3D: React.FC = () => {
  const [hasWebGL, setHasWebGL] = useState(true);
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    setHasWebGL(isWebGLAvailable());
    const handleScroll = () => {
      const p = Math.min(window.scrollY / 600, 1);
      setScrollProgress(p);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <section id="hero" className="relative min-h-screen w-full flex items-center justify-center overflow-hidden pt-20 pb-16 bg-[#0a0a12]">
      {/* Dark Studio Diagonal Spotlight Glow */}
      <div className="absolute inset-0 spotlight-beam pointer-events-none" />
      <div className="absolute inset-0 bg-grid-studio opacity-40 pointer-events-none" />

      {/* 3D WebGL Canvas Layer */}
      <div className="absolute inset-0 z-0">
        {hasWebGL ? (
          <Canvas camera={{ position: [0, 3, 10], fov: 45 }}>
            <fog attach="fog" args={['#0a0a12', 6, 24]} />
            <ambientLight intensity={0.3} />

            {/* Ford Mustang Style Spotlight Beam */}
            <spotLight
              position={[12, 18, 10]}
              angle={0.45}
              penumbra={0.8}
              intensity={3.5}
              color="#4d6dff"
              castShadow
            />
            <directionalLight position={[-10, 10, -5]} intensity={1.2} color="#e952d1" />
            <pointLight position={[0, 1, 3]} intensity={1.8} color="#8b5cf6" />

            <ClothAndExtrudedTerrain scrollProgress={scrollProgress} />
            <SpotlightDustParticles />

            <EffectComposer>
              <Bloom intensity={1.2} luminanceThreshold={0.2} luminanceSmoothing={0.9} />
            </EffectComposer>
          </Canvas>
        ) : (
          <WebGLFallback />
        )}
      </div>

      {/* Content Overlay */}
      <div className="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 w-full flex flex-col items-center text-center">
        {/* SIH 26175 Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-900/80 border border-violet-500/40 text-violet-300 text-xs font-mono tracking-wider uppercase backdrop-blur-md mb-8 shadow-[0_0_20px_rgba(139,92,246,0.3)]"
        >
          <Sparkles className="w-3.5 h-3.5 text-magenta-pink animate-pulse" />
          <span>SIH26175 • Single-View 3D Disaster Terrain AI</span>
        </motion.div>

        {/* Neurona Style Stacked Three-Word Headline */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="font-heading text-5xl sm:text-7xl md:text-8xl font-extrabold tracking-tight uppercase leading-[0.95] text-white max-w-4xl"
        >
          <div className="text-white">CAPTURE.</div>
          <div className="gradient-text-neurona">RECONSTRUCT.</div>
          <div className="text-white">FLY.</div>
        </motion.div>

        {/* Cinematic Subtitle Reveal */}
        <motion.p
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25 }}
          className="mt-8 text-lg sm:text-2xl font-heading font-medium tracking-wide max-w-2xl text-slate-300"
        >
          SEE BEYOND THE SURFACE — <span className="gradient-text-neurona font-semibold">DepthWizard</span> lifts flat 2D disaster imagery into volumetric 3D landscapes in real time.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.35 }}
          className="mt-10 flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto"
        >
          <a
            href="#how-it-works"
            className="w-full sm:w-auto px-8 py-4 rounded-xl bg-gradient-neurona text-white font-semibold text-xs uppercase tracking-wider shadow-[0_0_30px_rgba(139,92,246,0.5)] hover:shadow-[0_0_45px_rgba(233,82,209,0.7)] hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-2 group"
          >
            <Play className="w-4 h-4 fill-white group-hover:translate-x-0.5 transition-transform" />
            <span>Launch Solving Engine</span>
          </a>

          <a
            href="#live-demo"
            className="w-full sm:w-auto px-8 py-4 rounded-xl border border-violet-500/40 bg-slate-900/60 text-slate-200 font-semibold text-xs uppercase tracking-wider hover:border-magenta-pink/60 hover:bg-slate-800/80 backdrop-blur-md transition-all flex items-center justify-center gap-2"
          >
            <Cpu className="w-4 h-4 text-violet-400" />
            <span>Try Live Uploader</span>
          </a>
        </motion.div>
      </div>

      {/* Floating Neurona Glass Hero Badges (Left & Right) */}
      <div className="hidden lg:block absolute left-8 top-1/3 z-20">
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="glass-panel p-3.5 rounded-2xl border border-blue-500/30 flex items-center gap-3 backdrop-blur-xl shadow-lg"
        >
          <div className="w-9 h-9 rounded-xl bg-blue-500/20 border border-blue-500/40 flex items-center justify-center text-blue-400">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs font-bold text-white">Depth Accuracy</div>
            <div className="text-[11px] font-mono text-blue-300">96.4% Precision</div>
          </div>
        </motion.div>
      </div>

      <div className="hidden lg:block absolute left-12 top-2/3 z-20">
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="glass-panel p-3.5 rounded-2xl border border-magenta-pink/30 flex items-center gap-3 backdrop-blur-xl shadow-lg"
        >
          <div className="w-9 h-9 rounded-xl bg-magenta-pink/20 border border-magenta-pink/40 flex items-center justify-center text-magenta-pink">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs font-bold text-white">Zero Calibration</div>
            <div className="text-[11px] font-mono text-pink-300">Single Photo Input</div>
          </div>
        </motion.div>
      </div>

      {/* Neurona Stat Counter Card (Bottom Right) */}
      <div className="hidden md:block absolute right-8 bottom-8 z-20">
        <StatCounterCard />
      </div>
    </section>
  );
};
