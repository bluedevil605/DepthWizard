import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Eye, RotateCcw, Crosshair, Radio, ArrowLeft } from 'lucide-react';
import { generateMonochromeTerrainData, createReconDroneMesh, createCirclePointTexture } from '../utils/terrainPointGenerator';
import type { PointCloudData } from '../utils/terrainPointGenerator';
import { playScanPulseSound, playModeWhipSound } from '../utils/soundEffects';

// -------------------------------------------------------------
// 3D Monochrome Point Cloud & Terrain Mesh Component
// -------------------------------------------------------------
interface Terrain3DProps {
  data: PointCloudData;
  activeStage: number;
  terrainMode: string;
  mousePos: THREE.Vector2;
  isBreathing: boolean;
  isShattered: boolean;
}

const Terrain3D: React.FC<Terrain3DProps> = ({
  data,
  activeStage,
  terrainMode,
  mousePos,
  isBreathing,
  isShattered
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  const meshRef = useRef<THREE.Group>(null);
  const droneRef = useRef<THREE.Group>(null);
  const scanRingRef = useRef<THREE.Mesh>(null);
  const scanPulseScaleRef = useRef(0.1);

  // Recon drone model
  const droneMesh = useMemo(() => createReconDroneMesh(), []);

  // Smooth Circular Dot Texture Sprite
  const circleTexture = useMemo(() => createCirclePointTexture(), []);

  // Geometry attributes
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(data.positions.slice(), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(data.colors.slice(), 3));
    return geo;
  }, [data]);

  // Handle mode color alterations & torch light effect
  useFrame(({ clock }) => {
    const time = clock.getElapsedTime();

    if (meshRef.current && isBreathing) {
      // Subtle terrain breathing (sub-percent scale pulse)
      const scale = 1 + Math.sin(time * 0.8) * 0.008;
      meshRef.current.scale.set(scale, scale, scale);
    }

    // Drone flight path animation
    if (droneRef.current) {
      if (activeStage >= 2 && activeStage <= 5) {
        droneRef.current.position.x = Math.sin(time * 0.4) * 4;
        droneRef.current.position.y = 3.5 + Math.cos(time * 0.3) * 0.4;
        droneRef.current.position.z = Math.cos(time * 0.4) * 4;
        droneRef.current.rotation.y = time * 0.4 + Math.PI / 2;
      } else {
        // Trajectory orbit mode
        droneRef.current.position.x = Math.sin(time * 0.2) * 9;
        droneRef.current.position.y = 4.2 + Math.sin(time * 0.5) * 0.5;
        droneRef.current.position.z = Math.cos(time * 0.2) * 9;
        droneRef.current.rotation.y = time * 0.2 + Math.PI / 2;
      }
    }

    // Scan pulse ring expansion
    if (scanRingRef.current) {
      if (activeStage === 3 || activeStage === 4) {
        scanPulseScaleRef.current = (scanPulseScaleRef.current + 0.08) % 18;
        scanRingRef.current.scale.set(scanPulseScaleRef.current, scanPulseScaleRef.current, 1);
        (scanRingRef.current.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 1 - scanPulseScaleRef.current / 18);
      } else {
        scanRingRef.current.scale.set(0.01, 0.01, 1);
      }
    }

    // Point cloud position manipulation for Shatter & Cursor torch
    if (pointsRef.current) {
      const posAttr = pointsRef.current.geometry.attributes.position;
      const colAttr = pointsRef.current.geometry.attributes.color;
      const posArray = posAttr.array as Float32Array;
      const colArray = colAttr.array as Float32Array;

      for (let i = 0; i < data.positions.length / 3; i++) {
        const idx = i * 3;
        const origX = data.positions[idx];
        const origY = data.positions[idx + 1];
        const origZ = data.positions[idx + 2];

        if (isShattered) {
          // Shatter fragments outward
          posArray[idx] = origX + Math.sin(i + time * 5) * 1.5;
          posArray[idx + 1] = origY + Math.cos(i + time * 5) * 1.5;
          posArray[idx + 2] = origZ + Math.sin(i * 2 + time * 5) * 1.5;
        } else {
          posArray[idx] = origX;
          posArray[idx + 1] = origY;
          posArray[idx + 2] = origZ;
        }

        // Torch light effect near cursor
        const dx = origX - mousePos.x * 6;
        const dz = origZ - mousePos.y * 6;
        const dist = Math.sqrt(dx * dx + dz * dz);

        let baseColor = data.colors[idx];
        if (terrainMode === 'RISK') {
          // Risk mode uses warm off-white #F2E9DC for steep hazards
          if (origY > 1.2) {
            colArray[idx] = 0.95;     // #F2E9DC
            colArray[idx + 1] = 0.91;
            colArray[idx + 2] = 0.86;
          } else {
            colArray[idx] = 0.25;
            colArray[idx + 1] = 0.25;
            colArray[idx + 2] = 0.25;
          }
        } else if (terrainMode === 'DEPTH') {
          const depthVal = Math.min(1, Math.max(0.1, (origZ + 8) / 16));
          colArray[idx] = depthVal;
          colArray[idx + 1] = depthVal;
          colArray[idx + 2] = depthVal;
        } else {
          // Default elevation brightness + Cursor Torch boost
          if (dist < 2.5) {
            baseColor = Math.min(1, baseColor + (2.5 - dist) * 0.25);
          }
          colArray[idx] = baseColor;
          colArray[idx + 1] = baseColor;
          colArray[idx + 2] = baseColor;
        }
      }
      posAttr.needsUpdate = true;
      colAttr.needsUpdate = true;
    }
  });

  return (
    <group ref={meshRef}>
      {/* Dense Monochrome Circular Point Cloud (Dots only - no connecting lines!) */}
      <points ref={pointsRef} geometry={geometry}>
        <pointsMaterial
          size={0.12}
          map={circleTexture}
          vertexColors={true}
          transparent={true}
          alphaTest={0.2}
          depthWrite={false}
          opacity={activeStage >= 3 ? 0.95 : 0.2}
        />
      </points>

      {/* Rivers of Light Drainage Lines (Moonlit Circular Particles) */}
      {activeStage >= 5 && (
        <points>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[data.riverParticles, 3]} />
          </bufferGeometry>
          <pointsMaterial
            size={0.14}
            map={circleTexture}
            color="#FFFFFF"
            transparent={true}
            alphaTest={0.2}
            opacity={0.85}
          />
        </points>
      )}

      {/* Hairline Bounding Box Volume */}
      <lineSegments position={[0, 0, 0]}>
        <edgesGeometry args={[new THREE.BoxGeometry(16, 6, 16)]} />
        <lineBasicMaterial color="#2A2A2A" transparent={true} opacity={0.5} />
      </lineSegments>

      {/* Recon Drone Craft Mesh */}
      <group ref={droneRef}>
        <primitive object={droneMesh} />
      </group>

      {/* Concentric Scan Pulse Ring */}
      <mesh ref={scanRingRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
        <ringGeometry args={[0.95, 1.0, 64]} />
        <meshBasicMaterial color="#FFFFFF" transparent={true} opacity={0.8} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
};

// -------------------------------------------------------------
// Drone POV & Clamped Camera Controller
// -------------------------------------------------------------
interface CameraControllerProps {
  activeStage: number;
  isDronePOV: boolean;
  isDiving: boolean;
}

const CameraController: React.FC<CameraControllerProps> = ({ activeStage, isDronePOV, isDiving }) => {
  const { camera } = useThree();
  const scrollRatioRef = useRef(0);

  useEffect(() => {
    const handleScroll = () => {
      const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      scrollRatioRef.current = window.scrollY / maxScroll;
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useFrame(({ clock }) => {
    const time = clock.getElapsedTime();
    const scrollRatio = scrollRatioRef.current;

    if (isDiving) {
      // Clamped Enter Terrain Dive position (Zooms in smoothly to a fixed close-up, never unclamped)
      const targetPos = new THREE.Vector3(0, 1.1, 2.8);
      const targetLook = new THREE.Vector3(0, 0.3, 0);

      camera.position.lerp(targetPos, 0.05);
      camera.lookAt(targetLook);
    } else if (isDronePOV) {
      // DRONE POV: low altitude with stabilization jitter
      const jitterX = Math.sin(time * 8) * 0.03;
      const jitterY = Math.cos(time * 7) * 0.03;
      const targetPos = new THREE.Vector3(Math.sin(time * 0.3) * 5 + jitterX, 1.8 + jitterY, Math.cos(time * 0.3) * 5);
      camera.position.lerp(targetPos, 0.05);
      camera.lookAt(0, 0.2, 0);
    } else {
      // Stage sequence camera flight with scroll parallax effect across full background
      if (activeStage <= 2) {
        camera.position.set(0, 2.5 + scrollRatio * 2, 14 - scrollRatio * 2);
        camera.lookAt(0, 1, 0);
      } else if (activeStage === 3 || activeStage === 4) {
        camera.position.set(4, 5, 11);
        camera.lookAt(0, 0, 0);
      } else {
        // Stage 6+ Continuous dynamic orbit reacting to scroll position across full background
        const angle = time * 0.06 + scrollRatio * Math.PI * 0.7;
        const radius = 13 + Math.sin(scrollRatio * Math.PI) * 2;
        const height = 8 - scrollRatio * 3.5 + Math.sin(time * 0.4) * 0.4;
        const targetPos = new THREE.Vector3(Math.sin(angle) * radius, Math.max(3.5, height), Math.cos(angle) * radius);

        camera.position.lerp(targetPos, 0.04);
        camera.lookAt(0, scrollRatio * 1.5, 0);
      }
    }
  });

  return null;
};

// -------------------------------------------------------------
// Main DronePassHero3D Component
// -------------------------------------------------------------
interface DronePassHero3DProps {
  onEnterTerrain: () => void;
  onStartReconstruction: () => void;
  terrainMode: string;
}

export const DronePassHero3D: React.FC<DronePassHero3DProps> = ({
  onEnterTerrain,
  onStartReconstruction,
  terrainMode
}) => {
  const [activeStage, setActiveStage] = useState(1);
  const [isDronePOV, setIsDronePOV] = useState(false);
  const [isBreathing, setIsBreathing] = useState(true);
  const [isShattered, setIsShattered] = useState(false);
  const [isDiving, setIsDiving] = useState(false);
  const [mousePos, setMousePos] = useState<THREE.Vector2>(new THREE.Vector2(0, 0));

  const terrainData = useMemo(() => generateMonochromeTerrainData(120, 16), []);

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

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    setMousePos(new THREE.Vector2(x, y));
  };

  const handleResetView = () => {
    playModeWhipSound();
    setIsShattered(true);
    setIsDiving(false);
    setIsDronePOV(false);
    setTimeout(() => setIsShattered(false), 500);
  };

  const handleDiveClick = () => {
    setIsDiving(!isDiving);
    if (!isDiving) {
      onEnterTerrain();
    }
  };

  const stageNames = [
    '01. SILENCE',
    '02. CRAFT ENTRY',
    '03. SCAN PULSE',
    '04. GRID SURFACE',
    '05. RIVERS OF LIGHT',
    '06. PULL BACK',
    '07. TERRAIN DATASET READY'
  ];

  return (
    <section id="hero" className="relative min-h-screen w-full flex flex-col items-center justify-center overflow-hidden bg-transparent pt-16 pb-12" onMouseMove={handleMouseMove}>
      {/* Background Monochrome Grid - Fixed Fullscreen */}
      <div className="fixed inset-0 bg-grid-mono opacity-25 pointer-events-none z-0" />

      {/* 3D R3F WebGL Stage - FIXED FULLSCREEN BACKGROUND */}
      <div className="fixed inset-0 z-0 pointer-events-auto overflow-hidden">
        <Canvas camera={{ position: [0, 4, 13], fov: 45 }}>
          <ambientLight intensity={0.25} />
          <directionalLight position={[10, 15, 10]} intensity={1.2} color="#FFFFFF" />
          <pointLight position={[-10, -5, -10]} intensity={0.5} color="#B4B4B4" />

          <Terrain3D
            data={terrainData}
            activeStage={activeStage}
            terrainMode={terrainMode}
            mousePos={mousePos}
            isBreathing={isBreathing}
            isShattered={isShattered}
          />

          <CameraController activeStage={activeStage} isDronePOV={isDronePOV} isDiving={isDiving} />
        </Canvas>
      </div>

      {/* Sequence HUD Stage Readout (Top Left) */}
      <div className="absolute top-20 left-6 z-20 hidden sm:block">
        <div className="mono-panel p-3 rounded-none border border-[#2A2A2A] text-[11px] text-[#B4B4B4] space-y-1 font-mono">
          <div className="flex items-center gap-2 text-[#E8E8E8] font-semibold">
            <Radio className="w-3.5 h-3.5 text-[#FFFFFF] animate-pulse" />
            <span>RECON PASS SEQUENCE</span>
          </div>
          <div className="text-[#6E6E6E]">{isDiving ? 'ENTERED TERRAIN DIVE NODE' : stageNames[activeStage - 1]}</div>
          <div className="w-32 h-1 bg-[#2A2A2A] overflow-hidden mt-1">
            <div
              className="h-full bg-[#E8E8E8] transition-all duration-300"
              style={{ width: `${isDiving ? 100 : (activeStage / 7) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Top Monospace Telemetry HUD Strip */}
      <div className="absolute top-20 right-6 z-20 hidden md:block">
        <div className="mono-panel p-3.5 rounded-none border border-[#2A2A2A] text-right font-mono space-y-2">
          <div className="text-xl font-bold text-[#E8E8E8] tracking-widest">
            {isDiving ? '0210m' : '1842m'} &nbsp; {isDiving ? '12.1°' : '37.4°'} &nbsp; 94.6%
          </div>
          <div className="text-[10px] text-[#6E6E6E] tracking-widest uppercase flex justify-end gap-6">
            <span>ALTITUDE</span>
            <span>SLOPE</span>
            <span>CONFIDENCE</span>
          </div>
        </div>
      </div>

      {/* Hero Central Title Overlay */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 text-center flex flex-col items-center mt-12">
        <AnimatePresence>
          {activeStage >= 6 && !isDiving && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              className="flex flex-col items-center"
            >
              <div className="text-xs font-mono tracking-[0.3em] text-[#B4B4B4] uppercase mb-4 border-b border-[#2A2A2A] pb-1">
                DEPTH WIZARD &nbsp;•&nbsp; FIELD NODE 01
              </div>

              <h1 className="font-heading text-5xl sm:text-7xl md:text-8xl font-bold text-[#FFFFFF] tracking-tight uppercase leading-[0.95] mb-6">
                ONE PICTURE.<br />
                <span className="text-[#B4B4B4]">ONE TERRAIN.</span>
              </h1>

              <p className="text-sm sm:text-base text-[#B4B4B4] max-w-xl font-mono leading-relaxed mb-8">
                Monocular depth estimation & volumetric field reconstruction. Instant explorable 3D topography from a single uncalibrated video pass.
              </p>

              <div className="flex flex-wrap items-center justify-center gap-4">
                <button
                  onClick={onStartReconstruction}
                  className="px-6 py-3.5 bg-[#E8E8E8] text-[#050505] font-mono text-xs font-bold uppercase tracking-wider hover:bg-[#FFFFFF] transition-all flex items-center gap-2 border border-[#FFFFFF]"
                >
                  <Play className="w-4 h-4 fill-[#050505]" />
                  <span>[ START RECONSTRUCTION ]</span>
                </button>

                <button
                  onClick={handleDiveClick}
                  className="px-6 py-3.5 bg-[#0D0D0D] text-[#E8E8E8] font-mono text-xs font-bold uppercase tracking-wider border border-[#2A2A2A] hover:border-[#6E6E6E] hover:bg-[#151515] transition-all flex items-center gap-2"
                >
                  <Crosshair className="w-4 h-4 text-[#B4B4B4]" />
                  <span>[ ENTER TERRAIN DIVE ]</span>
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Exit Dive Floating HUD button when Diving */}
        {isDiving && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-8"
          >
            <button
              onClick={() => setIsDiving(false)}
              className="px-6 py-3 bg-[#E8E8E8] text-[#050505] font-mono text-xs font-bold uppercase tracking-widest hover:bg-[#FFFFFF] transition-all flex items-center gap-2 border border-[#FFFFFF]"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>[ EXIT TERRAIN DIVE ]</span>
            </button>
          </motion.div>
        )}
      </div>

      {/* Floating 3D Controls Bar */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 max-w-full px-4">
        <div className="mono-panel p-2 rounded-none border border-[#2A2A2A] flex flex-wrap items-center justify-center gap-3 text-xs font-mono">
          <button
            onClick={() => setIsDronePOV(!isDronePOV)}
            className={`px-3 py-1.5 border transition-all flex items-center gap-1.5 ${
              isDronePOV
                ? 'bg-[#E8E8E8] text-[#050505] border-[#FFFFFF] font-bold'
                : 'bg-[#0D0D0D] text-[#B4B4B4] border-[#2A2A2A] hover:border-[#6E6E6E]'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>DRONE POV: {isDronePOV ? 'ON' : 'OFF'}</span>
          </button>

          <button
            onClick={handleResetView}
            className="px-3 py-1.5 bg-[#0D0D0D] text-[#B4B4B4] border border-[#2A2A2A] hover:border-[#6E6E6E] hover:text-[#FFFFFF] transition-all flex items-center gap-1.5"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>[ RESET VIEW ]</span>
          </button>

          <button
            onClick={() => setIsBreathing(!isBreathing)}
            className={`px-3 py-1.5 border transition-all ${
              isBreathing
                ? 'bg-[#151515] text-[#E8E8E8] border-[#6E6E6E]'
                : 'bg-[#0D0D0D] text-[#6E6E6E] border-[#2A2A2A]'
            }`}
          >
            BREATHING: {isBreathing ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>
    </section>
  );
};
