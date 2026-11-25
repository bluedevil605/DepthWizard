import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { Upload, Sliders, RefreshCw, Sparkles, AlertTriangle, Layers, FileImage } from 'lucide-react';
import { processImageToDepthMap } from '../utils/depthMapGenerator';
import type { ColorMode } from '../utils/depthMapGenerator';
import { getSampleDisasterImages } from '../utils/sampleData';
import type { SampleImage } from '../utils/sampleData';

// -------------------------------------------------------------
// Interactive 3D Displaced Mesh Component for Uploader Preview
// -------------------------------------------------------------
interface DemoDisplacedMeshProps {
  depthTexture: THREE.CanvasTexture | null;
  imageTexture: THREE.Texture | null;
  heightScale: number;
  wireframe: boolean;
  colorMode: ColorMode;
  autoRotate: boolean;
}

const DemoDisplacedMesh: React.FC<DemoDisplacedMeshProps> = ({
  depthTexture,
  imageTexture,
  heightScale,
  wireframe,
  colorMode,
  autoRotate
}) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (meshRef.current && autoRotate) {
      meshRef.current.rotation.z += 0.005;
    }
  });

  const materialProps = useMemo(() => {
    if (colorMode === 'realistic' && imageTexture) {
      return {
        map: imageTexture,
        displacementMap: depthTexture || undefined,
        displacementScale: heightScale,
        wireframe: wireframe,
        roughness: 0.4,
        metalness: 0.1
      };
    } else if (colorMode === 'thermal') {
      return {
        displacementMap: depthTexture || undefined,
        displacementScale: heightScale,
        wireframe: wireframe,
        color: '#e952d1',
        emissive: '#8b5cf6',
        emissiveIntensity: 0.4,
        roughness: 0.2
      };
    } else if (colorMode === 'elevation') {
      return {
        displacementMap: depthTexture || undefined,
        displacementScale: heightScale,
        wireframe: wireframe,
        color: '#4d6dff',
        emissive: '#1e1b4b',
        emissiveIntensity: 0.3,
        roughness: 0.3
      };
    } else {
      // Neurona default gradient cyan/magenta
      return {
        displacementMap: depthTexture || undefined,
        displacementScale: heightScale,
        wireframe: wireframe,
        color: '#8b5cf6',
        emissive: '#4d6dff',
        emissiveIntensity: 0.4,
        roughness: 0.2,
        metalness: 0.7
      };
    }
  }, [colorMode, depthTexture, imageTexture, heightScale, wireframe]);

  return (
    <group rotation={[-Math.PI / 3, 0, 0]} position={[0, -0.5, 0]}>
      <mesh ref={meshRef} castShadow receiveShadow>
        <planeGeometry args={[7, 7, 128, 128]} />
        <meshStandardMaterial {...materialProps} />
      </mesh>

      {/* Grid Floor */}
      <gridHelper args={[14, 28, '#8b5cf6', '#1e1b4b']} position={[0, 0, -0.2]} rotation={[Math.PI / 2, 0, 0]} />
    </group>
  );
};

// -------------------------------------------------------------
// Demo Uploader Main Component
// -------------------------------------------------------------
export const DemoUploader: React.FC = () => {
  const samples = useMemo(() => getSampleDisasterImages(), []);
  const [selectedSample, setSelectedSample] = useState<SampleImage>(samples[0]);
  const [depthTex, setDepthTex] = useState<THREE.CanvasTexture | null>(null);
  const [rgbTex, setRgbTex] = useState<THREE.Texture | null>(null);
  const [heightScale, setHeightScale] = useState<number>(1.8);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const [colorMode, setColorMode] = useState<ColorMode>('cyan');
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadAndProcessImage(selectedSample.dataUrl);
  }, [selectedSample]);

  const loadAndProcessImage = (src: string) => {
    setIsProcessing(true);
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.src = src;
    img.onload = () => {
      try {
        const processed = processImageToDepthMap(img, 1.3);
        setDepthTex(processed.texture);

        const textureLoader = new THREE.TextureLoader();
        textureLoader.load(src, (tex) => {
          setRgbTex(tex);
          setIsProcessing(false);
        });
      } catch (err) {
        console.error('Failed to process image depth map:', err);
        setIsProcessing(false);
      }
    };
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        const customUrl = event.target.result as string;
        setSelectedSample({
          id: 'custom-' + Date.now(),
          name: file.name,
          category: 'Uploaded 2D Photo',
          description: 'Client-side real-time luminance depth displacement map generated from user photo.',
          dataUrl: customUrl,
          preset: 'flood',
          depthMetrics: {
            maxElevation: 'Custom Peak',
            waterVolume: 'Estimated Vol',
            hazardIndex: 'Calculated',
            confidence: 'Realtime Mesh'
          }
        });
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <section id="live-demo" className="relative py-28 bg-[#0a0a12] border-b border-violet-500/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-900/80 border border-violet-500/30 text-violet-300 text-xs font-mono mb-4">
            <Sparkles className="w-3.5 h-3.5 text-magenta-pink" />
            <span>Interactive WebGL Sandbox</span>
          </div>
          <h2 className="font-heading text-3xl sm:text-5xl font-extrabold text-white tracking-tight uppercase">
            Live 3D Depth <span className="gradient-text-neurona">Preview Engine</span>
          </h2>
          <p className="mt-4 text-slate-300 text-base sm:text-lg">
            Upload any 2D drone photo or select a disaster scenario to instantly generate and explore its 3D depth mesh in your browser.
          </p>
        </div>

        {/* Main Demo Layout Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Preset Selector & Uploader (4 cols) */}
          <div className="lg:col-span-4 space-y-6">
            {/* Upload Zone Card */}
            <div className="glass-panel p-6 rounded-3xl border border-violet-500/20">
              <h3 className="text-xs font-mono font-semibold text-slate-200 uppercase tracking-wider mb-3 flex items-center justify-between">
                <span>1. Upload 2D Image</span>
                <span className="text-[10px] text-magenta-pink font-bold">Client-side Processing</span>
              </h3>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />

              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full py-8 px-4 rounded-2xl border-2 border-dashed border-violet-500/40 bg-slate-900/50 hover:bg-violet-500/10 hover:border-magenta-pink transition-all flex flex-col items-center justify-center gap-3 group"
              >
                <div className="w-12 h-12 rounded-full bg-violet-500/20 flex items-center justify-center text-magenta-pink group-hover:scale-110 transition-transform">
                  <Upload className="w-6 h-6" />
                </div>
                <div className="text-center">
                  <span className="text-sm font-semibold text-white group-hover:text-violet-300 transition-colors">
                    Click or Drag Image Here
                  </span>
                  <p className="text-xs text-slate-400 mt-1">
                    Supports PNG, JPG, WebP (Max 10MB)
                  </p>
                </div>
              </button>
            </div>

            {/* Disaster Presets Selector */}
            <div className="glass-panel p-6 rounded-3xl border border-violet-500/20">
              <h3 className="text-xs font-mono font-semibold text-slate-200 uppercase tracking-wider mb-4 flex items-center justify-between">
                <span>Select Preset Disaster</span>
                <FileImage className="w-4 h-4 text-violet-400" />
              </h3>

              <div className="space-y-3">
                {samples.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setSelectedSample(s)}
                    className={`w-full p-3 rounded-2xl border text-left transition-all flex items-center gap-3 ${
                      selectedSample.id === s.id
                        ? 'bg-violet-500/20 border-magenta-pink/60 shadow-[0_0_20px_rgba(233,82,209,0.25)]'
                        : 'bg-slate-900/50 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40'
                    }`}
                  >
                    <img
                      src={s.dataUrl}
                      alt={s.name}
                      className="w-12 h-12 rounded-xl object-cover border border-violet-500/30"
                    />
                    <div className="overflow-hidden">
                      <div className="text-[10px] font-mono text-magenta-pink uppercase tracking-wide truncate">
                        {s.category}
                      </div>
                      <div className="text-xs font-bold text-white truncate">
                        {s.name}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Metrics Panel */}
            <div className="glass-panel p-6 rounded-3xl border border-violet-500/20">
              <h3 className="text-xs font-mono text-violet-300 uppercase tracking-wider mb-4 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Disaster Elevation Assessment
              </h3>
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800">
                  <div className="text-slate-400">Max Elevation</div>
                  <div className="text-sm font-bold text-white mt-1">{selectedSample.depthMetrics.maxElevation}</div>
                </div>
                <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800">
                  <div className="text-slate-400">Volume</div>
                  <div className="text-sm font-bold text-violet-400 mt-1">{selectedSample.depthMetrics.waterVolume}</div>
                </div>
                <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800">
                  <div className="text-slate-400">Hazard Level</div>
                  <div className="text-xs font-bold text-amber-400 mt-1">{selectedSample.depthMetrics.hazardIndex}</div>
                </div>
                <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800">
                  <div className="text-slate-400">Precision</div>
                  <div className="text-sm font-bold text-magenta-pink mt-1">{selectedSample.depthMetrics.confidence}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Interactive 3D WebGL Canvas & Controls (8 cols) */}
          <div className="lg:col-span-8 flex flex-col">
            <div className="glass-panel rounded-3xl border border-violet-500/20 flex-1 flex flex-col overflow-hidden relative shadow-[0_0_35px_rgba(139,92,246,0.15)] min-h-[520px]">
              {/* Canvas Controls Header */}
              <div className="p-4 border-b border-violet-500/20 bg-slate-950/80 flex flex-wrap items-center justify-between gap-4 z-10">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-magenta-pink animate-ping" />
                  <span className="text-xs font-mono text-slate-200 font-semibold">
                    WebGL 3D Displaced Mesh
                  </span>
                </div>

                {/* Shading Color Map Selector */}
                <div className="flex items-center gap-1.5 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
                  {(['cyan', 'elevation', 'thermal', 'realistic'] as ColorMode[]).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setColorMode(mode)}
                      className={`px-3 py-1 rounded-lg text-[11px] font-mono capitalize transition-all ${
                        colorMode === mode
                          ? 'bg-gradient-neurona text-white font-semibold shadow-[0_0_10px_rgba(233,82,209,0.4)]'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              {/* 3D Canvas Viewport */}
              <div className="relative flex-1 bg-[#06060c]">
                {isProcessing && (
                  <div className="absolute inset-0 z-20 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
                    <RefreshCw className="w-8 h-8 text-magenta-pink animate-spin" />
                    <span className="text-xs font-mono text-violet-300">Extracting Depth Displacement Matrix...</span>
                  </div>
                )}

                <Canvas camera={{ position: [0, 4, 7], fov: 45 }}>
                  <ambientLight intensity={0.7} />
                  <directionalLight position={[10, 15, 10]} intensity={1.5} color="#4d6dff" />
                  <pointLight position={[-5, -2, -5]} intensity={0.9} color="#e952d1" />

                  <DemoDisplacedMesh
                    depthTexture={depthTex}
                    imageTexture={rgbTex}
                    heightScale={heightScale}
                    wireframe={wireframe}
                    colorMode={colorMode}
                    autoRotate={autoRotate}
                  />

                  <OrbitControls enableZoom={true} enablePan={true} maxPolarAngle={Math.PI / 2} />
                </Canvas>
              </div>

              {/* Bottom Interactive Sliders & Toggles Footer */}
              <div className="p-4 bg-slate-950/90 border-t border-violet-500/20 grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
                {/* Height Scale Slider */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="flex items-center gap-1">
                      <Sliders className="w-3.5 h-3.5 text-violet-400" /> Height Scale
                    </span>
                    <span className="text-magenta-pink font-bold">{heightScale.toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="4.0"
                    step="0.1"
                    value={heightScale}
                    onChange={(e) => setHeightScale(parseFloat(e.target.value))}
                    className="accent-magenta-pink cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                  />
                </div>

                {/* Wireframe Toggle */}
                <div className="flex items-center justify-between bg-slate-900/60 px-3.5 py-2 rounded-xl border border-slate-800">
                  <span className="text-slate-300 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-blue-400" /> Wireframe Grid
                  </span>
                  <button
                    onClick={() => setWireframe(!wireframe)}
                    className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${
                      wireframe ? 'bg-gradient-neurona text-white' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {wireframe ? 'ACTIVE' : 'OFF'}
                  </button>
                </div>

                {/* Auto Rotate Toggle */}
                <div className="flex items-center justify-between bg-slate-900/60 px-3.5 py-2 rounded-xl border border-slate-800">
                  <span className="text-slate-300 flex items-center gap-1.5">
                    <RefreshCw className="w-3.5 h-3.5 text-magenta-pink" /> Auto Orbit
                  </span>
                  <button
                    onClick={() => setAutoRotate(!autoRotate)}
                    className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${
                      autoRotate ? 'bg-violet-600 text-white' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {autoRotate ? 'ON' : 'PAUSED'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
