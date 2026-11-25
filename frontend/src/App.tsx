// @ts-nocheck
import React, { useState, useEffect, useRef } from 'react';
import { Navbar } from './components/Navbar';
import { UnifiedTerrainScene } from './components/UnifiedTerrainScene';
import { HeroOverlay } from './components/HeroOverlay';
import { ScanningChamber } from './components/ScanningChamber';
import { ReconstructionEngine } from './components/ReconstructionEngine';
import { TerrainModesBar } from './components/TerrainModesBar';
import { ExplodeView3D } from './components/ExplodeView3D';
import { TopoMapMode } from './components/TopoMapMode';
import { MeasurementTool } from './components/MeasurementTool';
import { TechBlueprint } from './components/TechBlueprint';
import { FilmGrainOverlay } from './components/FilmGrainOverlay';
import { FlashTransition } from './components/FlashTransition';
import { Footer } from './components/Footer';
import { ElevationHUD } from './components/ElevationHUD';
import { RenderLegend } from './components/RenderLegend';
import {
  fetchHealth,
  fetchSamples,
  processSample,
  uploadImageAndProcess,
  fetchMeshMeta,
  fetchValidationBundle,
  fetchValidationMetrics
} from './utils/api';

export default function App() {
  const [flashKey, setFlashKey] = useState(0);
  const [terrainMode, setTerrainMode] = useState('ELEVATION');
  const [activeSection, setActiveSection] = useState('overview');
  const [isMuted, setIsMuted] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [isDronePOV, setIsDronePOV] = useState(false);
  const [isBreathing, setIsBreathing] = useState(true);
  const [isDiving, setIsDiving] = useState(false);
  const [heatmapBounds, setHeatmapBounds] = useState<{ min: number; max: number } | null>(null);

  // DepthWizard Backend State
  const [backendOnline, setBackendOnline] = useState(false);
  const [samples, setSamples] = useState<any[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressMessage, setProgressMessage] = useState('');
  const [processingStage, setProcessingStage] = useState('READY');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [nadirWarning, setNadirWarning] = useState<any | null>(null);

  // Active Terrain State
  const [activeModel, setActiveModel] = useState<any>(null);
  const [meshMeta, setMeshMeta] = useState<any>(null);
  const [activeDatasetName, setActiveDatasetName] = useState('');
  const [terrainSource, setTerrainSource] = useState<'demo' | 'processed'>('demo');

  // Validation & Telemetry
  const [validationBundle, setValidationBundle] = useState<any>(null);
  const [benchmarkData, setBenchmarkData] = useState<any>(null);
  const [calibrationMode, setCalibrationMode] = useState('global');
  const [selectedDetection, setSelectedDetection] = useState<any>(null);
  const [hoverData, setHoverData] = useState<any>(null);
  const metaCache = useRef<Record<string, any>>({});

  // Track scroll progress for the overlay elements with rAF throttling
  useEffect(() => {
    let ticking = false;
    const handleScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
          setScrollProgress(window.scrollY / maxScroll);
          ticking = false;
        });
        ticking = true;
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Health check and sample loading
  useEffect(() => {
    let intervalId: any;
    let initialLoaded = false;

    async function checkAndInit() {
      try {
        await fetchHealth();
        setBackendOnline(true);
        fetchValidationMetrics().then(setBenchmarkData).catch(() => {});
        
        if (!initialLoaded) {
          initialLoaded = true;
          const sampleList = await fetchSamples();
          setSamples(sampleList);
        }
      } catch (err) {
        setBackendOnline(false);
      }
    }

    checkAndInit();

    intervalId = setInterval(async () => {
      try {
        await fetchHealth();
        setBackendOnline((wasOnline) => {
          if (!wasOnline && !initialLoaded) {
            initialLoaded = true;
            fetchSamples().then(setSamples).catch(() => {});
          }
          return true;
        });
      } catch {
        setBackendOnline(false);
      }
    }, 5000);

    return () => clearInterval(intervalId);
  }, []);

  const triggerWhiteFlash = () => {
    setFlashKey((prev) => prev + 1);
  };

  const handleStartReconstruction = () => {
    triggerWhiteFlash();
    setActiveSection('chamber');
  };

  const handleEnterTerrain = () => {
    triggerWhiteFlash();
    setIsDiving(true);
  };

  // Upload Flow
  const handleUpload = async (file: File, referenceFile: File | null = null, verticalExaggeration = 1.0, force = false) => {
    setIsProcessing(true);
    setErrorMessage(null);
    if (force) setNadirWarning(null);
    setProcessingStage('UPLOADING');
    setProgressMessage(
      force
        ? 'Forcing pipeline on non-nadir input (strict YOLO confidence floor)...'
        : 'Analyzing georeferencing & domain safety...'
    );
    triggerWhiteFlash();
    setActiveSection('chamber');

    // Visual timers to simulate UI progression while real backend works
    const t1 = setTimeout(() => {
      setProcessingStage('ANALYZING IMAGE');
      setProgressMessage('Decoding optical raster & checking georeference...');
    }, 400);
    const t2 = setTimeout(() => {
      setProcessingStage('DEPTH ESTIMATION');
      setProgressMessage('Running Depth Anything V2 monocular depth inference...');
    }, 1200);
    const t3 = setTimeout(() => {
      setProcessingStage('CALIBRATION');
      setProgressMessage('Applying structure-aware filtering & DSM calibration...');
    }, 2800);
    const t4 = setTimeout(() => {
      setProcessingStage('DSM GENERATION');
      setProgressMessage('Generating 3D terrain heightfield mesh & GLB...');
    }, 4000);

    try {
      const result = await uploadImageAndProcess(file, referenceFile, verticalExaggeration, 18.0, force);
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); clearTimeout(t4);

      if (result.status === 'warning') {
        setProcessingStage('READY');
        setNadirWarning({
          message: result.message,
          assessment: result.nadir_assessment,
          filename: result.filename || file.name,
          pendingUpload: { file, referenceFile, verticalExaggeration }
        });
        setIsProcessing(false);
        return;
      }

      setProcessingStage('LOADING TERRAIN');
      setProgressMessage('Loading generated 3D terrain into viewer...');

      setValidationBundle(result.validation_bundle || null);
      setBenchmarkData(result.validation_metrics_summary || result.validation_metrics || null);
      setActiveModel(result);
      setActiveDatasetName(result.filename || file.name);

      if (result.meta_url) {
        const meta = await fetchMeshMeta(result.meta_url);
        setMeshMeta(meta);
      }

      setTerrainSource('processed');
      setActiveSection('overview');
      setProcessingStage('COMPLETE');
      setProgressMessage('3D terrain model loaded successfully.');
      triggerWhiteFlash();
    } catch (err: any) {
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); clearTimeout(t4);
      console.error('[UPLOAD ERROR]', err);
      setProcessingStage('ERROR');
      setErrorMessage(`Processing failed: ${err.message || 'Failed to process uploaded image'}`);
    } finally {
      setTimeout(() => setIsProcessing(false), 2000);
    }
  };

  const currentBundleMode = validationBundle?.modes?.[calibrationMode];
  const effectiveGlbUrl = currentBundleMode?.glb_url || activeModel?.glb_url;
  const effectiveStats = currentBundleMode?.stats || activeModel?.stats;
  const effectiveValidationMetrics = calibrationMode === 'relative'
    ? { skipped: true, message: 'Relative rDSM mode: Metric LiDAR validation unavailable.' }
    : (currentBundleMode?.metrics || activeModel?.validation_metrics);

  const errorOverlayUrl = (calibrationMode !== 'relative') ? (currentBundleMode?.error_url || null) : null;
  const confidenceOverlayUrl = (calibrationMode !== 'relative') ? (currentBundleMode?.conf_url || null) : null;
  const elevationOverlayUrl = (calibrationMode !== 'relative') ? (currentBundleMode?.elev_url || null) : null;
  const slopeOverlayUrl = validationBundle?.slope_url || null;
  const referenceElevUrl = (calibrationMode !== 'relative') ? (validationBundle?.reference_elevation_url || null) : null;
  const probeGrid = validationBundle?.probe_grid || null;

  const modelLoaded = terrainSource === 'processed';

  const renderUnifiedTerrain = (embedded: boolean) => (
    <UnifiedTerrainScene
      terrainMode={terrainMode}
      isDronePOV={isDronePOV}
      isBreathing={isBreathing}
      isDiving={isDiving}
      terrainSource={terrainSource}
      glbUrl={effectiveGlbUrl}
      activeModel={activeModel}
      errorOverlayUrl={errorOverlayUrl}
      confidenceOverlayUrl={confidenceOverlayUrl}
      elevationOverlayUrl={elevationOverlayUrl}
      slopeOverlayUrl={slopeOverlayUrl}
      referenceElevUrl={referenceElevUrl}
      calibrationMode={calibrationMode}
      probeGrid={probeGrid}
      meta={meshMeta}
      detections={activeModel?.detections || []}
      selectedDetection={selectedDetection}
      onSelectDetection={setSelectedDetection}
      onHover={setHoverData}
      landCoverUrl={activeModel?.land_cover_mask_url}
      buildingMaskUrl={activeModel?.building_mask_url}
      buildingOverlayUrl={activeModel?.building_overlay_url}
      detectionHeatmapUrl={activeModel?.detection_heatmap_url}
      isEmbedded={embedded}
      onHeatmapBoundsChange={setHeatmapBounds}
    />
  );

  return (
    <div className="min-h-screen bg-[#211E1B] text-[#D9D2C2] font-mono selection:bg-[#E0433C] selection:text-[#FFFFFF] flex flex-col">
      <FlashTransition triggerKey={flashKey} />
      <FilmGrainOverlay isDronePOV={isDronePOV} terrainSource={terrainSource} />

      {/* GLOBAL BACKGROUND ONLY WHEN NOT IN MODEL VIEWER */}
      {!(modelLoaded && activeSection === 'overview') && renderUnifiedTerrain(false)}

      <Navbar 
        isMuted={isMuted} 
        onToggleMute={() => setIsMuted(!isMuted)} 
        activeSection={activeSection}
        onNavChange={setActiveSection}
      />

      <main className="relative z-10 pointer-events-none flex-grow flex flex-col">
        {activeSection === 'overview' && !modelLoaded && (
          <>
            <HeroOverlay
              scrollProgress={scrollProgress}
              onStartReconstruction={handleStartReconstruction}
              onEnterTerrain={handleEnterTerrain}
              isDronePOV={isDronePOV}
              setIsDronePOV={setIsDronePOV}
              isBreathing={isBreathing}
              setIsBreathing={setIsBreathing}
              isDiving={isDiving}
              setIsDiving={setIsDiving}
              terrainSource={terrainSource}
            />
            <div id="hero" className="h-screen" />
          </>
        )}

        {activeSection === 'overview' && modelLoaded && (
          <div className="relative h-[calc(100vh-64px)] w-full overflow-hidden flex flex-col pointer-events-auto mt-16 bg-[#0F0E0C]">
            <div className="z-30 w-full bg-[#1A1816]/95 border-b border-[#5A564C]/50 relative">
               <TerrainModesBar activeMode={terrainMode} onModeChange={setTerrainMode} />
            </div>
            
            <div className="relative flex-1 w-full h-full">
               {renderUnifiedTerrain(true)}
               <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-40 pointer-events-none w-full flex justify-center">
                 <ElevationHUD hoverData={hoverData} currentMode={calibrationMode} />
               </div>
               <div className="absolute bottom-6 right-6 z-40 pointer-events-none">
                 <RenderLegend activeMode={terrainMode} heatmapBounds={heatmapBounds} />
               </div>
            </div>
          </div>
        )}

        {activeSection === 'chamber' && (
          <div className="relative pointer-events-auto min-h-screen pt-24">
            {isProcessing ? (
              <ReconstructionEngine
                stage={processingStage}
                message={progressMessage}
              />
            ) : (
              <ScanningChamber onUpload={handleUpload} />
            )}
          </div>
        )}

        {activeSection === 'exploded' && (
          <div className="pointer-events-auto min-h-screen pt-24">
            <ExplodeView3D />
          </div>
        )}

        {activeSection === 'topo' && (
          <div className="pointer-events-auto min-h-screen pt-24">
            <TopoMapMode meta={meshMeta} />
          </div>
        )}

        {activeSection === 'architecture' && (
          <div className="pointer-events-auto min-h-screen pt-24">
            <MeasurementTool />
            <TechBlueprint 
              activeModel={activeModel}
              stats={effectiveStats}
              validationMetrics={effectiveValidationMetrics}
              benchmarkData={benchmarkData}
              calibrationMode={calibrationMode}
              activeDatasetName={activeDatasetName}
            />
          </div>
        )}
      </main>

      {/* FOOTER ONLY IF NOT IN MODEL VIEWER */}
      {!(modelLoaded && activeSection === 'overview') && (
        <div className="relative z-10 pointer-events-auto">
          <Footer />
        </div>
      )}

      {errorMessage && (
        <div className="fixed bottom-10 left-1/2 -translate-x-1/2 bg-[#ff3366]/95 backdrop-blur-md text-white px-5 py-2.5 rounded shadow-2xl z-50 flex items-center gap-2">
          <span>{errorMessage}</span>
          <button onClick={() => setErrorMessage(null)} className="ml-3 font-bold">✕</button>
        </div>
      )}
    </div>
  );
}
