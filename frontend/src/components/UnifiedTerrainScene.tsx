// @ts-nocheck
import React, { useRef, useMemo, useEffect, useState, Suspense } from 'react';
import { Canvas, useFrame, useThree, useLoader } from '@react-three/fiber';
import * as THREE from 'three';
import { generateMonochromeTerrainData, createReconDroneMesh, createCirclePointTexture, createVermilionSunTexture } from '../utils/terrainPointGenerator';
import type { PointCloudData } from '../utils/terrainPointGenerator';
import { Line, Center, Bounds, OrbitControls } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { Detection3DOverlay } from './Detection3DOverlay';

// -------------------------------------------------------------
// Shared Scroll Interpolation Coordinator
// -------------------------------------------------------------
interface SceneContextProps {
  targetScrollRef: React.MutableRefObject<number>;
  smoothedScrollRef: React.MutableRefObject<number>;
}

const ScrollCoordinator: React.FC<SceneContextProps> = ({ targetScrollRef, smoothedScrollRef }) => {
  useFrame((_, delta) => {
    const factor = 1 - Math.exp(-3.5 * Math.min(delta, 0.1));
    smoothedScrollRef.current += (targetScrollRef.current - smoothedScrollRef.current) * factor;
  });
  return null;
};

// -------------------------------------------------------------
// Real Terrain 3D — Rendering actual processed GLB data
// -------------------------------------------------------------
interface RealTerrain3DProps {
  glbUrl: string;
  terrainMode: string;
  errorOverlayUrl?: string | null;
  confidenceOverlayUrl?: string | null;
  elevationOverlayUrl?: string | null;
  slopeOverlayUrl?: string | null;
  referenceElevUrl?: string | null;
  landCoverUrl?: string | null;
  buildingMaskUrl?: string | null;
  buildingOverlayUrl?: string | null;
  detectionHeatmapUrl?: string | null;
  probeGrid?: any;
  meta?: any;
  calibrationMode?: string;
  detections?: any[];
  selectedDetection?: any;
  onSelectDetection?: (det: any) => void;
  onHover?: (data: any) => void;
  smoothedScrollRef: React.MutableRefObject<number>;
  onHeatmapBoundsChange?: (bounds: {min: number, max: number} | null) => void;
}

const RealTerrain3D: React.FC<RealTerrain3DProps> = ({
  glbUrl,
  terrainMode,
  errorOverlayUrl,
  confidenceOverlayUrl,
  elevationOverlayUrl,
  slopeOverlayUrl,
  referenceElevUrl,
  landCoverUrl,
  buildingMaskUrl,
  buildingOverlayUrl,
  detectionHeatmapUrl,
  probeGrid,
  meta,
  calibrationMode = 'global',
  detections = [],
  selectedDetection,
  onSelectDetection,
  onHover,
  smoothedScrollRef,
  onHeatmapBoundsChange
}) => {
  const gltf = useLoader(GLTFLoader, glbUrl);
  const meshRef = useRef<THREE.Group>(null);
  
  const sceneClone = useMemo(() => gltf.scene.clone(true), [gltf.scene]);

  useEffect(() => {
    const loadedMeshes: Array<{ name: string; vertices: number; visible: boolean }> = [];
    sceneClone.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (mesh.isMesh) {
        loadedMeshes.push({
          name: child.name || '(unnamed)',
          vertices: mesh.geometry?.attributes?.position?.count || 0,
          visible: child.visible,
        });
      }
    });
    console.info('[GLB FRONTEND TRACE]', {
      url: glbUrl,
      meshCount: loadedMeshes.length,
      meshes: loadedMeshes,
    });
  }, [glbUrl, sceneClone]);

  const terrainMesh = useMemo(() => {
    let target: THREE.Mesh | null = null;
    sceneClone.traverse((child) => {
      if ((child as THREE.Mesh).isMesh && !target) {
        target = child as THREE.Mesh;
      }
    });
    return target;
  }, [sceneClone]);

  const originalMaterial = useMemo(() => {
    if (!terrainMesh) return null;
    return terrainMesh.material;
  }, [terrainMesh]);

  const [textures, setTextures] = useState<Record<string, THREE.Texture>>({});

  useEffect(() => {
    const urls = {
      land_cover: landCoverUrl,
      building_mask: terrainMode === 'BUILDING MASK' ? (buildingOverlayUrl || buildingMaskUrl) : null,
      detection_heatmap: detectionHeatmapUrl,
      accuracy_error: errorOverlayUrl,
      calibration_confidence: confidenceOverlayUrl,
      elevation: elevationOverlayUrl,
      reference: referenceElevUrl,
      slope: slopeOverlayUrl
    };

    const loader = new THREE.TextureLoader();
    Object.entries(urls).forEach(([key, url]) => {
      if (!url) return;
      loader.load(url, (tex) => {
        tex.flipY = false;
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.needsUpdate = true;
        setTextures((prev) => ({ ...prev, [key]: tex }));
      });
    });
  }, [
    landCoverUrl,
    buildingMaskUrl,
    buildingOverlayUrl,
    detectionHeatmapUrl,
    errorOverlayUrl,
    confidenceOverlayUrl,
    elevationOverlayUrl,
    referenceElevUrl,
    slopeOverlayUrl
  ]);

  useEffect(() => {
    if (!terrainMesh || !originalMaterial) return;

    // Clone the original material to preserve displacement maps, normal maps, and backend configurations
    const mat = (originalMaterial as THREE.Material).clone() as THREE.MeshStandardMaterial;
    
    // Reset any vertex colors fallback by default
    mat.vertexColors = false;
    mat.needsUpdate = true;
    mat.roughness = 1.0;
    mat.metalness = 0.0;
    mat.emissive = new THREE.Color(0x000000);
    mat.emissiveIntensity = 0;
    
    if (mat.map) {
      mat.map.colorSpace = THREE.SRGBColorSpace;
    }

    if (terrainMode === 'BUILDING MASK' && textures.building_mask) {
      mat.map = textures.building_mask;
      mat.vertexColors = false;
    } else if (terrainMode === 'STRUCTURE DEBUG' || terrainMode === 'GEOMETRY DEBUG' || terrainMode === 'WIREFRAME') {
      mat.map = null;
      mat.vertexColors = false;
      mat.color = new THREE.Color(terrainMode === 'WIREFRAME' ? 0x20252b : 0xb8b1a5);
      mat.wireframe = terrainMode === 'WIREFRAME';
    } else if (terrainMode === 'HEATMAP') {
      // Create a programmatic slope heatmap
      const geom = terrainMesh.geometry;
      if (geom && geom.attributes.position) {
        const tempGeom = geom.clone();
        tempGeom.computeVertexNormals();
        const normals = tempGeom.attributes.normal;
        
        const count = normals.count;
        const slopes = new Float32Array(count);
        let validCount = 0;
        
        for (let i = 0; i < count; i++) {
          const ny = THREE.MathUtils.clamp(normals.getY(i), -1, 1);
          const slopeDeg = THREE.MathUtils.radToDeg(Math.acos(Math.abs(ny)));
          if (!isNaN(slopeDeg) && isFinite(slopeDeg)) {
            slopes[i] = slopeDeg;
            validCount++;
          } else {
            slopes[i] = -1;
          }
        }
        
        if (validCount > 0) {
          const validSlopes = Array.from(slopes).filter(v => v >= 0).sort((a, b) => a - b);
          const minSlope = validSlopes[Math.floor(validCount * 0.02)] || 0;
          const maxSlope = validSlopes[Math.floor(validCount * 0.98)] || 90;
          const meanSlope = validSlopes.reduce((a,b)=>a+b, 0) / validCount;
          
          if (onHeatmapBoundsChange) {
            onHeatmapBoundsChange({ min: minSlope, max: maxSlope });
          }
          
          console.log(`[HEATMAP DEBUG]
mode: HEATMAP
data available: true
width: N/A (mesh based)
height: N/A (mesh based)
min: ${validSlopes[0]} (clamped to ${minSlope})
max: ${validSlopes[validCount-1]} (clamped to ${maxSlope})
mean: ${meanSlope}
finite values: ${validCount}
NaN values: ${count - validCount}`);
          
          const colors = new Float32Array(count * 3);
          const colorObj = new THREE.Color();
          const span = Math.max(0.1, maxSlope - minSlope);
          
          for (let i = 0; i < count; i++) {
            const val = slopes[i];
            if (val < 0) {
              colors[i*3] = 0; colors[i*3+1] = 0; colors[i*3+2] = 0;
              continue;
            }
            const norm = THREE.MathUtils.clamp((val - minSlope) / span, 0, 1);
            colorObj.setHSL((1.0 - norm) * 0.66, 1.0, 0.5);
            colors[i*3] = colorObj.r;
            colors[i*3+1] = colorObj.g;
            colors[i*3+2] = colorObj.b;
          }
          
          geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
          mat.vertexColors = true;
          mat.map = null; // Ensure underlying texture doesn't hide vertex colors
        }
      }
    } else if (terrainMode === 'LAND COVER' && textures.land_cover) {
      mat.map = textures.land_cover;
    } else if (terrainMode === 'OBJECTS') {
      mat.map = (originalMaterial as THREE.MeshStandardMaterial).map;
    } else if (terrainMode === 'ERROR' && textures.accuracy_error) {
      mat.map = textures.accuracy_error;
    } else if (terrainMode === 'CONFIDENCE' && textures.calibration_confidence) {
      mat.map = textures.calibration_confidence;
    } else if (terrainMode === 'ELEVATION' && textures.elevation) {
      mat.map = textures.elevation;
    } else if (terrainMode === 'SLOPE' && textures.slope) {
      mat.map = textures.slope;
    } else if (terrainMode === 'OPTICAL') {
      mat.map = (originalMaterial as THREE.MeshStandardMaterial).map;
      mat.emissive = new THREE.Color(0xffffff);
      mat.emissiveMap = mat.map;
      mat.emissiveIntensity = 0.25; // Subtle emissive lift to maintain texture visibility
    } else {
      mat.map = (originalMaterial as THREE.MeshStandardMaterial).map;
    }

    terrainMesh.material = mat;

    sceneClone.traverse((child) => {
      if (!(child as THREE.Mesh).isMesh || child === terrainMesh) return;
      const structureMaterial = ((child as THREE.Mesh).material as THREE.Material).clone() as THREE.MeshStandardMaterial;
      if (terrainMode === 'STRUCTURE DEBUG' || terrainMode === 'GEOMETRY DEBUG' || terrainMode === 'WIREFRAME') {
        structureMaterial.map = null;
        structureMaterial.vertexColors = false;
        structureMaterial.color = new THREE.Color(
          terrainMode === 'WIREFRAME' ? 0x20252b : terrainMode === 'STRUCTURE DEBUG' ? 0xe24a2a : 0xd0c7b8
        );
        structureMaterial.wireframe = terrainMode === 'WIREFRAME';
      }
      (child as THREE.Mesh).material = structureMaterial;
    });
  }, [terrainMode, textures, terrainMesh, originalMaterial]);

  const handlePointerMove = (e: any) => {
    e.stopPropagation();
    if (!e.point || !e.face) return;

    const point = e.point;
    const normal = e.face.normal;

    const ny = THREE.MathUtils.clamp(normal.y, -1, 1);
    const geomSlopeRad = Math.acos(Math.abs(ny));
    let slopeDeg = THREE.MathUtils.radToDeg(geomSlopeRad);

    const sceneW = meta?.scene_width || 10.0;
    const sceneD = meta?.scene_depth || 10.0;
    const u = THREE.MathUtils.clamp((point.x / sceneW) + 0.5, 0.0, 1.0);
    const v = THREE.MathUtils.clamp((point.z / sceneD) + 0.5, 0.0, 1.0);

    let computedElevation = point.y;
    let refElev = null;
    let absErr = null;
    let confPct = null;
    let utmCoords = null;

    if (probeGrid && probeGrid.grid_size) {
      const [gh, gw] = probeGrid.grid_size;
      const c = Math.min(gw - 1, Math.max(0, Math.floor(u * gw)));
      const r = Math.min(gh - 1, Math.max(0, Math.floor(v * gh)));

      if (calibrationMode === 'global' && probeGrid.pred_global) {
        computedElevation = probeGrid.pred_global[r][c];
        absErr = probeGrid.error_global ? probeGrid.error_global[r][c] : null;
        confPct = 75;
      } else if (calibrationMode === 'localized' && probeGrid.pred_localized) {
        computedElevation = probeGrid.pred_localized[r][c];
        absErr = probeGrid.error_localized ? probeGrid.error_localized[r][c] : null;
        confPct = probeGrid.conf_localized ? probeGrid.conf_localized[r][c] : 92;
      } else if (calibrationMode === 'relative' && probeGrid.pred_relative) {
        computedElevation = probeGrid.pred_relative[r][c];
        absErr = null;
        confPct = null;
      }

      if (calibrationMode !== 'relative' && probeGrid.reference) {
        refElev = probeGrid.reference[r][c];
      }

      if (probeGrid.slope) {
        slopeDeg = probeGrid.slope[r][c];
      }

      if (probeGrid.bounds) {
        const [minX, minY, maxX, maxY] = probeGrid.bounds;
        utmCoords = {
          x: minX + u * (maxX - minX),
          y: maxY - v * (maxY - minY)
        };
      }
    } else {
      if (meta && meta.min_elevation !== undefined && meta.max_elevation !== undefined) {
        if (meta.is_metric) {
          const scaleFactor = (meta.metric_scale_factor || 1.0) * (meta.vertical_exaggeration || 1.0);
          const baseElev = scaleFactor > 1e-8 ? (point.y / scaleFactor) : 0;
          computedElevation = baseElev + (meta.elevation_origin !== undefined ? meta.elevation_origin : meta.min_elevation);
        } else {
          const minE = meta.min_elevation;
          const maxE = meta.max_elevation;
          const span = maxE - minE;
          computedElevation = minE + (point.y + 1.0) * 0.5 * span;
        }
      }
    }

    if (onHover) {
      onHover({
        elevation: computedElevation,
        referenceElevation: refElev,
        absoluteError: absErr,
        slopeDeg: slopeDeg,
        confidencePct: confPct,
        utm: utmCoords,
        worldPos: { x: point.x, y: point.y, z: point.z },
        screenPos: { x: e.clientX, y: e.clientY }
      });
    }
  };

  const handlePointerOut = () => {
    if (onHover) onHover(null);
  };

  return (
    <group ref={meshRef}>
      <primitive 
        object={sceneClone} 
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
      />
      {terrainMode === 'OBJECTS' && detections && detections.length > 0 && (
        <Detection3DOverlay 
          detections={detections}
          meta={meta}
          selectedDetection={selectedDetection}
          onSelectDetection={onSelectDetection}
        />
      )}
    </group>
  );
};


// -------------------------------------------------------------
// Station definitions for scroll-driven camera path
// -------------------------------------------------------------
interface Station {
  scrollAt: number;
  camPos: [number, number, number];
  lookAt: [number, number, number];
  mode: 'overview' | 'points' | 'exploded' | 'topdown';
  pointOpacity: number;
  contourOpacity: number;
  explodeSpread: number;
}

const STATIONS: Station[] = [
  { scrollAt: 0.00, camPos: [0, 4.6, 17.5],    lookAt: [0, 0.6, 0],   mode: 'overview',  pointOpacity: 0.95, contourOpacity: 0.0, explodeSpread: 0 },
  { scrollAt: 0.22, camPos: [-2.0, 3.8, 16.5], lookAt: [0, 0.4, 0],   mode: 'points',    pointOpacity: 0.95, contourOpacity: 0.0, explodeSpread: 0 },
  { scrollAt: 0.45, camPos: [1.8, 6.0, 17.0],  lookAt: [0, 0.7, 0],   mode: 'exploded',  pointOpacity: 0.7,  contourOpacity: 0.4, explodeSpread: 1 },
  { scrollAt: 0.68, camPos: [0, 18.0, 1.2],    lookAt: [0, 0, 0],     mode: 'topdown',   pointOpacity: 0.3,  contourOpacity: 0.9, explodeSpread: 0 },
  { scrollAt: 0.90, camPos: [0, 4.8, 17.5],    lookAt: [0, 0.6, 0],   mode: 'overview',  pointOpacity: 0.85, contourOpacity: 0.0, explodeSpread: 0 },
];

function findStationLerp(progress: number): { a: Station; b: Station; t: number } {
  for (let i = 0; i < STATIONS.length - 1; i++) {
    if (progress <= STATIONS[i + 1].scrollAt) {
      const a = STATIONS[i];
      const b = STATIONS[i + 1];
      const range = b.scrollAt - a.scrollAt;
      const t = range > 0 ? (progress - a.scrollAt) / range : 0;
      const eased = t * t * t * (t * (t * 6 - 15) + 10);
      return { a, b, t: eased };
    }
  }
  return { a: STATIONS[STATIONS.length - 1], b: STATIONS[STATIONS.length - 1], t: 0 };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// -------------------------------------------------------------
// Scroll-driven Camera Controller
// -------------------------------------------------------------
interface ScrollCameraControllerProps {
  smoothedScrollRef: React.MutableRefObject<number>;
  introComplete: boolean;
  isDronePOV?: boolean;
  isDiving?: boolean;
  isBreathing?: boolean;
  disabled?: boolean;
}

const ScrollCameraController: React.FC<ScrollCameraControllerProps> = ({
  smoothedScrollRef,
  introComplete,
  isDronePOV = false,
  isDiving = false,
  isBreathing = true,
  disabled = false
}) => {
  const { camera } = useThree();
  const targetPosRef = useRef(new THREE.Vector3(0, 4.6, 17.5));
  const targetLookRef = useRef(new THREE.Vector3(0, 0.6, 0));
  const currentLookRef = useRef(new THREE.Vector3(0, 0.6, 0));

  useFrame(({ clock }, delta) => {
    if (disabled) return;
    const time = clock.getElapsedTime();
    const scrollProgress = smoothedScrollRef.current;

    if (!introComplete) {
      const introPos = new THREE.Vector3(
        Math.sin(time * 0.08) * 0.6,
        4.7 + Math.cos(time * 0.06) * 0.15,
        18.5 - Math.min(time * 0.15, 1.0)
      );
      camera.position.lerp(introPos, 0.03);
      currentLookRef.current.set(0, 0.6, 0);
      camera.lookAt(currentLookRef.current);
      return;
    }

    if (isDiving && scrollProgress < 0.2) {
      const diveTargetPos = new THREE.Vector3(0, 1.6, 6.5);
      const diveTargetLook = new THREE.Vector3(0, 0.4, 0);
      camera.position.lerp(diveTargetPos, 0.035);
      currentLookRef.current.lerp(diveTargetLook, 0.035);
      camera.lookAt(currentLookRef.current);
      return;
    }

    if (isDronePOV && scrollProgress < 0.2) {
      const floatX = Math.sin(time * 0.075) * 10.5;
      const floatZ = Math.cos(time * 0.075) * 10.5;
      const droneCamPos = new THREE.Vector3(
        floatX,
        4.0 + Math.sin(time * 0.4) * 0.08,
        floatZ
      );
      camera.position.lerp(droneCamPos, 0.035);
      currentLookRef.current.lerp(new THREE.Vector3(0, 0.3, 0), 0.035);
      camera.lookAt(currentLookRef.current);
      return;
    }

    const { a, b, t } = findStationLerp(scrollProgress);

    targetPosRef.current.set(
      lerp(a.camPos[0], b.camPos[0], t),
      lerp(a.camPos[1], b.camPos[1], t),
      lerp(a.camPos[2], b.camPos[2], t)
    );

    targetLookRef.current.set(
      lerp(a.lookAt[0], b.lookAt[0], t),
      lerp(a.lookAt[1], b.lookAt[1], t),
      lerp(a.lookAt[2], b.lookAt[2], t)
    );

    if (isBreathing) {
      targetPosRef.current.x += Math.sin(time * 0.05) * 0.12;
      targetPosRef.current.y += Math.cos(time * 0.04) * 0.06;
    }

    const factor = 1 - Math.exp(-4.2 * Math.min(delta, 0.1));
    camera.position.lerp(targetPosRef.current, factor);
    currentLookRef.current.lerp(targetLookRef.current, factor);
    camera.lookAt(currentLookRef.current);
  });

  return null;
}

// -------------------------------------------------------------
// Unified Procedural Demo Terrain
// -------------------------------------------------------------
// (Simplified extraction of contour segments to keep code manageable)
function extractContourSegments(
  heightGrid: Float32Array,
  positions: Float32Array,
  gridRes: number,
  targetH: number,
  minH: number,
  maxH: number,
  offsetY: number = 0
): { verts: number[]; rgb: [number, number, number] } {
  const verts: number[] = [];

  const getH = (r: number, c: number) => heightGrid[r * gridRes + c];

  const interp = (r1: number, c1: number, r2: number, c2: number): [number, number, number] => {
    const h1 = getH(r1, c1);
    const h2 = getH(r2, c2);
    const t = Math.abs(h2 - h1) < 0.0001 ? 0.5 : (targetH - h1) / (h2 - h1);
    const tc = Math.max(0, Math.min(1, t));
    const idx1 = (r1 * gridRes + c1) * 3;
    const idx2 = (r2 * gridRes + c2) * 3;
    return [
      positions[idx1] + (positions[idx2] - positions[idx1]) * tc,
      targetH + offsetY + 0.04,
      positions[idx1 + 2] + (positions[idx2 + 2] - positions[idx1 + 2]) * tc
    ];
  };

  for (let r = 0; r < gridRes - 1; r++) {
    for (let c = 0; c < gridRes - 1; c++) {
      const h00 = getH(r, c);
      const h10 = getH(r, c + 1);
      const h11 = getH(r + 1, c + 1);
      const h01 = getH(r + 1, c);

      let ci = 0;
      if (h00 >= targetH) ci |= 1;
      if (h10 >= targetH) ci |= 2;
      if (h11 >= targetH) ci |= 4;
      if (h01 >= targetH) ci |= 8;

      const table: [number, number][][] = [
        [],               // 0000
        [[3, 0]],         // 0001
        [[0, 1]],         // 0010
        [[3, 1]],         // 0011
        [[1, 2]],         // 0100
        [[3, 0], [1, 2]], // 0101 (saddle)
        [[0, 2]],         // 0110
        [[3, 2]],         // 0111
        [[2, 3]],         // 1000
        [[2, 0]],         // 1001
        [[0, 1], [2, 3]], // 1010 (saddle)
        [[2, 1]],         // 1011
        [[1, 3]],         // 1100
        [[1, 0]],         // 1101
        [[0, 3]],         // 1110
        [],               // 1111
      ];

      const getEdgePoint = (edge: number): [number, number, number] => {
        if (edge === 0) return interp(r, c, r, c + 1);
        if (edge === 1) return interp(r, c + 1, r + 1, c + 1);
        if (edge === 2) return interp(r + 1, c, r + 1, c + 1);
        return interp(r, c, r + 1, c);
      };

      for (const [e0, e1] of table[ci]) {
        const [ax, ay, az] = getEdgePoint(e0);
        const [bx, by, bz] = getEdgePoint(e1);
        verts.push(ax, ay, az, bx, by, bz);
      }
    }
  }

  const normH = Math.max(0, Math.min(1, (targetH - minH) / ((maxH - minH) || 1)));
  let r = 90 / 255, g = 86 / 255, b = 76 / 255;
  if (normH < 0.4) {
    const t = normH / 0.4;
    r = (90 + (139 - 90) * t) / 255;
    g = (86 + (132 - 86) * t) / 255;
    b = (76 + (120 - 76) * t) / 255;
  } else {
    const t = (normH - 0.4) / 0.6;
    r = (139 + (217 - 139) * t) / 255;
    g = (132 + (210 - 132) * t) / 255;
    b = (120 + (194 - 120) * t) / 255;
  }

  return { verts, rgb: [r, g, b] };
}

function buildContourLinesGeometry(data: PointCloudData, numLevels: number = 42): THREE.BufferGeometry {
  const { heightGrid, positions, gridResolution: gridRes, bounds } = data;
  const minH = bounds.minZ;
  const maxH = bounds.maxZ;
  const range = (maxH - minH) || 1;

  const allVerts: number[] = [];
  const allCols: number[] = [];

  for (let lvl = 1; lvl < numLevels; lvl++) {
    const targetH = minH + (lvl / numLevels) * range;
    const { verts, rgb } = extractContourSegments(heightGrid, positions, gridRes, targetH, minH, maxH, 0);
    for (let k = 0; k < verts.length; k += 3) {
      allVerts.push(verts[k], verts[k + 1], verts[k + 2]);
      allCols.push(rgb[0], rgb[1], rgb[2]);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(allVerts), 3));
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(allCols), 3));
  return geo;
}

function buildExplodedContourGeometry(data: PointCloudData): THREE.BufferGeometry {
  const { heightGrid, positions, gridResolution: gridRes, bounds } = data;
  const minH = bounds.minZ;
  const maxH = bounds.maxZ;
  const range = (maxH - minH) || 1;

  const allVerts: number[] = [];
  const allCols: number[] = [];

  const numBands = 5;
  const levelsPerBand = 8;

  for (let band = 0; band < numBands; band++) {
    const bandMinH = minH + (band / numBands) * range;
    const bandMaxH = minH + ((band + 1) / numBands) * range;
    const bandOffset = (band - 2) * 3.5;

    for (let lvl = 1; lvl <= levelsPerBand; lvl++) {
      const targetH = bandMinH + (lvl / (levelsPerBand + 1)) * (bandMaxH - bandMinH);
      if (targetH < minH || targetH > maxH) continue;
      const { verts, rgb } = extractContourSegments(heightGrid, positions, gridRes, targetH, minH, maxH, bandOffset);
      for (let k = 0; k < verts.length; k += 3) {
        allVerts.push(verts[k], verts[k + 1], verts[k + 2]);
        allCols.push(rgb[0], rgb[1], rgb[2]);
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(allVerts), 3));
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(allCols), 3));
  return geo;
}

interface UnifiedTerrain3DProps {
  data: PointCloudData;
  smoothedScrollRef: React.MutableRefObject<number>;
  terrainMode: string;
  introComplete: boolean;
  isBreathing?: boolean;
}

const UnifiedTerrain3D: React.FC<UnifiedTerrain3DProps> = ({
  data,
  smoothedScrollRef,
  terrainMode: _terrainMode,
  introComplete,
  isBreathing = true
}) => {
  const wireRef = useRef<THREE.LineSegments>(null);
  const explodedWireRef = useRef<THREE.LineSegments>(null);
  const pointsRef = useRef<THREE.Points>(null);
  const contourGroupRef = useRef<THREE.Group>(null);
  const droneRef = useRef<THREE.Group>(null);
  const scanRingRef = useRef<THREE.Mesh>(null);
  const meshRef = useRef<THREE.Group>(null);
  const lastContourOpacity = useRef<number>(-1);

  const droneMesh = useMemo(() => createReconDroneMesh(), []);
  const circleTexture = useMemo(() => createCirclePointTexture(), []);
  const sunTexture = useMemo(() => createVermilionSunTexture(), []);

  const wireGeometry = useMemo(() => buildContourLinesGeometry(data, 42), [data]);
  const explodedWireGeometry = useMemo(() => buildExplodedContourGeometry(data), [data]);

  const accentGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(data.positions.slice(), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(data.colors.slice(), 3));
    return geo;
  }, [data]);

  const trajectoryPositions = useMemo(() => {
    const segments = 100;
    const radius = 10.5;
    const pos = new Float32Array((segments + 1) * 3);
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      pos[i * 3] = Math.sin(theta) * radius;
      pos[i * 3 + 1] = 4.5;
      pos[i * 3 + 2] = Math.cos(theta) * radius;
    }
    return pos;
  }, []);

  useFrame(({ clock }) => {
    const time = clock.getElapsedTime();
    const scrollProgress = smoothedScrollRef.current;
    const { a, b, t } = findStationLerp(scrollProgress);

    const pointOpacity = lerp(a.pointOpacity, b.pointOpacity, t);
    const contourOpacity = lerp(a.contourOpacity, b.contourOpacity, t);
    const explodeSpread = lerp(a.explodeSpread, b.explodeSpread, t);

    if (meshRef.current) {
      const breath = isBreathing ? 1 + Math.sin(time * 0.3) * 0.0025 : 1.0;
      meshRef.current.scale.set(2.4 * breath, 1.4 * breath, 2.4 * breath);
      meshRef.current.position.set(0, -1.8, 0);
    }

    if (droneRef.current) {
      const radius = 10.5;
      const droneSpeed = 0.075;
      droneRef.current.position.x = Math.sin(time * droneSpeed) * radius;
      droneRef.current.position.y = 4.5;
      droneRef.current.position.z = Math.cos(time * droneSpeed) * radius;
      droneRef.current.rotation.y = time * droneSpeed + Math.PI / 2;
    }

    if (scanRingRef.current) {
      if (scrollProgress < 0.18) {
        scanRingRef.current.visible = true;
        const scanScale = (time * 0.28) % 15;
        scanRingRef.current.scale.set(scanScale, scanScale, 1);
        const heroFade = Math.max(0, (0.18 - scrollProgress) / 0.18);
        (scanRingRef.current.material as THREE.MeshBasicMaterial).opacity =
          Math.max(0, 0.4 - (scanScale / 15) * 0.4) * heroFade;
      } else {
        scanRingRef.current.visible = false;
      }
    }

    const wireOpacity = introComplete ? pointOpacity : Math.min(0.95, time * 0.12);
    if (wireRef.current) {
      (wireRef.current.material as THREE.LineBasicMaterial).opacity = wireOpacity;
      wireRef.current.visible = explodeSpread < 0.85;
    }
    if (explodedWireRef.current) {
      (explodedWireRef.current.material as THREE.LineBasicMaterial).opacity = wireOpacity * explodeSpread;
      explodedWireRef.current.visible = explodeSpread > 0.05;
    }

    if (pointsRef.current) {
      const mat = pointsRef.current.material as THREE.PointsMaterial;
      mat.opacity = (introComplete ? pointOpacity : Math.min(0.5, time * 0.07)) * 0.35;
    }

    if (contourGroupRef.current) {
      const isVisible = contourOpacity > 0.01;
      if (contourGroupRef.current.visible !== isVisible) {
        contourGroupRef.current.visible = isVisible;
      }
      if (isVisible && Math.abs(lastContourOpacity.current - contourOpacity) > 0.005) {
        lastContourOpacity.current = contourOpacity;
        contourGroupRef.current.children.forEach((child) => {
          if ((child as THREE.Line).material) {
            ((child as THREE.Line).material as THREE.LineBasicMaterial).opacity = contourOpacity;
          }
        });
      }
    }
  });

  return (
    <group ref={meshRef}>
      <lineSegments ref={wireRef} geometry={wireGeometry}>
        <lineBasicMaterial
          vertexColors={true}
          transparent={true}
          opacity={0.92}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </lineSegments>

      <lineSegments ref={explodedWireRef} geometry={explodedWireGeometry} visible={false}>
        <lineBasicMaterial
          vertexColors={true}
          transparent={true}
          opacity={0}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </lineSegments>

      <mesh position={[0, 4.2, -22]} rotation={[0, 0, 0]}>
        <planeGeometry args={[26, 26]} />
        <meshBasicMaterial
          map={sunTexture}
          transparent={true}
          opacity={0.92}
          depthWrite={false}
        />
      </mesh>

      <points ref={pointsRef} geometry={accentGeometry}>
        <pointsMaterial
          color="#D9D2C2"
          size={0.12}
          sizeAttenuation={true}
          map={circleTexture}
          vertexColors={true}
          transparent={true}
          alphaTest={0.2}
          depthWrite={false}
          opacity={0.28}
          blending={THREE.AdditiveBlending}
        />
      </points>

      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[data.riverParticles, 3]} />
        </bufferGeometry>
        <pointsMaterial
          size={0.18}
          map={circleTexture}
          color="#D9D2C2"
          transparent={true}
          alphaTest={0.2}
          opacity={0.8}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>

      <group ref={contourGroupRef} visible={false}>
        {data.contourLines.map((chain, idx) => (
          <Line
            key={idx}
            points={chain}
            color="#D9D2C2"
            lineWidth={1.5}
            transparent={true}
            opacity={0}
          />
        ))}
      </group>

      <group ref={droneRef}>
        <primitive object={droneMesh} />
      </group>

      <lineLoop>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[trajectoryPositions, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#8B8478" transparent opacity={0.45} />
      </lineLoop>

      <mesh ref={scanRingRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
        <ringGeometry args={[0.95, 1.0, 64]} />
        <meshBasicMaterial color="#E0433C" transparent={true} opacity={0.5} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
};


// -------------------------------------------------------------
// Main Exported Component
// -------------------------------------------------------------
interface UnifiedTerrainSceneProps {
  terrainMode: string;
  isDronePOV?: boolean;
  isBreathing?: boolean;
  isDiving?: boolean;
  terrainSource?: 'demo' | 'processed';
  glbUrl?: string | null;
  activeModel?: any;
  errorOverlayUrl?: string | null;
  confidenceOverlayUrl?: string | null;
  elevationOverlayUrl?: string | null;
  slopeOverlayUrl?: string | null;
  referenceElevUrl?: string | null;
  landCoverUrl?: string | null;
  buildingMaskUrl?: string | null;
  buildingOverlayUrl?: string | null;
  detectionHeatmapUrl?: string | null;
  probeGrid?: any;
  meta?: any;
  calibrationMode?: string;
  detections?: any[];
  selectedDetection?: any;
  onSelectDetection?: (det: any) => void;
  onHover?: (data: any) => void;
  isEmbedded?: boolean;
  onHeatmapBoundsChange?: (bounds: {min: number, max: number} | null) => void;
}

export const UnifiedTerrainScene: React.FC<UnifiedTerrainSceneProps> = React.memo(({
  terrainMode,
  isDronePOV = false,
  isBreathing = true,
  isDiving = false,
  terrainSource = 'demo',
  glbUrl = null,
  activeModel,
  errorOverlayUrl = null,
  confidenceOverlayUrl = null,
  elevationOverlayUrl = null,
  slopeOverlayUrl = null,
  referenceElevUrl = null,
  landCoverUrl = null,
  buildingMaskUrl = null,
  buildingOverlayUrl = null,
  detectionHeatmapUrl = null,
  probeGrid,
  meta,
  calibrationMode = 'global',
  detections = [],
  selectedDetection,
  onSelectDetection,
  onHover,
  isEmbedded = false,
  onHeatmapBoundsChange
}) => {
  const terrainData = useMemo(() => generateMonochromeTerrainData(120, 16), []);
  const [introComplete, setIntroComplete] = React.useState(false);

  const targetScrollRef = useRef<number>(0);
  const smoothedScrollRef = useRef<number>(0);

  useEffect(() => {
    const handleScroll = () => {
      const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      targetScrollRef.current = window.scrollY / maxScroll;
      if (window.scrollY > 5) {
        setIntroComplete(true);
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => setIntroComplete(true), 6000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={`${isEmbedded ? "absolute" : "fixed"} inset-0 z-0`} style={{ pointerEvents: 'none' }}>
      <Canvas
        flat
        camera={{ position: [0, 4.5, 17.5], fov: 52 }}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance', toneMapping: THREE.NoToneMapping, outputColorSpace: THREE.SRGBColorSpace }}
        onCreated={({ gl, scene }) => {
          gl.setClearColor('#211E1B');
          gl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
          scene.fog = new THREE.FogExp2('#211E1B', 0.012);
        }}
        style={{ pointerEvents: 'auto' }}
      >
        <ambientLight intensity={0.6} color="#ffffff" />
        <hemisphereLight skyColor="#ffffff" groundColor="#444444" intensity={0.8} />
        <directionalLight position={[10, 15, 10]} intensity={0.8} color="#ffffff" />
        <pointLight position={[1.5, 9, -15]} intensity={0.3} color="#ffffff" />

        <ScrollCoordinator
          targetScrollRef={targetScrollRef}
          smoothedScrollRef={smoothedScrollRef}
        />

        {terrainSource === 'demo' || !glbUrl ? (
          <UnifiedTerrain3D
            data={terrainData}
            smoothedScrollRef={smoothedScrollRef}
            terrainMode={terrainMode}
            introComplete={introComplete}
            isBreathing={isBreathing}
          />
        ) : (
          <Suspense fallback={null}>
            <Bounds fit clip observe margin={1.2}>
              <RealTerrain3D
                glbUrl={glbUrl}
                terrainMode={terrainMode}
                errorOverlayUrl={errorOverlayUrl}
                confidenceOverlayUrl={confidenceOverlayUrl}
                elevationOverlayUrl={elevationOverlayUrl}
                slopeOverlayUrl={slopeOverlayUrl}
                referenceElevUrl={referenceElevUrl}
                landCoverUrl={landCoverUrl}
                buildingMaskUrl={buildingMaskUrl}
                buildingOverlayUrl={buildingOverlayUrl}
                detectionHeatmapUrl={detectionHeatmapUrl}
                probeGrid={probeGrid}
                meta={meta}
                calibrationMode={calibrationMode}
                detections={detections}
                selectedDetection={selectedDetection}
                onSelectDetection={onSelectDetection}
                onHover={onHover}
                smoothedScrollRef={smoothedScrollRef}
                onHeatmapBoundsChange={onHeatmapBoundsChange}
              />
            </Bounds>
            <OrbitControls makeDefault autoRotate autoRotateSpeed={0.5} maxPolarAngle={Math.PI / 2.1} />
          </Suspense>
        )}

        <ScrollCameraController
          smoothedScrollRef={smoothedScrollRef}
          introComplete={introComplete}
          isDronePOV={isDronePOV}
          isBreathing={isBreathing}
          isDiving={isDiving}
          disabled={terrainSource === 'processed'}
        />
      </Canvas>
    </div>
  );
});
