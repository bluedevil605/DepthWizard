import * as THREE from 'three';
import { Html } from '@react-three/drei';

type Detection = {
  bbox: [number, number, number, number];
  class: string;
  confidence: number;
  raw_confidence?: number | null;
  size_sanity?: string;
  nadir_confidence?: number;
  cross_val_status?: string;
  obb_coords?: [number, number][];
};

type TerrainMeta = {
  image_width?: number;
  image_height?: number;
  scene_width?: number;
  scene_depth?: number;
  elevation_samples?: number[][];
  min_elevation?: number;
  relief_span?: number;
  target_max_height?: number;
  is_relative?: boolean;
};

type Detection3DOverlayProps = {
  detections: Detection[];
  meta?: TerrainMeta | null;
  selectedDetection?: Detection | null;
  onSelectDetection: (detection: Detection) => void;
};

/**
 * Renders 3D bounding box outlines and floating tactical labels
 * for aerial object detections directly on top of the terrain mesh.
 */
export function Detection3DOverlay({
  detections,
  meta,
  selectedDetection,
  onSelectDetection
}: Detection3DOverlayProps) {
  if (!detections || detections.length === 0) return null;

  const imgW = meta?.image_width || 512;
  const imgH = meta?.image_height || 512;
  const sceneW = meta?.scene_width || 10.0;
  const sceneD = meta?.scene_depth || 10.0;
  const elevSamples = meta?.elevation_samples;
  const minElev = meta?.min_elevation ?? 0.0;
  const reliefSpan = Math.max(0.01, meta?.relief_span ?? 1.0);
  const targetMaxHeight = meta?.target_max_height ?? (meta?.is_relative ? 2.5 : 3.0);

  // Helper to approximate 3D surface height at normalized UV (0..1)
  const getSurfaceY = (u: number, v: number) => {
    if (elevSamples && elevSamples.length > 0) {
      const rows = elevSamples.length;
      const cols = elevSamples[0].length;
      const r = Math.min(rows - 1, Math.max(0, Math.floor(v * rows)));
      const c = Math.min(cols - 1, Math.max(0, Math.floor(u * cols)));
      const sampleVal = elevSamples[r][c];
      const normY = Math.max(0.0, Math.min(1.0, (sampleVal - minElev) / reliefSpan));
      return normY * targetMaxHeight + 0.08;
    }
    return 0.2;
  };

  return (
    <group>
      {detections.map((det, idx) => {
        const [x1, y1, x2, y2] = det.bbox;
        const uCenter = (x1 + x2) / (2.0 * imgW);
        const vCenter = (y1 + y2) / (2.0 * imgH);
        const uWidth = Math.max(0.005, (x2 - x1) / imgW);
        const vHeight = Math.max(0.005, (y2 - y1) / imgH);

        const x3D = (uCenter - 0.5) * sceneW;
        const z3D = (vCenter - 0.5) * sceneD;
        const y3D = getSurfaceY(uCenter, vCenter);

        const boxW = Math.max(0.12, uWidth * sceneW);
        const boxD = Math.max(0.12, vHeight * sceneD);

        const isSelected =
          selectedDetection &&
          selectedDetection.class === det.class &&
          selectedDetection.confidence === det.confidence &&
          selectedDetection.bbox[0] === det.bbox[0] &&
          selectedDetection.bbox[1] === det.bbox[1];

        // Visual de-emphasis for low nadir confidence, flagged size sanity, or low model conf
        const isFlagged = det.size_sanity === 'flagged';
        const isLowNadir = det.nadir_confidence !== undefined && det.nadir_confidence < 0.50;
        const isMismatch = det.cross_val_status === 'mismatch';
        const isLowConf = det.confidence < 0.45;
        const isDeEmphasized = isLowNadir || isFlagged || isLowConf || isMismatch;

        const outlineColor = isSelected
          ? '#ff0055'
          : isFlagged
          ? '#f59e0b'
          : isMismatch
          ? '#f59e0b'
          : isLowNadir
          ? '#94a3b8'
          : isLowConf
          ? '#5599aa'
          : '#00f0ff';
        const confPercent = Math.round(det.confidence * 100);
        const rawPercent =
          det.raw_confidence != null ? Math.round(det.raw_confidence * 100) : null;
        const confLabel =
          rawPercent != null && rawPercent !== confPercent
            ? `${rawPercent}% → ${confPercent}%`
            : `${confPercent}%`;
        const materialOpacity = isSelected ? 1.0 : (isDeEmphasized ? 0.35 : 1.0);

        // Compute oriented 3D polygon if OBB coordinates available
        const obbPoints = det.obb_coords?.map(([px, py]) => {
          const u = px / imgW;
          const v = py / imgH;
          const ox = (u - 0.5) * sceneW;
          const oz = (v - 0.5) * sceneD;
          const oy = getSurfaceY(u, v);
          return new THREE.Vector3(ox, oy, oz);
        });

        return (
          <group key={`det-${idx}`}>
            {/* If OBB corners available, draw oriented polygon loop */}
            {obbPoints && obbPoints.length === 4 ? (
              <lineLoop>
                <bufferGeometry
                  attach="geometry"
                  onUpdate={(self) => self.setFromPoints(obbPoints)}
                />
                <lineBasicMaterial
                  attach="material"
                  color={outlineColor}
                  transparent={isDeEmphasized}
                  opacity={materialOpacity}
                  linewidth={isSelected ? 3 : (isDeEmphasized ? 1 : 2)}
                />
              </lineLoop>
            ) : (
              /* Axis-aligned 3D bounding box outline */
              <mesh
                position={[x3D, y3D, z3D]}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectDetection(det);
                }}
              >
                <boxGeometry args={[boxW, 0.12, boxD]} />
                <meshStandardMaterial
                  color={outlineColor}
                  wireframe
                  transparent={isDeEmphasized}
                  opacity={materialOpacity}
                  emissive={outlineColor}
                  emissiveIntensity={isSelected ? 0.8 : (isDeEmphasized ? 0.08 : 0.35)}
                />
              </mesh>
            )}

            {/* Clickable Floating Tactical Label */}
            <Html
              position={[x3D, y3D + 0.25, z3D]}
              center
              distanceFactor={18}
              zIndexRange={[100, 0]}
            >
              <div
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectDetection(det);
                }}
                style={{
                  background: isSelected
                    ? 'rgba(255, 0, 85, 0.92)'
                    : isDeEmphasized
                    ? 'rgba(15, 23, 42, 0.70)'
                    : 'rgba(7, 10, 19, 0.88)',
                  backdropFilter: 'blur(8px)',
                  border: isSelected
                    ? '1px solid #ff0055'
                    : isFlagged
                    ? '1px dashed #f59e0b'
                    : isDeEmphasized
                    ? '1px dashed rgba(148, 163, 184, 0.5)'
                    : '1px solid var(--accent-cyan)',
                  color: isDeEmphasized ? 'rgba(226, 232, 240, 0.8)' : '#ffffff',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  fontFamily: 'var(--font-mono)',
                  whiteSpace: 'nowrap',
                  cursor: 'pointer',
                  userSelect: 'none',
                  boxShadow: isSelected
                    ? '0 0 16px rgba(255, 0, 85, 0.6)'
                    : isDeEmphasized
                    ? 'none'
                    : '0 0 10px rgba(0, 240, 255, 0.4)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  transition: 'all 0.2s ease',
                  transform: isSelected ? 'scale(1.12)' : (isDeEmphasized ? 'scale(0.90)' : 'scale(1.0)'),
                  opacity: isDeEmphasized ? 0.65 : 1.0,
                }}
                title={`Click to inspect ${det.class} (${isFlagged ? 'Flagged Size Sanity' : (isLowNadir ? 'Low Nadir Domain' : 'Detection')})`}
              >
                <span
                  style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: isSelected ? '#ffffff' : (isFlagged ? '#f59e0b' : (isLowNadir ? '#94a3b8' : (isLowConf ? '#ffbb00' : '#00ff88'))),
                    display: 'inline-block',
                    opacity: isDeEmphasized ? 0.7 : 1.0,
                  }}
                />
                <span style={{ fontWeight: 600, textTransform: 'capitalize' }}>
                  {det.class}
                </span>
                <span style={{ opacity: isDeEmphasized ? 0.6 : 0.85, fontSize: '10px' }}>
                  {confLabel}
                </span>
                {isMismatch && (
                  <span style={{ fontSize: '9px', color: '#f59e0b', background: 'rgba(245, 158, 11, 0.2)', padding: '1px 3px', borderRadius: '2px' }}>
                    mismatch
                  </span>
                )}
                {isFlagged && (
                  <span style={{ fontSize: '9px', color: '#f59e0b', background: 'rgba(245, 158, 11, 0.2)', padding: '1px 3px', borderRadius: '2px' }}>
                    size?
                  </span>
                )}
                {isLowNadir && !isFlagged && (
                  <span style={{ fontSize: '9px', color: '#94a3b8', background: 'rgba(148, 163, 184, 0.2)', padding: '1px 3px', borderRadius: '2px' }}>
                    non-nadir
                  </span>
                )}
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}
