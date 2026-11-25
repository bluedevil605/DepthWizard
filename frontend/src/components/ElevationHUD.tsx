import { TrendingUp, MapPin } from 'lucide-react';

type HoverData = {
  elevation?: number | null;
  slopeDeg?: number | null;
  referenceElevation?: number | null;
  absoluteError?: number | null;
  utm?: { x: number; y: number } | null;
  worldPos?: { x: number; y: number; z: number } | null;
};

type ElevationHUDProps = {
  hoverData: HoverData | null;
  currentMode?: string;
  cameraElevation?: number | null;
};

export function ElevationHUD({ hoverData, currentMode = 'global', cameraElevation = null }: ElevationHUDProps) {
  const isRelative = currentMode === 'relative';
  const isLocalized = currentMode === 'localized';
  const isGlobal = currentMode === 'global' || currentMode === 'absolute';

  const elevationVal = hoverData?.elevation ?? cameraElevation;

  // Slope severity classification
  const slopeDeg = hoverData?.slopeDeg ?? null;
  let slopeColor = '#00ff9d';
  let slopeClass = '0–10° (Gentle)';
  if (slopeDeg !== null) {
    if (slopeDeg < 10) {
      slopeColor = '#00ff9d';
      slopeClass = '0–10° (Gentle)';
    } else if (slopeDeg < 20) {
      slopeColor = '#a3e635';
      slopeClass = '10–20° (Moderate)';
    } else if (slopeDeg < 30) {
      slopeColor = '#facc15';
      slopeClass = '20–30° (Steep)';
    } else if (slopeDeg < 45) {
      slopeColor = '#fb923c';
      slopeClass = '30–45° (Very Steep)';
    } else {
      slopeColor = '#f43f5e';
      slopeClass = '45–90° (Precipitous)';
    }
  }

  const calBadgeText = isGlobal
    ? 'GLOBAL (POLY2)'
    : isLocalized
    ? 'LOCALIZED (250m)'
    : 'RELATIVE (rDSM)';

  const calBadgeColor = isGlobal
    ? 'var(--accent-cyan)'
    : isLocalized
    ? '#00ff9d'
    : '#f59e0b';

  return (
    <div
      className="tactical-hud-bar"
      style={{
        gap: '10px',
        maxWidth: 'min(780px, calc(100vw - 32px))',
        whiteSpace: 'nowrap',
        overflowX: 'auto',
        userSelect: 'none'
      }}
    >
      {/* Calibration Tag */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
        <span
          className="pulse-dot"
          style={{
            width: '6px',
            height: '6px',
            background: calBadgeColor,
            boxShadow: `0 0 6px ${calBadgeColor}`
          }}
        />
        <span style={{
          fontSize: '0.64rem',
          fontWeight: 700,
          fontFamily: 'var(--font-mono)',
          color: calBadgeColor
        }}>
          {calBadgeText}
        </span>
      </div>

      <div style={{ width: '1px', height: '14px', background: 'rgba(255,255,255,0.18)', flexShrink: 0 }} />

      {/* Live Elevation */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
        <span style={{ color: 'var(--text-secondary)', fontSize: '0.70rem' }}>
          {hoverData ? (isRelative ? 'RELATIVE HEIGHT:' : 'ABSOLUTE ELEV:') : 'CAMERA ELEV:'}
        </span>
        <span style={{
          color: '#ffffff',
          fontWeight: 800,
          fontSize: '0.86rem',
          fontFamily: 'var(--font-mono)',
          textShadow: '0 0 10px rgba(0, 240, 255, 0.4)'
        }}>
          {elevationVal !== null && elevationVal !== undefined
            ? isRelative
              ? `${elevationVal.toFixed(2)} (norm)`
              : `${elevationVal.toFixed(1)} m MSL`
            : '—'}
        </span>
      </div>

      {/* Reference & Absolute Error (when available and not relative) */}
      {!isRelative && hoverData?.referenceElevation != null && (
        <>
          <div style={{ width: '1px', height: '14px', background: 'rgba(255,255,255,0.18)', flexShrink: 0 }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.70rem' }}>REF:</span>
            <span style={{ color: 'var(--accent-emerald)', fontWeight: 700, fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>
              {hoverData.referenceElevation.toFixed(1)} m
            </span>
          </div>

          <div style={{ width: '1px', height: '14px', background: 'rgba(255,255,255,0.18)', flexShrink: 0 }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.70rem' }}>ERROR:</span>
            <span style={{
              color: hoverData.absoluteError != null && hoverData.absoluteError > 150 ? '#f43f5e' : '#facc15',
              fontWeight: 700,
              fontSize: '0.78rem',
              fontFamily: 'var(--font-mono)'
            }}>
              {hoverData.absoluteError != null ? `±${hoverData.absoluteError.toFixed(1)} m` : 'N/A'}
            </span>
          </div>
        </>
      )}

      <div style={{ width: '1px', height: '14px', background: 'rgba(255,255,255,0.18)', flexShrink: 0 }} />

      {/* Live Slope Angle & Class */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
        <TrendingUp size={13} color={slopeColor} />
        <span style={{ color: 'var(--text-secondary)', fontSize: '0.70rem' }}>SLOPE:</span>
        <span style={{ color: slopeColor, fontWeight: 700, fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>
          {slopeDeg !== null ? `${slopeDeg.toFixed(1)}° (${slopeClass})` : '—'}
        </span>
      </div>

      <div style={{ width: '1px', height: '14px', background: 'rgba(255,255,255,0.18)', flexShrink: 0 }} />

      {/* Coordinates */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '0.70rem', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
        <MapPin size={11} />
        <span>
          {hoverData?.utm
            ? `E ${hoverData.utm.x.toFixed(0)} | N ${hoverData.utm.y.toFixed(0)}`
            : hoverData?.worldPos
            ? `X: ${hoverData.worldPos.x.toFixed(2)} | Z: ${hoverData.worldPos.z.toFixed(2)}`
            : 'SURFACE PROBE ACTIVE'}
        </span>
      </div>
    </div>
  );
}
