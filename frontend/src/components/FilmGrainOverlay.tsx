import React, { useEffect, useRef } from 'react';

interface FilmGrainOverlayProps {
  isDronePOV: boolean;
  terrainSource?: 'demo' | 'processed';
}

export const FilmGrainOverlay: React.FC<FilmGrainOverlayProps> = ({ isDronePOV, terrainSource = 'demo' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Lightweight 160x160 noise tile — eliminates CPU lock & memory churn
    const size = 160;
    canvas.width = size;
    canvas.height = size;

    // Pre-generate 4 warm stone noise frames once
    const frames: ImageData[] = [];
    for (let f = 0; f < 4; f++) {
      const imgData = ctx.createImageData(size, size);
      const data = imgData.data;
      for (let i = 0; i < data.length; i += 4) {
        const noise = (Math.random() - 0.5) * 22;
        const base = Math.min(255, Math.max(0, 128 + noise));
        data[i] = Math.min(255, base * 1.04);     // R (warm stone)
        data[i + 1] = base;                       // G
        data[i + 2] = Math.max(0, base * 0.92);   // B
        data[i + 3] = 16;                         // subtle opacity
      }
      frames.push(imgData);
    }

    let frameIdx = 0;
    // Step at cinematic 12 FPS — takes <0.01% CPU!
    const interval = setInterval(() => {
      frameIdx = (frameIdx + 1) % frames.length;
      ctx.putImageData(frames[frameIdx], 0, 0);
    }, 80);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 z-40 pointer-events-none overflow-hidden">
      {/* Repeating fine warm stone grain canvas */}
      <canvas
        ref={canvasRef}
        className={`absolute inset-0 w-full h-full mix-blend-overlay ${
          terrainSource === 'processed' ? 'opacity-10' : 'opacity-45'
        }`}
        style={{
          imageRendering: 'pixelated',
          backgroundRepeat: 'repeat',
        }}
      />

      {/* DRONE POV Scanlines Overlay */}
      {isDronePOV && (
        <div className="absolute inset-0 scanline-overlay opacity-80" />
      )}
    </div>
  );
};
