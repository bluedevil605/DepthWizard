import * as THREE from 'three';

export type TerrainPreset = 'flood' | 'landslide' | 'collapse' | 'mountain';
export type ColorMode = 'cyan' | 'elevation' | 'thermal' | 'realistic';

/**
 * Generates a procedural disaster heightmap texture on an HTML5 canvas.
 */
export function createProceduralTerrainCanvas(
  preset: TerrainPreset = 'flood',
  size: number = 256
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;

  const imgData = ctx.createImageData(size, size);
  const data = imgData.data;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const v = y / size;

      let height = 0;

      if (preset === 'flood') {
        // Flood plain: smooth low basin with sudden steep embankments & water channels
        const n1 = Math.sin(u * Math.PI * 4) * Math.cos(v * Math.PI * 4) * 0.3;
        const n2 = Math.sin(u * Math.PI * 10 + v * 5) * 0.15;
        const basin = Math.pow(Math.sin(u * Math.PI) * Math.sin(v * Math.PI), 1.5);
        height = (0.2 + n1 + n2) * basin;
        if (height < 0.1) height = 0.05; // flat flooded water plane
      } else if (preset === 'landslide') {
        // Landslide slope: diagonal cliff tear and fallen debris mounds
        const slope = u * 0.6 + v * 0.4;
        const tear = Math.sin(u * Math.PI * 6 - v * Math.PI * 3) * 0.25;
        const debris = Math.cos(u * 15) * Math.sin(v * 15) * 0.15;
        height = Math.max(0, slope + tear + debris);
      } else if (preset === 'collapse') {
        // Structural collapse: jagged blocky elevation spikes and rubble craters
        const gridX = Math.floor(u * 8) / 8;
        const gridY = Math.floor(v * 8) / 8;
        const block = Math.sin(gridX * 12 + gridY * 8) * 0.4;
        const rubble = (Math.random() - 0.5) * 0.2;
        const distFromCenter = Math.hypot(u - 0.5, v - 0.5);
        const crater = Math.exp(-distFromCenter * 8) * 0.5;
        height = Math.max(0, block - crater + rubble + 0.3);
      } else {
        // Mountainous disaster relief
        const n1 = Math.sin(u * Math.PI * 3) * Math.sin(v * Math.PI * 3) * 0.5;
        const n2 = Math.cos(u * Math.PI * 8) * Math.sin(v * Math.PI * 8) * 0.25;
        const n3 = Math.sin(u * 20 + v * 20) * 0.1;
        height = (n1 + n2 + n3 + 0.5);
      }

      // Clamp height [0, 1]
      height = Math.min(1, Math.max(0, height));
      const val = Math.floor(height * 255);

      const idx = (y * size + x) * 4;
      data[idx] = val;     // R
      data[idx + 1] = val; // G
      data[idx + 2] = val; // B
      data[idx + 3] = 255; // A
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

/**
 * Creates a Three.js CanvasTexture from a procedural terrain preset.
 */
export function createTerrainTexture(preset: TerrainPreset = 'flood', size: number = 256): THREE.CanvasTexture {
  const canvas = createProceduralTerrainCanvas(preset, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Computes a depth texture & height array from an uploaded image in browser canvas.
 */
export function processImageToDepthMap(
  img: HTMLImageElement,
  contrastBoost: number = 1.2
): { canvas: HTMLCanvasElement; texture: THREE.CanvasTexture; heightArray: Float32Array; width: number; height: number } {
  const targetSize = 256; // Standardized grid mesh size
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Could not obtain canvas 2D context');
  }

  // Draw image scaled to canvas
  ctx.drawImage(img, 0, 0, targetSize, targetSize);
  const imgData = ctx.getImageData(0, 0, targetSize, targetSize);
  const data = imgData.data;

  const heightArray = new Float32Array(targetSize * targetSize);

  // Compute luminance & boost contrast for depth estimation
  for (let i = 0; i < targetSize * targetSize; i++) {
    const idx = i * 4;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];

    // Standard perceptual luminance formula
    let lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    
    // Apply sigmoid / contrast curve
    lum = Math.pow(lum, contrastBoost);
    lum = Math.min(1, Math.max(0, lum));

    heightArray[i] = lum;

    const val = Math.floor(lum * 255);
    data[idx] = val;
    data[idx + 1] = val;
    data[idx + 2] = val;
  }

  ctx.putImageData(imgData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;

  return { canvas, texture, heightArray, width: targetSize, height: targetSize };
}
