import * as THREE from 'three';

export interface PointCloudData {
  positions: Float32Array;
  colors: Float32Array;
  contourLines: THREE.Vector3[][];
  riverParticles: Float32Array;
  wireGridPositions: Float32Array;   // line-strip grid: rows then cols, connected ridge-to-ridge
  wireGridColors: Float32Array;
  heightGrid: Float32Array;
  gridResolution: number;
  size: number;
  bounds: { minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number };
}

/**
 * Creates a smooth circular dot sprite texture for THREE.PointsMaterial
 */
export function createCirclePointTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.beginPath();
    ctx.arc(32, 32, 28, 0, Math.PI * 2);
    ctx.fillStyle = '#D9D2C2';
    ctx.fill();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

/**
 * Creates a graphic Japanese vermilion sun texture with radial gradient:
 * Bright vermilion red (#E0433C) at center, transitioning to #C4272B and #9E1B1F at edge
 */
export function createVermilionSunTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    const cx = 256;
    const cy = 256;
    const radius = 240;
    const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, radius);
    grad.addColorStop(0.0, '#FA6B64');     // Lighter radiant vermilion core
    grad.addColorStop(0.25, '#E0433C');    // Bright vermilion
    grad.addColorStop(0.70, '#C4272B');    // Classic vermilion
    grad.addColorStop(0.92, '#9E1B1F');    // Deeper crimson edge
    grad.addColorStop(1.0, 'rgba(92, 19, 21, 0)'); // Soft maroon fade
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

/**
 * Marching-squares iso-contour extraction.
 * For each cell in the height grid, finds edges where height crosses the threshold
 * and produces interpolated line segments forming proper closed contour curves.
 */
function extractMarchingSquaresContours(
  heightGrid: Float32Array,
  gridRes: number,
  size: number,
  targetH: number
): THREE.Vector3[][] {
  const segments: [THREE.Vector3, THREE.Vector3][] = [];

  // Helper to get grid height at (row, col)
  const getH = (r: number, c: number) => heightGrid[r * gridRes + c];

  // Helper to get world coords from grid indices
  const toWorld = (r: number, c: number) => {
    const u = (r / (gridRes - 1) - 0.5) * size;
    const v = (c / (gridRes - 1) - 0.5) * size;
    return { u, v };
  };

  // Interpolate between two grid points along an edge
  const interp = (r1: number, c1: number, r2: number, c2: number): THREE.Vector3 => {
    const h1 = getH(r1, c1);
    const h2 = getH(r2, c2);
    const t = Math.abs(h2 - h1) < 0.0001 ? 0.5 : (targetH - h1) / (h2 - h1);
    const clampedT = Math.max(0, Math.min(1, t));
    const w1 = toWorld(r1, c1);
    const w2 = toWorld(r2, c2);
    const u = w1.u + (w2.u - w1.u) * clampedT;
    const v = w1.v + (w2.v - w1.v) * clampedT;
    return new THREE.Vector3(u, targetH + 0.04, v);
  };

  // Walk each cell of the grid
  for (let r = 0; r < gridRes - 1; r++) {
    for (let c = 0; c < gridRes - 1; c++) {
      // Corner heights: TL=0, TR=1, BR=2, BL=3
      const h0 = getH(r, c);       // top-left
      const h1 = getH(r, c + 1);   // top-right
      const h2 = getH(r + 1, c + 1); // bottom-right
      const h3 = getH(r + 1, c);   // bottom-left

      // Build case index (4-bit)
      let caseIndex = 0;
      if (h0 >= targetH) caseIndex |= 1;
      if (h1 >= targetH) caseIndex |= 2;
      if (h2 >= targetH) caseIndex |= 4;
      if (h3 >= targetH) caseIndex |= 8;

      // Skip empty or full cells
      if (caseIndex === 0 || caseIndex === 15) continue;

      // Edge midpoints (interpolated)
      // Edge 0: top (TL→TR)
      // Edge 1: right (TR→BR)
      // Edge 2: bottom (BL→BR)
      // Edge 3: left (TL→BL)
      const edgePoints: Record<number, THREE.Vector3> = {};
      const getEdge = (edge: number): THREE.Vector3 => {
        if (edgePoints[edge]) return edgePoints[edge];
        switch (edge) {
          case 0: edgePoints[edge] = interp(r, c, r, c + 1); break;
          case 1: edgePoints[edge] = interp(r, c + 1, r + 1, c + 1); break;
          case 2: edgePoints[edge] = interp(r + 1, c, r + 1, c + 1); break;
          case 3: edgePoints[edge] = interp(r, c, r + 1, c); break;
        }
        return edgePoints[edge];
      };

      // Marching squares lookup table: maps case index to edge pairs
      const edgePairs: number[] = [];
      switch (caseIndex) {
        case 1:  edgePairs.push(0, 3); break;
        case 2:  edgePairs.push(0, 1); break;
        case 3:  edgePairs.push(1, 3); break;
        case 4:  edgePairs.push(1, 2); break;
        case 5:  edgePairs.push(0, 1, 2, 3); break; // saddle
        case 6:  edgePairs.push(0, 2); break;
        case 7:  edgePairs.push(2, 3); break;
        case 8:  edgePairs.push(2, 3); break;
        case 9:  edgePairs.push(0, 2); break;
        case 10: edgePairs.push(0, 3, 1, 2); break; // saddle
        case 11: edgePairs.push(1, 2); break;
        case 12: edgePairs.push(1, 3); break;
        case 13: edgePairs.push(0, 1); break;
        case 14: edgePairs.push(0, 3); break;
      }

      // Push segment pairs
      for (let p = 0; p < edgePairs.length; p += 2) {
        segments.push([getEdge(edgePairs[p]), getEdge(edgePairs[p + 1])]);
      }
    }
  }

  // Chain segments into polylines for smoother rendering
  if (segments.length === 0) return [];

  const chains: THREE.Vector3[][] = [];
  const used = new Array(segments.length).fill(false);
  const EPSILON = 0.01;

  const near = (a: THREE.Vector3, b: THREE.Vector3) =>
    Math.abs(a.x - b.x) < EPSILON && Math.abs(a.z - b.z) < EPSILON;

  for (let i = 0; i < segments.length; i++) {
    if (used[i]) continue;
    used[i] = true;
    const chain = [segments[i][0], segments[i][1]];

    let extended = true;
    while (extended) {
      extended = false;
      for (let j = 0; j < segments.length; j++) {
        if (used[j]) continue;
        const [a, b] = segments[j];
        const tail = chain[chain.length - 1];
        const head = chain[0];

        if (near(tail, a)) { chain.push(b); used[j] = true; extended = true; }
        else if (near(tail, b)) { chain.push(a); used[j] = true; extended = true; }
        else if (near(head, a)) { chain.unshift(b); used[j] = true; extended = true; }
        else if (near(head, b)) { chain.unshift(a); used[j] = true; extended = true; }
      }
    }

    if (chain.length >= 3) {
      chains.push(chain);
    }
  }

  return chains;
}

/**
 * Generates a high-density monochrome 3D point cloud terrain dataset
 * with altitude-based brightness gradients and stippled topography.
 */
export function generateMonochromeTerrainData(
  gridResolution: number = 140,
  size: number = 16
): PointCloudData {
  const count = gridResolution * gridResolution;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const heightGrid = new Float32Array(count);
  const riverPositions: number[] = [];

  let minZ = Infinity;
  let maxZ = -Infinity;

  // Generate height field using layered sinusoidal frequencies
  for (let i = 0; i < gridResolution; i++) {
    for (let j = 0; j < gridResolution; j++) {
      const idx = (i * gridResolution + j) * 3;
      const gridIdx = i * gridResolution + j;

      const u = (i / gridResolution - 0.5) * size;
      const v = (j / gridResolution - 0.5) * size;

      // Multi-octave terrain height
      const freq1 = Math.sin(u * 0.4) * Math.cos(v * 0.4) * 2.2;
      const freq2 = Math.sin(u * 1.2 + v * 0.8) * 0.8;
      const freq3 = Math.cos(u * 2.5) * Math.sin(v * 2.5) * 0.3;
      const basin = Math.exp(-(u * u + v * v) * 0.02) * 1.5;

      let height = freq1 + freq2 + freq3 + basin;

      // Create a river channel groove
      const riverDist = Math.abs(u - Math.sin(v * 0.5) * 3);
      if (riverDist < 1.2) {
        height -= (1.2 - riverDist) * 1.2;
        // Collect river flow particle coords
        riverPositions.push(u, height + 0.05, v);
      }

      positions[idx] = u;
      positions[idx + 1] = height;
      positions[idx + 2] = v;
      heightGrid[gridIdx] = height;

      if (height < minZ) minZ = height;
      if (height > maxZ) maxZ = height;
    }
  }

  // Color helper: Japanese landscape stone gradient
  // #211E1B (charcoal) -> #5A564C (warm slate) -> #8B8478 (muted taupe) -> #D9D2C2 (mist off-white)
  const getStoneColor = (normH: number): [number, number, number] => {
    const c0 = [33 / 255, 30 / 255, 27 / 255];    // #211E1B deep charcoal
    const c1 = [90 / 255, 86 / 255, 76 / 255];    // #5A564C warm slate
    const c2 = [139 / 255, 132 / 255, 120 / 255]; // #8B8478 muted taupe
    const c3 = [217 / 255, 210 / 255, 194 / 255]; // #D9D2C2 mist off-white

    if (normH < 0.3) {
      const t = normH / 0.3;
      return [c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t];
    } else if (normH < 0.7) {
      const t = (normH - 0.3) / 0.4;
      return [c1[0] + (c2[0] - c1[0]) * t, c1[1] + (c2[1] - c1[1]) * t, c1[2] + (c2[2] - c1[2]) * t];
    } else {
      const t = (normH - 0.7) / 0.3;
      return [c2[0] + (c3[0] - c2[0]) * t, c2[1] + (c3[1] - c2[1]) * t, c2[2] + (c3[2] - c2[2]) * t];
    }
  };

  // Assign Japanese landscape stone colors based on altitude
  const range = maxZ - minZ || 1;
  for (let i = 0; i < count; i++) {
    const idx = i * 3;
    const h = positions[idx + 1];
    const normH = Math.max(0, Math.min(1, (h - minZ) / range));

    const [r, g, b] = getStoneColor(normH);
    colors[idx] = r;
    colors[idx + 1] = g;
    colors[idx + 2] = b;
  }

  // Generate proper iso-elevation contour lines using marching squares
  const contourLines: THREE.Vector3[][] = [];
  const contourLevels = 14;
  for (let lvl = 1; lvl < contourLevels; lvl++) {
    const targetH = minZ + (lvl / contourLevels) * range;
    const chains = extractMarchingSquaresContours(heightGrid, gridResolution, size, targetH);
    contourLines.push(...chains);
  }

  // Build wireframe grid: every N-th row and column as a continuous line strip
  const wireStride = 3;
  const wireRowLines: number[] = [];
  const wireColLines: number[] = [];
  const wireRowColors: number[] = [];
  const wireColColors: number[] = [];

  // Row lines: for each sampled row, walk across all columns
  for (let i = 0; i < gridResolution; i += wireStride) {
    for (let j = 0; j < gridResolution; j++) {
      const gIdx = (i * gridResolution + j) * 3;
      wireRowLines.push(positions[gIdx], positions[gIdx + 1], positions[gIdx + 2]);
      const h = heightGrid[i * gridResolution + j];
      const normH = Math.max(0, Math.min(1, (h - minZ) / (range || 1)));
      const [r, g, b] = getStoneColor(normH);
      wireRowColors.push(r, g, b);
    }
  }

  // Column lines: for each sampled column, walk down all rows
  for (let j = 0; j < gridResolution; j += wireStride) {
    for (let i = 0; i < gridResolution; i++) {
      const gIdx = (i * gridResolution + j) * 3;
      wireColLines.push(positions[gIdx], positions[gIdx + 1], positions[gIdx + 2]);
      const h = heightGrid[i * gridResolution + j];
      const normH = Math.max(0, Math.min(1, (h - minZ) / (range || 1)));
      const [r, g, b] = getStoneColor(normH);
      wireColColors.push(r, g, b);
    }
  }

  const wireGridPositions = new Float32Array([...wireRowLines, ...wireColLines]);
  const wireGridColors = new Float32Array([...wireRowColors, ...wireColColors]);

  return {
    positions,
    colors,
    contourLines,
    riverParticles: new Float32Array(riverPositions),
    wireGridPositions,
    wireGridColors,
    heightGrid,
    gridResolution,
    size,
    bounds: { minX: -size / 2, maxX: size / 2, minY: -size / 2, maxY: size / 2, minZ, maxZ }
  };
}

/**
 * Creates a 3D minimal quadcopter drone survey craft mesh (no weapons, sleek recon craft)
 */
export function createReconDroneMesh(): THREE.Group {
  const group = new THREE.Group();

  // Central sleek fuselage body - Near-black charcoal
  const bodyGeo = new THREE.BoxGeometry(0.6, 0.12, 0.6);
  const bodyMat = new THREE.MeshStandardMaterial({ color: '#211E1B', roughness: 0.4, metalness: 0.6 });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  group.add(body);

  // 4 Arm extensions & Rotors
  const armPositions = [
    [-0.5, 0, -0.5],
    [0.5, 0, -0.5],
    [-0.5, 0, 0.5],
    [0.5, 0, 0.5]
  ];

  const armMat = new THREE.MeshStandardMaterial({ color: '#5A564C', roughness: 0.5, metalness: 0.5 });

  armPositions.forEach(([x, y, z]) => {
    const armGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.7);
    const arm = new THREE.Mesh(armGeo, armMat);
    arm.rotation.z = Math.PI / 2;
    arm.position.set(x * 0.5, y, z * 0.5);
    group.add(arm);

    // Rotor disc - Muted taupe
    const rotorGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.02, 16);
    const rotorMat = new THREE.MeshBasicMaterial({ color: '#8B8478', transparent: true, opacity: 0.6, wireframe: true });
    const rotor = new THREE.Mesh(rotorGeo, rotorMat);
    rotor.position.set(x, y + 0.06, z);
    group.add(rotor);

    // Vermilion red LED running light pinprick (#E0433C)
    const lightGeo = new THREE.SphereGeometry(0.045, 8, 8);
    const lightMat = new THREE.MeshBasicMaterial({ color: '#E0433C' });
    const light = new THREE.Mesh(lightGeo, lightMat);
    light.position.set(x * 1.1, y - 0.05, z * 1.1);
    group.add(light);
  });

  // Sensor pod below - Soft mist highlights (#D9D2C2)
  const sensorGeo = new THREE.SphereGeometry(0.12, 16, 16);
  const sensorMat = new THREE.MeshBasicMaterial({ color: '#D9D2C2', wireframe: true });
  const sensor = new THREE.Mesh(sensorGeo, sensorMat);
  sensor.position.set(0, -0.12, 0.1);
  group.add(sensor);

  group.scale.set(0.7, 0.7, 0.7);
  return group;
}
