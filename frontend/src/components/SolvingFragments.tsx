import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { useScroll, Line } from '@react-three/drei';
import * as THREE from 'three';
import { playFragmentLockSound } from '../utils/soundEffects';

const FRAGMENT_COUNT = 24;

// --- Generate fragment geometries by slicing an icosphere ---
function useFragmentGeometries(count: number) {
  return useMemo(() => {
    const base = new THREE.IcosahedronGeometry(2.4, 2);
    const posAttr = base.attributes.position;
    const triCount = posAttr.count / 3;
    const trisPerFragment = Math.max(1, Math.floor(triCount / count));

    const fragments = [];
    for (let f = 0; f < count; f++) {
      const startTri = f * trisPerFragment;
      const endTri = f === count - 1 ? triCount : startTri + trisPerFragment;

      const positions: number[] = [];
      const normals: number[] = [];
      for (let i = startTri; i < endTri; i++) {
        for (let v = 0; v < 3; v++) {
          const idx = i * 3 + v;
          positions.push(posAttr.getX(idx), posAttr.getY(idx), posAttr.getZ(idx));
          normals.push(
            base.attributes.normal.getX(idx),
            base.attributes.normal.getY(idx),
            base.attributes.normal.getZ(idx)
          );
        }
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));

      // Centroid = fragment's solved position on sphere surface
      const centroid = new THREE.Vector3();
      for (let i = 0; i < positions.length; i += 3) {
        centroid.add(new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]));
      }
      centroid.divideScalar(positions.length / 3);

      fragments.push({ geometry: geo, solvedPosition: centroid.clone() });
    }
    return fragments;
  }, [count]);
}

// --- Precompute scrambled start state: scattered position + random rotation ---
function useScrambleStates(fragments: { solvedPosition: THREE.Vector3 }[]) {
  return useMemo(() => {
    return fragments.map((_, i) => {
      // Deterministic pseudo-randomness based on index for stable transforms
      const seed1 = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
      const seed2 = Math.cos(i * 4.1414 + 12.434) * 23421.2341;
      const seed3 = Math.sin(i * 8.2341 + 45.123) * 12345.6789;

      const rX = ((seed1 - Math.floor(seed1)) - 0.5) * 14;
      const rY = ((seed2 - Math.floor(seed2)) - 0.5) * 14;
      const rZ = ((seed3 - Math.floor(seed3)) - 0.5) * 14;

      return {
        position: new THREE.Vector3(rX, rY, rZ),
        rotation: new THREE.Euler(
          (seed1 % 1) * Math.PI * 2,
          (seed2 % 1) * Math.PI * 2,
          (seed3 % 1) * Math.PI * 2
        )
      };
    });
  }, [fragments]);
}

// --- Single Fragment Mesh ---
interface FragmentProps {
  geometry: THREE.BufferGeometry;
  scrambled: { position: THREE.Vector3; rotation: THREE.Euler };
  solvedPosition: THREE.Vector3;
  progress: { current: number };
  index: number;
  onLock?: (index: number) => void;
  onPositionUpdate?: (index: number, currentPos: THREE.Vector3, isLocked: boolean) => void;
}

const Fragment: React.FC<FragmentProps> = ({
  geometry,
  scrambled,
  solvedPosition,
  progress,
  index,
  onLock,
  onPositionUpdate
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const lockedRef = useRef<boolean>(false);

  // 3-color gradient across fragments: #4d6dff -> #8b5cf6 -> #e952d1
  const color = useMemo(() => {
    const t = index / FRAGMENT_COUNT;
    const c1 = new THREE.Color('#4d6dff');
    const c2 = new THREE.Color('#8b5cf6');
    const c3 = new THREE.Color('#e952d1');
    return t < 0.5 ? c1.clone().lerp(c2, t * 2) : c2.clone().lerp(c3, (t - 0.5) * 2);
  }, [index]);

  useFrame(() => {
    if (!meshRef.current) return;
    const p = progress.current;

    // Stagger solve timing slightly per fragment
    const localT = THREE.MathUtils.clamp((p - index * 0.015) / 0.45, 0, 1);
    const eased = THREE.MathUtils.smoothstep(localT, 0, 1);

    meshRef.current.position.lerpVectors(scrambled.position, solvedPosition, eased);

    meshRef.current.rotation.x = THREE.MathUtils.lerp(scrambled.rotation.x, 0, eased);
    meshRef.current.rotation.y = THREE.MathUtils.lerp(scrambled.rotation.y, 0, eased);
    meshRef.current.rotation.z = THREE.MathUtils.lerp(scrambled.rotation.z, 0, eased);

    const isLockedNow = eased > 0.98;
    onPositionUpdate?.(index, meshRef.current.position, isLockedNow);

    // Fire lock-in callback once when snapped into place
    if (isLockedNow && !lockedRef.current) {
      lockedRef.current = true;
      playFragmentLockSound(index, FRAGMENT_COUNT);
      onLock?.(index);
    } else if (!isLockedNow && lockedRef.current) {
      lockedRef.current = false;
    }
  });

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.35}
        roughness={0.25}
        metalness={0.7}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

// --- Main SolvingFragments Component ---
interface SolvingFragmentsProps {
  onFragmentLock?: (index: number) => void;
  groupRotationSpeed?: number;
  scrollProgressOverride?: number;
}

export const SolvingFragments: React.FC<SolvingFragmentsProps> = ({
  onFragmentLock,
  groupRotationSpeed = 0.05,
  scrollProgressOverride
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const scrollData = useScroll();
  const progressRef = useRef<{ current: number }>({ current: 0 });

  const fragments = useFragmentGeometries(FRAGMENT_COUNT);
  const scrambleStates = useScrambleStates(fragments);

  const positionsRef = useRef<THREE.Vector3[]>(
    Array.from({ length: FRAGMENT_COUNT }, () => new THREE.Vector3())
  );
  const lockedFlagsRef = useRef<boolean[]>(Array(FRAGMENT_COUNT).fill(false));

  const handlePositionUpdate = (index: number, pos: THREE.Vector3, isLocked: boolean) => {
    positionsRef.current[index].copy(pos);
    lockedFlagsRef.current[index] = isLocked;
  };

  useFrame((_, delta) => {
    if (typeof scrollProgressOverride === 'number') {
      progressRef.current.current = scrollProgressOverride;
    } else if (scrollData) {
      progressRef.current.current = scrollData.offset;
    }

    // Slow turntable rotation scaling down as fragments settle
    if (groupRef.current) {
      const settleFactor = 1 - THREE.MathUtils.clamp(progressRef.current.current, 0, 1) * 0.6;
      groupRef.current.rotation.y += delta * groupRotationSpeed * settleFactor;
    }
  });

  // Calculate light trail connector lines between solved adjacent fragments
  const linesToDraw = useMemo(() => {
    const lines = [];
    for (let i = 0; i < FRAGMENT_COUNT - 1; i++) {
      lines.push({ from: i, to: i + 1 });
    }
    return lines;
  }, []);

  return (
    <group ref={groupRef}>
      {fragments.map((frag, i) => (
        <Fragment
          key={i}
          index={i}
          geometry={frag.geometry}
          solvedPosition={frag.solvedPosition}
          scrambled={scrambleStates[i]}
          progress={progressRef.current}
          onLock={onFragmentLock}
          onPositionUpdate={handlePositionUpdate}
        />
      ))}

      {/* Light-trail connector lines when adjacent fragments snap */}
      {linesToDraw.map((line, idx) => (
        <Line
          key={idx}
          points={[
            positionsRef.current[line.from] || [0, 0, 0],
            positionsRef.current[line.to] || [0, 0, 0]
          ]}
          color={idx % 2 === 0 ? '#4d6dff' : '#e952d1'}
          lineWidth={1.5}
          transparent={true}
          opacity={0.5}
        />
      ))}
    </group>
  );
};
