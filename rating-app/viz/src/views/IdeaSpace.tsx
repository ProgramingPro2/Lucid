import { OrbitControls } from "@react-three/drei";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import ClusterHulls from "../components/ClusterHulls";
import ClusterLegend from "../components/ClusterLegend";
import DetailPanel from "../components/DetailPanel";
import { clusterColor } from "../lib/colors";
import { findNearestNeighbors } from "../lib/neighbors";
import type { NormalizedEmbedding } from "../lib/normalize";
import type { EmbeddingPoint } from "../types";

type Props = {
  points: NormalizedEmbedding[];
  raw: EmbeddingPoint[];
};

const DIM_COLOR = new THREE.Color("#52525b");
const HIGHLIGHT_COLOR = new THREE.Color("#fbbf24");
const HOVER_SCALE = 1.8;

function EmbeddingInstances({
  points,
  hiddenClusters,
  searchQuery,
  hoveredId,
  selectedId,
  onHover,
  onSelect,
}: {
  points: NormalizedEmbedding[];
  hiddenClusters: Set<number>;
  searchQuery: string;
  hoveredId: number | null;
  selectedId: number | null;
  onHover: (id: number | null) => void;
  onSelect: (id: number) => void;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const query = searchQuery.trim().toLowerCase();
  const hasSearch = query.length > 0;

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tempColor = useMemo(() => new THREE.Color(), []);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const hidden = hiddenClusters.has(p.cluster);
      dummy.position.set(p.nx, p.ny, p.nz);

      let scale = 1;
      if (p.id === hoveredId || p.id === selectedId) scale = HOVER_SCALE;
      dummy.scale.setScalar(scale * 0.015);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      if (hidden) {
        tempColor.copy(DIM_COLOR).multiplyScalar(0.15);
      } else if (hasSearch && !p.text.toLowerCase().includes(query)) {
        tempColor.copy(DIM_COLOR).multiplyScalar(0.35);
      } else if (hasSearch) {
        tempColor.copy(HIGHLIGHT_COLOR);
      } else {
        tempColor.copy(clusterColor(p.cluster));
      }

      if (p.id === selectedId) tempColor.lerp(HIGHLIGHT_COLOR, 0.45);
      if (p.id === hoveredId) tempColor.lerp(new THREE.Color("#ffffff"), 0.35);

      mesh.setColorAt(i, tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (!mesh.instanceColor) {
      mesh.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(points.length * 3),
        3,
      );
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [points, hiddenClusters, query, hasSearch, hoveredId, selectedId, dummy, tempColor]);

  const handlePointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();
      const id = e.instanceId;
      onHover(id !== undefined ? points[id]?.id ?? null : null);
    },
    [onHover, points],
  );

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      const id = e.instanceId;
      if (id !== undefined && points[id]) onSelect(points[id].id);
    },
    [onSelect, points],
  );

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, points.length]}
      onPointerMove={handlePointerMove}
      onPointerOut={() => onHover(null)}
      onClick={handleClick}
    >
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

export default function IdeaSpace({ points, raw }: Props) {
  const [searchQuery, setSearchQuery] = useState("");
  const [showHulls, setShowHulls] = useState(false);
  const [hiddenClusters, setHiddenClusters] = useState<Set<number>>(new Set());
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const visibleClusters = useMemo(() => {
    const all = new Set<number>();
    for (const p of points) all.add(p.cluster);
    for (const c of hiddenClusters) all.delete(c);
    return all;
  }, [points, hiddenClusters]);

  const clusterCounts = useMemo(() => {
    const counts = new Map<number, number>();
    for (const p of points) counts.set(p.cluster, (counts.get(p.cluster) ?? 0) + 1);
    return counts;
  }, [points]);

  const selectedPoint =
    selectedId !== null ? (raw.find((p) => p.id === selectedId) ?? null) : null;
  const neighbors = useMemo(
    () => (selectedId !== null ? findNearestNeighbors(raw, selectedId, 10) : []),
    [raw, selectedId],
  );

  const hoveredText =
    hoveredId !== null ? (raw.find((p) => p.id === hoveredId)?.text ?? null) : null;

  const toggleCluster = (cluster: number) => {
    setHiddenClusters((prev) => {
      const next = new Set(prev);
      if (next.has(cluster)) next.delete(cluster);
      else next.add(cluster);
      return next;
    });
  };

  return (
    <div className="relative h-[calc(100vh-3.5rem)] w-full">
      <div className="absolute left-0 right-0 top-0 z-10 border-b border-amber-900/50 bg-amber-950/30 px-4 py-2 text-sm text-amber-100">
        <strong>Lossy 3D projection.</strong> Coordinates are PCA-reduced from 512-d sentence
        embeddings. Distances in this view approximate, not preserve, true semantic geometry.
      </div>

      <div className="absolute left-4 top-16 z-10 w-64 space-y-3">
        <input
          type="search"
          placeholder="Search sentences..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full rounded-lg border border-zinc-700 bg-zinc-900/90 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 backdrop-blur"
        />
        <ClusterLegend
          counts={clusterCounts}
          hiddenClusters={hiddenClusters}
          onToggle={toggleCluster}
        />
        <label className="flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900/90 px-3 py-2 text-xs text-zinc-300 backdrop-blur">
          <input
            type="checkbox"
            checked={showHulls}
            onChange={(e) => setShowHulls(e.target.checked)}
          />
          Show cluster convex hulls
        </label>
        <p className="text-xs text-zinc-500">{points.length.toLocaleString()} points loaded</p>
      </div>

      {hoveredText && (
        <div className="pointer-events-none absolute bottom-4 left-1/2 z-10 max-w-xl -translate-x-1/2 rounded-lg border border-zinc-700 bg-zinc-900/95 px-4 py-2 text-sm text-zinc-200 shadow-lg backdrop-blur">
          {hoveredText.length > 200 ? `${hoveredText.slice(0, 200)}…` : hoveredText}
        </div>
      )}

      <DetailPanel
        point={selectedPoint}
        neighbors={neighbors}
        onClose={() => setSelectedId(null)}
      />

      <Canvas camera={{ position: [2.2, 1.6, 2.2], fov: 50 }} className="!h-full !w-full">
        <color attach="background" args={["#09090b"]} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
        <EmbeddingInstances
          points={points}
          hiddenClusters={hiddenClusters}
          searchQuery={searchQuery}
          hoveredId={hoveredId}
          selectedId={selectedId}
          onHover={setHoveredId}
          onSelect={setSelectedId}
        />
        {showHulls && <ClusterHulls points={points} visibleClusters={visibleClusters} />}
      </Canvas>
    </div>
  );
}
