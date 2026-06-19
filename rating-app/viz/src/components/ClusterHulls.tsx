import { useMemo } from "react";
import * as THREE from "three";
import quickhull from "quickhull3d";
import type { NormalizedEmbedding } from "../lib/normalize";
import { clusterHex } from "../lib/colors";

type Props = {
  points: NormalizedEmbedding[];
  visibleClusters: Set<number>;
};

export default function ClusterHulls({ points, visibleClusters }: Props) {
  const segments = useMemo(() => {
    const byCluster = new Map<number, NormalizedEmbedding[]>();
    for (const p of points) {
      if (!visibleClusters.has(p.cluster)) continue;
      const list = byCluster.get(p.cluster) ?? [];
      list.push(p);
      byCluster.set(p.cluster, list);
    }

    const result: { cluster: number; geometry: THREE.BufferGeometry }[] = [];

    for (const [cluster, clusterPoints] of byCluster) {
      if (clusterPoints.length < 4) continue;
      try {
        const coords = clusterPoints.map((p) => [p.nx, p.ny, p.nz] as [number, number, number]);
        const faces = quickhull(coords) as number[][];
        const edgeSet = new Set<string>();
        const positions: number[] = [];

        for (const face of faces) {
          for (let i = 0; i < face.length; i++) {
            const a = face[i];
            const b = face[(i + 1) % face.length];
            const key = a < b ? `${a}-${b}` : `${b}-${a}`;
            if (edgeSet.has(key)) continue;
            edgeSet.add(key);
            positions.push(...coords[a], ...coords[b]);
          }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        result.push({ cluster, geometry });
      } catch {
        // skip degenerate hulls
      }
    }

    return result;
  }, [points, visibleClusters]);

  return (
    <>
      {segments.map(({ cluster, geometry }) => (
        <lineSegments key={cluster} geometry={geometry}>
          <lineBasicMaterial color={clusterHex(cluster)} transparent opacity={0.3} />
        </lineSegments>
      ))}
    </>
  );
}
