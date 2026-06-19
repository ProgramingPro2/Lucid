import type { EmbeddingPoint } from "../types";

export type Neighbor = {
  point: EmbeddingPoint;
  distance: number;
};

export function findNearestNeighbors(
  points: EmbeddingPoint[],
  targetId: number,
  k = 10,
): Neighbor[] {
  const target = points.find((p) => p.id === targetId);
  if (!target) return [];

  const heap: Neighbor[] = [];

  for (const p of points) {
    if (p.id === targetId) continue;
    const dx = p.x - target.x;
    const dy = p.y - target.y;
    const dz = p.z - target.z;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (heap.length < k) {
      heap.push({ point: p, distance });
      heap.sort((a, b) => b.distance - a.distance);
      continue;
    }

    if (distance < heap[0].distance) {
      heap[0] = { point: p, distance };
      heap.sort((a, b) => b.distance - a.distance);
    }
  }

  return heap.sort((a, b) => a.distance - b.distance);
}
