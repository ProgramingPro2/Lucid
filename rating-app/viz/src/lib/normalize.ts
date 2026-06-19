import type { EmbeddingPoint } from "../types";

export type NormalizedEmbedding = EmbeddingPoint & {
  nx: number;
  ny: number;
  nz: number;
};

export function normalizeEmbeddings(points: EmbeddingPoint[]): NormalizedEmbedding[] {
  if (points.length === 0) return [];

  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (const p of points) {
    cx += p.x;
    cy += p.y;
    cz += p.z;
  }
  cx /= points.length;
  cy /= points.length;
  cz /= points.length;

  let maxR = 0;
  for (const p of points) {
    const dx = p.x - cx;
    const dy = p.y - cy;
    const dz = p.z - cz;
    maxR = Math.max(maxR, Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  const scale = maxR > 0 ? 1 / maxR : 1;

  return points.map((p) => ({
    ...p,
    nx: (p.x - cx) * scale,
    ny: (p.y - cy) * scale,
    nz: (p.z - cz) * scale,
  }));
}
