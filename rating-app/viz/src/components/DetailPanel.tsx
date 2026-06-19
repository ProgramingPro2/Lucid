import type { Neighbor } from "../lib/neighbors";
import type { EmbeddingPoint } from "../types";

type Props = {
  point: EmbeddingPoint | null;
  neighbors: Neighbor[];
  onClose: () => void;
};

export default function DetailPanel({ point, neighbors, onClose }: Props) {
  if (!point) return null;

  return (
    <aside className="absolute right-0 top-0 z-20 flex h-full w-96 flex-col border-l border-zinc-800 bg-zinc-900/95 shadow-xl backdrop-blur">
      <div className="flex items-start justify-between gap-2 border-b border-zinc-800 p-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-zinc-500">Selected sentence</p>
          <p className="mt-1 text-xs text-zinc-400">Cluster {point.cluster} · ID {point.id}</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded px-2 py-1 text-sm text-zinc-400 hover:bg-zinc-800"
        >
          Close
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <p className="text-sm leading-relaxed text-zinc-100">{point.text}</p>
        <h3 className="mb-2 mt-6 text-xs font-semibold uppercase tracking-wide text-zinc-500">
          10 nearest neighbors (3D PCA space)
        </h3>
        <p className="mb-3 text-xs text-zinc-500">
          Approximate — Euclidean distance in the lossy 3D projection, not true semantic similarity.
        </p>
        <ul className="space-y-2">
          {neighbors.map(({ point: n, distance }) => (
            <li
              key={n.id}
              className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2 text-xs text-zinc-300"
            >
              <span className="font-mono text-zinc-500">d={distance.toFixed(4)}</span>
              <p className="mt-1 text-zinc-200">{n.text}</p>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
