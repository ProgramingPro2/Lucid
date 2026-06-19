import type { ArchitectureNodeData } from "../types";

type Props = {
  node: ArchitectureNodeData | null;
  onClose: () => void;
};

export default function NodeDetailPanel({ node, onClose }: Props) {
  if (!node) return null;

  return (
    <aside className="absolute right-4 top-4 z-20 w-80 rounded-xl border border-zinc-700 bg-zinc-900/95 p-4 shadow-xl backdrop-blur">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-zinc-500">{node.kind}</p>
          <h3 className="text-lg font-semibold text-zinc-100">{node.label}</h3>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded px-2 py-1 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          Close
        </button>
      </div>
      <p className="text-sm leading-relaxed text-zinc-300">{node.description}</p>
      <div className="mt-4">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Tensor shapes
        </h4>
        <ul className="space-y-1 text-sm text-zinc-300">
          {node.tensors.map((t) => (
            <li key={t} className="rounded bg-zinc-950/80 px-2 py-1 font-mono text-xs">
              {t}
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
