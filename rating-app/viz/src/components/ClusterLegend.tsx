import { clusterHex, CLUSTER_COUNT } from "../lib/colors";

type Props = {
  counts: Map<number, number>;
  hiddenClusters: Set<number>;
  onToggle: (cluster: number) => void;
};

export default function ClusterLegend({ counts, hiddenClusters, onToggle }: Props) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/90 p-3 backdrop-blur">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-400">
        Clusters
      </h3>
      <ul className="max-h-64 space-y-1 overflow-y-auto text-xs">
        {Array.from({ length: CLUSTER_COUNT }, (_, cluster) => {
          const count = counts.get(cluster) ?? 0;
          if (count === 0) return null;
          const hidden = hiddenClusters.has(cluster);
          return (
            <li key={cluster}>
              <button
                type="button"
                onClick={() => onToggle(cluster)}
                className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-zinc-800 ${
                  hidden ? "opacity-40" : ""
                }`}
              >
                <span
                  className="inline-block h-3 w-3 rounded-full"
                  style={{ backgroundColor: clusterHex(cluster) }}
                />
                <span className="text-zinc-300">Cluster {cluster}</span>
                <span className="ml-auto font-mono text-zinc-500">{count}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
