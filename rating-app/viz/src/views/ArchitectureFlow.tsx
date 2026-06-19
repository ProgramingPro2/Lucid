import { useCallback, useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useEdgesState,
  useNodesState,
  type Node,
} from "reactflow";
import "reactflow/dist/style.css";

import ArchNode from "../components/ArchNode";
import FlowDot from "../components/FlowDot";
import GeneratePanel from "../components/GeneratePanel";
import NodeDetailPanel from "../components/NodeDetailPanel";
import { edgePathForSequence, FLOW_SEQUENCES, getGraph } from "../data/architectureGraph";
import type { ArchitectureNodeData, CodecMode } from "../types";

const nodeTypes = { archNode: ArchNode };

export default function ArchitectureFlow() {
  const [codecMode, setCodecMode] = useState<CodecMode>("jepa");
  const graph = useMemo(() => getGraph(codecMode), [codecMode]);
  const [nodes, setNodes, onNodesChange] = useNodesState(graph.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(graph.edges);
  const [selected, setSelected] = useState<ArchitectureNodeData | null>(null);
  const [activeEdgeId, setActiveEdgeId] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setSelected(null);
  }, [graph, setNodes, setEdges]);

  const highlightNodes = useCallback(
    (ids: string[]) => {
      setNodes((nds) =>
        nds.map((n) => ({
          ...n,
          selected: ids.includes(n.id),
        })),
      );
    },
    [setNodes],
  );

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node<ArchitectureNodeData>) => {
    setSelected(node.data);
  }, []);

  const playTrainingFlow = useCallback(async () => {
    setPlaying(true);
    for (const sequence of FLOW_SEQUENCES.training) {
      highlightNodes([...sequence]);
      const edgeIds = edgePathForSequence(sequence);
      for (const edgeId of edgeIds) {
        setActiveEdgeId(edgeId);
        await sleep(1200);
      }
    }
    highlightNodes([]);
    setActiveEdgeId(null);
    setPlaying(false);
  }, [highlightNodes]);

  return (
    <div className="relative h-[calc(100vh-3.5rem)] w-full">
      <div className="absolute left-4 top-4 z-20 flex flex-wrap items-center gap-2">
        <div className="rounded-lg border border-zinc-700 bg-zinc-900/90 p-1">
          <button
            type="button"
            onClick={() => setCodecMode("jepa")}
            className={`rounded px-3 py-1.5 text-xs font-medium ${
              codecMode === "jepa" ? "bg-sky-600 text-white" : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            JEPA codec
          </button>
          <button
            type="button"
            onClick={() => setCodecMode("frozen")}
            className={`rounded px-3 py-1.5 text-xs font-medium ${
              codecMode === "frozen" ? "bg-sky-600 text-white" : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Frozen codec
          </button>
        </div>
        <button
          type="button"
          disabled={playing}
          onClick={() => void playTrainingFlow()}
          className="rounded-lg bg-violet-600 px-3 py-2 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
        >
          {playing ? "Playing flow..." : "Play training flow"}
        </button>
      </div>

      <aside className="absolute left-4 bottom-4 z-20 w-56 rounded-xl border border-zinc-700 bg-zinc-900/95 p-3 text-xs shadow-xl backdrop-blur">
        <h4 className="mb-2 font-semibold text-zinc-200">Training facts</h4>
        <Metric label="reconstruction_cosine" value="~0.99" healthy />
        <Metric label="next_vector_cosine" value="~0.58" healthy />
        <Metric label="std_z (collapse)" value="~0.62" healthy threshold={0.1} />
      </aside>

      <NodeDetailPanel node={selected} onClose={() => setSelected(null)} />
      <GeneratePanel onHighlightNodes={highlightNodes} onActiveEdge={setActiveEdgeId} />
      <FlowDot activeEdgeId={activeEdgeId} nodes={nodes} edges={edges} />

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.4}
        maxZoom={1.5}
        className="bg-zinc-950"
      >
        <Background color="#3f3f46" gap={20} />
        <Controls className="!bg-zinc-900 !border-zinc-700" />
        <MiniMap
          className="!bg-zinc-900 !border-zinc-700"
          nodeColor={(n) => {
            const kind = (n.data as ArchitectureNodeData)?.kind;
            if (kind === "loss") return "#f59e0b";
            if (kind === "target") return "#10b981";
            if (kind === "input") return "#38bdf8";
            return "#a78bfa";
          }}
        />
      </ReactFlow>
    </div>
  );
}

function Metric({
  label,
  value,
  healthy,
  threshold,
}: {
  label: string;
  value: string;
  healthy?: boolean;
  threshold?: number;
}) {
  return (
    <div className="mb-2 flex items-center justify-between gap-2">
      <span className="text-zinc-400">{label}</span>
      <span className="flex items-center gap-1.5 font-mono text-zinc-100">
        {value}
        {healthy !== undefined && (
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              healthy ? "bg-emerald-400" : "bg-amber-400"
            }`}
            title={threshold !== undefined ? `healthy when > ${threshold}` : undefined}
          />
        )}
      </span>
    </div>
  );
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
