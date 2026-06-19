import { memo } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import type { ArchitectureNodeData } from "../types";

const kindStyles: Record<ArchitectureNodeData["kind"], string> = {
  input: "border-sky-500/60 bg-sky-950/40",
  module: "border-violet-500/60 bg-violet-950/40",
  loss: "border-amber-500/60 bg-amber-950/40",
  target: "border-emerald-500/60 bg-emerald-950/40",
};

function ArchNodeComponent({ data, selected }: NodeProps<ArchitectureNodeData>) {
  return (
    <div
      className={`min-w-[140px] rounded-lg border px-3 py-2 shadow-lg transition ${
        kindStyles[data.kind]
      } ${selected ? "ring-2 ring-sky-400" : ""}`}
    >
      <Handle type="target" position={Position.Left} className="!bg-zinc-400" />
      <div className="text-xs uppercase tracking-wide text-zinc-400">{data.kind}</div>
      <div className="text-sm font-semibold text-zinc-100">{data.label}</div>
      <Handle type="source" position={Position.Right} className="!bg-zinc-400" />
    </div>
  );
}

export default memo(ArchNodeComponent);
