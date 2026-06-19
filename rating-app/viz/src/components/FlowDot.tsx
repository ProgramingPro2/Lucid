import { useEffect, useRef, useState } from "react";
import { getBezierPath, type Edge, type Node } from "reactflow";

type Props = {
  activeEdgeId: string | null;
  nodes: Node[];
  edges: Edge[];
};

export default function FlowDot({ activeEdgeId, nodes, edges }: Props) {
  const dotRef = useRef<SVGCircleElement>(null);
  const [pathD, setPathD] = useState<string>("");

  useEffect(() => {
    if (!activeEdgeId) {
      setPathD("");
      return;
    }

    const edge = edges.find((e) => e.id === activeEdgeId);
    if (!edge) return;

    const source = nodes.find((n) => n.id === edge.source);
    const target = nodes.find((n) => n.id === edge.target);
    if (!source || !target) return;

    const [d] = getBezierPath({
      sourceX: source.position.x + 140,
      sourceY: source.position.y + 30,
      sourcePosition: undefined,
      targetX: target.position.x,
      targetY: target.position.y + 30,
      targetPosition: undefined,
    });
    setPathD(d);
  }, [activeEdgeId, nodes, edges]);

  useEffect(() => {
    const dot = dotRef.current;
    if (!dot || !pathD) return;

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pathD);
    const length = path.getTotalLength();
    let start: number | null = null;

    const step = (ts: number) => {
      if (start === null) start = ts;
      const t = ((ts - start) % 2000) / 2000;
      const pt = path.getPointAtLength(t * length);
      dot.setAttribute("cx", String(pt.x));
      dot.setAttribute("cy", String(pt.y));
      requestAnimationFrame(step);
    };

    const id = requestAnimationFrame(step);
    return () => cancelAnimationFrame(id);
  }, [pathD]);

  if (!pathD) return null;

  return (
    <svg className="pointer-events-none absolute inset-0 z-10 h-full w-full">
      <path d={pathD} fill="none" stroke="transparent" strokeWidth={1} />
      <circle ref={dotRef} r={6} fill="#38bdf8" stroke="#0ea5e9" strokeWidth={2} />
    </svg>
  );
}
