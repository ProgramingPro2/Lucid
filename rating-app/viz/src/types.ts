export type EmbeddingPoint = {
  id: number;
  text: string;
  x: number;
  y: number;
  z: number;
  cluster: number;
};

export type CodecMode = "jepa" | "frozen";

export type ArchitectureNodeData = {
  label: string;
  kind: "input" | "module" | "loss" | "target";
  description: string;
  tensors: string[];
};
