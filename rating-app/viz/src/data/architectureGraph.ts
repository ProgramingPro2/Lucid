import type { Edge, Node } from "reactflow";
import type { ArchitectureNodeData, CodecMode } from "../types";

const BASE_NODES: Omit<Node<ArchitectureNodeData>, "data">[] = [
  { id: "text", position: { x: 0, y: 120 }, type: "archNode" },
  { id: "encoder", position: { x: 220, y: 120 }, type: "archNode" },
  { id: "z", position: { x: 440, y: 120 }, type: "archNode" },
  { id: "decoder", position: { x: 660, y: 120 }, type: "archNode" },
  { id: "text-prime", position: { x: 880, y: 120 }, type: "archNode" },
  { id: "recon-loss", position: { x: 880, y: 280 }, type: "archNode" },
  { id: "z-seq", position: { x: 440, y: 400 }, type: "archNode" },
  { id: "predictor", position: { x: 660, y: 400 }, type: "archNode" },
  { id: "z-hat", position: { x: 880, y: 400 }, type: "archNode" },
  { id: "next-sentence", position: { x: 0, y: 520 }, type: "archNode" },
  { id: "ema", position: { x: 220, y: 520 }, type: "archNode" },
  { id: "z-target", position: { x: 440, y: 520 }, type: "archNode" },
  { id: "pred-loss", position: { x: 880, y: 520 }, type: "archNode" },
  { id: "vicreg", position: { x: 660, y: 280 }, type: "archNode" },
];

const EDGES: Edge[] = [
  { id: "e-text-enc", source: "text", target: "encoder", label: "token ids", animated: true },
  { id: "e-enc-z", source: "encoder", target: "z", label: "512-d L2 norm", animated: true },
  { id: "e-z-dec", source: "z", target: "decoder", label: "concept vector", animated: true },
  { id: "e-dec-tp", source: "decoder", target: "text-prime", label: "autoregressive", animated: true },
  { id: "e-tp-recon", source: "text-prime", target: "recon-loss", label: "compare", animated: true },
  { id: "e-text-recon", source: "text", target: "recon-loss", label: "target", animated: true },
  { id: "e-z-vicreg", source: "z", target: "vicreg", label: "batch stats", animated: true },
  { id: "e-zseq-pred", source: "z-seq", target: "predictor", label: "context (8)", animated: true },
  { id: "e-pred-zhat", source: "predictor", target: "z-hat", label: "z_hat(t+1)", animated: true },
  {
    id: "e-next-ema",
    source: "next-sentence",
    target: "ema",
    label: "next sentence",
    animated: true,
    style: { strokeDasharray: "6 4" },
  },
  {
    id: "e-ema-ztarget",
    source: "ema",
    target: "z-target",
    label: "stop-grad",
    animated: true,
    style: { strokeDasharray: "6 4" },
  },
  { id: "e-ztarget-ploss", source: "z-target", target: "pred-loss", label: "target", animated: true },
  { id: "e-zhat-ploss", source: "z-hat", target: "pred-loss", label: "prediction", animated: true },
  { id: "e-z-zseq", source: "z", target: "z-seq", label: "history", animated: false },
  { id: "e-zhat-dec", source: "z-hat", target: "decoder", label: "inference", animated: true },
];

function nodeData(
  label: string,
  kind: ArchitectureNodeData["kind"],
  description: string,
  tensors: string[],
): ArchitectureNodeData {
  return { label, kind, description, tensors };
}

function codecDescriptions(mode: CodecMode) {
  if (mode === "jepa") {
    return {
      encoder: {
        description:
          "Transformer encoder (6 layers, 8 heads, d_model 512). Maps tokenized sentences to a single L2-normalized concept vector per sentence.",
        tensors: ["Input: (B, T) token ids", "Output: (B, 512) L2-normalized z"],
      },
      decoder: {
        description:
          "Autoregressive Transformer decoder (6 layers, 8 heads). Generates novel sentences token-by-token from a concept vector (GPT-2 vocab, 50257 tokens).",
        tensors: ["Input: (B, 512) concept vector", "Output: (B, T', 50257) logits"],
      },
    };
  }

  return {
    encoder: {
      description:
        "Frozen sentence-transformer (all-MiniLM-L6-v2, 384-d). Produces fixed pretrained embeddings; predictor operates in a projected 512-d space.",
      tensors: ["Input: raw sentence strings", "Output: (B, 384) frozen embeddings"],
    },
    decoder: {
      description:
        "Retrieval decoder: nearest-neighbor lookup in a corpus bank (~50k sentences). Decode = snap z to the closest real sentence by cosine similarity.",
      tensors: ["Input: (B, 512) query vector", "Output: retrieved sentence text (no generation)"],
    },
  };
}

export function getGraph(mode: CodecMode): { nodes: Node<ArchitectureNodeData>[]; edges: Edge[] } {
  const codec = codecDescriptions(mode);

  const dataById: Record<string, ArchitectureNodeData> = {
    text: nodeData(
      "Text (sentence)",
      "input",
      "Raw sentence input segmented from corpus documents. Each sentence becomes one concept vector.",
      ["Shape: (batch, seq_len) token ids after BPE/char tokenization"],
    ),
    encoder: nodeData("Encoder", "module", codec.encoder.description, codec.encoder.tensors),
    z: nodeData(
      "Idea vector z",
      "module",
      "512-dimensional L2-normalized concept embedding representing one sentence-level idea.",
      ["Shape: (512,) unit vector", "Dot products = cosine similarity"],
    ),
    decoder: nodeData("Decoder", "module", codec.decoder.description, codec.decoder.tensors),
    "text-prime": nodeData(
      "Text' (reconstruction)",
      "input",
      "Decoder output: reconstructed or retrieved sentence compared against the original.",
      ["Shape: decoded token sequence or retrieved string"],
    ),
    "recon-loss": nodeData(
      "Reconstruction loss",
      "loss",
      "Anchors the concept space: Text' must match original Text. Prevents collapse to a constant vector.",
      ["Cosine similarity in embedding space", "Token-level cross-entropy for JEPA decoder"],
    ),
    "z-seq": nodeData(
      "z_1..z_t",
      "input",
      "Window of prior concept vectors within a document (context length 8 for JEPA predictor).",
      ["Shape: (B, 8, 512) padded sequence"],
    ),
    predictor: nodeData(
      "Predictor",
      "module",
      "Causal Transformer (6 layers, 8 heads) over concept vectors. Predicts the next idea vector from context.",
      ["Input: (B, 8, 512) context", "Output: (B, 512) predicted z_hat(t+1)"],
    ),
    "z-hat": nodeData(
      "z_hat(t+1)",
      "module",
      "Predicted next concept vector from the predictor head.",
      ["Shape: (B, 512) L2-normalized prediction"],
    ),
    "next-sentence": nodeData(
      "Next sentence",
      "input",
      "The actual next sentence in the document — used only as a stop-gradient target.",
      ["Raw text of sentence t+1"],
    ),
    ema: nodeData(
      "EMA target encoder",
      "target",
      "Slow momentum copy of the encoder (momentum 0.99). Gradients are stopped — the encoder cannot make targets trivial.",
      ["Updated: theta_ema = 0.99 * theta_ema + 0.01 * theta", "stop-grad on forward pass"],
    ),
    "z-target": nodeData(
      "z_target(t+1)",
      "target",
      "Target concept vector from the EMA encoder applied to the next sentence.",
      ["Shape: (B, 512) stop-grad target"],
    ),
    "pred-loss": nodeData(
      "Prediction loss",
      "loss",
      "Compares predicted z_hat(t+1) to EMA target z_target(t+1) using cosine distance and MSE in latent space.",
      ["L_pred = MSE(z_hat, z_target) + (1 - cos(z_hat, z_target))"],
    ),
    vicreg: nodeData(
      "VICReg",
      "loss",
      "Variance + covariance regularization on unnormalized projections. Anti-collapse: pushes per-dimension std above margin and decorrelates dimensions.",
      ["L_var: hinge on per-dim std", "L_cov: off-diagonal covariance penalty"],
    ),
  };

  const nodes = BASE_NODES.map((n) => ({
    ...n,
    data: dataById[n.id],
  }));

  return { nodes, edges: EDGES.map((e) => ({ ...e })) };
}

export const FLOW_SEQUENCES = {
  training: [
    ["text", "encoder", "z"],
    ["z", "decoder", "text-prime", "recon-loss"],
    ["z-seq", "predictor", "z-hat", "pred-loss"],
    ["z", "vicreg"],
  ],
  inference: [
    ["text", "encoder", "z"],
    ["z-seq", "predictor", "z-hat"],
    ["z-hat", "decoder", "text-prime"],
  ],
} as const;

export function edgePathForSequence(sequence: readonly string[]): string[] {
  const paths: string[] = [];
  for (let i = 0; i < sequence.length - 1; i++) {
    const source = sequence[i];
    const target = sequence[i + 1];
    const edge = EDGES.find((e) => e.source === source && e.target === target);
    if (edge) paths.push(edge.id);
  }
  return paths;
}
