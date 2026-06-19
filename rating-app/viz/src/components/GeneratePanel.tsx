import { useCallback, useEffect, useState } from "react";
import { edgePathForSequence } from "../data/architectureGraph";

const MOCK_CONTINUATIONS = [
  "The river bent slowly through the valley below.",
  "She remembered the letter and its unfinished promise.",
  "A cold wind moved the branches against the window.",
  "Nothing in the room had changed, yet everything felt different.",
  "He waited until the sound of footsteps faded away.",
];

type Props = {
  onHighlightNodes: (ids: string[]) => void;
  onActiveEdge: (edgeId: string | null) => void;
};

export default function GeneratePanel({ onHighlightNodes, onActiveEdge }: Props) {
  const [prompt, setPrompt] = useState("The old house stood at the end of the lane.");
  const [steps, setSteps] = useState(3);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);

  const runRollout = useCallback(async () => {
    setRunning(true);
    setLog([]);

    const sequence = ["text", "encoder", "z"];
    onHighlightNodes(sequence);
    onActiveEdge(edgePathForSequence(sequence)[0] ?? null);
    setLog((l) => [...l, `Encode: "${prompt}" -> z_0`]);
    await sleep(900);

    for (let i = 0; i < steps; i++) {
      const predSeq = ["z-seq", "predictor", "z-hat"];
      onHighlightNodes(predSeq);
      onActiveEdge(edgePathForSequence(predSeq)[0] ?? null);
      setLog((l) => [...l, `Predict step ${i + 1}: z_${i + 1}`]);
      await sleep(900);

      const decSeq = ["z-hat", "decoder", "text-prime"];
      onHighlightNodes(decSeq);
      onActiveEdge(edgePathForSequence(decSeq)[0] ?? null);
      const text = MOCK_CONTINUATIONS[i % MOCK_CONTINUATIONS.length];
      setLog((l) => [...l, `Decode z_${i + 1}: "${text}"`]);
      await sleep(900);
    }

    onHighlightNodes([]);
    onActiveEdge(null);
    setRunning(false);
  }, [prompt, steps, onHighlightNodes, onActiveEdge]);

  useEffect(() => {
    return () => {
      onHighlightNodes([]);
      onActiveEdge(null);
    };
  }, [onHighlightNodes, onActiveEdge]);

  return (
    <div className="absolute bottom-4 right-4 z-20 w-96 rounded-xl border border-zinc-700 bg-zinc-900/95 p-4 shadow-xl backdrop-blur">
      <h3 className="text-sm font-semibold text-zinc-100">Generate (inference)</h3>
      <p className="mt-1 text-xs text-zinc-400">
        Simulated rollout: encode prompt, predict next ideas, decode each to text.
      </p>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={2}
        className="mt-3 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100"
      />
      <label className="mt-2 flex items-center gap-2 text-xs text-zinc-400">
        Steps
        <input
          type="number"
          min={1}
          max={6}
          value={steps}
          onChange={(e) => setSteps(Number(e.target.value))}
          className="w-16 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-100"
        />
      </label>
      <button
        type="button"
        disabled={running}
        onClick={() => void runRollout()}
        className="mt-3 rounded-lg bg-sky-600 px-3 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
      >
        {running ? "Running..." : "Run rollout"}
      </button>
      {log.length > 0 && (
        <ul className="mt-3 max-h-32 space-y-1 overflow-y-auto text-xs text-zinc-300">
          {log.map((line, i) => (
            <li key={`${line}-${i}`}>{line}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
