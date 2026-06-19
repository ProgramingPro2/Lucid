import { useEffect, useState } from "react";
import { loadEmbeddings } from "../lib/embeddingsLoader";
import { normalizeEmbeddings, type NormalizedEmbedding } from "../lib/normalize";
import type { EmbeddingPoint } from "../types";

export function useEmbeddings() {
  const [raw, setRaw] = useState<EmbeddingPoint[]>([]);
  const [points, setPoints] = useState<NormalizedEmbedding[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    loadEmbeddings()
      .then((data) => {
        if (cancelled) return;
        setRaw(data);
        setPoints(normalizeEmbeddings(data));
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load embeddings");
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { raw, points, loading, error };
}
