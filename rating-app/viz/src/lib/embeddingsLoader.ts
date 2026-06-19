import type { EmbeddingPoint } from "../types";

async function fetchJson(url: string): Promise<EmbeddingPoint[]> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return (await res.json()) as EmbeddingPoint[];
}

export async function loadEmbeddings(): Promise<EmbeddingPoint[]> {
  const sources = ["/api/embeddings", "/viz/embeddings.json", "/embeddings.json"];

  for (const url of sources) {
    try {
      return await fetchJson(url);
    } catch {
      // try next source
    }
  }

  throw new Error("Could not load embeddings from any source.");
}
