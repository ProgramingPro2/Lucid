import { NavLink, Route, Routes } from "react-router-dom";
import { useEmbeddings } from "./hooks/useEmbeddings";
import ArchitectureFlow from "./views/ArchitectureFlow";
import IdeaSpace from "./views/IdeaSpace";

function NavTab({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        `rounded-lg px-4 py-2 text-sm font-medium transition ${
          isActive
            ? "bg-sky-600 text-white"
            : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        }`
      }
    >
      {children}
    </NavLink>
  );
}

export default function App() {
  const { points, raw, loading, error } = useEmbeddings();

  return (
    <div className="flex min-h-full flex-col">
      <header className="flex h-14 items-center justify-between border-b border-zinc-800 bg-zinc-950 px-4">
        <div>
          <h1 className="text-base font-semibold text-zinc-100">Concept Model Visualizer</h1>
          <p className="text-xs text-zinc-500">Idea-space explorer & architecture diagram</p>
        </div>
        <nav className="flex gap-2">
          <NavTab to="/">Idea Space</NavTab>
          <NavTab to="/architecture">Architecture</NavTab>
        </nav>
      </header>

      <main className="flex-1">
        <Routes>
          <Route
            path="/"
            element={
              loading ? (
                <LoadingState message="Loading embeddings..." />
              ) : error ? (
                <ErrorState message={error} />
              ) : (
                <IdeaSpace points={points} raw={raw} />
              )
            }
          />
          <Route path="/architecture" element={<ArchitectureFlow />} />
        </Routes>
      </main>
    </div>
  );
}

function LoadingState({ message }: { message: string }) {
  return (
    <div className="flex h-[calc(100vh-3.5rem)] items-center justify-center text-zinc-400">
      {message}
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col items-center justify-center gap-2 text-center">
      <p className="text-red-400">{message}</p>
      <p className="max-w-md text-sm text-zinc-500">
        Ensure <code className="text-zinc-300">public/embeddings.json</code> exists or run{" "}
        <code className="text-zinc-300">python tools/generate_dev_embeddings.py</code>.
      </p>
    </div>
  );
}
