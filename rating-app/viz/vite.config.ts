import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
  const isFastApi = mode === "fastapi";

  return {
    plugins: [react(), tailwindcss()],
    base: isFastApi ? "/viz/" : "/",
    build: {
      outDir: isFastApi ? "../static/viz" : "dist",
      emptyOutDir: true,
    },
  };
});
