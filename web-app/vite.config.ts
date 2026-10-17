import path from "path"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ command }) => ({
  // GitHub Pages serves the app under the repository's name; another site sets its own path (RENAME_APP_BASE)
  base: command === 'build' ? (process.env.RENAME_APP_BASE || '/rename_photos_AI/') : '/',
  plugins: [react(), tailwindcss()],
  // onnxruntime-web loads its .wasm next to its own module; pre-bundling moves it and breaks that
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  // the photo workers load onnxruntime's WebGPU or WebAssembly build when they start (a dynamic import)
  worker: { format: 'es' },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}))
