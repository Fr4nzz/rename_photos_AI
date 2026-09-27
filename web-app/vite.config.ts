import path from "path"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/rename_photos_AI/' : '/',
  plugins: [react(), tailwindcss()],
  // onnxruntime-web loads its .wasm next to its own module; pre-bundling moves it and breaks that
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}))
