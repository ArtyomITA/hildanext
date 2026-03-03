import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Forward /api/* to the Python FastAPI server.
      "/api": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
        configure: (proxy) => {
          // Suppress ALL proxy errors when backend is offline.
          // No more ECONNREFUSED spam in terminal.
          proxy.on("error", (_err, _req, res) => {
            if ("writeHead" in res) {
              try { res.writeHead(503).end(); } catch { /* already sent */ }
            } else {
              (res as import("net").Socket).destroy();
            }
          });
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    exclude: ["src/test/e2e/**"],
  },
});
