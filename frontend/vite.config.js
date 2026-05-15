import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite is the build tool that:
//  - Runs a fast local dev server (npm run dev)
//  - Bundles everything into /dist for deployment (npm run build)
export default defineConfig({
  plugins: [react()],

  // Base path — keep "/" for Vercel/Netlify.
  // If deploying to GitHub Pages in a subfolder e.g. github.com/user/ogdc,
  // change this to "/ogdc/"
  base: "/",

  build: {
    // Output folder — this is what you upload/deploy
    outDir: "dist",

    // Increase warning threshold for large chunk sizes
    // (our app embeds chart images so chunks can be big)
    chunkSizeWarningLimit: 1000,
  },
});
