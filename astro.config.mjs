// @ts-check
import { defineConfig, fontProviders } from 'astro/config';
import tailwindcss from "@tailwindcss/vite";

// https://astro.build/config
export default defineConfig({
    vite: {
        plugins: [tailwindcss()],
    },
    experimental: {
        fonts: [
            {
                name: "Hina-Mincho",
                cssVariable: "--font-hina-mincho",
                fallbacks: ["sans-serif"],
                provider: fontProviders.local(),
                options: {
                    variants: [
                        {
                            src: ["./src/assets/fonts/Hina-Mincho-Regular.woff2"],
                        }
                    ]
                }
            }
        ]
    }
});
