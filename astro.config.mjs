// @ts-check
import { defineConfig, envField, fontProviders } from 'astro/config';
import tailwindcss from "@tailwindcss/vite";

// https://astro.build/config
export default defineConfig({
    vite: {
        plugins: [tailwindcss()],
    },
    env: {
        schema: {
            PUBLIC_GOOGLE_SITE_VERIFICATION: envField.string({
                access: "public",
                context: "client",
                optional: true,
            }),
        },
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
