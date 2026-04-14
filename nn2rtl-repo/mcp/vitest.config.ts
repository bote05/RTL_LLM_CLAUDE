import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    coverage: {
      provider: "v8",
      all: true,
      exclude: [
        "dist/**",
        "main.ts",
        "types.ts",
      ],
      include: [
        "schemas.ts",
        "server.ts",
        "tools.ts",
      ],
      thresholds: {
        branches: 95,
        functions: 100,
        lines: 100,
        statements: 100,
      },
    },
  },
});
