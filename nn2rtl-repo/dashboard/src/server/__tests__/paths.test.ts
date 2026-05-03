import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { readAllowlistedFile, repoRoot } from "../paths.js";

const testPath = path.join(repoRoot, "output", "dashboard", "read-limit-test.txt");

afterEach(async () => {
  await rm(testPath, { force: true });
});

describe("readAllowlistedFile", () => {
  it("reads only up to the requested byte limit", async () => {
    await mkdir(path.dirname(testPath), { recursive: true });
    await writeFile(testPath, "abcdef", "utf8");

    const result = await readAllowlistedFile("output/dashboard/read-limit-test.txt", 3);

    expect(result).toMatchObject({
      path: "output/dashboard/read-limit-test.txt",
      content: "abc",
      sizeBytes: 6,
      truncated: true,
    });
  });
});
