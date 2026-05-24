import { describe, expect, test } from "bun:test";

// Re-implement the diff detector + token formatter standalone so the
// test doesn't have to import the full Ink component tree (Ink renders
// would try to acquire a tty during test setup).

function looksLikeDiff(text: string): boolean {
  return /^@@ .* @@/m.test(text) || (/^--- /m.test(text) && /^\+\+\+ /m.test(text));
}

function formatTokens(tokens: number | null): string {
  if (tokens === null || tokens === 0) return "";
  if (tokens >= 1000) return `~${(tokens / 1000).toFixed(1)}k tok`;
  return `~${tokens} tok`;
}

function tokenColor(tokens: number | null): string {
  if (tokens === null) return "dim";
  if (tokens >= 100_000) return "red";
  if (tokens >= 50_000) return "yellow";
  return "dim";
}

describe("looksLikeDiff", () => {
  test("detects unified diff hunk header", () => {
    expect(looksLikeDiff("@@ -1,3 +1,4 @@\n line\n-old\n+new")).toBe(true);
  });

  test("detects --- / +++ pair", () => {
    expect(looksLikeDiff("--- a/foo.txt\n+++ b/foo.txt\n @@ noisy")).toBe(true);
  });

  test("rejects plain text", () => {
    expect(looksLikeDiff("hello world")).toBe(false);
  });

  test("rejects markdown that uses + and -", () => {
    expect(looksLikeDiff("- bullet\n+ another\n")).toBe(false);
  });
});

describe("formatTokens", () => {
  test("hides zero / null", () => {
    expect(formatTokens(0)).toBe("");
    expect(formatTokens(null)).toBe("");
  });

  test("uses k suffix above 1000", () => {
    expect(formatTokens(1500)).toBe("~1.5k tok");
    expect(formatTokens(123_456)).toBe("~123.5k tok");
  });

  test("falls back to raw count below 1000", () => {
    expect(formatTokens(42)).toBe("~42 tok");
  });
});

describe("tokenColor", () => {
  test("dim under 50k", () => {
    expect(tokenColor(0)).toBe("dim");
    expect(tokenColor(40_000)).toBe("dim");
  });

  test("yellow at 50k+", () => {
    expect(tokenColor(50_000)).toBe("yellow");
    expect(tokenColor(99_999)).toBe("yellow");
  });

  test("red at 100k+", () => {
    expect(tokenColor(100_000)).toBe("red");
    expect(tokenColor(500_000)).toBe("red");
  });
});
