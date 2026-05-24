import { describe, expect, test } from "bun:test";
import { fuzzyRank } from "../components/Autocomplete.js";

interface Item {
  label: string;
  insert: string;
}

const items: Item[] = [
  { label: "/help", insert: "/help" },
  { label: "/status", insert: "/status" },
  { label: "/skills", insert: "/skills" },
  { label: "/exit", insert: "/exit" },
  { label: "/reset", insert: "/reset" },
];

describe("fuzzyRank", () => {
  test("empty query returns input unchanged", () => {
    expect(fuzzyRank(items, "")).toEqual(items);
  });

  test("prefix match outranks substring match", () => {
    const ranked = fuzzyRank(items, "ex");
    expect(ranked[0].label).toBe("/exit");
  });

  test("subsequence fuzzy matches when no prefix/substring fits", () => {
    // "skl" is a subsequence of "/skills"
    const ranked = fuzzyRank(items, "skl");
    expect(ranked.some((i) => i.label === "/skills")).toBe(true);
  });

  test("unrelated query yields empty list", () => {
    expect(fuzzyRank(items, "zzzzz")).toEqual([]);
  });
});
