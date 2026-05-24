import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";

export interface CompletionItem {
  /** What gets inserted when the user accepts. */
  insert: string;
  /** What's shown on the left of the row. */
  label: string;
  /** Optional muted descriptor on the right. */
  hint?: string;
  /** Optional category color name (e.g. "cyan"). */
  category?: string;
}

interface Props {
  items: CompletionItem[];
  cursor: number;
  visible: boolean;
}

/**
 * Inline dropdown rendered just above the input bar. Used for both
 * `/`-command discovery and `@`-file completion.
 */
export const Autocomplete: React.FC<Props> = ({ items, cursor, visible }) => {
  if (!visible || items.length === 0) return null;

  const max = Math.min(items.length, 8);
  // Window the list so the cursor is always visible.
  let start = 0;
  if (cursor >= max) start = cursor - max + 1;
  if (start + max > items.length) start = Math.max(0, items.length - max);
  const window = items.slice(start, start + max);

  return (
    <Box
      flexDirection="column"
      marginX={2}
      paddingX={1}
      borderStyle="round"
      borderColor="gray"
    >
      {window.map((item, i) => {
        const realIndex = start + i;
        const selected = realIndex === cursor;
        return (
          <Box key={realIndex}>
            <Text color={selected ? "cyan" : undefined} bold={selected}>
              {selected ? "›" : " "}{" "}
            </Text>
            <Text
              color={selected ? "cyan" : item.category}
              bold={selected}
            >
              {item.label}
            </Text>
            {item.hint && (
              <>
                <Text>  </Text>
                <Text dimColor>{item.hint}</Text>
              </>
            )}
          </Box>
        );
      })}
      {items.length > max && (
        <Text dimColor italic>
          {`  +${items.length - max} more  ↑/↓ navigate · tab complete · enter submit`}
        </Text>
      )}
    </Box>
  );
};


/**
 * Fuzzy-rank `items` against `query`. Scoring is simple but covers the
 * common case: prefix matches beat substring matches; characters
 * appearing earlier in the label outrank later ones.
 */
export function fuzzyRank<T extends { label: string }>(items: T[], query: string): T[] {
  if (!query) return items;
  const needle = query.toLowerCase();
  const scored: { item: T; score: number }[] = [];
  for (const item of items) {
    const haystack = item.label.toLowerCase();
    if (haystack.startsWith(needle)) {
      scored.push({ item, score: 1000 - haystack.indexOf(needle) });
      continue;
    }
    const idx = haystack.indexOf(needle);
    if (idx >= 0) {
      scored.push({ item, score: 500 - idx });
      continue;
    }
    // Subsequence fuzzy fallback.
    let cursor = 0;
    let score = 0;
    let lastMatchIndex = -2;
    for (let i = 0; i < haystack.length && cursor < needle.length; i++) {
      if (haystack[i] === needle[cursor]) {
        score += i === lastMatchIndex + 1 ? 5 : 1;
        lastMatchIndex = i;
        cursor++;
      }
    }
    if (cursor === needle.length) {
      scored.push({ item, score });
    }
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.map((s) => s.item);
}
