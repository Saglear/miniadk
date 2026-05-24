import React from "react";
import { Box, Text } from "ink";
import { marked } from "marked";
import { highlight, supportsLanguage } from "cli-highlight";
import chalk from "chalk";

/**
 * Render markdown to ANSI for display in Ink.
 *
 * Strategy: marked tokenises, we map each token to an Ink element and
 * compose ANSI for inline runs through `chalk`. Code fences route to
 * `cli-highlight` (a curated highlight.js bundle).
 *
 * We deliberately avoid `marked-terminal` — it stacks ANSI styles in
 * ways Ink's renderer fights with, producing dropped colours and
 * orphan reset codes.
 */

interface Props {
  text: string;
}

export const Markdown: React.FC<Props> = ({ text }) => {
  const tokens = marked.lexer(text);
  return (
    <Box flexDirection="column">
      {tokens.map((token, i) => (
        <TokenView key={i} token={token} depth={0} />
      ))}
    </Box>
  );
};

const TokenView: React.FC<{ token: marked.Token; depth: number }> = ({
  token,
  depth,
}) => {
  switch (token.type) {
    case "heading":
      return <Heading token={token as marked.Tokens.Heading} />;
    case "paragraph":
      return <Paragraph token={token as marked.Tokens.Paragraph} />;
    case "code":
      return <CodeBlock token={token as marked.Tokens.Code} />;
    case "list":
      return <ListBlock token={token as marked.Tokens.List} depth={depth} />;
    case "table":
      return <TableBlock token={token as marked.Tokens.Table} />;
    case "blockquote":
      return <Blockquote token={token as marked.Tokens.Blockquote} />;
    case "hr":
      return <HR />;
    case "space":
      return null;
    case "text":
      return (
        <Text wrap="wrap">{renderInline((token as { text?: string }).text ?? "")}</Text>
      );
    default:
      return <Text>{(token as { raw?: string }).raw ?? ""}</Text>;
  }
};

// ── headings ─────────────────────────────────────────────────────────
// Six tones — H1 grabs attention, lower levels step down in saturation.
const HEADING_COLORS = ["magenta", "cyan", "yellow", "green", "blue", "white"] as const;

const Heading: React.FC<{ token: marked.Tokens.Heading }> = ({ token }) => {
  const depth = Math.min(Math.max(token.depth, 1), 6);
  const color = HEADING_COLORS[depth - 1];
  // A short accent bar gives the heading visual weight without the
  // noise of repeated `#` chars.
  const accent = depth === 1 ? "▎▎ " : depth === 2 ? "▎ " : depth === 3 ? "› " : "";
  return (
    <Box marginTop={depth <= 2 ? 1 : 0} marginBottom={1}>
      <Text bold color={color}>
        {accent}
        {renderInline(token.text)}
      </Text>
    </Box>
  );
};

// ── paragraph / text ─────────────────────────────────────────────────
const Paragraph: React.FC<{ token: marked.Tokens.Paragraph }> = ({ token }) => (
  <Box marginBottom={1}>
    <Text wrap="wrap">{renderInline(token.text)}</Text>
  </Box>
);

// ── code fence ───────────────────────────────────────────────────────
const CodeBlock: React.FC<{ token: marked.Tokens.Code }> = ({ token }) => {
  const lang = (token.lang || "").trim();
  const safeLang = lang && supportsLanguage(lang) ? lang : undefined;
  let body = token.text;
  try {
    if (safeLang) body = highlight(token.text, { language: safeLang, ignoreIllegals: true });
  } catch {
    // Fall through to raw text on highlighter errors (unknown grammar
    // tokens, malformed snippets) — better than crashing the render.
  }
  // marked sometimes leaves a trailing newline; trim it so the closing
  // border line isn't preceded by a blank row.
  const lines = body.replace(/\n+$/, "").split("\n");
  return (
    <Box
      flexDirection="column"
      marginBottom={1}
      marginLeft={2}
      borderStyle="round"
      borderColor="gray"
      paddingX={1}
    >
      {lang ? (
        <Text dimColor italic>
          {lang}
        </Text>
      ) : null}
      {lines.map((line, i) => (
        <Text key={i}>{line.length === 0 ? " " : line}</Text>
      ))}
    </Box>
  );
};

// ── lists ────────────────────────────────────────────────────────────
const BULLETS = ["•", "◦", "▪"] as const;

const ListBlock: React.FC<{ token: marked.Tokens.List; depth: number }> = ({
  token,
  depth,
}) => (
  <Box flexDirection="column" marginBottom={depth === 0 ? 1 : 0}>
    {token.items.map((item, i) => (
      <ListItem
        key={i}
        item={item}
        index={i}
        ordered={token.ordered}
        depth={depth}
      />
    ))}
  </Box>
);

const ListItem: React.FC<{
  item: marked.Tokens.ListItem;
  index: number;
  ordered: boolean;
  depth: number;
}> = ({ item, index, ordered, depth }) => {
  // GitHub task-list items: `[ ] todo` / `[x] done`. marked sets
  // `task` and `checked` directly when it detects them.
  if (item.task) {
    return (
      <Box marginLeft={depth * 2}>
        <Text color={item.checked ? "green" : "cyan"}>
          {`  ${item.checked ? "✔" : "·"} `}
        </Text>
        <Box flexGrow={1}>
          <Text dimColor={item.checked} strikethrough={item.checked} wrap="wrap">
            {renderInline(stripTaskMarker(item.text))}
          </Text>
        </Box>
      </Box>
    );
  }
  const bullet = ordered ? `${index + 1}.` : BULLETS[Math.min(depth, BULLETS.length - 1)];
  // The first paragraph-like child holds the leading text; nested
  // lists / code etc. follow.
  const leadIdx = (item.tokens ?? []).findIndex(
    (t) => t.type === "text" || t.type === "paragraph",
  );
  const lead = leadIdx >= 0 ? (item.tokens![leadIdx] as { text?: string }).text ?? "" : item.text;
  const rest = (item.tokens ?? []).filter((_, i) => i !== leadIdx);
  return (
    <Box flexDirection="column" marginLeft={depth * 2}>
      <Box>
        <Text color="cyan">{`  ${bullet} `}</Text>
        <Box flexGrow={1}>
          <Text wrap="wrap">{renderInline(lead)}</Text>
        </Box>
      </Box>
      {rest.map((sub, i) => (
        <Box key={i} marginLeft={4}>
          <TokenView token={sub} depth={depth + 1} />
        </Box>
      ))}
    </Box>
  );
};

function stripTaskMarker(text: string): string {
  return text.replace(/^\[[ xX]\]\s+/, "");
}

// ── tables ───────────────────────────────────────────────────────────
const TableBlock: React.FC<{ token: marked.Tokens.Table }> = ({ token }) => {
  const headers = token.header.map((h) => renderInline(h.text));
  const rows = token.rows.map((row) => row.map((cell) => renderInline(cell.text)));
  // CJK characters render at width 2 in monospaced terminals. Compute
  // visible width from the unstyled string so columns line up
  // regardless of ANSI/colour bytes.
  const widths = headers.map((h, ci) =>
    Math.max(visibleWidth(h), ...rows.map((r) => visibleWidth(r[ci] ?? ""))),
  );
  const fmt = (cells: string[]) =>
    cells.map((c, i) => padToWidth(c, widths[i] ?? 0)).join("  ");
  const sep = widths.map((w) => "─".repeat(w)).join("  ");
  return (
    <Box flexDirection="column" marginBottom={1} marginLeft={2}>
      <Text bold color="cyan">{fmt(headers)}</Text>
      <Text dimColor>{sep}</Text>
      {rows.map((row, i) => (
        <Text key={i}>{fmt(row)}</Text>
      ))}
    </Box>
  );
};

// ── blockquote ───────────────────────────────────────────────────────
const Blockquote: React.FC<{ token: marked.Tokens.Blockquote }> = ({ token }) => {
  const children = token.tokens ?? [];
  return (
    <Box marginBottom={1}>
      <Text color="cyan">▎ </Text>
      <Box flexDirection="column" flexGrow={1}>
        {children.length > 0 ? (
          children.map((t, i) => <TokenView key={i} token={t} depth={0} />)
        ) : (
          <Text dimColor wrap="wrap">{renderInline(token.text)}</Text>
        )}
      </Box>
    </Box>
  );
};

// ── horizontal rule ──────────────────────────────────────────────────
const HR: React.FC = () => (
  <Box marginY={1}>
    <Text dimColor>{"─".repeat(40)}</Text>
  </Box>
);

// ── inline ───────────────────────────────────────────────────────────
/**
 * Render inline markdown to a chalk-coloured string. Returned as a
 * plain string so callers can drop it into a single <Text>, which Ink
 * needs for its `wrap` behaviour to do anything useful.
 *
 * Order matters: triple-emphasis must run before double, double before
 * single, otherwise `**foo**` is shredded by the italic regex.
 */
function renderInline(text: string): string {
  if (!text) return "";
  let out = text;
  out = out.replace(/\*\*\*([^*\n]+)\*\*\*/g, (_, m) => chalk.bold.italic(m));
  out = out.replace(/___([^_\n]+)___/g, (_, m) => chalk.bold.italic(m));
  out = out.replace(/\*\*([^*\n]+)\*\*/g, (_, m) => chalk.bold(m));
  out = out.replace(/__([^_\n]+)__/g, (_, m) => chalk.bold(m));
  out = out.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, (_, m) => chalk.italic(m));
  out = out.replace(/(?<![\w_])_([^_\n]+)_(?![\w_])/g, (_, m) => chalk.italic(m));
  out = out.replace(/~~([^~\n]+)~~/g, (_, m) => chalk.strikethrough(m));
  // Inline code gets a subtle background so it stands apart from the
  // surrounding prose — much more legible than plain yellow text.
  out = out.replace(/`([^`\n]+)`/g, (_, m) => chalk.bgBlackBright.yellowBright(` ${m} `));
  // Markdown links → OSC-8 hyperlinks. We replace with a placeholder
  // first so the bare-URL fallback below can't re-match the URL we
  // just embedded in the escape sequence.
  const linkPlaceholders: string[] = [];
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) => {
    const idx = linkPlaceholders.length;
    linkPlaceholders.push(`\x1b]8;;${url}\x1b\\${chalk.cyan.underline(label)}\x1b]8;;\x1b\\`);
    return `\x00LINK${idx}\x00`;
  });
  // Bare URLs that aren't already inside a markdown link.
  out = out.replace(/(?<![("\w])(https?:\/\/[^\s)]+)/g, (_, url) =>
    `\x1b]8;;${url}\x1b\\${chalk.cyan.underline(url)}\x1b]8;;\x1b\\`,
  );
  out = out.replace(/\x00LINK(\d+)\x00/g, (_, n) => linkPlaceholders[Number(n)] ?? "");
  return out;
}

// ── width helpers ────────────────────────────────────────────────────
const ANSI_CSI_RE = /\x1b\[[0-9;]*m/g;
const ANSI_OSC8_RE = /\x1b\]8;;[^\x07\x1b]*(\x07|\x1b\\)/g;

function stripAnsi(s: string): string {
  return s.replace(ANSI_CSI_RE, "").replace(ANSI_OSC8_RE, "");
}

/**
 * East-Asian-width-aware visible length. Covers the CJK ranges that
 * matter in practice — full-width punctuation, CJK ideographs, hangul,
 * kana, and the supplementary CJK planes. Anything outside those falls
 * back to width 1, which is correct for ASCII and most Latin/Cyrillic.
 */
function visibleWidth(s: string): number {
  const plain = stripAnsi(s);
  let w = 0;
  for (const ch of plain) {
    const cp = ch.codePointAt(0) ?? 0;
    if (
      cp >= 0x1100 &&
      ((cp <= 0x115f) ||
        (cp >= 0x2e80 && cp <= 0x9fff) ||
        (cp >= 0xa000 && cp <= 0xa4cf) ||
        (cp >= 0xac00 && cp <= 0xd7a3) ||
        (cp >= 0xf900 && cp <= 0xfaff) ||
        (cp >= 0xfe30 && cp <= 0xfe4f) ||
        (cp >= 0xff00 && cp <= 0xff60) ||
        (cp >= 0xffe0 && cp <= 0xffe6) ||
        (cp >= 0x20000 && cp <= 0x2fffd) ||
        (cp >= 0x30000 && cp <= 0x3fffd))
    ) {
      w += 2;
    } else {
      w += 1;
    }
  }
  return w;
}

function padToWidth(s: string, width: number): string {
  return s + " ".repeat(Math.max(0, width - visibleWidth(s)));
}
