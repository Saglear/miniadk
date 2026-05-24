import React from "react";
import { Box, Text } from "ink";
import { TranscriptItem } from "../protocol.js";
import { Markdown } from "./Markdown.js";

interface Props {
  items: TranscriptItem[];
  /** Optional in-flight assistant text rendered after committed items. */
  streamingAssistant?: string | null;
  /** When true, the last tool_result block renders in full instead of folded. */
  expandLast?: boolean;
}

export const Transcript: React.FC<Props> = ({ items, streamingAssistant, expandLast }) => {
  // Find index of the last tool_result so we can render it expanded.
  let lastToolResult = -1;
  for (let i = items.length - 1; i >= 0; i--) {
    if (items[i].kind === "tool_result") {
      lastToolResult = i;
      break;
    }
  }
  return (
    <Box flexDirection="column" paddingX={2}>
      {items.map((item, i) => (
        <TranscriptBlock
          key={i}
          item={item}
          expanded={Boolean(expandLast) && i === lastToolResult}
        />
      ))}
      {streamingAssistant && (
        <Box marginBottom={1}>
          <Markdown text={streamingAssistant} />
        </Box>
      )}
    </Box>
  );
};

interface BlockProps {
  item: TranscriptItem;
  expanded: boolean;
}

const TranscriptBlock: React.FC<BlockProps> = ({ item, expanded }) => {
  switch (item.kind) {
    case "user":
      return (
        <Box marginBottom={1}>
          <Box marginRight={1}>
            <Text color="cyan" bold>▌</Text>
          </Box>
          <Box flexDirection="column">
            {item.text.split("\n").map((line, i) => (
              <Text key={i}>{line}</Text>
            ))}
          </Box>
        </Box>
      );
    case "assistant":
      return (
        <Box marginBottom={1}>
          <Markdown text={item.text} />
        </Box>
      );
    case "tool_call":
      return (
        <Box marginTop={1}>
          <Text color="yellow">● {item.name}</Text>
          {Object.keys(item.arguments).length > 0 && (
            <Text dimColor>  {compactArgs(item.arguments)}</Text>
          )}
        </Box>
      );
    case "tool_result":
      return <ToolResultBlock item={item} expanded={expanded} />;
    case "tool_progress":
      return (
        <Text>
          <Text color="yellow">  {item.name}</Text>
          <Text dimColor> · {item.message}</Text>
        </Text>
      );
    case "tool_denied":
      return <Text color="red">  × {item.message}</Text>;
    case "error":
      return <Text color="red" bold>  × {item.message}</Text>;
    case "notice":
      return <Text dimColor>  · {item.text}</Text>;
  }
};


const ToolResultBlock: React.FC<{
  item: Extract<TranscriptItem, { kind: "tool_result" }>;
  expanded: boolean;
}> = ({ item, expanded }) => {
  const lines = item.text.split("\n");
  const isDiff = looksLikeDiff(item.text);
  const PREVIEW = 5;
  const visible = expanded ? lines : lines.slice(0, PREVIEW);
  const more = expanded ? 0 : lines.length - visible.length;

  return (
    <Box flexDirection="column" marginBottom={1} marginLeft={2}>
      <Box>
        <Text color="green">  ↳ </Text>
        <Text dimColor>
          {lines.length} lines · {item.text.length} chars
          {isDiff ? " · diff" : ""}
          {more > 0 ? "  ·  ctrl+r to expand" : expanded ? "  ·  ctrl+r to fold" : ""}
        </Text>
      </Box>
      <Box flexDirection="column" marginLeft={4}>
        {visible.map((line, i) =>
          isDiff ? (
            <DiffLine key={i} line={line} />
          ) : (
            <Text key={i} dimColor>
              {line}
            </Text>
          ),
        )}
        {more > 0 && (
          <Text dimColor italic>
            {`… (${more} more lines)`}
          </Text>
        )}
      </Box>
    </Box>
  );
};


const DiffLine: React.FC<{ line: string }> = ({ line }) => {
  if (line.startsWith("+++") || line.startsWith("---")) {
    return <Text bold dimColor>{line}</Text>;
  }
  if (line.startsWith("@@")) {
    return <Text color="cyan">{line}</Text>;
  }
  if (line.startsWith("+")) {
    return <Text color="green">{line}</Text>;
  }
  if (line.startsWith("-")) {
    return <Text color="red">{line}</Text>;
  }
  return <Text dimColor>{line}</Text>;
};


function looksLikeDiff(text: string): boolean {
  // Cheap heuristic — diffs almost always have a hunk header. Avoid
  // expensive parsing: Ink rerenders on every token, and most outputs
  // are NOT diffs.
  return /^@@ .* @@/m.test(text) || (/^--- /m.test(text) && /^\+\+\+ /m.test(text));
}


function compactArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args).slice(0, 3);
  const parts = entries.map(([k, v]) => {
    let s = String(v).replace(/\n/g, "\\n");
    if (s.length > 48) s = s.slice(0, 45) + "...";
    return `${k}=${s}`;
  });
  if (Object.keys(args).length > 3) parts.push("...");
  return parts.join(" ");
}
