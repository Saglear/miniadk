import React from "react";
import { Box, Text } from "ink";
import { PermissionMode } from "../protocol.js";

interface Props {
  agent: string;
  model: string;
  cwd: string;
  permissionMode?: PermissionMode;
  tokens?: number | null;
  /** Two-stage ctrl+c hint shown right of the bar when armed. */
  exitHint?: string | null;
}

const MODE_LABEL: Record<PermissionMode, string> = {
  default: "default",
  accept_edits: "accept edits",
  plan: "plan",
};

const MODE_COLOR: Record<PermissionMode, string> = {
  default: "gray",
  accept_edits: "green",
  plan: "yellow",
};

const MODE_GLYPH: Record<PermissionMode, string> = {
  default: "○",
  accept_edits: "●",
  plan: "◆",
};

export const StatusBar: React.FC<Props> = ({
  agent,
  model,
  cwd,
  permissionMode,
  tokens,
  exitHint,
}) => {
  const cwdShort = shortenCwd(cwd);
  const tokensColor = tokenColor(tokens ?? null);
  const tokensLabel = formatTokens(tokens ?? null);

  return (
    <Box justifyContent="space-between" paddingX={2}>
      <Box>
        {permissionMode && (
          <>
            <Text color={MODE_COLOR[permissionMode]}>
              {MODE_GLYPH[permissionMode]} {MODE_LABEL[permissionMode]}
            </Text>
            <Text dimColor>  ·  </Text>
          </>
        )}
        <Text dimColor>miniadk · {agent}</Text>
        <Text dimColor>  ·  shift+tab mode</Text>
      </Box>
      <Box>
        {exitHint && (
          <>
            <Text color="yellow">{exitHint}</Text>
            <Text dimColor>  ·  </Text>
          </>
        )}
        {tokensLabel && (
          <>
            <Text {...(tokensColor === "dim" ? { dimColor: true } : { color: tokensColor })}>
              {tokensLabel}
            </Text>
            <Text dimColor>  ·  </Text>
          </>
        )}
        <Text dimColor>
          {model} · {cwdShort}
        </Text>
      </Box>
    </Box>
  );
};

function shortenCwd(cwd: string): string {
  // Compress $HOME → ~ for compactness; truncate from the front so the
  // tail (the active project name) is always visible.
  const home = process.env.HOME;
  let display = cwd;
  if (home && (display === home || display.startsWith(home + "/"))) {
    display = "~" + display.slice(home.length);
  }
  if (display.length > 28) display = "…" + display.slice(-27);
  return display;
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
