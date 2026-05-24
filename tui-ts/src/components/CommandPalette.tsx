import React, { useState, useMemo } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";

export interface PaletteCommand {
  name: string;
  description: string;
  group?: string;
}

interface Props {
  commands: PaletteCommand[];
  onSelect: (command: PaletteCommand) => void;
  onClose: () => void;
}

/** Full-screen-ish modal: search box + scrolling list of slash commands. */
export const CommandPalette: React.FC<Props> = ({ commands, onSelect, onClose }) => {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase().replace(/^\//, "");
    if (!q) return commands;
    return commands.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        c.description.toLowerCase().includes(q),
    );
  }, [commands, query]);

  useInput((_input, key) => {
    if (key.escape) {
      onClose();
      return;
    }
    if (key.return) {
      const choice = filtered[cursor];
      if (choice) onSelect(choice);
      return;
    }
    if (key.downArrow) setCursor((c) => Math.min(c + 1, filtered.length - 1));
    if (key.upArrow) setCursor((c) => Math.max(c - 1, 0));
  });

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="cyan"
      paddingX={1}
      paddingY={0}
      marginX={4}
      marginY={1}
    >
      <Box paddingX={1}>
        <Text color="cyan">› </Text>
        <TextInput
          value={query}
          onChange={(v) => {
            setQuery(v);
            setCursor(0);
          }}
          placeholder="search commands…"
        />
      </Box>
      <Box flexDirection="column" marginTop={1}>
        {filtered.length === 0 ? (
          <Text dimColor>  no matches</Text>
        ) : (
          filtered.slice(0, 12).map((cmd, i) => (
            <Box key={cmd.name}>
              <Text color={i === cursor ? "cyan" : undefined} bold={i === cursor}>
                {i === cursor ? "›" : " "}{" "}/{cmd.name}
              </Text>
              <Text dimColor>{"  "}{cmd.description}</Text>
            </Box>
          ))
        )}
      </Box>
      <Box marginTop={1} paddingX={1}>
        <Text dimColor italic>
          enter to run · esc to close · ↑/↓ to navigate
        </Text>
      </Box>
    </Box>
  );
};
