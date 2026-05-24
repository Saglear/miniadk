import React from "react";
import { Box, Text } from "ink";
import { Intro } from "../protocol.js";

const LOGO = [
  "  ▄▄▄▄ ▗▖",
  " █    █▐▌",
  " █    █▐▌",
  " █    █▐▌",
  "  ▀▀▀▀ ▝▘",
];

interface Props {
  intro: Intro;
}

/** Two-column splash shown on startup. Hidden after the first user
 * submission so the transcript has the full height. */
export const Welcome: React.FC<Props> = ({ intro }) => {
  let cwdShort = intro.cwd;
  if (cwdShort.length > 48) cwdShort = "…" + cwdShort.slice(-47);

  return (
    <Box
      flexDirection="row"
      borderStyle="round"
      borderColor="cyan"
      paddingX={2}
      paddingY={1}
      marginX={2}
      marginY={1}
      alignSelf="flex-start"
    >
      <Box flexDirection="column" marginRight={3}>
        {LOGO.map((line, i) => (
          <Text key={i} color="cyan" bold>
            {line}
          </Text>
        ))}
      </Box>
      <Box flexDirection="column">
        <Text bold>Welcome to miniadk · {intro.agent}</Text>
        <Box marginTop={1} flexDirection="column">
          <Text dimColor>/help to list commands</Text>
          <Text dimColor>/status for session info</Text>
          <Text dimColor>ctrl+p to open the command palette</Text>
          <Text dimColor>ctrl+c to exit</Text>
        </Box>
        <Box marginTop={1}>
          <Text dimColor>
            {intro.model} · {intro.toolCount} tools
          </Text>
        </Box>
        <Text dimColor>{cwdShort}</Text>
      </Box>
    </Box>
  );
};
