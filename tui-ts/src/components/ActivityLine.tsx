import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";

interface Props {
  text: string | null;
}

export const ActivityLine: React.FC<Props> = ({ text }) => {
  if (!text) return <Box height={1} />;
  return (
    <Box paddingX={2}>
      <Text color="cyan">
        <Spinner type="dots" />
      </Text>
      <Text dimColor> {text}</Text>
    </Box>
  );
};
