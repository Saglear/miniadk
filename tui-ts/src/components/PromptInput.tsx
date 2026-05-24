import React from "react";
import { Box, Text } from "ink";
import TextInput from "ink-text-input";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: (v: string) => void;
  disabled?: boolean;
  /** Inline status hint shown next to the prompt when disabled. */
  busyHint?: string;
  /** When true, the input border / chevron use accent colour. Visual
   * cue that the input is the focused interactive surface. */
  focused?: boolean;
}

/**
 * The composer at the bottom of the screen.
 *
 * Visual contract:
 *   - rounded border, cyan when focused / gray when blurred
 *   - bright `❯` chevron so the user immediately sees this is an input
 *   - explicit cursor (showCursor=true) so the caret blinks
 *   - placeholder uses a different style from the typed text so it's
 *     obvious the field is empty
 */
export const PromptInput: React.FC<Props> = ({
  value,
  onChange,
  onSubmit,
  disabled,
  busyHint,
  focused = true,
}) => {
  const accent = focused ? "cyan" : "gray";
  return (
    <Box borderStyle="round" borderColor={accent} paddingX={1}>
      <Text color={accent} bold>
        ❯{" "}
      </Text>
      {disabled ? (
        <Text dimColor>
          {value || busyHint || "running…"}
        </Text>
      ) : (
        <TextInput
          value={value}
          onChange={onChange}
          onSubmit={onSubmit}
          placeholder="ask me anything · /help for commands · @ for files"
          showCursor
        />
      )}
    </Box>
  );
};
