import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import { PendingPermission } from "../protocol.js";

interface Props {
  request: PendingPermission;
  onResolve: (allow: boolean) => void;
}

/**
 * Permission decision modal.
 *
 * Bindings (all clearly labelled in the footer so the user never has to
 * guess):
 *   - y / Y / Right       → allow
 *   - n / N / Left / Esc  → deny
 *   - ↑ / ↓ / Tab          → move selection
 *   - Enter                → confirm current selection
 *
 * The default selection is **deny** for safety: if the user reflexively
 * hits Enter we don't run a destructive action they didn't read.
 */
export const PermissionModal: React.FC<Props> = ({ request, onResolve }) => {
  const [selected, setSelected] = useState<"deny" | "allow">("deny");

  useInput((input, key) => {
    if (input === "y" || input === "Y" || key.rightArrow) {
      onResolve(true);
      return;
    }
    if (input === "n" || input === "N" || key.leftArrow || key.escape) {
      onResolve(false);
      return;
    }
    if (key.upArrow || key.downArrow || key.tab) {
      setSelected((s) => (s === "deny" ? "allow" : "deny"));
      return;
    }
    if (key.return) {
      onResolve(selected === "allow");
      return;
    }
  });

  const argsText = (() => {
    try {
      return JSON.stringify(request.arguments, null, 2);
    } catch {
      return String(request.arguments);
    }
  })();

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="yellow"
      paddingX={2}
      paddingY={1}
      marginX={4}
      marginY={1}
    >
      <Text bold color="yellow">
        ⚠  Allow {request.tool}?
      </Text>
      <Box marginTop={1}>
        <Text dimColor>{request.reason}</Text>
      </Box>
      <Box marginTop={1} flexDirection="column">
        <Text dimColor>args:</Text>
        {argsText.split("\n").slice(0, 12).map((line, i) => (
          <Text key={i} dimColor>
            {"  "}
            {line}
          </Text>
        ))}
        {argsText.split("\n").length > 12 && (
          <Text dimColor italic>
            {`  … (${argsText.split("\n").length - 12} more lines)`}
          </Text>
        )}
      </Box>
      <Box marginTop={1}>
        <Choice label="Allow" hotkey="y" active={selected === "allow"} color="green" />
        <Box marginX={2}>
          <Text dimColor>·</Text>
        </Box>
        <Choice label="Deny" hotkey="n" active={selected === "deny"} color="red" />
      </Box>
      <Box marginTop={1}>
        <Text dimColor italic>
          y/n · ←→ to choose · enter to confirm · esc denies
        </Text>
      </Box>
    </Box>
  );
};


const Choice: React.FC<{
  label: string;
  hotkey: string;
  active: boolean;
  color: string;
}> = ({ label, hotkey, active, color }) => {
  if (active) {
    return (
      <Text color={color} bold>
        ▶ {label} ({hotkey})
      </Text>
    );
  }
  return (
    <Text dimColor>
      {"  "}
      {label} ({hotkey})
    </Text>
  );
};
