import { useEffect, useMemo, useRef, useState } from "react";
import { CompletionItem, fuzzyRank } from "../components/Autocomplete.js";

export interface CommandCatalogItem {
  name: string;
  description: string;
  category?: "command" | "skill";
}

export type FilesProvider = (prefix: string) => Promise<string[]>;

interface AutocompleteState {
  visible: boolean;
  items: CompletionItem[];
  cursor: number;
  /** Trigger character (`/` or `@`) at the active token, if any. */
  trigger: "/" | "@" | null;
  /** Index where the trigger character starts in the input. */
  triggerStart: number;
  /** Active partial after the trigger. */
  partial: string;
}

const empty: AutocompleteState = {
  visible: false,
  items: [],
  cursor: 0,
  trigger: null,
  triggerStart: -1,
  partial: "",
};

/**
 * Drive the inline autocomplete popup based on the current input.
 *
 * Triggers:
 *   - `/` at the start of the input or after whitespace → command palette.
 *   - `@` anywhere a path could plausibly start → file completion.
 *
 * The hook owns:
 *   - parsing the input to detect the active trigger,
 *   - debouncing the network-style file lookup,
 *   - clamping the cursor when the item set changes.
 *
 * It does **not** own keyboard handling — `App.tsx` does that, so it can
 * route Tab / Enter / Escape keys appropriately.
 */
export function useAutocomplete(
  value: string,
  caret: number,
  commands: CommandCatalogItem[],
  filesProvider: FilesProvider,
): {
  state: AutocompleteState;
  setCursor: (next: number) => void;
  apply: () => string | null;
} {
  const [state, setState] = useState<AutocompleteState>(empty);
  const requestId = useRef(0);

  // Derive trigger from the input.
  const trigger = useMemo(() => detectTrigger(value, caret), [value, caret]);

  useEffect(() => {
    if (trigger === null) {
      setState(empty);
      return;
    }

    if (trigger.kind === "/") {
      const items: CompletionItem[] = commands.map((c) => ({
        insert: `/${c.name}`,
        label: `/${c.name}`,
        hint: c.description,
        category: c.category === "skill" ? "magenta" : undefined,
      }));
      const ranked = fuzzyRank(items, trigger.partial);
      setState({
        visible: ranked.length > 0,
        items: ranked,
        cursor: 0,
        trigger: "/",
        triggerStart: trigger.start,
        partial: trigger.partial,
      });
      return;
    }

    // @ trigger — async file lookup. Bump request id so out-of-order
    // responses are dropped.
    const myId = ++requestId.current;
    setState((prev) => ({ ...prev, visible: true, trigger: "@", triggerStart: trigger.start, partial: trigger.partial }));
    filesProvider(trigger.partial).then((paths) => {
      if (myId !== requestId.current) return;
      const items: CompletionItem[] = paths.map((p) => ({
        insert: `@${p}`,
        label: `@${p}`,
        category: p.endsWith("/") ? "cyan" : undefined,
      }));
      setState({
        visible: items.length > 0,
        items,
        cursor: 0,
        trigger: "@",
        triggerStart: trigger.start,
        partial: trigger.partial,
      });
    });
  }, [trigger, commands, filesProvider]);

  return {
    state,
    setCursor: (next: number) =>
      setState((prev) =>
        prev.items.length === 0
          ? prev
          : { ...prev, cursor: ((next % prev.items.length) + prev.items.length) % prev.items.length },
      ),
    /**
     * Apply the highlighted completion. Returns the new input value, or
     * ``null`` if no completion is active.
     */
    apply: () => {
      if (!state.visible || state.items.length === 0 || state.trigger === null) return null;
      const choice = state.items[state.cursor];
      const before = value.slice(0, state.triggerStart);
      const after = value.slice(state.triggerStart + 1 + state.partial.length);
      return before + choice.insert + after;
    },
  };
}


interface DetectedTrigger {
  kind: "/" | "@";
  start: number;
  partial: string;
}

function detectTrigger(value: string, caret: number): DetectedTrigger | null {
  if (caret <= 0 || caret > value.length) return null;
  // Walk left from the caret to find the trigger character or a stop.
  for (let i = caret - 1; i >= 0; i--) {
    const ch = value[i];
    if (ch === "/" || ch === "@") {
      const isStart = i === 0;
      const prev = i > 0 ? value[i - 1] : " ";
      // `/` only fires at line start; `@` after any whitespace.
      if (ch === "/" && !isStart) return null;
      if (ch === "@" && !isStart && !/\s/.test(prev)) return null;
      const partial = value.slice(i + 1, caret);
      // Trigger only fires while the partial has no whitespace.
      if (/\s/.test(partial)) return null;
      return { kind: ch, start: i, partial };
    }
    if (/\s/.test(ch)) return null;
  }
  return null;
}
