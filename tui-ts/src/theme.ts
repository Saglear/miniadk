/**
 * Default theme tokens.
 *
 * Front-end-friendly: pass an alternate object to the components that
 * accept ``theme`` props, or wrap the tree with your own context. We
 * deliberately keep this small — colour, accent, spacing — instead of
 * shipping a giant design system. If you need more knobs, fork it.
 */

export interface Theme {
  /** Primary accent (active prompt, focused button). */
  accent: string;
  /** Secondary accent (info, links). */
  info: string;
  /** Success / positive (tool result, completed task). */
  success: string;
  /** Warning (permission prompts, mode banner). */
  warning: string;
  /** Danger (errors, denied tools). */
  danger: string;
  /** Subdued text (timestamps, hints). */
  muted: string;
  /** Foreground for prose. */
  foreground: string;
  /** Borders for cards / fences. */
  border: string;
}

export const theme: Theme = {
  accent: "cyan",
  info: "blue",
  success: "green",
  warning: "yellow",
  danger: "red",
  muted: "gray",
  foreground: "white",
  border: "gray",
};
