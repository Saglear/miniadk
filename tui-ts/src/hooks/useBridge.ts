/**
 * React context + hooks for the MiniADK bridge.
 *
 * The lower-level alternative is to thread the ``BridgeApi`` directly
 * through props. The hooks here are convenience for trees that prefer
 * idiomatic React state-from-context.
 */

import React, { createContext, useContext, useEffect, useState } from "react";
import type { BridgeApi, DownEvent, UpEvent } from "../bootstrap.js";

const BridgeContext = createContext<BridgeApi | null>(null);

export interface BridgeProviderProps {
  bridge: BridgeApi;
  children: React.ReactNode;
}

/**
 * Make a bridge available to descendants via ``useBridge*`` hooks.
 *
 * ```tsx
 * mount((bridge) => (
 *   <BridgeProvider bridge={bridge}>
 *     <YourApp />
 *   </BridgeProvider>
 * ));
 * ```
 */
export function BridgeProvider({ bridge, children }: BridgeProviderProps): React.ReactElement {
  return React.createElement(BridgeContext.Provider, { value: bridge }, children);
}

/** Get the full bridge API. Throws if no provider is mounted. */
export function useBridge(): BridgeApi {
  const ctx = useContext(BridgeContext);
  if (ctx === null) {
    throw new Error(
      "useBridge() called outside <BridgeProvider>. " +
        "Wrap your tree with <BridgeProvider bridge={bridge}> from `mount`.",
    );
  }
  return ctx;
}

/** Get just the upstream sender (sugar over ``useBridge().send``). */
export function useBridgeSend(): (event: UpEvent) => void {
  return useBridge().send;
}

/**
 * Subscribe to DownEvents matching ``filter``. The hook unsubscribes
 * on unmount automatically.
 *
 * ```tsx
 * useBridgeEvents("message_delta", (event) => {
 *   setText((prev) => prev + event.data.text);
 * });
 * ```
 *
 * Pass ``"*"`` to receive every event.
 */
export function useBridgeEvents<T extends DownEvent["type"] | "*">(
  filter: T,
  handler: (
    event: T extends "*" ? DownEvent : Extract<DownEvent, { type: T }>,
  ) => void,
): void {
  const bridge = useBridge();
  useEffect(() => {
    const unsubscribe = bridge.subscribe((event) => {
      if (filter === "*" || event.type === filter) {
        handler(event as never);
      }
    });
    return unsubscribe;
    // The handler can change every render; we re-subscribe each time so
    // closures see the latest props/state. This matches typical hook
    // patterns and is fine for a low-volume terminal UI.
  });
}

/** Track the latest event of ``type`` as React state (re-render on update). */
export function useLatestEvent<T extends DownEvent["type"]>(
  type: T,
): Extract<DownEvent, { type: T }> | null {
  const [event, setEvent] = useState<Extract<DownEvent, { type: T }> | null>(null);
  useBridgeEvents(type, (e) => setEvent(e));
  return event;
}
