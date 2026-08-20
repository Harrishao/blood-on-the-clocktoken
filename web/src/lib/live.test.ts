import { describe, expect, it, vi } from "vitest"
import { connectLive, type EventSourceLike } from "./live"
import { orderedFixture } from "../test/fixtures"

class FakeEventSource implements EventSourceLike {
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  closed = false
  constructor(readonly url: string) {}
  close() { this.closed = true }
  emit(record: unknown) { this.onmessage?.({ data: JSON.stringify(record) } as MessageEvent<string>) }
  fail() { this.onerror?.(new Event("error")) }
}

const flushPromises = () => new Promise((resolve) => window.setTimeout(resolve, 0))

describe("connectLive", () => {
  it("deduplicates replay and reconnects from the last accepted sequence", async () => {
    const sources: FakeEventSource[] = []
    const received: number[] = []
    const disconnect = connectLive(0, (event) => received.push(event.seq), {
      createEventSource: (url) => {
        const source = new FakeEventSource(url)
        sources.push(source)
        return source
      },
      scheduleReconnect: (callback) => { callback(); return 1 },
      cancelReconnect: () => undefined,
      probeCursor: async () => "valid",
    })

    sources[0].emit(orderedFixture[0])
    sources[0].emit(orderedFixture[0])
    sources[0].fail()
    await flushPromises()

    expect(received).toEqual([1])
    expect(sources[1].url).toBe("/api/events?after_seq=1")
    disconnect()
    expect(sources[1].closed).toBe(true)
  })

  it("reconnects from the cursor when a gap is observed instead of delivering out of order", () => {
    const sources: FakeEventSource[] = []
    const received = vi.fn()
    connectLive(1, received, {
      createEventSource: (url) => {
        const source = new FakeEventSource(url)
        sources.push(source)
        return source
      },
      scheduleReconnect: (callback) => { callback(); return 1 },
      cancelReconnect: () => undefined,
    })

    sources[0].emit({ ...orderedFixture[2], seq: 3 })
    expect(received).not.toHaveBeenCalled()
    expect(sources[1].url).toBe("/api/events?after_seq=1")
  })

  it("resets to zero only when a cursor probe identifies a future generation", async () => {
    const sources: FakeEventSource[] = []
    const resets: number[] = []
    const options = {
      createEventSource: (url: string) => {
        const source = new FakeEventSource(url)
        sources.push(source)
        return source
      },
      probeCursor: async () => "future" as const,
      onGenerationReset: () => resets.push(1),
      scheduleReconnect: (callback: () => void) => { callback(); return 1 },
      cancelReconnect: () => undefined,
    }
    connectLive(5, () => undefined, options)

    sources[0].fail()
    await flushPromises()

    expect(sources[0].closed).toBe(true)
    expect(resets).toEqual([1])
    expect(sources[1].url).toBe("/api/events?after_seq=0")
  })

  it("keeps the accepted cursor when the probe is valid or temporarily unreachable", async () => {
    for (const result of ["valid", "network"] as const) {
      const sources: FakeEventSource[] = []
      const options = {
        createEventSource: (url: string) => {
          const source = new FakeEventSource(url)
          sources.push(source)
          return source
        },
        probeCursor: result === "valid"
          ? async () => "valid" as const
          : async () => { throw new TypeError("offline") },
        onGenerationReset: vi.fn(),
        scheduleReconnect: (callback: () => void) => { callback(); return 1 },
        cancelReconnect: () => undefined,
      }
      connectLive(4, () => undefined, options)
      sources[0].fail()
      await flushPromises()

      expect(options.onGenerationReset).not.toHaveBeenCalled()
      expect(sources[1].url).toBe("/api/events?after_seq=4")
    }
  })

  it("disconnect aborts a pending probe and prevents retry", async () => {
    const sources: FakeEventSource[] = []
    let probeSignal: AbortSignal | undefined
    let finishProbe!: (result: "valid" | "future") => void
    const options = {
      createEventSource: (url: string) => {
        const source = new FakeEventSource(url)
        sources.push(source)
        return source
      },
      probeCursor: (_url: string, signal: AbortSignal) => {
        probeSignal = signal
        return new Promise<"valid" | "future">((resolve) => { finishProbe = resolve })
      },
      scheduleReconnect: (callback: () => void) => window.setTimeout(callback, 0),
      cancelReconnect: (handle: number) => window.clearTimeout(handle),
    }
    const disconnect = connectLive(3, () => undefined, options)
    sources[0].fail()
    expect(probeSignal).toBeDefined()
    disconnect()
    finishProbe?.("future")
    await flushPromises()

    expect(probeSignal?.aborted).toBe(true)
    expect(sources).toHaveLength(1)
  })
})
