import { describe, expect, it, vi } from "vitest"
import { connectLive, type EventSourceLike } from "./live"
import { orderedFixture } from "../test/fixtures"

class FakeEventSource implements EventSourceLike {
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  onerror: (() => void) | null = null
  closed = false
  constructor(readonly url: string) {}
  close() { this.closed = true }
  emit(record: unknown) { this.onmessage?.({ data: JSON.stringify(record) } as MessageEvent<string>) }
  fail() { this.onerror?.() }
}

describe("connectLive", () => {
  it("deduplicates replay and reconnects from the last accepted sequence", () => {
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
    })

    sources[0].emit(orderedFixture[0])
    sources[0].emit(orderedFixture[0])
    sources[0].fail()

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
})
