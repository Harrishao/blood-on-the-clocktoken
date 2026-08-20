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

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((fulfill, fail) => { resolve = fulfill; reject = fail })
  return { promise, resolve, reject }
}

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

  it("serializes repeated errors from one failed source behind a single reconnect", async () => {
    const sources: FakeEventSource[] = []
    const probes: Array<ReturnType<typeof deferred<"valid" | "future">>> = []
    const scheduled: Array<() => void> = []
    connectLive(2, () => undefined, {
      createEventSource: (url) => {
        const source = new FakeEventSource(url)
        sources.push(source)
        return source
      },
      probeCursor: () => {
        const probe = deferred<"valid" | "future">()
        probes.push(probe)
        return probe.promise
      },
      scheduleReconnect: (callback) => { scheduled.push(callback); return scheduled.length },
      cancelReconnect: () => undefined,
    })

    sources[0].fail()
    probes[0].resolve("valid")
    await flushPromises()
    expect(scheduled).toHaveLength(1)

    sources[0].fail()
    expect(probes).toHaveLength(1)
    scheduled[0]()
    expect(sources).toHaveLength(2)
    expect(sources.filter((source) => !source.closed)).toHaveLength(1)

    sources[0].fail()
    scheduled[0]()
    expect(probes).toHaveLength(1)
    expect(sources).toHaveLength(2)
    expect(sources.filter((source) => !source.closed)).toHaveLength(1)
  })

  it("ignores an old source error after its replacement is open", async () => {
    const sources: FakeEventSource[] = []
    const probes: Array<ReturnType<typeof deferred<"valid" | "future">>> = []
    const scheduled: Array<() => void> = []
    connectLive(1, () => undefined, {
      createEventSource: (url) => {
        const source = new FakeEventSource(url)
        sources.push(source)
        return source
      },
      probeCursor: () => {
        const probe = deferred<"valid" | "future">()
        probes.push(probe)
        return probe.promise
      },
      scheduleReconnect: (callback) => { scheduled.push(callback); return scheduled.length },
      cancelReconnect: () => undefined,
    })

    const lateError = sources[0].onerror
    sources[0].fail()
    probes[0].resolve("valid")
    await flushPromises()
    scheduled[0]()
    expect(sources).toHaveLength(2)

    lateError?.(new Event("error"))
    await flushPromises()
    scheduled[1]?.()
    expect(probes).toHaveLength(1)
    expect(sources).toHaveLength(2)
    expect(sources.filter((source) => !source.closed)).toHaveLength(1)
  })

  it("makes source and timer callbacks inert after disconnect", async () => {
    const sources: FakeEventSource[] = []
    const scheduled: Array<() => void> = []
    const received = vi.fn()
    const disconnect = connectLive(0, received, {
      createEventSource: (url) => {
        const source = new FakeEventSource(url)
        sources.push(source)
        return source
      },
      probeCursor: async () => "valid",
      scheduleReconnect: (callback) => { scheduled.push(callback); return scheduled.length },
      cancelReconnect: () => undefined,
    })

    const lateMessage = sources[0].onmessage
    const lateError = sources[0].onerror
    sources[0].emit({ ...orderedFixture[2], seq: 3 })
    expect(scheduled).toHaveLength(1)
    disconnect()
    scheduled[0]()
    lateMessage?.({ data: JSON.stringify(orderedFixture[0]) } as MessageEvent<string>)
    lateError?.(new Event("error"))
    await flushPromises()

    expect(received).not.toHaveBeenCalled()
    expect(sources).toHaveLength(1)
    expect(sources[0].closed).toBe(true)
  })
})
