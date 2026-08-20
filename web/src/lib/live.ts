import { parseEventRecord, type EventRecord } from "../types"

export interface EventSourceLike {
  onmessage: ((event: MessageEvent<string>) => void) | null
  onerror: ((event: Event) => void) | null
  close(): void
}

type LiveOptions = {
  createEventSource?: (url: string) => EventSourceLike
  scheduleReconnect?: (callback: () => void) => number
  cancelReconnect?: (handle: number) => void
  probeCursor?: (url: string, signal: AbortSignal) => Promise<"valid" | "future">
  onGenerationReset?: () => void
  onError?: (message: string) => void
}

export async function probeLiveCursor(url: string, signal: AbortSignal): Promise<"valid" | "future"> {
  const response = await fetch(url, {
    signal,
    cache: "no-store",
    headers: { Accept: "text/event-stream" },
  })
  try {
    return response.status === 422 ? "future" : "valid"
  } finally {
    await response.body?.cancel().catch(() => undefined)
  }
}

export function connectLive(
  afterSeq: number,
  onEvent: (event: EventRecord) => void,
  options: LiveOptions = {},
): () => void {
  const create = options.createEventSource ?? ((url) => new EventSource(url))
  const schedule = options.scheduleReconnect ?? ((callback) => window.setTimeout(callback, 800))
  const cancel = options.cancelReconnect ?? ((handle) => window.clearTimeout(handle))
  const probe = options.probeCursor ?? probeLiveCursor
  let cursor = afterSeq
  let source: EventSourceLike | null = null
  let retryHandle: number | null = null
  let retryScheduled = false
  let probeController: AbortController | null = null
  let closed = false

  const scheduleOpen = () => {
    if (closed || retryScheduled) return
    retryScheduled = true
    const handle = schedule(() => {
      retryScheduled = false
      retryHandle = null
      open()
    })
    if (retryScheduled) retryHandle = handle
  }

  const reconnect = () => {
    if (closed) return
    source?.close()
    scheduleOpen()
  }

  const probeAndReconnect = (failedSource: EventSourceLike) => {
    if (closed || source !== failedSource || probeController !== null) return
    failedSource.close()
    const controller = new AbortController()
    probeController = controller
    void probe(`/api/events?after_seq=${cursor}`, controller.signal)
      .then((result) => {
        if (closed || probeController !== controller) return
        probeController = null
        if (result === "future") {
          cursor = 0
          options.onGenerationReset?.()
        }
        scheduleOpen()
      })
      .catch((error) => {
        if (closed || probeController !== controller) return
        probeController = null
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          scheduleOpen()
        }
      })
  }

  const open = () => {
    if (closed) return
    const nextSource = create(`/api/events?after_seq=${cursor}`)
    source = nextSource
    nextSource.onmessage = (message) => {
      try {
        const event = parseEventRecord(JSON.parse(message.data))
        if (event.seq <= cursor) return
        if (event.seq !== cursor + 1) {
          options.onError?.(`Live sequence gap after ${cursor}`)
          reconnect()
          return
        }
        cursor = event.seq
        onEvent(event)
      } catch (error) {
        options.onError?.(error instanceof Error ? error.message : "Invalid live event")
        reconnect()
      }
    }
    nextSource.onerror = () => probeAndReconnect(nextSource)
  }

  open()
  return () => {
    closed = true
    source?.close()
    probeController?.abort()
    probeController = null
    if (retryHandle !== null) cancel(retryHandle)
    retryScheduled = false
  }
}
