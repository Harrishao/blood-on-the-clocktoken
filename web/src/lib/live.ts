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
  let retryOwner: number | null = null
  let probeController: AbortController | null = null
  let closed = false
  let owner = 0

  const owns = (token: number, candidate?: EventSourceLike) =>
    !closed && owner === token && (candidate === undefined || source === candidate)

  const retire = (token: number, candidate: EventSourceLike) => {
    if (!owns(token, candidate)) return false
    candidate.onmessage = null
    candidate.onerror = null
    candidate.close()
    source = null
    return true
  }

  const scheduleOpen = (token: number) => {
    if (!owns(token) || retryScheduled) return
    retryScheduled = true
    retryOwner = token
    const handle = schedule(() => {
      if (!owns(token) || !retryScheduled || retryOwner !== token) return
      retryScheduled = false
      retryOwner = null
      retryHandle = null
      open(token)
    })
    if (retryScheduled && retryOwner === token) retryHandle = handle
  }

  const reconnect = (token: number, failedSource: EventSourceLike) => {
    if (!retire(token, failedSource)) return
    scheduleOpen(token)
  }

  const probeAndReconnect = (token: number, failedSource: EventSourceLike) => {
    if (probeController !== null || !retire(token, failedSource)) return
    const controller = new AbortController()
    probeController = controller
    void probe(`/api/events?after_seq=${cursor}`, controller.signal)
      .then((result) => {
        if (!owns(token) || probeController !== controller) return
        probeController = null
        if (result === "future") {
          cursor = 0
          options.onGenerationReset?.()
        }
        scheduleOpen(token)
      })
      .catch((error) => {
        if (!owns(token) || probeController !== controller) return
        probeController = null
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          scheduleOpen(token)
        }
      })
  }

  const open = (expectedOwner = owner) => {
    if (!owns(expectedOwner)) return
    if (source !== null) {
      source.onmessage = null
      source.onerror = null
      source.close()
      source = null
    }
    const token = ++owner
    const nextSource = create(`/api/events?after_seq=${cursor}`)
    source = nextSource
    nextSource.onmessage = (message) => {
      if (!owns(token, nextSource)) return
      try {
        const event = parseEventRecord(JSON.parse(message.data))
        if (event.seq <= cursor) return
        if (event.seq !== cursor + 1) {
          options.onError?.(`Live sequence gap after ${cursor}`)
          reconnect(token, nextSource)
          return
        }
        cursor = event.seq
        onEvent(event)
      } catch (error) {
        options.onError?.(error instanceof Error ? error.message : "Invalid live event")
        reconnect(token, nextSource)
      }
    }
    nextSource.onerror = () => probeAndReconnect(token, nextSource)
  }

  open()
  return () => {
    closed = true
    owner += 1
    if (source !== null) {
      source.onmessage = null
      source.onerror = null
      source.close()
      source = null
    }
    probeController?.abort()
    probeController = null
    if (retryHandle !== null) cancel(retryHandle)
    retryHandle = null
    retryScheduled = false
    retryOwner = null
  }
}
