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
  onError?: (message: string) => void
}

export function connectLive(
  afterSeq: number,
  onEvent: (event: EventRecord) => void,
  options: LiveOptions = {},
): () => void {
  const create = options.createEventSource ?? ((url) => new EventSource(url))
  const schedule = options.scheduleReconnect ?? ((callback) => window.setTimeout(callback, 800))
  const cancel = options.cancelReconnect ?? ((handle) => window.clearTimeout(handle))
  let cursor = afterSeq
  let source: EventSourceLike | null = null
  let retryHandle: number | null = null
  let closed = false

  const reconnect = () => {
    if (closed || retryHandle !== null) return
    source?.close()
    retryHandle = schedule(() => {
      retryHandle = null
      open()
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
    nextSource.onerror = reconnect
  }

  open()
  return () => {
    closed = true
    source?.close()
    if (retryHandle !== null) cancel(retryHandle)
  }
}
