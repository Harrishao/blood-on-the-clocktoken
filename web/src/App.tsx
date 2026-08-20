import { useEffect, useMemo, useRef, useState } from "react"
import { CheckpointPanel } from "./components/CheckpointPanel"
import { EventStream } from "./components/EventStream"
import { Toolbar } from "./components/Toolbar"
import { parseHistory } from "./lib/history"
import { connectLive } from "./lib/live"
import type { EventFilter, EventRecord, RuntimeStatus } from "./types"

type Mode = { kind: "live" } | { kind: "history"; name: string; events: EventRecord[] }
type ViewerError = { message: string; liveOnly: boolean }
export type LiveConnector = (
  afterSeq: number,
  onEvent: (event: EventRecord) => void,
  onGenerationReset: () => void,
) => () => void

const connectAppLive: LiveConnector = (afterSeq, onEvent, onGenerationReset) =>
  connectLive(afterSeq, onEvent, { onGenerationReset })

const defaultFetchRuntime = async (): Promise<RuntimeStatus> => {
  const response = await fetch("/api/state")
  if (!response.ok) throw new Error(`State request failed (${response.status})`)
  return response.json() as Promise<RuntimeStatus>
}

const defaultSendControl = async (action: "stop" | "continue") => {
  const response = await fetch(`/api/control/${action}`, { method: "POST" })
  if (!response.ok) throw new Error(`${action} request failed (${response.status})`)
}

export function App({ connect = connectAppLive, fetchRuntime = defaultFetchRuntime, sendControl = defaultSendControl }: {
  connect?: LiveConnector
  fetchRuntime?: () => Promise<RuntimeStatus>
  sendControl?: (action: "stop" | "continue") => Promise<unknown>
}) {
  const [liveEvents, setLiveEvents] = useState<EventRecord[]>([])
  const [mode, setMode] = useState<Mode>({ kind: "live" })
  const [filter, setFilter] = useState<EventFilter>("all")
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null)
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<EventRecord | null>(null)
  const [error, setError] = useState<ViewerError | null>(null)
  const lastSeq = useRef(0)
  const modeRef = useRef(mode)
  modeRef.current = mode

  useEffect(() => connect(lastSeq.current, (event) => {
    if (event.seq <= lastSeq.current) return
    lastSeq.current = event.seq
    setLiveEvents((current) => [...current, event])
  }, () => {
    lastSeq.current = 0
    setLiveEvents([])
    if (modeRef.current.kind === "live") setSelectedCheckpoint(null)
  }), [connect])

  useEffect(() => {
    let mounted = true
    const refresh = () => fetchRuntime()
      .then((value) => { if (mounted) setRuntime(value) })
      .catch((reason) => { if (mounted) setError({ message: String(reason), liveOnly: true }) })
    refresh()
    const handle = window.setInterval(refresh, 1500)
    return () => { mounted = false; window.clearInterval(handle) }
  }, [fetchRuntime])

  const events = mode.kind === "live" ? liveEvents : mode.events
  const checkpoints = useMemo(() => events.filter((event) => event.type === "checkpoint").length, [events])

  const openHistory = async (file: File | undefined) => {
    if (!file) return
    setError(null)
    try {
      const parsed = await parseHistory(file)
      setSelectedCheckpoint(null)
      setMode({ kind: "history", name: file.name, events: parsed })
    } catch (reason) {
      setError({ message: reason instanceof Error ? reason.message : "Could not read history", liveOnly: false })
    }
  }

  const control = async (action: "stop" | "continue") => {
    setError(null)
    try {
      await sendControl(action)
      setRuntime(await fetchRuntime())
    } catch (reason) {
      setError({ message: reason instanceof Error ? reason.message : `Could not ${action}`, liveOnly: false })
    }
  }

  const locate = (seq: number) => document.getElementById(`event-${seq}`)?.scrollIntoView({ behavior: "smooth", block: "center" })

  return <main className={`app ${selectedCheckpoint ? "with-panel" : ""}`}>
    <section className="conversation-pane">
      <Toolbar
        mode={mode.kind}
        runtime={runtime?.state ?? "connecting"}
        statusTitle={mode.kind === "live" ? `Live · ${runtime?.state ?? "connecting"}` : `History · ${mode.name}`}
        statusDetail={mode.kind === "live" ? `Day ${runtime?.day ?? "–"} · ${runtime?.phase ?? "waiting"}` : `${events.length} events · ${checkpoints} checkpoints`}
        onOpenHistory={(file) => void openHistory(file)}
        onBackToLive={() => { setMode({ kind: "live" }); setSelectedCheckpoint(null) }}
        onStop={() => void control("stop")}
        onContinue={() => void control("continue")}
      />
      <nav className="filterbar" aria-label="Event filters">
        {(["all", "text", "reasoning", "tools", "rules"] as EventFilter[]).map((item) => <button key={item} aria-pressed={filter === item} onClick={() => setFilter(item)}>{item}</button>)}
      </nav>
      {error && (!error.liveOnly || mode.kind === "live") && <div className="error-banner" role="alert">{error.message}</div>}
      <EventStream events={events} filter={filter} onCheckpoint={setSelectedCheckpoint} />
    </section>
    {selectedCheckpoint && <CheckpointPanel checkpoint={selectedCheckpoint} onClose={() => setSelectedCheckpoint(null)} onLocate={locate} />}
  </main>
}
