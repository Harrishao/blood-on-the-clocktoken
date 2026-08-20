import type { RuntimeState } from "../types"

export type ToolbarMode = "live" | "history"

type Props = {
  mode: ToolbarMode
  runtime: RuntimeState | "connecting"
  statusTitle?: string
  statusDetail?: string
  onStop: () => void
  onContinue: () => void
  onOpenHistory: (file: File | undefined) => void
  onBackToLive: () => void
}

export function Toolbar({
  mode,
  runtime,
  statusTitle,
  statusDetail,
  onStop,
  onContinue,
  onOpenHistory,
  onBackToLive,
}: Props) {
  return <header className="topbar">
    <div className="brand"><span className="brand-mark">CT</span><div><h1>Clocktower Observer</h1><p>Autonomous Trouble Brewing</p></div></div>
    <div className="session-status">
      <span className={`status-dot ${runtime}`} />
      <div><strong>{statusTitle ?? (mode === "live" ? `Live · ${runtime}` : "History")}</strong><small>{statusDetail ?? "Single-game observer"}</small></div>
    </div>
    <div className="controls">
      <label className="file-button">Open history<input aria-label="Open history" type="file" accept=".jsonl,application/x-ndjson" onChange={(event) => onOpenHistory(event.target.files?.[0])} /></label>
      {mode === "history" && <button onClick={onBackToLive}>Back to live</button>}
      {mode === "live" && runtime === "running" && <button className="control-primary" onClick={onStop}>Stop</button>}
      {mode === "live" && runtime === "stopped" && <button className="control-primary" onClick={onContinue}>Continue</button>}
    </div>
  </header>
}
