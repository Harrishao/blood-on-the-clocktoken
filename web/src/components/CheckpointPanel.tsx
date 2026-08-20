import { useEffect, useMemo, useState } from "react"
import { checkpointOf, type EventRecord } from "../types"

const titleCase = (value: string) => value.split(/[_-]/).map((part) => part ? part[0].toUpperCase() + part.slice(1) : part).join(" ")

export function CheckpointPanel({ checkpoint, onClose, onLocate }: { checkpoint: EventRecord; onClose: () => void; onLocate: (seq: number) => void }) {
  const snapshot = checkpointOf(checkpoint)
  const orderedPlayers = useMemo(() => snapshot ? Object.values(snapshot.players).sort((a, b) => a.seat - b.seat) : [], [snapshot])
  const [playerId, setPlayerId] = useState(orderedPlayers[0]?.player_id ?? "")
  useEffect(() => setPlayerId(orderedPlayers[0]?.player_id ?? ""), [checkpoint.seq])
  if (!snapshot) return null
  const player = snapshot.players[playerId] ?? orderedPlayers[0]

  return <aside className="checkpoint-panel" aria-label="Checkpoint details">
    <header><div><span className="eyebrow">Checkpoint #{checkpoint.seq}</span><h2>Day {snapshot.day} · {snapshot.phase}</h2></div><button onClick={onClose} aria-label="Close checkpoint">Close</button></header>
    <dl className="checkpoint-facts">
      <div><dt>Latest seq</dt><dd>{snapshot.latest_event_seq}</dd></div>
      <div><dt>Active scene</dt><dd>{snapshot.active_scene ?? "None"}</dd></div>
      <div><dt>Triggered by</dt><dd>{snapshot.trigger_player_id}</dd></div>
    </dl>
    <section><h3>Players at this moment</h3><div className="player-snapshot-grid">
      {orderedPlayers.map((item) => <div className={`player-snapshot ${item.alive ? "alive" : "dead"}`} key={item.player_id}>
        <strong>{titleCase(item.player_id)}</strong><span>{titleCase(item.role)}</span><small>{item.alignment} · {item.alive ? "alive" : "dead"}</small>
      </div>)}
    </div></section>
    <section><label htmlFor="checkpoint-player">Player notebook</label><select id="checkpoint-player" value={player?.player_id} onChange={(event) => setPlayerId(event.target.value)}>
      {orderedPlayers.map((item) => <option key={item.player_id} value={item.player_id}>{titleCase(item.player_id)} notebook</option>)}
    </select>
    <div className="notebook"><pre>{player?.notebook.notes || "No notes."}</pre><div className="attention-row"><span>Watching: {player?.notebook.attention.players.join(", ") || "none"}</span><span>Pending: {player?.notebook.attention.pending_actions.join(", ") || "none"}</span></div></div></section>
    <section><h3>Role state</h3><pre className="role-state">{JSON.stringify(snapshot.role_state, null, 2)}</pre></section>
    <button className="locate-button" onClick={() => onLocate(snapshot.trigger_event_seq)}>Locate event {snapshot.trigger_event_seq}</button>
  </aside>
}
