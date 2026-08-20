import { useLayoutEffect, useMemo, useRef, useState } from "react"
import { checkpointOf, isObject, modelSegmentOf, type EventFilter, type EventRecord } from "../types"
import { ModelTrace } from "./ModelTrace"

type EventGroup = { kind: "model"; callId: string; events: EventRecord[] } | { kind: "event"; event: EventRecord }
type PlayerMeta = { playerId: string; displayName: string; seat: number; role?: string; alignment?: string }

const ROLE_ALIGNMENT: Record<string, string> = {
  poisoner: "evil", spy: "evil", scarlet_woman: "evil", baron: "evil", imp: "evil",
}

export function groupEventRecords(events: EventRecord[]): EventGroup[] {
  const groups: EventGroup[] = []
  for (const event of events) {
    const segment = modelSegmentOf(event)
    const previous = groups.at(-1)
    if (segment && previous?.kind === "model" && previous.callId === segment.call_id) {
      previous.events.push(event)
    } else if (segment) {
      groups.push({ kind: "model", callId: segment.call_id, events: [event] })
    } else {
      groups.push({ kind: "event", event })
    }
  }
  return groups
}

const titleCase = (value: string) => value.split(/[_-]/).map((part) => part ? part[0].toUpperCase() + part.slice(1) : part).join(" ")

function playerMetadata(events: EventRecord[]): Map<string, PlayerMeta> {
  const metadata = new Map<string, PlayerMeta>()
  const ensure = (playerId: string) => {
    if (!metadata.has(playerId)) metadata.set(playerId, { playerId, displayName: titleCase(playerId), seat: metadata.size })
    return metadata.get(playerId)!
  }
  for (const event of events) {
    if (event.actor) ensure(event.actor)
    if (event.type === "game.header" && Array.isArray(event.payload.players)) {
      for (const entry of event.payload.players) if (isObject(entry) && typeof entry.player_id === "string") {
        const item = ensure(entry.player_id)
        if (Number.isSafeInteger(entry.seat)) item.seat = Number(entry.seat)
        if (typeof entry.display_name === "string") item.displayName = entry.display_name
      }
    }
    if (event.type === "setup.completed" && isObject(event.payload.roles_by_player)) {
      for (const [playerId, role] of Object.entries(event.payload.roles_by_player)) if (typeof role === "string") {
        const item = ensure(playerId); item.role = role; item.alignment = ROLE_ALIGNMENT[role] ?? "good"
      }
    }
    const checkpoint = checkpointOf(event)
    if (checkpoint) for (const [playerId, value] of Object.entries(checkpoint.players)) {
      if (!isObject(value)) continue
      const item = ensure(playerId)
      if (Number.isSafeInteger(value.seat)) item.seat = Number(value.seat)
      if (typeof value.role === "string") item.role = value.role
      if (typeof value.alignment === "string") item.alignment = value.alignment
    }
  }
  return metadata
}

function eventCategory(event: EventRecord): EventFilter {
  const segment = modelSegmentOf(event)
  if (segment?.kind === "reasoning") return "reasoning"
  if (segment?.kind === "tool_call" || segment?.kind === "tool_result") return "tools"
  if (segment?.kind === "final_message") return "text"
  if (segment?.kind === "provider_metadata") return "all"
  if (["player.public_message", "chat.public_message", "chat.private_message"].includes(event.type)) return "text"
  if (event.type.startsWith("tool.")) return "tools"
  return "rules"
}

function Identity({ player }: { player?: PlayerMeta }) {
  if (!player) return <div className="identity system"><span className="avatar">CT</span><div><strong>Clocktower</strong><small>rules · scheduler</small></div></div>
  return <div className="identity">
    <span className="avatar">{player.displayName.slice(0, 2).toUpperCase()}</span>
    <div><strong>{player.displayName}</strong><small>{player.role ? titleCase(player.role) : "Role pending"} · {player.alignment ?? "unknown"}</small></div>
  </div>
}

function EventCard({ event, player, onCheckpoint }: { event: EventRecord; player?: PlayerMeta; onCheckpoint?: (event: EventRecord) => void }) {
  const checkpoint = checkpointOf(event)
  const isMessage = ["player.public_message", "chat.public_message", "chat.private_message"].includes(event.type)
  const isPrivate = event.type.includes("private") || event.audience.kind === "players" || event.audience.kind === "player"
  const label = event.type === "chat.private_message" ? "Private message" : isMessage ? "Public message" : event.type
  return <article id={`event-${event.seq}`} data-testid={`event-${event.seq}`} className={`event-card ${isPrivate ? "private" : "rule"} ${player ? `seat-${player.seat % 8}` : "system-card"}`}>
    <Identity player={player} />
    <div className="event-body">
      <div className="event-heading"><span data-testid="trace-kind">{label}</span><small>#{event.seq} · {event.phase}</small></div>
      {checkpoint ? <button className="checkpoint-bookmark" onClick={() => onCheckpoint?.(event)} aria-label={`Checkpoint ${event.seq}`}>
        <span>Checkpoint</span><strong>Notebook snapshot after event {checkpoint.trigger_event_seq}</strong>
      </button> : isMessage && typeof event.payload.text === "string" ? <p>{event.payload.text}</p> : <pre>{JSON.stringify(event.payload, null, 2)}</pre>}
    </div>
  </article>
}

export function EventStream({ events, filter, onCheckpoint }: { events: EventRecord[]; filter: EventFilter; onCheckpoint?: (event: EventRecord) => void }) {
  const groups = useMemo(() => groupEventRecords(events), [events])
  const players = useMemo(() => playerMetadata(events), [events])
  const scroller = useRef<HTMLDivElement>(null)
  const [atLatest, setAtLatest] = useState(true)
  const previousCount = useRef(events.length)

  useLayoutEffect(() => {
    const node = scroller.current
    if (events.length > previousCount.current && atLatest && node) {
      if (typeof node.scrollTo === "function") node.scrollTo({ top: node.scrollHeight })
      else node.scrollTop = node.scrollHeight
    }
    previousCount.current = events.length
  }, [events.length, atLatest])

  const onScroll = () => {
    const node = scroller.current
    if (!node) return
    setAtLatest(node.scrollHeight - node.scrollTop - node.clientHeight < 48)
  }

  const backToLatest = () => {
    const node = scroller.current
    if (node && typeof node.scrollTo === "function") node.scrollTo({ top: node.scrollHeight, behavior: "smooth" })
    else if (node) node.scrollTop = node.scrollHeight
    setAtLatest(true)
  }

  return <div className="stream-shell">
    <div className="event-stream" ref={scroller} onScroll={onScroll} aria-label="Game event conversation">
      {groups.map((group) => {
        if (group.kind === "model") {
          const visible = group.events.some((event) => filter === "all" || eventCategory(event) === filter)
          if (!visible) return null
          const actor = modelSegmentOf(group.events[0])?.player_id ?? group.events[0].actor ?? undefined
          const player = actor ? players.get(actor) : undefined
          return <article className={`event-card model ${player ? `seat-${player.seat % 8}` : "system-card"}`} key={`call-${group.events[0].seq}`}>
            <Identity player={player} />
            <div className="event-body"><ModelTrace events={group.events} filter={filter} /></div>
          </article>
        }
        if (filter !== "all" && eventCategory(group.event) !== filter) return null
        return <EventCard key={group.event.seq} event={group.event} player={group.event.actor ? players.get(group.event.actor) : undefined} onCheckpoint={onCheckpoint} />
      })}
      {events.length === 0 && <div className="empty-state"><strong>Waiting for the bell</strong><span>Live game events will appear here.</span></div>}
    </div>
    {!atLatest && <button className="back-latest" onClick={backToLatest}>Back to latest</button>}
  </div>
}
