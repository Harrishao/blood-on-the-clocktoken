export type AudienceKind = "public" | "players" | "player" | "observer"
export type SegmentKind = "reasoning" | "tool_call" | "tool_result" | "final_message" | "provider_metadata"
export type EventFilter = "all" | "text" | "reasoning" | "tools" | "rules"
export type RuntimeState = "ready" | "running" | "stopped" | "ended"

export type EventRecord = {
  seq: number
  time: string
  phase: string
  type: string
  actor: string | null
  audience: { kind: AudienceKind; player_ids: string[] }
  payload: Record<string, unknown>
}

export type RuntimeStatus = {
  state: RuntimeState
  reason: string | null
  phase: string
  day: number
  winner: string | null
  history_path: string
}

export type ModelSegmentPayload = {
  call_id: string
  player_id: string
  call_purpose: string
  segment_index: number
  kind: SegmentKind
  source_field: string
  text: string
  incomplete: boolean
  tool_index?: number
  tool_call_id?: string
  tool_name?: string
  tool_type?: string
}

export type NotebookSnapshot = {
  notes: string
  attention: { players: string[]; pending_actions: string[]; watch_triggers: string[] }
}

export type PlayerSnapshot = {
  player_id: string
  seat: number
  role: string
  perceived_identity: string
  alignment: string
  known_alignment: string
  alive: boolean
  dead_vote_available: boolean
  notebook: NotebookSnapshot
  perceived_ability_text: string
  reminders: string[]
}

export type CheckpointPayload = {
  trigger_player_id: string
  trigger_event_seq: number
  day: number
  phase: string
  active_scene: string | null
  players: Record<string, PlayerSnapshot>
  role_state: Record<string, unknown>
  latest_event_seq: number
}

export const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string")

export function parseEventRecord(value: unknown): EventRecord {
  if (!isObject(value)) throw new Error("event must be an object")
  const audience = value.audience
  const validKinds: AudienceKind[] = ["public", "players", "player", "observer"]
  if (
    !Number.isSafeInteger(value.seq) || Number(value.seq) < 1 ||
    typeof value.time !== "string" || Number.isNaN(Date.parse(value.time)) ||
    typeof value.phase !== "string" || typeof value.type !== "string" || !value.type ||
    !(value.actor === null || typeof value.actor === "string") ||
    !isObject(audience) || !validKinds.includes(audience.kind as AudienceKind) ||
    !isStringArray(audience.player_ids) || !isObject(value.payload)
  ) {
    throw new Error("invalid EventRecord envelope")
  }
  const recipients = audience.player_ids
  if (
    ((audience.kind === "public" || audience.kind === "observer") && recipients.length !== 0) ||
    (audience.kind === "player" && recipients.length !== 1) ||
    (audience.kind === "players" && recipients.length === 0)
  ) {
    throw new Error("invalid EventRecord audience")
  }
  return value as EventRecord
}

export function modelSegmentOf(event: EventRecord): ModelSegmentPayload | null {
  if (event.type !== "model.output_segment") return null
  const payload = event.payload
  const kinds: SegmentKind[] = ["reasoning", "tool_call", "tool_result", "final_message", "provider_metadata"]
  if (
    typeof payload.call_id !== "string" || typeof payload.player_id !== "string" ||
    typeof payload.call_purpose !== "string" || !Number.isSafeInteger(payload.segment_index) ||
    !kinds.includes(payload.kind as SegmentKind) || typeof payload.source_field !== "string" ||
    typeof payload.text !== "string" || typeof payload.incomplete !== "boolean"
  ) return null
  return payload as ModelSegmentPayload
}

export function checkpointOf(event: EventRecord): CheckpointPayload | null {
  if (event.type !== "checkpoint") return null
  const payload = event.payload
  if (
    typeof payload.trigger_player_id !== "string" || !Number.isSafeInteger(payload.trigger_event_seq) ||
    !Number.isSafeInteger(payload.day) || typeof payload.phase !== "string" ||
    !(payload.active_scene === null || typeof payload.active_scene === "string") ||
    !isObject(payload.players) || !isObject(payload.role_state) || !Number.isSafeInteger(payload.latest_event_seq)
  ) return null
  return payload as CheckpointPayload
}
