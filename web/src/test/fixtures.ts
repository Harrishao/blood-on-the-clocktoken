import type { EventRecord } from "../types"

const audience = { kind: "observer" as const, player_ids: [] }

export const orderedFixture: EventRecord[] = [
  {
    seq: 1,
    time: "2026-08-20T12:00:00+08:00",
    phase: "day.discussion",
    type: "model.output_segment",
    actor: "alice",
    audience,
    payload: {
      call_id: "call-42",
      player_id: "alice",
      call_purpose: "formal_action",
      segment_index: 0,
      kind: "reasoning",
      source_field: "reasoning_content",
      text: "I should ask <script>alert('no')</script> Bob.",
      incomplete: false,
    },
  },
  {
    seq: 2,
    time: "2026-08-20T12:00:01+08:00",
    phase: "day.discussion",
    type: "model.output_segment",
    actor: "alice",
    audience,
    payload: {
      call_id: "call-42",
      player_id: "alice",
      call_purpose: "formal_action",
      segment_index: 1,
      kind: "tool_call",
      source_field: "tool_calls",
      text: "{\"text\":\"I am the Washerwoman.\"}",
      tool_call_id: "tool-1",
      tool_name: "speak_public",
      incomplete: false,
    },
  },
  {
    seq: 3,
    time: "2026-08-20T12:00:02+08:00",
    phase: "day.discussion",
    type: "model.output_segment",
    actor: "alice",
    audience,
    payload: {
      call_id: "call-42",
      player_id: "alice",
      call_purpose: "formal_action",
      segment_index: 2,
      kind: "tool_result",
      source_field: "tool",
      text: "accepted",
      tool_call_id: "tool-1",
      tool_name: "speak_public",
      incomplete: false,
    },
  },
  {
    seq: 4,
    time: "2026-08-20T12:00:03+08:00",
    phase: "day.discussion",
    type: "player.public_message",
    actor: "alice",
    audience: { kind: "public", player_ids: [] },
    payload: { text: "I am the Washerwoman." },
  },
]

export const checkpointFixture: EventRecord = {
  seq: 8,
  time: "2026-08-20T12:00:08+08:00",
  phase: "day.private",
  type: "checkpoint",
  actor: null,
  audience,
  payload: {
    trigger_player_id: "alice",
    trigger_event_seq: 7,
    day: 1,
    phase: "day.private",
    active_scene: "private-chat-17-1",
    latest_event_seq: 8,
    players: {
      alice: {
        player_id: "alice", seat: 0, role: "drunk", perceived_identity: "librarian",
        alignment: "good", known_alignment: "good", alive: true, dead_vote_available: true,
        notebook: { notes: "Bob may be the Chef.", attention: { players: ["bob"], pending_actions: ["request_private_chat"], watch_triggers: ["claim.public"] } },
        perceived_ability_text: "You start knowing that one of two players is a particular Outsider.", reminders: ["is_drunk"],
      },
      bob: {
        player_id: "bob", seat: 1, role: "poisoner", perceived_identity: "poisoner",
        alignment: "evil", known_alignment: "evil", alive: false, dead_vote_available: false,
        notebook: { notes: "Alice is probing me.", attention: { players: ["alice"], pending_actions: ["nominate"], watch_triggers: ["vote.cast"] } },
        perceived_ability_text: "Each night, choose a player: they are poisoned tonight and tomorrow day.", reminders: ["poisoned", "dead_vote_spent"],
      },
    },
    role_state: { poisoned_player_id: "alice", protected_player_id: "eve", game_ended: false },
  },
}

export const fileOf = (text: string) => new File([text], "game.jsonl", { type: "application/x-ndjson" })
