import type { EventRecord, EventFilter, ModelSegmentPayload } from "../types"
import { modelSegmentOf } from "../types"

const kindLabel: Record<ModelSegmentPayload["kind"], string> = {
  reasoning: "Think",
  tool_call: "Tool",
  tool_result: "Tool result",
  final_message: "Message",
  provider_metadata: "Provider metadata",
}

function visible(kind: ModelSegmentPayload["kind"], filter: EventFilter): boolean {
  if (filter === "all") return true
  if (filter === "reasoning") return kind === "reasoning"
  if (filter === "tools") return kind === "tool_call" || kind === "tool_result"
  if (filter === "text") return kind === "final_message"
  return false
}

export function ModelTrace({ events, filter }: { events: EventRecord[]; filter: EventFilter }) {
  const segments = events
    .map((event) => ({ event, segment: modelSegmentOf(event) }))
    .filter((item): item is { event: EventRecord; segment: ModelSegmentPayload } =>
      item.segment !== null && visible(item.segment.kind, filter))
  if (segments.length === 0) return null

  return <section className="model-trace" data-call-id={segments[0].segment.call_id}>
    {segments.map(({ event, segment }) => {
      const content = <>
        <div className="trace-meta">
          <span className="source-field">{segment.source_field}</span>
          <span>seq {event.seq} · segment {segment.segment_index}</span>
          {segment.incomplete && <span className="incomplete">incomplete</span>}
        </div>
        {(segment.tool_name || segment.tool_call_id) && <div className="tool-identity">
          {segment.tool_name && <strong>{segment.tool_name}</strong>}
          {segment.tool_call_id && <code>{segment.tool_call_id}</code>}
        </div>}
        <pre className="trace-text">{segment.text}</pre>
      </>
      return segment.kind === "reasoning" ? (
        <details className="trace-card reasoning" open key={event.seq} id={`event-${event.seq}`}>
          <summary><span data-testid="trace-kind">{kindLabel[segment.kind]}</span></summary>
          {content}
        </details>
      ) : (
        <article className={`trace-card ${segment.kind}`} key={event.seq} id={`event-${event.seq}`}>
          <div className="trace-title" data-testid="trace-kind">{kindLabel[segment.kind]}</div>
          {content}
        </article>
      )
    })}
  </section>
}
