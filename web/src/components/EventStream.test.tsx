import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { EventStream, groupEventRecords } from "./EventStream"
import { checkpointFixture, orderedFixture } from "../test/fixtures"

describe("EventStream", () => {
  it("renders reasoning, tool call, result, and public text in segment order", () => {
    render(<EventStream events={orderedFixture} filter="all" />)
    expect(screen.getAllByTestId("trace-kind").map((node) => node.textContent)).toEqual([
      "Think", "Tool", "Tool result", "Public message",
    ])
  })

  it("shows the exact reasoning field and raw text through safe React text nodes", () => {
    const { container } = render(<EventStream events={orderedFixture} filter="reasoning" />)
    expect(screen.getByText("reasoning_content")).toBeVisible()
    expect(screen.getByText("I should ask <script>alert('no')</script> Bob.")).toBeVisible()
    expect(container.querySelector("script")).toBeNull()
  })

  it("groups only adjacent model segments from the same call", () => {
    const interrupted = [orderedFixture[0], orderedFixture[3], orderedFixture[1]]
    expect(groupEventRecords(interrupted)).toHaveLength(3)
    expect(groupEventRecords(orderedFixture)).toHaveLength(2)
  })

  it("filters body, reasoning, tools, and rules without changing event order", () => {
    const rule = { ...orderedFixture[3], seq: 5, type: "vote.cast", actor: "bob", payload: { vote: true } }
    const { rerender } = render(<EventStream events={[...orderedFixture, rule]} filter="text" />)
    expect(screen.getAllByTestId("trace-kind").map((node) => node.textContent)).toEqual(["Public message"])
    rerender(<EventStream events={[...orderedFixture, rule]} filter="tools" />)
    expect(screen.getAllByTestId("trace-kind").map((node) => node.textContent)).toEqual(["Tool", "Tool result"])
    rerender(<EventStream events={[...orderedFixture, rule]} filter="rules" />)
    expect(screen.getByText("vote.cast")).toBeVisible()
  })

  it("keeps provider metadata out of rules and classifies tool errors as tools", () => {
    const metadata = {
      ...orderedFixture[0], seq: 5,
      payload: { ...orderedFixture[0].payload, segment_index: 3, kind: "provider_metadata", source_field: "usage", text: "{}" },
    }
    const toolError = { ...orderedFixture[3], seq: 6, type: "tool.error", payload: { reason: "invalid target" } }
    const { rerender } = render(<EventStream events={[metadata, toolError]} filter="rules" />)
    expect(screen.queryByText("Provider metadata")).toBeNull()
    expect(screen.queryByText("tool.error")).toBeNull()
    rerender(<EventStream events={[metadata, toolError]} filter="tools" />)
    expect(screen.getByText("tool.error")).toBeVisible()
  })

  it("distinguishes player identity and exposes checkpoint bookmarks", () => {
    const onCheckpoint = vi.fn()
    render(<EventStream events={[...orderedFixture, checkpointFixture]} filter="all" onCheckpoint={onCheckpoint} />)
    const alice = screen.getByTestId("event-4")
    expect(within(alice).getByText("Alice")).toBeVisible()
    expect(alice).toHaveClass("seat-0")
    fireEvent.click(screen.getByRole("button", { name: /checkpoint 8/i }))
    expect(onCheckpoint).toHaveBeenCalledWith(checkpointFixture)
  })
})
