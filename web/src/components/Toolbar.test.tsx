import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { Toolbar } from "./Toolbar"

const handlers = {
  onStop: vi.fn(),
  onContinue: vi.fn(),
  onOpenHistory: vi.fn(),
  onBackToLive: vi.fn(),
}

describe("Toolbar", () => {
  it("offers Stop while running and Continue while stopped, never step or speed controls", () => {
    const { rerender } = render(<Toolbar mode="live" runtime="running" {...handlers} />)
    expect(screen.getByRole("button", { name: "Stop" })).toBeVisible()
    expect(screen.queryByText(/speed|step|next phase/i)).toBeNull()

    rerender(<Toolbar mode="live" runtime="stopped" {...handlers} />)
    expect(screen.getByRole("button", { name: "Continue" })).toBeVisible()
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull()
  })

  it("opens a local history file and hides process controls in history mode", () => {
    const onOpenHistory = vi.fn()
    const { rerender } = render(<Toolbar mode="live" runtime="running" {...handlers} onOpenHistory={onOpenHistory} />)
    const file = new File(["{}"], "game.jsonl", { type: "application/x-ndjson" })
    fireEvent.change(screen.getByLabelText("Open history"), { target: { files: [file] } })
    expect(onOpenHistory).toHaveBeenCalledWith(file)

    rerender(<Toolbar mode="history" runtime="stopped" {...handlers} onOpenHistory={onOpenHistory} />)
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull()
    expect(screen.getByRole("button", { name: "Back to live" })).toBeVisible()
  })
})
