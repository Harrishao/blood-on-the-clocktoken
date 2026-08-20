import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { CheckpointPanel } from "./CheckpointPanel"
import { checkpointFixture } from "../test/fixtures"

it("shows complete read-only checkpoint state and can locate its trigger event", () => {
  const onLocate = vi.fn()
  render(<CheckpointPanel checkpoint={checkpointFixture} onClose={() => undefined} onLocate={onLocate} />)
  expect(screen.getByText(/Day 1/)).toBeVisible()
  expect(screen.getByText("Washerwoman")).toBeVisible()
  expect(screen.getByText("Poisoner")).toBeVisible()
  fireEvent.change(screen.getByLabelText("Player notebook"), { target: { value: "bob" } })
  expect(screen.getByText("Alice is probing me.")).toBeVisible()
  expect(screen.getByText(/private-chat-17-1/)).toBeVisible()
  expect(screen.queryByRole("button", { name: /restore|resume|edit|export/i })).toBeNull()
  fireEvent.click(screen.getByRole("button", { name: /locate event 7/i }))
  expect(onLocate).toHaveBeenCalledWith(7)
})
