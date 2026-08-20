import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { App, type LiveConnector } from "./App"
import { fileOf, orderedFixture } from "./test/fixtures"

const runtime = { state: "running" as const, reason: null, phase: "day.discussion", day: 1, winner: null, history_path: "history/game.jsonl" }

function setup() {
  const subscriptions: Array<{ after: number; emit: (event: typeof orderedFixture[number]) => void; reset?: () => void }> = []
  const connector: LiveConnector = (after, emit, reset) => {
    subscriptions.push({ after, emit, reset })
    return () => undefined
  }
  render(<App connect={connector} fetchRuntime={async () => runtime} sendControl={vi.fn()} />)
  return subscriptions
}

describe("App", () => {
  it("shows Stop as the only live process control and hides controls in history mode", async () => {
    setup()
    expect(await screen.findByRole("button", { name: "Stop" })).toBeVisible()
    expect(screen.queryByText(/step|speed|next phase/i)).toBeNull()

    const input = screen.getByLabelText("Open history")
    fireEvent.change(input, { target: { files: [fileOf(orderedFixture.map((event) => JSON.stringify(event)).join("\n"))] } })
    await screen.findByText("History · game.jsonl")
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull()
  })

  it("keeps the current live view when every history line does not validate", async () => {
    const subscriptions = setup()
    await waitFor(() => expect(subscriptions).toHaveLength(1))
    act(() => subscriptions[0].emit(orderedFixture[0]))
    expect(await screen.findByText(/I should ask/)).toBeVisible()

    fireEvent.change(screen.getByLabelText("Open history"), { target: { files: [fileOf("not-json")] } })
    expect(await screen.findByRole("alert")).toHaveTextContent("line 1")
    expect(screen.getByText(/I should ask/)).toBeVisible()
    expect(screen.getByText(/Live/)).toBeVisible()
  })

  it("keeps receiving live events in history mode and reveals them on return", async () => {
    const subscriptions = setup()
    await waitFor(() => expect(subscriptions).toHaveLength(1))
    act(() => subscriptions[0].emit(orderedFixture[0]))
    fireEvent.change(screen.getByLabelText("Open history"), { target: { files: [fileOf(JSON.stringify(orderedFixture[0]))] } })
    await screen.findByText("History · game.jsonl")

    subscriptions[0].emit(orderedFixture[1])
    fireEvent.click(screen.getByRole("button", { name: "Back to live" }))
    expect(await screen.findByText("speak_public")).toBeVisible()
  })

  it("does not let a late live-status error cover a valid history session", async () => {
    let rejectRuntime!: (reason: Error) => void
    const fetchRuntime = () => new Promise<typeof runtime>((_resolve, reject) => { rejectRuntime = reject })
    const connector: LiveConnector = () => () => undefined
    render(<App connect={connector} fetchRuntime={fetchRuntime} sendControl={vi.fn()} />)

    fireEvent.change(screen.getByLabelText("Open history"), { target: { files: [fileOf(JSON.stringify(orderedFixture[0]))] } })
    await screen.findByText("History · game.jsonl")
    await act(async () => rejectRuntime(new Error("backend offline")))
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("atomically replaces an old live generation and accepts its new sequence one", async () => {
    const subscriptions = setup()
    await waitFor(() => expect(subscriptions).toHaveLength(1))
    act(() => subscriptions[0].emit(orderedFixture[0]))
    expect(await screen.findByText(/I should ask/)).toBeVisible()
    expect(subscriptions[0].reset).toBeTypeOf("function")

    act(() => {
      subscriptions[0].reset!()
      subscriptions[0].emit({
        ...orderedFixture[0],
        payload: { ...orderedFixture[0].payload, call_id: "new-generation", text: "Fresh game reasoning." },
      })
    })

    expect(await screen.findByText("Fresh game reasoning.")).toBeVisible()
    expect(screen.queryByText(/I should ask/)).toBeNull()
  })

  it("keeps history visible while the background live generation resets", async () => {
    const subscriptions = setup()
    await waitFor(() => expect(subscriptions).toHaveLength(1))
    act(() => subscriptions[0].emit(orderedFixture[0]))
    fireEvent.change(screen.getByLabelText("Open history"), { target: { files: [fileOf(JSON.stringify(orderedFixture[0]))] } })
    await screen.findByText("History · game.jsonl")
    expect(subscriptions[0].reset).toBeTypeOf("function")

    act(() => {
      subscriptions[0].reset!()
      subscriptions[0].emit({
        ...orderedFixture[0],
        payload: { ...orderedFixture[0].payload, call_id: "new-generation", text: "Fresh game reasoning." },
      })
    })

    expect(screen.getByText(/I should ask/)).toBeVisible()
    expect(screen.queryByText("Fresh game reasoning.")).toBeNull()
    expect(screen.queryByRole("alert")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Back to live" }))
    expect(await screen.findByText("Fresh game reasoning.")).toBeVisible()
    expect(screen.queryByText(/I should ask/)).toBeNull()
  })
})
