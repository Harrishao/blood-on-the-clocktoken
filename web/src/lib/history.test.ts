import { describe, expect, it } from "vitest"
import { parseHistory } from "./history"
import { fileOf, orderedFixture } from "../test/fixtures"

describe("parseHistory", () => {
  it("rejects malformed JSON with its physical line number", async () => {
    const valid = JSON.stringify(orderedFixture[0])
    await expect(parseHistory(fileOf(`${valid}\nnot-json`))).rejects.toThrow("line 2")
  })

  it("rejects an invalid event envelope instead of accepting a partial history", async () => {
    const lines = [JSON.stringify(orderedFixture[0]), JSON.stringify({ seq: 2 })]
    await expect(parseHistory(fileOf(lines.join("\n")))).rejects.toThrow("line 2")
  })

  it("rejects a sequence gap so reconnect and history share one ordered view", async () => {
    const second = { ...orderedFixture[1], seq: 3 }
    await expect(parseHistory(fileOf([JSON.stringify(orderedFixture[0]), JSON.stringify(second)].join("\n"))))
      .rejects.toThrow("line 2")
  })

  it("returns all validated records in file order", async () => {
    const result = await parseHistory(fileOf(orderedFixture.map((event) => JSON.stringify(event)).join("\n")))
    expect(result.map((event) => event.seq)).toEqual([1, 2, 3, 4])
  })
})
