import { parseEventRecord, type EventRecord } from "../types"

export async function parseHistory(file: File): Promise<EventRecord[]> {
  const text = await file.text()
  const lines = text.split(/\r?\n/)
  if (lines.at(-1) === "") lines.pop()
  if (lines.length === 0) throw new Error("History file is empty")

  const records: EventRecord[] = []
  for (let index = 0; index < lines.length; index += 1) {
    const lineNumber = index + 1
    const line = lines[index]
    if (!line.trim()) throw new Error(`Invalid history at line ${lineNumber}: blank line`)
    try {
      const record = parseEventRecord(JSON.parse(line))
      const expected = records.length === 0 ? 1 : records.at(-1)!.seq + 1
      if (record.seq !== expected) throw new Error(`expected sequence ${expected}, received ${record.seq}`)
      records.push(record)
    } catch (error) {
      const detail = error instanceof Error ? error.message : "invalid record"
      throw new Error(`Invalid history at line ${lineNumber}: ${detail}`)
    }
  }
  return records
}
