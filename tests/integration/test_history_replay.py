from __future__ import annotations

import json
from collections import defaultdict

from tests.integration.test_headless_game import run_complete_game


async def test_completed_history_reopens_with_ordered_traces_checkpoints_and_isolated_prompts(tmp_path):
    """Dropping a trace field, checkpoint, or prompt audience must break replay acceptance."""

    path = tmp_path / "completed.jsonl"
    _orchestrator, script = await run_complete_game(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert records[0]["type"] == "game.header"
    assert records[-1]["type"] == "game.ended"
    assert [record["seq"] for record in records] == list(range(1, len(records) + 1))

    notebook_indexes = [
        index for index, record in enumerate(records) if record["type"] == "notebook.updated"
    ]
    assert notebook_indexes, "the complete fake-provider path must exercise notebook checkpoints"
    for index in notebook_indexes:
        assert records[index + 1]["type"] == "checkpoint"
        assert records[index + 1]["payload"]["trigger_event_seq"] == records[index]["seq"]

    calls: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record["type"] == "model.output_segment":
            calls[record["payload"]["call_id"]].append(record["payload"])
    assert calls, "the complete fake-provider path must persist raw model segments"
    for segments in calls.values():
        assert [segment["segment_index"] for segment in segments] == [0, 1, 2, 3, 4]
        assert [segment["source_field"] for segment in segments] == [
            "reasoning_content",
            "thinking",
            "tool_calls",
            "tool_result",
            "content",
        ]

    by_seq = {record["seq"]: record for record in records}
    assert script.prompt_event_seqs
    for player_id, prompt_seqs in script.prompt_event_seqs.items():
        for seq in prompt_seqs:
            record = by_seq[seq]
            assert record["type"] not in {"checkpoint", "model.output_segment", "game.header", "setup.completed"}
            audience = record["audience"]
            assert audience["kind"] == "public" or player_id in audience["player_ids"]

    private_records = [record for record in records if record["type"] == "chat.private_message"]
    assert private_records
    for record in private_records:
        recipients = set(record["audience"]["player_ids"])
        for player_id, prompt_seqs in script.prompt_event_seqs.items():
            if player_id not in recipients:
                assert record["seq"] not in prompt_seqs
