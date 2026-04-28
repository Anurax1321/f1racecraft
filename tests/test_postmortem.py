"""Postmortem memory storage and classification tests."""

from server.postmortem import PostmortemMemory


def test_postmortem_record_and_retrieve(tmp_path, monkeypatch):
    path = tmp_path / "postmortems.jsonl"
    monkeypatch.setattr(PostmortemMemory, "PATH", path)
    PostmortemMemory.record({"scenario_family": "dry_strategy_sprint", "final_score": 0.4})
    PostmortemMemory.record({"scenario_family": "dry_strategy_sprint", "final_score": 0.2})
    rows = PostmortemMemory.retrieve("dry_strategy_sprint", k=1)
    assert len(rows) == 1
    assert rows[0]["final_score"] == 0.2


def test_postmortem_classifies_panic_pit():
    audit = [
        {"lap": 2, "action": "PIT_NOW soft"},
        {"lap": 4, "action": "PIT_NOW hard"},
    ]
    assert PostmortemMemory.classify_failure(audit, {"fuel_management": 1.0}) == "panic_pit"
