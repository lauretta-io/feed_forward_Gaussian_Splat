from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ariadne.common import ModelVersion, Timestamp
from ariadne.communications import MessageClass, TransportMessage, UplinkPackager
from ariadne.intelligence import ObservationJournal, ObservationRegistry
from ariadne.object_state import KeyframeRecord, LocalObjectRecord


def messages(count: int) -> tuple[TransportMessage, ...]:
    record = LocalObjectRecord(
        "wingman_01:tower",
        "wingman_01",
        Timestamp(100),
        np.array([4.0, 2.0, 0.0]),
        np.array([1.0, 0.2, 0.1]),
        np.array([0.04, 0.09, 0.01]),
        0.9,
        ModelVersion("reference", "1"),
        1,
        (KeyframeRecord(1, Timestamp(100), 0.9, "tower"),),
    )
    packager = UplinkPackager()
    result = []
    for index in range(count):
        timestamp = Timestamp(100 + index)
        packet = packager.package("wingman_01", timestamp, (record,))
        result.append(
            TransportMessage(
                f"message-{index}",
                "wingman_01",
                "intelligence_01",
                MessageClass.OBSERVATION,
                timestamp,
                10_000,
                packet,
            )
        )
    return tuple(result)


def test_journal_round_trip_rotation_and_bounded_retention(tmp_path: Path) -> None:
    journal = ObservationJournal(
        tmp_path / "journal",
        max_entries_per_segment=1,
        max_segments=2,
    )
    source = messages(3)
    assert all(journal.append(message) for message in source)

    assert [path.name for path in journal.segment_paths] == [
        "segment-00000000000000000001.jsonl",
        "segment-00000000000000000002.jsonl",
    ]
    assert [message.message_id for message in journal.messages()] == [
        "message-1",
        "message-2",
    ]
    assert journal.entry_count == 2
    assert journal.metrics["pruned_segments"] == 1
    assert not journal.append(source[-1])
    assert journal.metrics["duplicates"] == 1


def test_journal_repairs_only_an_interrupted_final_write(tmp_path: Path) -> None:
    directory = tmp_path / "journal"
    journal = ObservationJournal(directory)
    first, second = messages(2)
    journal.append(first)
    with journal.segment_paths[-1].open("ab") as stream:
        stream.write(b'{"schema":"interrupted')

    recovered = ObservationJournal(directory)
    assert recovered.metrics["recovered_tails"] == 1
    assert recovered.messages() == (first,)
    assert recovered.append(second)
    assert recovered.messages() == (first, second)


def test_journal_fails_closed_on_committed_entry_corruption(tmp_path: Path) -> None:
    directory = tmp_path / "journal"
    journal = ObservationJournal(directory)
    journal.append(messages(1)[0])
    path = journal.segment_paths[0]
    path.write_text(
        path.read_text(encoding="utf-8").replace("wingman_01", "wingman_99", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="corruption"):
        ObservationJournal(directory)


def test_registry_journals_before_mutating_derived_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = ObservationJournal(tmp_path / "journal")
    registry = ObservationRegistry(journal=journal)
    message = messages(1)[0]

    def fail_append(_message: TransportMessage) -> bool:
        raise OSError("disk unavailable")

    monkeypatch.setattr(journal, "append", fail_append)
    with pytest.raises(OSError, match="unavailable"):
        registry.ingest(message, Timestamp(100))
    assert registry.observations == ()
    assert registry.metrics["packets"] == 0


def test_registry_rebuilds_derived_state_from_raw_journal(tmp_path: Path) -> None:
    journal = ObservationJournal(tmp_path / "journal")
    original = ObservationRegistry(journal=journal)
    message = messages(1)[0]
    assert len(original.ingest(message, Timestamp(100))) == 1
    assert original.metrics["journaled"] == 1

    rebuilt = ObservationRegistry()
    accepted = rebuilt.replay_journal(journal, now=Timestamp(100))
    assert len(accepted) == 1
    assert accepted[0].local_id == "wingman_01:tower"
    assert rebuilt.metrics["journal_replays"] == 1
    assert journal.messages()[0] == message
