"""Bounded append-only raw observation journal with integrity verification."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

from ariadne.common import Timestamp
from ariadne.communications import MessageClass, TransportMessage, UplinkPacket

JOURNAL_SCHEMA = "ariadne.observation-journal.v1"


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


class ObservationJournal:
    def __init__(
        self,
        directory: str | Path,
        *,
        max_entries_per_segment: int = 1024,
        max_segments: int = 8,
    ) -> None:
        if max_entries_per_segment <= 0 or max_segments <= 0:
            raise ValueError("journal bounds must be positive")
        self.directory = Path(directory)
        self.max_entries_per_segment = max_entries_per_segment
        self.max_segments = max_segments
        self.directory.mkdir(parents=True, exist_ok=True)
        self._message_ids: set[str] = set()
        self._last_checksum = ""
        self._active_segment = 0
        self._active_entries = 0
        self.metrics = {
            "appended": 0,
            "duplicates": 0,
            "replayed": 0,
            "recovered_tails": 0,
            "pruned_segments": 0,
        }
        self._load_state(repair_tail=True)

    @property
    def segment_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self.directory.glob("segment-*.jsonl")))

    @property
    def entry_count(self) -> int:
        return len(self._message_ids)

    def append(self, message: TransportMessage) -> bool:
        if message.message_id in self._message_ids:
            self.metrics["duplicates"] += 1
            return False
        if self._active_entries >= self.max_entries_per_segment:
            self._active_segment += 1
            self._active_entries = 0
        content = self._message_to_content(message, self._last_checksum)
        checksum = hashlib.sha256(_canonical_json(content)).hexdigest()
        entry = {**content, "checksum_sha256": checksum}
        line = _canonical_json(entry) + b"\n"
        path = self._segment_path(self._active_segment)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            written = os.write(descriptor, line)
            if written != len(line):
                raise OSError("observation journal write was incomplete")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._message_ids.add(message.message_id)
        self._last_checksum = checksum
        self._active_entries += 1
        self.metrics["appended"] += 1
        self._prune()
        return True

    def messages(self) -> tuple[TransportMessage, ...]:
        entries = self._read_entries(repair_tail=False)
        messages = tuple(self._content_to_message(entry) for entry in entries)
        self.metrics["replayed"] += len(messages)
        return messages

    def _load_state(self, *, repair_tail: bool) -> None:
        entries = self._read_entries(repair_tail=repair_tail)
        self._message_ids = {str(entry["message_id"]) for entry in entries}
        self._last_checksum = (
            str(entries[-1]["checksum_sha256"]) if entries else ""
        )
        paths = self.segment_paths
        if paths:
            self._active_segment = self._segment_number(paths[-1])
            self._active_entries = sum(
                self._segment_number_path(entry) == self._active_segment
                for entry in entries
            )
        else:
            self._active_segment = 0
            self._active_entries = 0

    def _read_entries(self, *, repair_tail: bool) -> list[dict[str, Any]]:
        paths = self.segment_paths
        entries: list[dict[str, Any]] = []
        previous_checksum: str | None = None
        seen_ids: set[str] = set()
        for path_index, path in enumerate(paths):
            data = path.read_bytes()
            offset = 0
            lines = data.splitlines(keepends=True)
            for line_index, line in enumerate(lines):
                is_final_line = (
                    path_index == len(paths) - 1 and line_index == len(lines) - 1
                )
                try:
                    if not line.endswith(b"\n"):
                        raise ValueError("journal line is incomplete")
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("journal entry must be an object")
                    entry = cast(dict[str, Any], raw)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                    if repair_tail and is_final_line:
                        with path.open("r+b") as stream:
                            stream.truncate(offset)
                            stream.flush()
                            os.fsync(stream.fileno())
                        self.metrics["recovered_tails"] += 1
                        break
                    raise ValueError(
                        f"observation journal corruption in {path.name}"
                    ) from error
                try:
                    self._validate_entry(entry, previous_checksum, seen_ids)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"observation journal corruption in {path.name}"
                    ) from error
                entries.append(entry)
                seen_ids.add(str(entry["message_id"]))
                previous_checksum = str(entry["checksum_sha256"])
                offset += len(line)
        return entries

    @staticmethod
    def _validate_entry(
        entry: dict[str, Any],
        previous_checksum: str | None,
        seen_ids: set[str],
    ) -> None:
        if entry.get("schema") != JOURNAL_SCHEMA:
            raise ValueError("unsupported observation journal schema")
        checksum = str(entry.get("checksum_sha256", ""))
        content = {key: value for key, value in entry.items() if key != "checksum_sha256"}
        if hashlib.sha256(_canonical_json(content)).hexdigest() != checksum:
            raise ValueError("observation journal checksum mismatch")
        if previous_checksum is not None and entry.get("previous_checksum") != previous_checksum:
            raise ValueError("observation journal chain mismatch")
        message_id = str(entry.get("message_id", ""))
        if not message_id or message_id in seen_ids:
            raise ValueError("observation journal message identifiers must be unique")

    def _prune(self) -> None:
        paths = self.segment_paths
        expired = paths[: max(0, len(paths) - self.max_segments)]
        if not expired:
            return
        for path in expired:
            path.unlink()
            self.metrics["pruned_segments"] += 1
        self._load_state(repair_tail=False)

    def _segment_path(self, number: int) -> Path:
        return self.directory / f"segment-{number:020d}.jsonl"

    @staticmethod
    def _segment_number(path: Path) -> int:
        return int(path.stem.removeprefix("segment-"))

    @staticmethod
    def _segment_number_path(entry: dict[str, Any]) -> int:
        return int(entry["journal_segment"])

    def _message_to_content(
        self, message: TransportMessage, previous_checksum: str
    ) -> dict[str, object]:
        packet = message.packet
        return {
            "schema": JOURNAL_SCHEMA,
            "journal_segment": self._active_segment,
            "previous_checksum": previous_checksum,
            "message_id": message.message_id,
            "source": message.source,
            "destination": message.destination,
            "message_class": int(message.message_class),
            "created_at_ns": message.created_at.monotonic_ns,
            "expires_at_ns": message.expires_at_ns,
            "packet": {
                "agent_id": packet.agent_id,
                "timestamp_ns": packet.timestamp.monotonic_ns,
                "sequence": packet.sequence,
                "payload_base64": base64.b64encode(packet.payload).decode("ascii"),
                "checksum_sha256": packet.checksum_sha256,
                "object_count": packet.object_count,
            },
        }

    @staticmethod
    def _content_to_message(entry: dict[str, Any]) -> TransportMessage:
        raw_packet = cast(dict[str, Any], entry["packet"])
        packet = UplinkPacket(
            agent_id=str(raw_packet["agent_id"]),
            timestamp=Timestamp(int(raw_packet["timestamp_ns"])),
            sequence=int(raw_packet["sequence"]),
            payload=base64.b64decode(str(raw_packet["payload_base64"]), validate=True),
            checksum_sha256=str(raw_packet["checksum_sha256"]),
            object_count=int(raw_packet["object_count"]),
        )
        return TransportMessage(
            message_id=str(entry["message_id"]),
            source=str(entry["source"]),
            destination=str(entry["destination"]),
            message_class=MessageClass(int(entry["message_class"])),
            created_at=Timestamp(int(entry["created_at_ns"])),
            expires_at_ns=int(entry["expires_at_ns"]),
            packet=packet,
        )
