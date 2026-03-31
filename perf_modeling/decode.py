"""Program decoding entry points."""

from __future__ import annotations

from perf_modeling.program import Program


class Decoder:
    """Placeholder decoder for assembly or compiler-emitted traces."""

    def decode_lines(self, lines: list[str]) -> Program:
        """Decode textual instruction lines into a program."""
        raise NotImplementedError("Instruction decoding has not been implemented yet.")

    def decode_bytes(self, blob: bytes) -> Program:
        """Decode a binary program image into a program."""
        raise NotImplementedError("Binary decoding has not been implemented yet.")
