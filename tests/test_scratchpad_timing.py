"""Scratchpad banking and latency helper tests."""

from __future__ import annotations

from perf_modeling.config import ScratchpadConfig
from perf_modeling.state.scratchpad import ScratchpadMemory
from perf_modeling.timing.latency import scratchpad_access_latency


class TestScratchpadTiming:
    """Validate bank enumeration and bank-aware scratchpad latency."""

    def test_bank_indices_for_range_tracks_contiguous_bank_footprint(self) -> None:
        """Contiguous local address ranges should enumerate the touched banks in order."""
        scratchpad = ScratchpadMemory(
            capacity_bytes=1024,
            name="scratchpad",
            num_banks=4,
            bank_width_bytes=16,
        )

        assert scratchpad.bank_indices_for_range(address=0, size_bytes=4) == (0,)
        assert scratchpad.bank_indices_for_range(address=0, size_bytes=20) == (0, 1)
        assert scratchpad.bank_indices_for_range(address=24, size_bytes=24) == (1, 2)

    def test_scratchpad_access_latency_scales_with_banks_touched(self) -> None:
        """Wider bank footprints should expose higher scratchpad bandwidth."""
        config = ScratchpadConfig(
            capacity_bytes=1024,
            num_banks=4,
            bank_width_bytes=16,
        )

        assert scratchpad_access_latency(config, banks_touched=1, num_bytes=32) == 2
        assert scratchpad_access_latency(config, banks_touched=2, num_bytes=32) == 1
        assert scratchpad_access_latency(config, banks_touched=8, num_bytes=64) == 1
