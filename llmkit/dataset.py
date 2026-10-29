"""Dataloader Pretraining (weighted, resumable, memmap) dan SFT Dataset (Bagian 10.3 & 12.3)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from llmkit.chat_template import render_conversation, PAD_TOKEN


class PretrainSourceCursor:
    """Mengelola pembacaan token sekuensial dari shard-shard suatu sumber data."""

    def __init__(
        self,
        source_name: str,
        tokenized_dir: Path,
        shard_paths: list[str],
        base_seed: int = 1337
    ) -> None:
        self.source_name = source_name
        self.tokenized_dir = tokenized_dir
        self.shard_paths = shard_paths
        self.base_seed = base_seed

        self.epoch = 0
        self.shard_index_in_order = 0
        self.offset = 0

        self.ordered_shards: list[str] = []
        self._update_order()
        self.current_mmap: Optional[np.memmap] = None
        self._open_current_mmap()

    def _update_order(self) -> None:
        """Kocok urutan shard berdasarkan epoch."""
        rng = np.random.default_rng(self.base_seed + self.epoch)
        indices = rng.permutation(len(self.shard_paths))
        self.ordered_shards = [self.shard_paths[i] for i in indices]

    def _open_current_mmap(self) -> None:
        """Buka file memmap shard saat ini."""
        if not self.ordered_shards:
            self.current_mmap = None
            return
        shard_rel = self.ordered_shards[self.shard_index_in_order]
        full_path = self.tokenized_dir / shard_rel
        self.current_mmap = np.memmap(full_path, dtype=np.uint16, mode="r")

    def read_tokens(self, n: int) -> np.ndarray:
        """Baca n token berturut-turut. Jika shard habis, maju ke shard berikutnya atau epoch berikutnya."""
        tokens = np.empty(n, dtype=np.uint16)
        filled = 0

        while filled < n:
            if self.current_mmap is None or len(self.current_mmap) == 0:
                break

            remaining_in_shard = len(self.current_mmap) - self.offset
            to_read = min(n - filled, remaining_in_shard)

            tokens[filled : filled + to_read] = self.current_mmap[self.offset : self.offset + to_read]
            filled += to_read
            self.offset += to_read

            if self.offset >= len(self.current_mmap):
                # Pindah ke shard berikutnya
                self.offset = 0
                self.shard_index_in_order += 1
                if self.shard_index_in_order >= len(self.ordered_shards):
                    self.epoch += 1
                    self.shard_index_in_order = 0
                    self._update_order()
                self._open_current_mmap()

        return tokens

    def get_state(self) -> dict[str, Any]:
        """Ekspor state cursor untuk checkpoint."""
        return {
            "epoch": self.epoch,
            "shard_index_in_order": self.shard_index_in_order,
            "offset": self.offset,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Pulihkan state cursor dari checkpoint."""
        self.epoch = state.get("epoch", 0)
        self._update_order()
        self.shard_index_in_order = state.get("shard_index_in_order", 0)
        self.offset = state.get("offset", 0)
        self._open_current_mmap()


class PretrainDataLoader:
    """
    Dataloader Pretraining dwitahap/tiga-tahap (Stage A/B/C) berbobot dan resumable secara presisi.
    """

    def __init__(
        self,
        project_dir: Path,
        plan: dict[str, Any],
        seed: int = 1337
    ) -> None:
        self.project_dir = project_dir
        self.plan = plan
        self.seed = seed
        self.tokenized_dir = project_dir / "tokenized"

        # Muat manifest
        manifest_path = self.tokenized_dir / "_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.sources_info = self.manifest.get("sources", {})
        self.cursors: dict[str, PretrainSourceCursor] = {}
        for src, info in self.sources_info.items():
            shards = info.get("shards", [])
            if shards:
                self.cursors[src] = PretrainSourceCursor(src, self.tokenized_dir, shards, base_seed=seed)

        # Stage info
        self.pretrain_comp = plan.get("pretrain_data", {})
        self.stages = self.pretrain_comp.get("stages", {})
        self.tokens_target = plan.get("tokens", {}).get("target", 0.0)

        # Hitung ambang batas perpindahan tahap
        self.stage_A_tokens = self.stages.get("A", {}).get("tokens", 0.8 * self.tokens_target)
        self.stage_B_tokens = self.stage_A_tokens + self.stages.get("B", {}).get("tokens", 0.2 * self.tokens_target)

        self.current_stage = "A"
        self.tokens_seen = 0
        self.step = 0

    def get_current_stage(self) -> str:
        """Tentukan tahap saat ini berdasarkan tokens_seen."""
        if self.tokens_seen < self.stage_A_tokens:
            return "A"
        elif self.tokens_seen < self.stage_B_tokens:
            return "B"
        else:
            return "C"

    def next_batch(
        self,
        micro_batch: int,
        seq_len: int,
        device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ambil satu batch (input_ids, labels) dengan ukuran (micro_batch, seq_len).
        Sequence length adalah seq_len + 1 (input_ids = [:-1], labels = [1:]).
        """
        self.current_stage = self.get_current_stage()
        stage_weights = self.stages.get(self.current_stage, {}).get("weights", {})

        # Filter sumber yang benar-benar ada di manifest
        valid_sources = [s for s in stage_weights if s in self.cursors]
        if not valid_sources:
            # Fallback ke semua cursor yang ada
            valid_sources = list(self.cursors.keys())

        weights_arr = np.array([stage_weights.get(s, 1.0) for s in valid_sources], dtype=np.float64)
        weights_arr /= weights_arr.sum()

        # Seed deterministik per step
        rng = np.random.default_rng(self.seed + self.step)
        chosen_sources = rng.choice(valid_sources, size=micro_batch, p=weights_arr)

        seq_tokens_list = []
        for src in chosen_sources:
            cursor = self.cursors[src]
            raw_tokens = cursor.read_tokens(seq_len + 1)
            seq_tokens_list.append(raw_tokens)

        batch_arr = np.array(seq_tokens_list, dtype=np.int64)
        input_ids = torch.from_numpy(batch_arr[:, :-1]).to(device)
        labels = torch.from_numpy(batch_arr[:, 1:]).to(device)

        batch_tokens = micro_batch * seq_len
        self.tokens_seen += batch_tokens
        self.step += 1

        return input_ids, labels

    def get_state(self) -> dict[str, Any]:
        """Ekspor state dataloader lengkap untuk checkpoint."""
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "current_stage": self.current_stage,
            "cursors": {src: c.get_state() for src, c in self.cursors.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Pulihkan state dataloader persis seperti sebelum disimpan."""
        self.step = state.get("step", 0)
        self.tokens_seen = state.get("tokens_seen", 0)
        self.current_stage = state.get("current_stage", "A")
        cursors_state = state.get("cursors", {})
        for src, c_state in cursors_state.items():
            if src in self.cursors:
                self.cursors[src].load_state(c_state)


class SFTDataset(Dataset):
    """Dataset untuk Supervised Fine-Tuning dengan loss masking pada giliran assistant."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Tokenizer,
        max_seq_len: int = 4096,
        pad_token_id: int = 11
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.encoded_data: list[tuple[list[int], list[int]]] = []

        self._preprocess()

    def _preprocess(self) -> None:
        """Tokenisasi sampel dan terapkan masking label -100 pada token non-assistant."""
        for s in self.samples:
            rendered_text, spans = render_conversation(s)
            enc = self.tokenizer.encode(rendered_text)
            tokens = enc.ids
            offsets = enc.offsets

            if len(tokens) > self.max_seq_len:
                # Discard sampel yang melebihi max_seq_len (Bagian 12.3)
                continue

            labels = [-100] * len(tokens)
            for idx, (tok_start, tok_end) in enumerate(offsets):
                # Periksa apakah token berada dalam span assistant
                for span_start, span_end in spans:
                    if tok_start >= span_start and tok_end <= span_end:
                        labels[idx] = tokens[idx]
                        break

            self.encoded_data.append((tokens, labels))

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.encoded_data[idx]


def sft_collate_fn(
    batch: list[tuple[list[int], list[int]]],
    pad_token_id: int = 11
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function untuk SFT dengan padding dinamis hingga sequence terpanjang di batch."""
    max_len = max(len(item[0]) for item in batch)
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for tokens, labels in batch:
        pad_len = max_len - len(tokens)
        input_ids = tokens + [pad_token_id] * pad_len
        att_mask = [1] * len(tokens) + [0] * pad_len
        padded_labels = labels + [-100] * pad_len

        input_ids_list.append(input_ids)
        attention_mask_list.append(att_mask)
        labels_list.append(padded_labels)

    return (
        torch.tensor(input_ids_list, dtype=torch.long),
        torch.tensor(attention_mask_list, dtype=torch.long),
        torch.tensor(labels_list, dtype=torch.long),
    )
