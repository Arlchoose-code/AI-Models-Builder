"""Advisor: Aturan validasi, saran, dan peringatan A01-A18 (Bagian 18.6).
Mendukung dwibahasa (Bahasa Indonesia & English).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from llmkit.gpu_table import get_gpu_spec
from llmkit.i18n import get_lang, tr
from llmkit.planner import compute_memory_estimate


@dataclass
class Finding:
    id: str
    level: str  # "error", "warn", "info"
    message: str
    suggestion: str
    related_setting: str


def run_advisor(
    answers: dict[str, Any],
    plan: dict[str, Any],
    runtime_context: Optional[dict[str, Any]] = None
) -> list[Finding]:
    """
    Evaluasi rencana dan setting berdasarkan aturan A01 sampai A18.
    Mengembalikan daftar Finding terurut: error, warn, info.
    """
    findings: list[Finding] = []
    runtime_context = runtime_context or {}

    settings = plan.get("settings", {})
    arch = plan.get("architecture", {})
    params = arch.get("total_params", 0)
    size_b = float(answers.get("size_b", params / 1e9))

    gpu_name = answers.get("gpu", "H100")
    gpu_spec = get_gpu_spec(gpu_name)
    vram_gb = gpu_spec.vram_gb if gpu_spec else 24.0

    tokens_target = plan.get("tokens", {}).get("target", 0.0)
    hours = plan.get("compute", {}).get("hours", 0.0)
    cost = plan.get("compute", {}).get("cost", 0.0)

    # Nilai-nilai setting
    context_pretrain = int(settings.get("context_pretrain", {}).get("value", 2048))
    context_final = int(settings.get("context_final", {}).get("value", 4096))
    tool_call = bool(settings.get("tool_call", {}).get("value", True))
    think = bool(settings.get("think", {}).get("value", True))
    think_max_tokens = int(settings.get("think_max_tokens", {}).get("value", 1024))
    vocab_size = int(settings.get("vocab_size", {}).get("value", 64000))
    languages = str(settings.get("languages", {}).get("value", "en+id"))
    profile = str(settings.get("profile", {}).get("value", "cloud"))
    vision = bool(settings.get("vision", {}).get("value", False))

    mem_static_gb = (params * 16.0) / (1024 ** 3)

    # A01: mem_static > 85% VRAM
    if mem_static_gb > 0.85 * vram_gb:
        max_b = round((0.85 * vram_gb * (1024 ** 3) / 16.0) / 1e9, 2)
        findings.append(Finding(
            id="A01",
            level="error",
            message=tr(
                f"Model tidak muat untuk full training di GPU ini (statis {mem_static_gb:.1f} GB > {0.85 * vram_gb:.1f} GB).",
                f"Model does not fit for full training on this GPU (static {mem_static_gb:.1f} GB > {0.85 * vram_gb:.1f} GB)."
            ),
            suggestion=tr(f"Pilih GPU lain atau ukuran <= {max_b} B.", f"Choose another GPU or size <= {max_b} B."),
            related_setting="size_b"
        ))

    # A02: estimasi total memori > VRAM walau grad checkpoint aktif dan micro_batch=1
    min_mem = compute_memory_estimate(params, 1, context_pretrain, arch.get("d", 1024), arch.get("L", 24), vocab_size, True)
    if min_mem["total_gb"] > vram_gb:
        findings.append(Finding(
            id="A02",
            level="error",
            message=tr(
                f"Total memori minimal ({min_mem['total_gb']:.1f} GB) melebihi VRAM ({vram_gb:.1f} GB) meskipun gradient checkpointing aktif.",
                f"Minimum total memory ({min_mem['total_gb']:.1f} GB) exceeds VRAM ({vram_gb:.1f} GB) even with gradient checkpointing active."
            ),
            suggestion=tr("Turunkan context_pretrain atau ukuran model.", "Lower context_pretrain or model size."),
            related_setting="context_pretrain"
        ))

    # A03: context_final >= 16384 dan VRAM < 24 GB
    if context_final >= 16384 and vram_gb < 24.0:
        findings.append(Finding(
            id="A03",
            level="warn",
            message=tr(
                f"Konteks panjang ({context_final}) di GPU ini (VRAM {vram_gb:.0f} GB) berisiko OOM.",
                f"Long context ({context_final}) on this GPU (VRAM {vram_gb:.0f} GB) risks OOM."
            ),
            suggestion=tr("Saran: set context_final=4096 atau 8192.", "Recommendation: set context_final=4096 or 8192."),
            related_setting="context_final"
        ))

    # A04: tool_call=true dan context_final < 4096
    if tool_call and context_final < 4096:
        findings.append(Finding(
            id="A04",
            level="warn",
            message=tr(
                f"tool_call aktif tetapi context_final={context_final} < 4096.",
                f"tool_call is active but context_final={context_final} < 4096."
            ),
            suggestion=tr(
                "Hasil search/tool sering panjang; saran context_final=8192.",
                "Search/tool outputs are often long; recommend context_final=8192."
            ),
            related_setting="context_final"
        ))

    # A05: think=true dan N < 0.1B
    if think and size_b < 0.1:
        findings.append(Finding(
            id="A05",
            level="warn",
            message=tr(
                f"Penalaran (think) aktif pada model {size_b:.2f}B (< 0.1B).",
                f"Reasoning (think) enabled on {size_b:.2f}B model (< 0.1B)."
            ),
            suggestion=tr(
                "Penalaran model sekecil ini biasanya buruk; saran think=false.",
                "Reasoning for models this small is usually poor; recommend think=false."
            ),
            related_setting="think"
        ))

    # A06: epochs suatu folder > 4 (error jika > 8)
    folder_epochs = runtime_context.get("folder_epochs", {})
    id_unique_b = answers.get("id_unique_tokens_b")
    if id_unique_b is not None:
        try:
            id_u = float(id_unique_b) * 1e9
            needed_id = plan.get("pretrain_data", {}).get("needed", {}).get("id_web", 0.0)
            if id_u > 0:
                ep = needed_id / id_u
                folder_epochs["id_web"] = max(folder_epochs.get("id_web", 0.0), ep)
        except Exception:
            pass

    for folder, ep in folder_epochs.items():
        if ep > 8.0:
            findings.append(Finding(
                id="A06",
                level="error",
                message=tr(
                    f"Data {folder} diulang {ep:.1f}x (> 8x epoch maksimum).",
                    f"Data {folder} repeated {ep:.1f}x (> 8x maximum epochs)."
                ),
                suggestion=tr("Tambah data unik/sintetis atau turunkan bobotnya.", "Add unique/synthetic data or lower its weight."),
                related_setting="token_mode"
            ))
        elif ep > 4.0:
            findings.append(Finding(
                id="A06",
                level="warn",
                message=tr(
                    f"Data {folder} diulang {ep:.1f}x (> 4x epoch).",
                    f"Data {folder} repeated {ep:.1f}x (> 4x epochs)."
                ),
                suggestion=tr("Tambah data unik/sintetis atau turunkan bobotnya.", "Add unique/synthetic data or lower its weight."),
                related_setting="token_mode"
            ))

    # A07: porsi Indonesia total < 10% dan languages=en+id
    pct_id = plan.get("pretrain_data", {}).get("pct_id", 0.0)
    if languages == "en+id" and pct_id < 10.0:
        findings.append(Finding(
            id="A07",
            level="warn",
            message=tr(
                f"Porsi bahasa Indonesia total ({pct_id:.1f}%) < 10%.",
                f"Total Indonesian language portion ({pct_id:.1f}%) < 10%."
            ),
            suggestion=tr(
                "Kemampuan bahasa Indonesia akan lemah; naikkan data id_web/id_hq.",
                "Indonesian fluency will be weak; increase id_web/id_hq data."
            ),
            related_setting="languages"
        ))

    # A08: rasio token/param > 1000 dan data unik total < 30% dari token latih
    ratio = plan.get("tokens", {}).get("ratio", 0.0)
    total_unique_tokens = runtime_context.get("total_unique_tokens")
    if ratio > 1000.0 and total_unique_tokens is not None:
        if total_unique_tokens < 0.3 * tokens_target:
            findings.append(Finding(
                id="A08",
                level="warn",
                message=tr(
                    "Terlalu banyak pengulangan data (rasio > 1000x dan data unik < 30%).",
                    "Excessive data repetition (ratio > 1000x and unique data < 30%)."
                ),
                suggestion=tr(
                    "Pertimbangkan mode token lebih rendah (balanced/chinchilla).",
                    "Consider lower token mode (balanced/chinchilla)."
                ),
                related_setting="token_mode"
            ))

    # A09: biaya estimasi > budget
    budget_usd = answers.get("budget_usd")
    if budget_usd is not None and float(budget_usd) > 0:
        b_val = float(budget_usd)
        if cost > (b_val + 1.0):
            findings.append(Finding(
                id="A09",
                level="error",
                message=tr(
                    f"Biaya estimasi ${cost:.0f} melebihi budget ${b_val:.0f}.",
                    f"Estimated cost ${cost:.0f} exceeds budget ${b_val:.0f}."
                ),
                suggestion=tr("Kurangi jumlah token atau ukuran model.", "Reduce token count or model size."),
                related_setting="budget_usd"
            ))

    # A10: tool_call=true tetapi folder sft/tool_call kosong saat sft
    if runtime_context.get("phase") == "sft" and tool_call:
        if runtime_context.get("tool_call_samples_count", 0) == 0:
            findings.append(Finding(
                id="A10",
                level="error",
                message=tr(
                    "Folder sft/tool_call kosong padahal fitur tool_call aktif.",
                    "Folder sft/tool_call is empty while tool_call feature is enabled."
                ),
                suggestion=tr("Isi data tool call dulu, atau matikan fitur tool_call.", "Add tool call data first, or disable tool_call."),
                related_setting="tool_call"
            ))

    # A11: vocab_size > 65535
    if vocab_size > 65535:
        findings.append(Finding(
            id="A11",
            level="error",
            message=tr(
                f"vocab_size={vocab_size} > 65.535 (uint16 limit).",
                f"vocab_size={vocab_size} > 65,535 (uint16 limit)."
            ),
            suggestion=tr("Set vocab_size=32000 atau 64000.", "Set vocab_size=32000 or 64000."),
            related_setting="vocab_size"
        ))

    # A12: profile=colab dan checkpoint tidak di Drive
    runs_dir = str(answers.get("runs_dir", "runs"))
    if profile == "colab" and not ("drive" in runs_dir.lower() or "mydrive" in runs_dir.lower()):
        findings.append(Finding(
            id="A12",
            level="warn",
            message=tr("Sesi Colab bisa terputus sewaktu-waktu.", "Colab sessions can disconnect at any time."),
            suggestion=tr(
                "Gunakan --out-dir ke Google Drive (misal /content/drive/MyDrive/runs).",
                "Use --out-dir pointing to Google Drive (e.g. /content/drive/MyDrive/runs)."
            ),
            related_setting="profile"
        ))

    # A13: profile=laptop dan estimasi waktu > 14 hari
    if profile == "laptop" and hours > 336.0:
        findings.append(Finding(
            id="A13",
            level="warn",
            message=tr(
                f"Estimasi training di laptop {hours / 24.0:.1f} hari (> 14 hari).",
                f"Estimated training on laptop is {hours / 24.0:.1f} days (> 14 days)."
            ),
            suggestion=tr(
                "Terlalu lama untuk laptop; pertimbangkan sewa GPU cloud atau kurangi token.",
                "Too long for laptop; consider renting cloud GPU or reducing tokens."
            ),
            related_setting="profile"
        ))

    # A14: vision=true
    if vision:
        findings.append(Finding(
            id="A14",
            level="info",
            message=tr(
                "Vision dikerjakan setelah model teks selesai dan stabil: encoder gambar (misalnya SigLIP) + projector + fine-tune pada data gambar-instruksi (Indonesia perlu diterjemahkan/dibuat sintetis).",
                "Vision should be tackled after text model is stable: image encoder (e.g. SigLIP) + projector + fine-tune on image-instruction data."
            ),
            suggestion=tr(
                "Pada model < 0.6B hasilnya biasanya lemah. Encoder bisa lebih besar dari LLM-nya sendiri.",
                "On models < 0.6B results are usually weak. Encoder may be larger than the LLM itself."
            ),
            related_setting="vision"
        ))

    # A15: ukuran >= 0.3B, token < 20x parameter
    if size_b >= 0.3 and ratio < 20.0:
        findings.append(Finding(
            id="A15",
            level="warn",
            message=tr(
                f"Rasio token/param ({ratio:.1f}x) di bawah Chinchilla (20x).",
                f"Token/param ratio ({ratio:.1f}x) is below Chinchilla (20x)."
            ),
            suggestion=tr("Model akan undertrained; naikkan jumlah token.", "Model will be undertrained; increase token count."),
            related_setting="token_mode"
        ))

    # A16: N=0.6B dan token <= 100B
    if abs(size_b - 0.6) < 0.05 and tokens_target <= 100e9:
        findings.append(Finding(
            id="A16",
            level="info",
            message=tr(
                "Ekspektasi realistis: bahasa lancar, tool call dasar, pengetahuan terbatas; kemungkinan masih di bawah Qwen3-0.6B.",
                "Realistic expectation: fluent language, basic tool calls, limited knowledge; likely below official Qwen3-0.6B."
            ),
            suggestion=tr("Gunakan continued pretraining jika menginginkan performa maksimal.", "Use continued pretraining if maximum performance is desired."),
            related_setting="size_b"
        ))

    # A17: think_max_tokens + typical_prompt_len > context_final
    typical_prompt_len = 1024
    if (think_max_tokens + typical_prompt_len) > context_final:
        findings.append(Finding(
            id="A17",
            level="warn",
            message=tr(
                f"Batas think ({think_max_tokens}) + prompt tipikal ({typical_prompt_len}) > context_final ({context_final}).",
                f"Think limit ({think_max_tokens}) + typical prompt ({typical_prompt_len}) > context_final ({context_final})."
            ),
            suggestion=tr("Turunkan think_max_tokens atau naikkan context_final.", "Lower think_max_tokens or raise context_final."),
            related_setting="think_max_tokens"
        ))

    # A18: setting manual berbeda dari saran auto
    for k, v in settings.items():
        if v.get("mode") == "manual":
            manual_val = v.get("value")
            findings.append(Finding(
                id="A18",
                level="info",
                message=tr(f"Kamu menimpa auto: {k}={manual_val}.", f"You overrode auto setting: {k}={manual_val}."),
                suggestion=tr(
                    f"Nilai auto dapat dikembalikan dengan: python main.py config auto {k}",
                    f"Auto value can be restored with: python main.py config auto {k}"
                ),
                related_setting=k
            ))

    # Urutkan: error duluan, lalu warn, lalu info
    level_order = {"error": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: level_order.get(f.level, 3))
    return findings
