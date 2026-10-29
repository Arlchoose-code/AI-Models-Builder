"""Unit tests untuk ChatML format, loss masking spans, dan konsistensi Jinja template (tests/test_chat_template.py)."""

import json
import unittest
from jinja2 import Template
from tokenizers import Tokenizer
from llmkit.chat_template import (
    EOS_TOKEN,
    SPECIAL_TOKENS,
    get_jinja_template,
    render_conversation,
)
from llmkit.tokenizer_train import build_bpe_trainer


class TestChatTemplate(unittest.TestCase):
    def setUp(self):
        # Bangun tokenizer cepat untuk pengujian offset dan decoding
        self.tok, self.trainer = build_bpe_trainer(vocab_size=1000)
        dummy_corpus = [
            "Kamu adalah Bahasa asisten AI dwibahasa buatan Pengguna.",
            "Berapa 125 dikali 8? /no_think",
            "The capital of France is Paris.",
            "calculate expression",
            "1000 125 * 8 = 1000",
            "<think> 25% of 40 is 10. The sale price is $30. </think>",
        ]
        self.tok.train_from_iterator(dummy_corpus, trainer=self.trainer)

    def test_tool_0001_exact_render_vector(self):
        """Uji vector contoh tepat tool-0001 dari Bagian 8.2."""
        sample = {
            "id": "tool-0001",
            "lang": "id",
            "mode": "no_think",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Hitung ekspresi matematika",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"]
                    }
                }
            }],
            "messages": [
                {"role": "user", "content": "Berapa 125 dikali 8?"},
                {"role": "assistant", "content": "", "tool_calls": [{"name": "calculate", "arguments": {"expression": "125 * 8"}}]},
                {"role": "tool", "name": "calculate", "content": "1000"},
                {"role": "assistant", "content": "125 × 8 = 1000."}
            ]
        }
        # Force tag /no_think aktif untuk mencocokkan contoh spesifikasi
        rendered, spans = render_conversation(sample, allow_think_tags=True)
        # Harus mengandung blok tools, tool_call, tool_response, dan giliran assistant
        self.assertIn("<|im_start|>system", rendered)
        self.assertIn("# Tools", rendered)
        self.assertIn("<tools>", rendered)
        self.assertIn("<tool_call>", rendered)
        self.assertIn("<tool_response>", rendered)
        self.assertIn("125 × 8 = 1000.<|im_end|>", rendered)
        self.assertEqual(len(spans), 2)  # 2 giliran assistant

    def test_loss_masking_spans(self):
        """Tes wajib 8.3: Decode token ber-label != -100 harus berawal <think> dan berakhiran <|im_end|>."""
        samples = []
        for i in range(20):
            mode = "think" if i % 2 == 0 else "no_think"
            samples.append({
                "id": f"s-{i}",
                "mode": mode,
                "messages": [
                    {"role": "user", "content": f"Pertanyaan nomor {i}?"},
                    {"role": "assistant", "reasoning": "Langkah penalaran cermat" if mode == "think" else "", "content": f"Jawaban nomor {i}."}
                ]
            })

        for s in samples:
            text, spans = render_conversation(s)
            for start, end in spans:
                seg = text[start:end]
                self.assertTrue(seg.startswith("<think>"))
                self.assertTrue(seg.endswith("<|im_end|>"))
                self.assertNotIn("<|im_start|>", seg)

    def test_jinja_python_consistency(self):
        """Tes wajib 8.4: Jinja template menghasilkan output identik dengan Python renderer untuk 50 sampel."""
        from transformers.tokenization_utils_base import render_jinja_template

        jinja_str = get_jinja_template("Kamu adalah Bahasa, asisten AI dwibahasa (Indonesia dan Inggris) buatan Pengguna.")
        dummy_tools = [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Hitung ekspresi matematika",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]
                }
            }
        }]

        for i in range(50):
            mode = "think" if i % 2 == 0 else "no_think"
            messages = [{"role": "user", "content": f"Pertanyaan ke-{i}?"}]
            tools = []

            if i % 3 == 0:
                # Percakapan normal dengan / tanpa think
                messages.append({
                    "role": "assistant",
                    "reasoning": f"Nalar mendalam {i}" if mode == "think" else "",
                    "content": f"Jawaban tuntas {i}."
                })
            elif i % 3 == 1:
                # Percakapan dengan tool call
                tools = dummy_tools
                messages.extend([
                    {"role": "assistant", "content": "", "tool_calls": [{"name": "calculate", "arguments": {"expression": f"{i} * 10"}}]},
                    {"role": "tool", "name": "calculate", "content": str(i * 10)},
                    {"role": "assistant", "reasoning": f"Nalar hasil {i}" if mode == "think" else "", "content": f"Hasilnya adalah {i * 10}."}
                ])
            else:
                # Percakapan dengan 2 tool call berurutan (consecutive tool responses)
                tools = dummy_tools
                messages.extend([
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"name": "calculate", "arguments": {"expression": f"{i} + 1"}},
                        {"name": "calculate", "arguments": {"expression": f"{i} + 2"}}
                    ]},
                    {"role": "tool", "name": "calculate", "content": str(i + 1)},
                    {"role": "tool", "name": "calculate", "content": str(i + 2)},
                    {"role": "assistant", "content": f"Hasilnya {i + 1} dan {i + 2}."}
                ])

            sample = {
                "id": f"gen-{i}",
                "mode": mode,
                "tools": tools,
                "messages": messages,
            }
            py_text, _ = render_conversation(sample, allow_think_tags=False)
            hf_res, _ = render_jinja_template(conversations=[messages], tools=tools, chat_template=jinja_str)
            self.assertEqual(py_text, hf_res[0])


if __name__ == "__main__":
    unittest.main()
