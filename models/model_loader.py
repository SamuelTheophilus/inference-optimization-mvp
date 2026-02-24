"""
model_loader.py
---------------
Loads encoder, re-ranker, and speech-to-text models in FP32 or FP16.
Mirrors the quantization approach used in production to reduce compute costs
while keeping accuracy within 5% of baseline.
"""

import time
from dataclasses import dataclass
from typing import Literal

import torch
import whisper
from sentence_transformers import CrossEncoder, SentenceTransformer

Precision = Literal["fp32", "fp16"]


@dataclass
class ModelBundle:
    encoder: SentenceTransformer
    multilingual: SentenceTransformer
    reranker: CrossEncoder
    stt: object  # whisper model
    precision: Precision
    device: str


def load_models(precision: Precision = "fp32", device: str = "cpu") -> ModelBundle:
    """
    Load all models at the specified precision.

    Args:
        precision: 'fp32' (baseline) or 'fp16' (optimized)
        device: 'cpu', 'cuda', or 'cuda:0' / 'cuda:1'

    Returns:
        ModelBundle with all models ready for inference
    """
    dtype = torch.float16 if precision == "fp16" else torch.float32

    if precision == "fp16" and device == "cpu":
        import os

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if "qnnpack" in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = "qnnpack"

    print(f"[loader] Loading models — precision={precision}, device={device}")

    # Encoder
    t0 = time.perf_counter()
    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device
    )
    if precision == "fp16":
        if device == "cpu":
            encoder = encoder.to("cpu")
            # Dynamic quantization is much faster on CPU than .half()
            encoder = torch.quantization.quantize_dynamic(
                encoder, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            encoder = encoder.half()
    print(f"[loader] Encoder ready ({time.perf_counter() - t0:.2f}s)")

    # Multilingual Encoder
    t0 = time.perf_counter()
    multilingual = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device
    )
    if precision == "fp16":
        if device == "cpu":
            multilingual = torch.quantization.quantize_dynamic(
                multilingual, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            multilingual = multilingual.half()
    print(f"[loader] Multilingual ready ({time.perf_counter() - t0:.2f}s)")

    # Re-ranker
    t0 = time.perf_counter()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    if precision == "fp16":
        if device == "cpu":
            reranker.model = reranker.model.to("cpu")
            reranker.model = torch.quantization.quantize_dynamic(
                reranker.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            reranker.model = reranker.model.half().to(device)
    elif device != "cpu":
        reranker.model = reranker.model.to(device)
    print(f"[loader] Re-ranker ready ({time.perf_counter() - t0:.2f}s)")

    # Speech-to-text (Whisper)
    t0 = time.perf_counter()
    stt = whisper.load_model("base", device=device)
    if precision == "fp16":
        if device != "cpu":
            stt = stt.half()
    elif device != "cpu":
        stt = stt.to(device)
    print(f"[loader] STT ready ({time.perf_counter() - t0:.2f}s)")

    return ModelBundle(
        encoder=encoder,
        multilingual=multilingual,
        reranker=reranker,
        stt=stt,
        precision=precision,
        device=device,
    )
