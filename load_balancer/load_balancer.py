"""
load_balancer.py
----------------
Distributes inference requests across N worker processes, each holding
a full model bundle. Mirrors the dual-GPU distribution strategy used in
production to recover latency after FP16 quantization on a single device.

Strategy: least-connections routing — requests go to the worker with the
fewest in-flight jobs, falling back to round-robin on ties.

Usage:
    balancer = LoadBalancer(num_workers=2, precision="fp16", devices=["cpu", "cpu"])
    balancer.start()
    result, latency = balancer.encode(texts)
    balancer.shutdown()
"""

import time
import queue
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from models.model_loader import load_models, ModelBundle
from models.inference import run_encoder, run_reranker


@dataclass
class WorkerState:
    worker_id: int
    bundle: ModelBundle
    in_flight: int = 0
    total_requests: int = 0
    total_latency_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def avg_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


class LoadBalancer:
    """
    Thread-safe load balancer over multiple ModelBundle workers.
    """

    def __init__(
        self,
        num_workers: int = 2,
        precision: str = "fp16",
        devices: Optional[List[str]] = None,
    ):
        self.num_workers = num_workers
        self.precision = precision
        self.devices = devices or ["cpu"] * num_workers
        self.workers: List[WorkerState] = []
        self._lock = threading.Lock()

    def start(self):
        """Initialize all workers (loads models into memory)."""
        print(f"[balancer] Starting {self.num_workers} workers (precision={self.precision})")
        for i in range(self.num_workers):
            device = self.devices[i] if i < len(self.devices) else "cpu"
            bundle = load_models(precision=self.precision, device=device)
            self.workers.append(WorkerState(worker_id=i, bundle=bundle))
        print(f"[balancer] All workers ready")

    def shutdown(self):
        """Release model memory."""
        self.workers.clear()
        print("[balancer] Shutdown complete")

    def _pick_worker(self) -> WorkerState:
        """Least-connections routing with round-robin tie-breaking."""
        with self._lock:
            return min(self.workers, key=lambda w: (w.in_flight, w.worker_id))

    def _record(self, worker: WorkerState, latency_ms: float):
        with worker.lock:
            worker.in_flight -= 1
            worker.total_requests += 1
            worker.total_latency_ms += latency_ms

    def encode(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, float]:
        """Route an encoding request to the best available worker."""
        worker = self._pick_worker()
        with worker.lock:
            worker.in_flight += 1
        try:
            result, latency = run_encoder(worker.bundle.encoder, texts, batch_size)
        finally:
            self._record(worker, latency)
        return result, latency

    def multilingual_encode(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, float]:
        """Route a multilingual encoding request to the best available worker."""
        worker = self._pick_worker()
        with worker.lock:
            worker.in_flight += 1
        try:
            result, latency = run_encoder(worker.bundle.multilingual, texts, batch_size)
        finally:
            self._record(worker, latency)
        return result, latency

    def rerank(self, query: str, passages: List[str]) -> Tuple[np.ndarray, float]:
        """Route a re-ranking request to the best available worker."""
        worker = self._pick_worker()
        with worker.lock:
            worker.in_flight += 1
        try:
            result, latency = run_reranker(worker.bundle.reranker, query, passages)
        finally:
            self._record(worker, latency)
        return result, latency

    def stats(self) -> List[dict]:
        """Return per-worker stats for observability."""
        return [
            {
                "worker_id": w.worker_id,
                "device": w.bundle.device,
                "precision": w.bundle.precision,
                "total_requests": w.total_requests,
                "avg_latency_ms": round(w.avg_latency(), 2),
            }
            for w in self.workers
        ]
