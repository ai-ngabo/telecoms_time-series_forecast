"""
memory_profile.py
__________________

Lightweight memory profiler for benchmarking pandas pipelines, 
tracking peak resident set size (RSS).

Why use psutil RSS instead of tracemalloc?
- tracemalloc only measures Python-level allocations.
- Pandas/NumPy rely heavily on C-extension buffers (e.g., NumPy arrays backing DataFrame columns).
- These allocations are invisible to tracemalloc, leading to severe under-reporting.
- psutil’s RSS reflects the actual physical memory mapped by the OS, 
  which is the true RAM footprint of a workload.

Outputs:
- baseline_mb → memory at start
- peak_mb → maximum observed memory
- delta_mb → increase during the operation
"""

import os
import time
import threading
import psutil
 
 
class PeakMemoryMonitor:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread = None
        self.baseline_mb = None
        self.peak_mb = None
        self.delta_mb = None
 
    def _rss_mb(self):
        return self.process.memory_info().rss / (1024 ** 2)
 
    def _sample_loop(self):
        while not self._stop_event.is_set():
            current = self._rss_mb()
            if current > self.peak_mb:
                self.peak_mb = current
            time.sleep(self.interval)
 
    def __enter__(self):
        # Encourage a clean baseline reading
        import gc
        gc.collect()
        self.baseline_mb = self._rss_mb()
        self.peak_mb = self.baseline_mb
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join()
        self.delta_mb = self.peak_mb - self.baseline_mb
        return False
 
    def report(self, label=""):
        print(f"[{label}] baseline={self.baseline_mb:.1f} MB  "
              f"peak={self.peak_mb:.1f} MB  "
              f"delta={self.delta_mb:.1f} MB")
 