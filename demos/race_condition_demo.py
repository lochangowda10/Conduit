"""Experiment 8, live: a race condition, its fix, a thread pool, and the GIL.

Run:  python -m demos.race_condition_demo
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from conduit.ledger import CostLedger
from conduit.types import Usage

THREADS = 8
ADDS_PER_THREAD = 2_000
ONE_CALL = Usage(prompt_tokens=10, completion_tokens=20, cost_usd=0.0001)


def run_threads(target) -> float:
    threads = [threading.Thread(target=target) for _ in range(THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0


def demo_race() -> None:
    expected = THREADS * ADDS_PER_THREAD
    print(f"{THREADS} threads x {ADDS_PER_THREAD} updates each -> expected {expected:,} recorded calls\n")

    unsafe = CostLedger()
    run_threads(lambda: [unsafe.unsafe_add("sim", ONE_CALL) for _ in range(ADDS_PER_THREAD)])
    lost = expected - unsafe._by_provider["sim"].prompt_tokens // 10
    print(f"  WITHOUT lock : recorded {expected - lost:>6,}  -> {lost:,} updates LOST "
          f"({lost / expected:.1%}); total cost is under-reported")

    safe = CostLedger()
    run_threads(lambda: [safe.add("sim", ONE_CALL) for _ in range(ADDS_PER_THREAD)])
    got = safe.total.prompt_tokens // 10
    print(f"  WITH lock    : recorded {got:>6,}  -> {expected - got} lost")
    print("""
Why: `total = total + usage` is read -> compute -> write. Two threads can both
read the same old total; the second write erases the first. The GIL only makes
single bytecodes atomic, not this 3-step sequence. A Lock makes the whole
sequence a critical section: only one thread at a time can be inside it.
(The demo calls time.sleep(0) between read and write to make the bad
interleaving happen reliably; in production it happens rarely - which is worse,
because it's hard to reproduce.)
""")


def demo_pool() -> None:
    ledger = CostLedger()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = [pool.submit(ledger.add, "sim", ONE_CALL) for _ in range(THREADS * ADDS_PER_THREAD)]
        for f in futures:
            f.result()  # re-raises any exception from the worker
    print(f"ThreadPoolExecutor: {len(ledger):,} calls recorded, total = {ledger.total}")


def demo_gil() -> None:
    def io_task():
        time.sleep(0.2)  # like waiting for an LLM API: releases the GIL

    def cpu_task():
        sum(i * i for i in range(400_000))  # pure Python bytecode: holds the GIL

    for name, fn in (("I/O-bound (network wait)", io_task), ("CPU-bound (pure Python)", cpu_task)):
        t0 = time.perf_counter()
        for _ in range(8):
            fn()
        serial = time.perf_counter() - t0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: fn(), range(8)))
        threaded = time.perf_counter() - t0
        print(f"  {name:<26} serial {serial:5.2f}s   8 threads {threaded:5.2f}s   speed-up x{serial / threaded:4.1f}")
    print("""
Threads speed up the I/O-bound work ~8x but not the CPU-bound work: only one
thread can execute Python bytecode at a time (the GIL), but a thread that is
waiting on a socket releases it. A gateway spends almost all its time waiting
on providers, so threads are the right tool. CPU-heavy work (e.g. embedding
a million documents) would use multiprocessing, NumPy (which releases the
GIL inside its C loops), or Python 3.13's free-threaded build.
""")


def banner(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    banner("1. Race condition and the Lock fix")
    demo_race()
    banner("2. Same work through a ThreadPoolExecutor")
    demo_pool()
    print()
    banner("3. The GIL: I/O-bound vs CPU-bound")
    demo_gil()


if __name__ == "__main__":
    main()
