from __future__ import annotations

import multiprocessing as mp
import textwrap
from dataclasses import dataclass
from typing import Callable


@dataclass
class SandboxResult:
    ok: bool
    assignment_fn: Callable[[list[int], list[int]], list[int]] | None
    error: str


def _compile_worker(source: str, queue: mp.Queue) -> None:
    namespace: dict = {}
    try:
        exec(textwrap.dedent(source), {"__builtins__": __builtins__}, namespace)  # noqa: S102
        fn = namespace.get("assignment")
        if not callable(fn):
            queue.put(("error", "missing_assignment_function"))
            return
        queue.put(("ok", source))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", str(exc)))


def compile_assignment_function(source: str, timeout_seconds: int = 6) -> SandboxResult:
    queue: mp.Queue = mp.Queue()
    p = mp.Process(target=_compile_worker, args=(source, queue))
    p.start()
    p.join(timeout=timeout_seconds)
    if p.is_alive():
        p.terminate()
        return SandboxResult(ok=False, assignment_fn=None, error="compile_timeout")
    if queue.empty():
        return SandboxResult(ok=False, assignment_fn=None, error="empty_compile_result")
    status, payload = queue.get()
    if status != "ok":
        return SandboxResult(ok=False, assignment_fn=None, error=str(payload))
    namespace: dict = {}
    exec(payload, {"__builtins__": __builtins__}, namespace)  # noqa: S102
    fn = namespace["assignment"]
    return SandboxResult(ok=True, assignment_fn=fn, error="")
