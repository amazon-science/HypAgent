# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Code-execution sandbox for LLM agent exploration — a persistent REPL.

The agent writes Python to explore the data. Each cell runs in the *same*
namespace so state persists across turns, and a compact summary of the
user-defined variables is appended after every call so the model always knows
what is available.

Each cell runs in a forked child process that the parent kills on a wall-clock
deadline. Unlike a ``SIGALRM`` this is uncatchable by the cell and stops
C-extension spins (an accidental O(n^2) scan or a runaway matmul), which a
Python-level signal cannot interrupt. Only small, pickleable variables are
shipped back to the parent so a per-row array over a large frame does not get
re-pickled through the pipe every turn.
"""

import os
import sys
from io import StringIO

DEFAULT_MAX_OUTPUT = 60000

# Wall-clock cap on a single agent-generated cell.
DEFAULT_EXEC_TIMEOUT = int(os.environ.get("SANDBOX_EXEC_TIMEOUT", "600"))

# Largest variable shipped back from the forked child to the parent. Row-scale
# objects (a 1M-row float64 array is ~8MB) stay in the child; the model consumes
# them via printed output anyway.
_MAX_VAR_BYTES = int(os.environ.get("SANDBOX_MAX_VAR_BYTES", str(1 * 1024 * 1024)))

# Variables excluded from the printed state summary.
_BUILTIN_KEYS = frozenset({
    'df', 'pd', 'np', 're', 'stats', '__builtins__', '__name__',
    '__doc__', '__package__', '__loader__', '__spec__',
})


def _namespace_summary(namespace):
    """Return a compact summary of the user-defined variables in ``namespace``."""
    import pandas as pd
    import numpy as np

    lines = []
    for k, v in sorted(namespace.items()):
        if k in _BUILTIN_KEYS or k.startswith('_'):
            continue
        if isinstance(v, pd.DataFrame):
            lines.append(f"  {k}: DataFrame {v.shape} cols={list(v.columns[:6])}")
        elif isinstance(v, pd.Series):
            lines.append(f"  {k}: Series len={len(v)} dtype={v.dtype}")
        elif isinstance(v, np.ndarray):
            lines.append(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, (int, float, bool, str)):
            lines.append(f"  {k} = {repr(v)[:60]}")
        elif isinstance(v, (list, tuple)):
            lines.append(f"  {k}: {type(v).__name__} len={len(v)}")
        elif isinstance(v, dict):
            lines.append(f"  {k}: dict keys={list(v.keys())[:5]}")
        # functions, modules, classes are skipped
    return lines


def _run_cell_in_namespace(code, namespace, max_output):
    """Execute one cell in ``namespace`` and return ``(output_str, updated_vars)``.

    This is the body that runs inside the forked child. It captures stdout, and
    if the cell produced none it evaluates a trailing bare expression (REPL
    behaviour). ``updated_vars`` is the pickleable subset of namespace mutations
    shipped back to the parent so state persists across cells.
    """
    import pandas as pd
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.min_rows', 25)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 40)
    before = set(namespace.keys())
    try:
        exec(code, namespace)
        output = sys.stdout.getvalue()
        if not output.strip():
            lines = code.strip().split('\n')
            last = lines[-1].strip()
            if last and not last.startswith(('import', 'from', 'def', 'class',
                                            'if', 'for', 'while', '#', 'print')):
                try:
                    r = eval(last, namespace)
                    if r is not None:
                        output = repr(r)
                except:  # noqa: E722 — REPL fallback: swallow ANYTHING (incl. a
                    pass  # stray exit()/SystemExit) so a bad last line is a no-op
    except Exception as e:
        output = f"ERROR: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout

    if len(output) > max_output:
        output = output[:max_output] + "\n... TRUNCATED."

    # Ship back only small user-defined vars the cell created/rebound. Larger
    # ones stay in the child (the model already saw their printed output).
    import pickle
    updated = {}
    for k in set(namespace.keys()) | before:
        if k in _BUILTIN_KEYS or k.startswith('_') or k in ('resid', 'residual'):
            continue
        v = namespace.get(k)
        try:
            blob = pickle.dumps(v)
        except Exception:
            continue  # functions, modules, closures — not persisted across cells
        if len(blob) > _MAX_VAR_BYTES:
            continue  # too big to round-trip; leave it in the child
        updated[k] = v
    return output, updated


def _child_target(conn, code, namespace, max_output):
    try:
        result = _run_cell_in_namespace(code, namespace, max_output)
        conn.send(("ok", result))
    except Exception as e:  # defensive; child crashes otherwise surface as timeouts
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def execute_code(code, namespace, max_output=DEFAULT_MAX_OUTPUT, timeout=DEFAULT_EXEC_TIMEOUT):
    """Execute ``code`` in the persistent ``namespace``, capturing stdout.

    Returns the cell output followed by a REPL state summary of the currently
    available user-defined variables. A runaway cell is stopped by running it in
    a forked child the parent kills after ``timeout`` seconds. The child inherits
    ``df``/``resid`` copy-on-write (no pickling of large frames) and ships back
    small pickleable vars so state persists across cells. If fork is unavailable
    the cell falls back to in-process exec with no hard timeout.
    """
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = None

    if ctx is None or not (timeout and timeout > 0):
        # No isolation available — run in-process (best effort).
        output, updated = _run_cell_in_namespace(code, namespace, max_output)
        namespace.update(updated)
    else:
        import threading
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_child_target,
                           args=(child_conn, code, namespace, max_output), daemon=True)
        proc.start()
        child_conn.close()  # parent keeps only the read end

        # Receive the child's result under a hard deadline. poll(timeout) only
        # bounds the wait for the first byte; recv() itself is unbounded, and a
        # forked child can deadlock mid-send on a lock a background thread held at
        # fork time. Capping the whole receive turns that into a single failed
        # cell (which the agent tolerates) rather than a wedged run.
        box = {}
        def _recv():
            try:
                if parent_conn.poll(timeout):
                    box["r"] = parent_conn.recv()
            except (EOFError, OSError):
                pass
        rt = threading.Thread(target=_recv, daemon=True)
        rt.start()
        rt.join(timeout + 15)  # poll(timeout) + margin to drain the payload
        status, payload = box["r"] if "r" in box else (None, None)

        if status == "ok":
            output, updated = payload
            namespace.update(updated)  # persist child's state into parent namespace
        elif status == "err":
            output = f"ERROR: {payload}"
        else:
            output = (f"ERROR: TimeoutError: code exceeded {timeout}s wall-clock limit and "
                      f"was killed. Use a subsample or a cheaper computation.")

        # Always reap the child so it can never linger as a 0%-CPU orphan. Killing
        # it closes the pipe write end, unblocking a still-stuck _recv thread.
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
        else:
            proc.join(1)
        try:
            parent_conn.close()
        except Exception:
            pass

    output = output if output.strip() else "(no output)"

    state_lines = _namespace_summary(namespace)
    if state_lines:
        output += "\n\n[REPL state — variables available for next code block:]\n"
        output += "\n".join(state_lines)

    return output
