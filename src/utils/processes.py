"""Signal handling and process cleanup for experiment entry points."""

import multiprocessing
import os
import signal
import time


def _exit_on_sigterm(signum, _frame):
    raise SystemExit(128 + signum)


def handle_sigterm():
    """Let normal exception cleanup run when a process receives SIGTERM."""
    signal.signal(signal.SIGTERM, _exit_on_sigterm)


def terminate_process_group(process, timeout=5):
    """Stop a child-owned process group, including descendants of its leader."""
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader so a zombie does not keep the group alive.
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def terminate_active_children(timeout=5):
    """Stop multiprocessing workers without waiting for their queued work."""
    deadline = time.monotonic() + timeout
    quiet_since = None
    while time.monotonic() < deadline:
        children = multiprocessing.active_children()
        if not children:
            if quiet_since is None:
                quiet_since = time.monotonic()
            if time.monotonic() - quiet_since >= 0.1:
                return
            time.sleep(0.02)
            continue

        quiet_since = None
        for child in children:
            child.terminate()
        for child in children:
            child.join(timeout=0.05)

    children = multiprocessing.active_children()
    for child in children:
        child.kill()
    for child in children:
        child.join(timeout=1)
