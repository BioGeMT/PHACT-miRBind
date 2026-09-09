#!/usr/bin/env python3
"""Run the fixed 18-condition/seed queue serially on node 4, stopping on failure."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fcntl
import json
import os
import subprocess
import time

from run_condition import ROOT, sha256
from summarize_runs import summarize


def write_state(state):
    state["updated_unix"] = time.time()
    temporary = ROOT / "suite_state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(ROOT / "suite_state.json")


def verify_frozen_inputs():
    hashes = json.loads((ROOT / "frozen_inputs.json").read_text())
    for name, expected in hashes.items():
        if sha256(name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def main():
    lock = (ROOT / "suite.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / "suite.pid").write_text(str(os.getpid()) + "\n")
    verify_frozen_inputs()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    (ROOT / "logs").mkdir(exist_ok=True)
    state = dict(status="running", pid=os.getpid(), completed=[], total=18)
    try:
        for seed in protocol["seeds"]:
            for condition in protocol["conditions"]:
                verify_frozen_inputs()
                name = f"seed_{seed}/{condition}"
                output = ROOT / "runs" / name
                if (output / "result.json").exists():
                    summarize()
                    state["completed"].append(name)
                    continue
                state.update(current=name, child_pid=None)
                write_state(state)
                with (ROOT / "logs" / f"seed_{seed}_{condition}.log").open("a") as log:
                    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("run_condition.py")),
                                                "--condition", condition, "--seed", str(seed),
                                                "--output", str(output)], stdout=log, stderr=subprocess.STDOUT,
                                               env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
                    state["child_pid"] = process.pid
                    write_state(state)
                    if process.wait() != 0:
                        raise RuntimeError(f"Run failed: {name}; see its log")
                summarize()
                state["completed"].append(name)
                write_state(state)
        if summarize() != 18:
            raise ValueError("Queue ended without all 18 verified results")
        state.update(status="complete", current=None, child_pid=None)
        write_state(state)
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_state(state)
        raise


if __name__ == "__main__":
    main()
