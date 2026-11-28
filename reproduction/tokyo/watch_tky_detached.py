"""Show the progress of an existing detached Tokyo run in PyCharm."""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time


def trainer_is_running(pid, expected_args):
    process_dir = Path("/proc") / str(pid)
    try:
        # The command line is empty for an exited process awaiting cleanup.
        args = (process_dir / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return False
    # Compare the full command so a reused PID cannot attach to a different run.
    return [arg.decode("utf-8", errors="surrogateescape") for arg in args] == expected_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True, help="Existing launcher directory")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Select the Linux SSH interpreter in PyCharm")

    manifest = args.job_dir / "launcher.json"
    try:
        job = json.loads(manifest.read_text(encoding="utf-8"))
        pid = job["pid"]
        command = job["command"]
        log_path = Path(job["console_log"])
        output_dir = Path(job["trainer_output_dir"])
        report_name = job.get("report_name")
        if report_name is None:
            # Existing launch manifests belong to the archived combined runner.
            mode = command[command.index("--mode") + 1]
            report_name = {"train": "test_report.json", "smoke": "smoke_report.json"}[mode]
        log = log_path.open(encoding="utf-8", errors="replace")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.error(f"Cannot open existing job {manifest}: {error}")

    print(f"Watching {output_dir.name} (PID {pid})", flush=True)
    with log:
        for line in deque(log, maxlen=25):
            print(line, end="", flush=True)
        while trainer_is_running(pid, command):
            for line in log:
                print(line, end="", flush=True)
            time.sleep(2)
        for line in log:
            print(line, end="", flush=True)

    report = output_dir / report_name
    try:
        json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"Trainer exited without a readable report at {report}: {error}", flush=True)
        return 1
    print(f"Trainer exited and wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
