"""Launch a Tokyo run in a detached server-side session from PyCharm."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = PROJECT_ROOT / "model" / "train.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-dir", type=Path, required=True,
        help="New server-side directory for launcher metadata and console output; keep it separate from trainer --output-dir",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for model/train.py after --")
    parsed = parser.parse_args()
    if os.name != "posix":
        parser.error("Detached server launch requires the Linux remote interpreter")
    if not parsed.args or parsed.args[0] != "--":
        parser.error("Separate trainer arguments with --")

    trainer_args = parsed.args[1:]
    if "--output-dir" not in trainer_args:
        parser.error("Trainer arguments must include --output-dir")
    output_index = trainer_args.index("--output-dir")
    if output_index + 1 >= len(trainer_args):
        parser.error("--output-dir requires a path")
    job_dir = parsed.job_dir.resolve()
    output_dir = Path(trainer_args[output_index + 1]).resolve()
    if output_dir == job_dir or job_dir in output_dir.parents or output_dir in job_dir.parents:
        parser.error("Keep launcher job directory and trainer output directory separate")
    if job_dir.exists():
        parser.error(f"Job directory already exists: {job_dir}")
    if output_dir.exists():
        parser.error(f"Trainer output directory already exists: {output_dir}")
    if not RUN_SCRIPT.is_file():
        parser.error(f"Trainer script is missing: {RUN_SCRIPT}")

    job_dir.mkdir(parents=True)
    command = [sys.executable, "-u", str(RUN_SCRIPT), *trainer_args]
    log_path = job_dir / "console.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
            env=os.environ.copy(),
        )
    (job_dir / "launcher.json").write_text(
        json.dumps(
            {
                "pid": process.pid,
                "cwd": str(PROJECT_ROOT),
                "command": command,
                "trainer_output_dir": str(output_dir),
                "console_log": str(log_path),
                "report_name": "train_report.json",
                "detached_session": True,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Detached Tokyo run started: pid={process.pid}")
    print(f"Console log: {log_path}")
    print(f"Trainer output: {output_dir}")


if __name__ == "__main__":
    main()
