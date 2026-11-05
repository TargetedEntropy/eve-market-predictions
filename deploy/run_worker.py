"""GPU-only batch wrapper, atomic low-cardinality Prometheus textfiles."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    directory = Path(os.environ.get("JITA_ARTIFACT_DIR", "/artifacts"))
    start = time.time()
    code = 1
    try:
        subprocess.run([sys.executable, str(Path(__file__).with_name("gpu_preflight.py"))],
                       timeout=60, check=True)
        code = subprocess.run([sys.executable, "-m", "src.jita.cli", "batch",
                               "--output-dir", str(directory), "--require-cuda"],
                              timeout=2700, check=False).returncode
    except Exception as exc:
        # No source URL, credentials, response bodies or exception strings in logs.
        print(json.dumps({"event": "batch_failed", "error_type": type(exc).__name__}), flush=True)
    finally:
        finished = time.time()
        metrics = (f"jita_batch_last_attempt_unixtime {finished}\n"
                   f"jita_batch_last_duration_seconds {finished - start}\n"
                   f"jita_batch_last_success {int(code == 0)}\n")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if code == 0:
                temporary = directory / "last-success.prom.tmp"
                temporary.write_text(f"jita_batch_last_success_unixtime {finished}\n")
                os.replace(temporary, directory / "last-success.prom")
            temporary = directory / "batch.prom.tmp"
            temporary.write_text(metrics)
            os.replace(temporary, directory / "batch.prom")
        except OSError:
            code = 1
            print('{"event":"metrics_write_failed"}', flush=True)
        print(json.dumps({"event": "batch_finished", "success": code == 0,
                          "duration_seconds": round(finished - start, 3)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
