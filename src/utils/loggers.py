from datetime import datetime
import sys
from pathlib import Path


class _Tee:
    """Redirect stdout to both the terminal and a log file."""

    def __init__(self, log_path: Path):
        self._terminal = sys.stdout
        self._logs = [open(log_path, "a")]

    def write(self, msg):
        self._terminal.write(msg)
        for line in msg.splitlines(keepends=True):
            if line.strip():
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for _log in self._logs:
                     _log.write(f"[{ts}] {line}")
            else:
                for _log in self._logs:
                    _log.write(line)
        for _log in self._logs:
            _log.flush()

    def flush(self):
        self._terminal.flush()
        for _log in self._logs:
            _log.flush()

    def add_log(self, log_path: Path):
        new_log = open(log_path, 'a')
        self._logs.append(new_log)

    def remove_log(self, log_path: Path):
        for _log in self._logs:
            if _log.name == str(log_path):
                _log.close()
                self._logs.remove(_log)
                break
        else:
            print(f"Warning: log file {log_path} not found in _Tee logs.")


def setup_logging(log_file: Path | str, prefix: str = ""):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(sys.stdout, _Tee):
        sys.stdout.add_log(log_file)
    else:
        sys.stdout = _Tee(log_file)
    if prefix:
        print(prefix)

def remove_logging(log_file: Path | str):
    log_file = Path(log_file)
    if isinstance(sys.stdout, _Tee):
        sys.stdout.remove_log(log_file)
    else:
        print("Warning: sys.stdout is not a _Tee instance. Cannot remove log.")
