"""Bootstrap the isolated MLX worker from a checkout or installed wheel."""
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mozhiflow.pipeline import worker_main
    raise SystemExit(worker_main())
