"""Bundle current evaluation source/assets for Colab, excluding private data.

Run with any Python 3 interpreter: python experiments/package_eval_source.py
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memact.provenance import code_revision, file_sha256


def package_source(output):
    paths = [ROOT / "requirements.txt", ROOT / "LICENSE"]
    for directory in ("app", "memact", "experiments"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".npy", ".model", ".txt", ".task"}:
                paths.append(path)
    manifest = {
        "purpose": "Memact evaluation source; no user recordings, profiles, or VSR checkpoint",
        "code": code_revision(ROOT),
        "files": {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(paths)},
    }
    output = Path(output)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT))
        archive.writestr("evaluation_source_manifest.json", json.dumps(manifest, indent=2))
    return output.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "memact_eval_source.zip")
    print(package_source(parser.parse_args().output))
