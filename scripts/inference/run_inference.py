"""DEPRECATED — use `jannus predict` instead.

This script was the external-site entry point through v1.40. It is retained
only so existing site scripts and documentation keep working, and it now
delegates to the supported CLI.

Why it was replaced
-------------------
The v1.40 implementation carried three defects that specifically hurt external
sites, all of which are fixed in `jannus predict`:

* It defaulted to the BrainMetShare-internal sequence naming (`bravo.nii.gz`),
  so a site with conventionally-named data was told "No patient directories
  found" and given no indication why.
* Its missing-file error path referenced an undefined `SEQUENCES` name, so the
  first problem a new site hit raised a `NameError` from inside the error
  handler instead of an actionable message.
* It wrote no provenance: nothing recorded which checkpoints, threshold, code
  revision or seeds produced a given mask, which makes returned results
  unverifiable.

Migration
---------
    OLD: python scripts/inference/run_inference.py --input DATA --output OUT
    NEW: jannus predict --input DATA --output OUT

`jannus predict` additionally runs dataset QC first, isolates per-case
failures, and refuses to run with unverified weights.
"""

from __future__ import annotations

import sys
import warnings


def main() -> int:
    warnings.warn(
        "scripts/inference/run_inference.py is deprecated and will be removed in "
        "v1.6. Use `jannus predict` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "NOTE: this script is deprecated; delegating to `jannus predict`.\n"
        "      Update your scripts to call `jannus predict` directly.\n",
        file=sys.stderr,
    )

    from jannus.cli.main import main as cli_main

    # argv is forwarded unchanged: --input/--output/--config/--threshold all
    # keep their v1.40 meanings under the new command.
    return cli_main(["predict", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
