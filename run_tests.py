#!/usr/bin/env python3
"""
run_tests.py — Test runner for the Wild Life Python interpreter.

Replicates the behaviour of the original C-version `check` / `check_all`
shell scripts (tests_original/check, tests_original/check_all).

Usage
-----
  # Run all tests (from the repo root):
  python run_tests.py

  # Run specific tests:
  python run_tests.py append alias arith1

  # Quiet mode (only print failures):
  python run_tests.py -q

  # Show diffs for failures:
  python run_tests.py -d

  # Write a check.log summary file:
  python run_tests.py --log check.log

  # Run from a different directory:
  python run_tests.py --test-dir tests_original

Options
-------
  tests               Test base-names to run (without .lf). Defaults to all.
  -q, --quiet         Only print failures, not every test name.
  -d, --diff          Show unified diffs for failing tests.
  -j N, --jobs N      Run N tests in parallel (default: 1).
  --test-dir DIR      Directory containing the test suite (default: tests_original).
  --log FILE          Write a summary log to FILE (like check.log).
  --timeout SECS      Per-test timeout in seconds (default: 30).
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Patterns to strip from interpreter output (mirrors the original egrep -v)
# ---------------------------------------------------------------------------
_STRIP_STDOUT = re.compile(
    r"Loading|already loaded|Version|customizing|Copyright|Garbage|Exiting|X interface"
)

# Strip CPU-time lines from stderr (mirrors: sed "s/.......s cpu (.*)//" )
_STRIP_STDERR = re.compile(r".{7}s cpu \(.*?\)")


def _filter_stdout(text: str) -> str:
    """Remove banner / loading lines from interpreter stdout."""
    lines = [ln for ln in text.splitlines(keepends=True)
             if not _STRIP_STDOUT.search(ln)]
    return "".join(lines)


def _filter_stderr(text: str) -> str:
    """Remove CPU-time info from interpreter stderr."""
    return _STRIP_STDERR.sub("", text)


# ---------------------------------------------------------------------------
# Single test runner
# ---------------------------------------------------------------------------

def run_one(
    name: str,
    test_dir: Path,
    timeout: int,
) -> Tuple[str, bool, bool, str, str]:
    """
    Run one test.

    Returns
    -------
    (name, out_ok, err_ok, out_diff, err_diff)
    """
    lf_path   = test_dir / "LF"   / f"{name}.lf"
    in_path   = test_dir / "IN"   / f"{name}.in"
    refout    = test_dir / "REFOUT"  / f"{name}.refout"
    referr    = test_dir / "REFERR"  / f"{name}.referr"
    out_path  = test_dir / "OUT"     / f"{name}.out"
    err_path  = test_dir / "ERR"     / f"{name}.err"
    refdiff   = test_dir / "REFDIFF" / f"{name}.refdiff"
    errdiff   = test_dir / "ERRDIFF" / f"{name}.errdiff"

    # Compose stdin for the interpreter:
    #   load("LF/name.lf")?
    #   <contents of IN/name.in>
    #   halt?
    load_cmd = f'load("LF/{name}.lf")?\n'
    in_text  = in_path.read_text(errors="replace") if in_path.exists() else ""
    # The original check script adds an echo "" (blank line) after load.
    # No extra blank line before halt? (mirrors: echo ""; cat in; echo "halt?").
    stdin_data = (load_cmd + "\n" + in_text + "halt?\n").encode()

    # Run the Python interpreter as a subprocess (no -q: prompts appear in refout)
    # PYTHONPATH must include the repo root so 'wild_life' package is importable.
    # cwd is test_dir so relative paths like LF/name.lf work from there.
    import os as _os
    env = dict(_os.environ)
    env['PYTHONPATH'] = str(test_dir.parent) + _os.pathsep + env.get('PYTHONPATH', '')
    cmd = [sys.executable, "-m", "wild_life"]
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            timeout=timeout,
            cwd=test_dir,          # tests_original dir so 'LF/name.lf' resolves
            env=env,
        )
    except subprocess.TimeoutExpired:
        out_diff = f"TIMEOUT after {timeout}s\n"
        err_diff = ""
        return name, False, True, out_diff, err_diff
    except Exception as exc:
        return name, False, True, f"ERROR: {exc}\n", ""

    # Filter outputs
    stdout_text = _filter_stdout(proc.stdout.decode(errors="replace"))
    stderr_text = _filter_stderr(proc.stderr.decode(errors="replace"))

    # Write OUT / ERR files (mirrors the original scripts)
    out_path.parent.mkdir(exist_ok=True)
    err_path.parent.mkdir(exist_ok=True)
    out_path.write_text(stdout_text)
    err_path.write_text(stderr_text)

    # Compare against reference
    # When a reference file does not exist, treat the test as passing for that
    # channel (mirrors the original csh check script: diff against /dev/null
    # gives empty output → no diff → pass).
    ref_out_exists = refout.exists()
    ref_err_exists = referr.exists()
    ref_out = refout.read_text(errors="replace") if ref_out_exists else ""
    ref_err = referr.read_text(errors="replace") if ref_err_exists else ""

    def _diff(a: str, b: str, from_file: str, to_file: str) -> str:
        a_lines = a.splitlines(keepends=True)
        b_lines = b.splitlines(keepends=True)
        return "".join(difflib.unified_diff(a_lines, b_lines,
                                            fromfile=from_file,
                                            tofile=to_file))

    if ref_out_exists:
        out_diff_text = _diff(stdout_text, ref_out,
                              str(out_path.relative_to(test_dir)),
                              str(refout.relative_to(test_dir)))
    else:
        out_diff_text = ""  # no REFOUT → treat stdout as passing

    if ref_err_exists:
        err_diff_text = _diff(stderr_text, ref_err,
                              str(err_path.relative_to(test_dir)),
                              str(referr.relative_to(test_dir)))
    else:
        err_diff_text = ""  # no REFERR → treat stderr as passing

    out_ok = not out_diff_text
    err_ok = not err_diff_text

    # Write diff files (only if there are differences, mirror original)
    refdiff.parent.mkdir(exist_ok=True)
    errdiff.parent.mkdir(exist_ok=True)
    if out_diff_text:
        refdiff.write_text(out_diff_text)
    elif refdiff.exists():
        refdiff.unlink()

    if err_diff_text:
        errdiff.write_text(err_diff_text)
    elif errdiff.exists():
        errdiff.unlink()

    return name, out_ok, err_ok, out_diff_text, err_diff_text


# ---------------------------------------------------------------------------
# Pretty status line
# ---------------------------------------------------------------------------

def _status(name: str, out_ok: bool, err_ok: bool) -> str:
    if out_ok and err_ok:
        return f"  {name:40s}  OK"
    parts = []
    if not out_ok:
        parts.append("output mismatch")
    if not err_ok:
        parts.append("stderr mismatch")
    return f"  {name:40s}  FAIL  ({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Wild Life Python interpreter test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tests", nargs="*", metavar="TEST",
        help="Base-names of tests to run (without .lf). Default: all.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Only print failures.",
    )
    parser.add_argument(
        "-d", "--diff", action="store_true",
        help="Show unified diffs for failing tests.",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, metavar="N",
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--test-dir", default="tests_original", metavar="DIR",
        help="Test suite directory (default: tests_original).",
    )
    parser.add_argument(
        "--log", default=None, metavar="FILE",
        help="Write summary log to FILE.",
    )
    parser.add_argument(
        "--timeout", type=int, default=30, metavar="SECS",
        help="Per-test timeout in seconds (default: 30).",
    )
    args = parser.parse_args(argv)

    # Locate test directory
    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        # Resolve relative to the directory containing this script
        test_dir = Path(__file__).parent / test_dir
    test_dir = test_dir.resolve()

    if not test_dir.is_dir():
        print(f"Error: test directory not found: {test_dir}", file=sys.stderr)
        return 1

    lf_dir = test_dir / "LF"
    if not lf_dir.is_dir():
        print(f"Error: LF/ subdirectory not found in {test_dir}", file=sys.stderr)
        return 1

    # Decide which tests to run
    if args.tests:
        # Strip .lf suffix if user supplied it
        names = [t[:-3] if t.endswith(".lf") else t for t in args.tests]
    else:
        names = sorted(p.stem for p in lf_dir.glob("*.lf"))

    if not names:
        print("No tests found.", file=sys.stderr)
        return 1

    # Header
    header = (
        f"Wild Life Python test runner\n"
        f"Interpreter : {sys.executable} -m wild_life\n"
        f"Test dir    : {test_dir}\n"
        f"Date        : {time.strftime('%c')}\n"
        f"Tests       : {len(names)}\n"
    )
    print(header)

    log_lines: List[str] = [header]

    # Run tests (optionally in parallel)
    total = len(names)
    passed = 0
    failed_out: List[str] = []
    failed_err: List[str] = []

    results: dict[str, tuple] = {}

    def _run(name):
        return run_one(name, test_dir, args.timeout)

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_run, n): n for n in names}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
    else:
        for n in names:
            results[n] = _run(n)

    # Print results in sorted order
    for name in names:
        n, out_ok, err_ok, out_diff, err_diff = results[name]
        line = _status(name, out_ok, err_ok)

        if out_ok and err_ok:
            passed += 1
            if not args.quiet:
                print(line)
        else:
            if not out_ok:
                failed_out.append(name)
            if not err_ok:
                failed_err.append(name)
            print(line)
            if args.diff:
                if out_diff:
                    print("    -- stdout diff --")
                    for dl in out_diff.splitlines():
                        print("    " + dl)
                if err_diff:
                    print("    -- stderr diff --")
                    for dl in err_diff.splitlines():
                        print("    " + dl)

        log_lines.append(line + "\n")

    # Summary
    all_failed = sorted(set(failed_out) | set(failed_err))
    summary = (
        f"\n{'='*60}\n"
        f"Results: {passed}/{total} passed"
        + (f", {len(all_failed)} failed" if all_failed else "")
        + f"\n"
    )
    if all_failed:
        summary += "Failed tests:\n"
        for f in all_failed:
            tags = []
            if f in failed_out:
                tags.append("stdout")
            if f in failed_err:
                tags.append("stderr")
            summary += f"  {f} ({', '.join(tags)})\n"

    print(summary)
    log_lines.append(summary)

    # Write log file
    if args.log:
        log_path = Path(args.log)
        log_path.write_text("".join(log_lines))
        print(f"Log written to: {log_path}")

    return 0 if not all_failed else 1


if __name__ == "__main__":
    sys.exit(main())
