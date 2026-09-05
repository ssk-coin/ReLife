"""
wild_life/main.py — Main REPL loop for the Wild Life interpreter.

Corresponds to Life.c (and Info.c) from the original C source.
Implements the Read-Evaluate-Print loop for the LIFE language.

This implements the "nested constraint session" model:
  - Each successful query at depth D opens depth D+1
  - Blank line at depth D>0 pops one level (prints *** No + parent bindings)
  - Period '.' exits all levels silently back to depth 0
  - Semicolon ';' backtracks for another solution at current depth
  - Variables persist across depth levels (WAM trail)
"""

from __future__ import annotations

import sys
import os
import argparse
import traceback
import io
from collections import namedtuple
from typing import Optional

# ---------------------------------------------------------------------------
# Title / version banner
# ---------------------------------------------------------------------------

_VERSION = "1.02"
_BANNER = (
    f"Wild_Life Interpreter Version {_VERSION} (Python port)\n"
    "Copyright (C) 1991-93 DEC Paris Research Laboratory\n"
    "Copyright (C) 1994-1995 Intelligent Software Group, SFU\n"
)


def title(quiet: bool = False) -> None:
    """Print the startup banner (mirrors C's title() in Info.c)."""
    if not quiet:
        sys.stdout.write(_BANNER)
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Nested constraint session model helpers
# ---------------------------------------------------------------------------

# A frame on the depth stack records:
#   pre_mark     - trail mark BEFORE proving this query (for undo on pop)
#   bindings_str - formatted variable bindings string (e.g. "A = 1, B = 2.")
#   cs_before    - choice_stack BEFORE proving (for restoring on pop)
Frame = namedtuple('Frame', ['pre_mark', 'bindings_str', 'cs_before'])


def _prompt(depth: int) -> str:
    """Return the prompt string for the given depth level.

    depth=0  -> '> '
    depth=1  -> '--1> '
    depth=2  -> '----2> '
    """
    if depth == 0:
        return "> "
    return "--" * depth + str(depth) + "> "


def _format_bindings(var_tree: dict, engine) -> str:
    """Format all named variables as a single-line string like 'A = 1, B = 2.'

    Returns empty string if there are no named variables (or all anonymous).
    """
    from wild_life.print_term import print_variables

    if not var_tree:
        return ""

    buf = io.StringIO()
    had_vars = print_variables(var_tree, outfile=buf, wl=engine.wl)
    result = buf.getvalue()
    # print_variables now writes "." at end without newline
    if not had_vars:
        return ""
    return result


# ---------------------------------------------------------------------------
# Top-level REPL
# ---------------------------------------------------------------------------

def run_repl(
    *,
    quiet: bool = False,
    load_files: Optional[list] = None,
    interactive: bool = True,
) -> int:
    """
    Main Read-Evaluate-Print loop.

    Parameters
    ----------
    quiet       : suppress banner (prompts are always printed)
    load_files  : extra .lf files to load from command line
    interactive : True when stdin is a terminal
    """
    # Deferred imports to avoid circular-import issues at module level
    from wild_life.runtime import WL
    from wild_life.built_ins import register_all
    from wild_life.inference import Engine
    from wild_life.parser_ import parse_string
    from wild_life.unification import HaltException, AbortException
    from wild_life.data_structures import QUERY, FACT, ERROR

    # ---- Initialise runtime (modules, types, operators) --------------------
    if not WL._initialized:
        WL.initialize()

    # ---- Register built-ins ------------------------------------------------
    register_all(WL)

    # ---- Create the inference engine ---------------------------------------
    engine = Engine(WL)
    engine.noisy = False   # REPL handles all output itself

    # ---- Print banner -------------------------------------------------------
    title(quiet)

    # ---- Load system initialisation file (.set_up) --------------------------
    # Note: built_ins.lf uses complex module syntax not yet supported by the
    # parser. All Python built-ins are registered via built_ins.py, so we only
    # load a .set_up file if one exists.
    setup_candidates = [
        os.path.join(os.path.dirname(__file__), ".set_up"),
        "/usr/local/lib/life/Source/.set_up",
    ]
    for candidate in setup_candidates:
        if os.path.isfile(candidate):
            try:
                engine.load_file(candidate)
            except Exception as exc:
                sys.stderr.write(f"Warning: could not load {candidate}: {exc}\n")
            break

    # ---- Load files given on command line -----------------------------------
    for path in (load_files or []):
        try:
            engine.load_file(path)
        except HaltException:
            return 0
        except Exception as exc:
            sys.stderr.write(f"Error loading {path}: {exc}\n")

    # ---- Nested constraint session state -----------------------------------
    frame_stack: list[Frame] = []
    depth = 0

    def _pop_frame() -> str:
        """Pop one depth level: undo trail, restore choice_stack, return parent bindings."""
        nonlocal depth
        if not frame_stack:
            return ""
        frame = frame_stack.pop()
        engine.trail.undo_to(frame.pre_mark)
        engine.choice_stack = frame.cs_before
        engine.goal_stack = None
        depth -= 1
        # Return parent frame's bindings (if any)
        return frame_stack[-1].bindings_str if frame_stack else ""

    def _pop_all():
        """Pop all frames back to depth 0."""
        nonlocal depth
        if frame_stack:
            root_frame = frame_stack[0]
            engine.trail.undo_to(root_frame.pre_mark)
            engine.choice_stack = root_frame.cs_before
            engine.goal_stack = None
            frame_stack.clear()
        depth = 0

    def _write_prompt(d: int):
        sys.stdout.write(_prompt(d))
        sys.stdout.flush()

    # ---- Main REPL ---------------------------------------------------------
    # No initial prompt — the first output comes from the load or the first query.
    exit_code = 0

    while True:
        try:
            try:
                line = input()
            except EOFError:
                break

            line_stripped = line.strip()

            # ---- Blank line or comment ------------------------------------
            if not line_stripped or line_stripped.startswith('%'):
                if depth == 0:
                    # At top level, blank just re-shows the prompt
                    _write_prompt(0)
                else:
                    # Pop one depth level
                    parent_bindings = _pop_frame()
                    sys.stdout.write("\n*** No\n")
                    if parent_bindings:
                        sys.stdout.write(parent_bindings + "\n")
                    _write_prompt(depth)
                continue

            # ---- Period: exit ALL depth levels ----------------------------
            if line_stripped == '.':
                _pop_all()
                _write_prompt(0)
                continue

            # ---- Semicolon: backtrack for another solution ----------------
            if line_stripped == ';':
                if depth == 0 or not frame_stack:
                    _write_prompt(depth)
                    continue
                if not engine.choice_stack:
                    # No more alternatives — pop one level
                    parent_bindings = _pop_frame()
                    sys.stdout.write("\n*** No\n")
                    if parent_bindings:
                        sys.stdout.write(parent_bindings + "\n")
                    _write_prompt(depth)
                    continue
                # Backtrack and find next solution (do NOT undo to pre_mark —
                # engine.backtrack() handles its own trail undo)
                saved_noisy = engine.noisy
                engine.noisy = False
                try:
                    engine.backtrack()
                    success = engine.run()
                finally:
                    engine.noisy = saved_noisy

                if success:
                    var_tree = getattr(engine, '_last_var_tree', {})
                    bindings_str = _format_bindings(var_tree, engine)
                    # Update current frame's stored bindings
                    if frame_stack:
                        frame_stack[-1] = frame_stack[-1]._replace(
                            bindings_str=bindings_str)
                    sys.stdout.write("\n*** Yes\n")
                    if bindings_str:
                        sys.stdout.write(bindings_str + "\n")
                    _write_prompt(depth)
                else:
                    # Exhausted alternatives — pop one level
                    parent_bindings = _pop_frame()
                    sys.stdout.write("\n*** No\n")
                    if parent_bindings:
                        sys.stdout.write(parent_bindings + "\n")
                    _write_prompt(depth)
                continue

            # ---- Parse the input ------------------------------------------
            try:
                term, sort, var_tree = parse_string(line_stripped)
            except Exception as exc:
                sys.stderr.write(f"Parse error: {exc}\n")
                _write_prompt(depth)
                continue

            if term is None:
                _write_prompt(depth)
                continue

            # EOF sentinel from parser
            if hasattr(term, 'type') and term.type is not None and term.type is WL.eof:
                break

            if sort == ERROR:
                sys.stderr.write(f"*** Syntax error: {line_stripped!r}\n")
                sys.stderr.write(
                    "    (Hint: facts end with '.' and queries end with '?')\n")
                _write_prompt(depth)
                continue

            # ---- Query: ?- Goal -------------------------------------------
            if sort == QUERY:
                engine._last_var_tree = var_tree
                cs_before = engine.choice_stack
                pre_mark = engine.trail.mark()

                saved_noisy = engine.noisy
                engine.noisy = False
                try:
                    success = engine.prove(term)
                except HaltException:
                    return 0
                except AbortException:
                    engine.goal_stack = None
                    engine.trail.undo_to(pre_mark)
                    engine.choice_stack = cs_before
                    sys.stdout.write("\n")
                    _write_prompt(depth)
                    continue
                except KeyboardInterrupt:
                    engine.trail.undo_to(pre_mark)
                    engine.choice_stack = cs_before
                    engine.goal_stack = None
                    sys.stdout.write("\n")
                    _write_prompt(depth)
                    continue
                except Exception as exc:
                    engine.trail.undo_to(pre_mark)
                    engine.choice_stack = cs_before
                    engine.goal_stack = None
                    sys.stderr.write(f"Error: {exc}\n")
                    _write_prompt(depth)
                    continue
                finally:
                    engine.noisy = saved_noisy

                if success:
                    bindings_str = _format_bindings(var_tree, engine)
                    if bindings_str:
                        # Query has variable bindings → enter a new depth level
                        frame_stack.append(Frame(pre_mark, bindings_str, cs_before))
                        depth += 1
                        sys.stdout.write("\n*** Yes\n")
                        sys.stdout.write(bindings_str + "\n")
                    else:
                        # No variables → stay at current depth, undo trail
                        engine.trail.undo_to(pre_mark)
                        engine.choice_stack = cs_before
                        engine.goal_stack = None
                        sys.stdout.write("\n*** Yes\n")
                    _write_prompt(depth)
                else:
                    # Failure: undo failed attempt
                    engine.trail.undo_to(pre_mark)
                    engine.choice_stack = cs_before
                    engine.goal_stack = None
                    sys.stdout.write("\n*** No\n")
                    if depth > 0:
                        # In a nested session: pop one frame on failure
                        parent_bindings = _pop_frame()
                        if parent_bindings:
                            sys.stdout.write(parent_bindings + "\n")
                    _write_prompt(depth)

            # ---- Fact / rule: assert into database ------------------------
            elif sort == FACT:
                try:
                    engine.assert_first = False
                    engine.assert_clause(term)
                    sys.stdout.write("\n*** Yes\n")
                except HaltException:
                    return 0
                except Exception as exc:
                    sys.stderr.write(f"Assert error: {exc}\n")
                    sys.stdout.write("\n*** No\n")
                _write_prompt(depth)

            else:
                # Unknown sort — treat as fact
                try:
                    engine.assert_clause(term)
                    sys.stdout.write("\n*** Yes\n")
                except Exception as exc:
                    sys.stderr.write(f"Error: {exc}\n")
                    sys.stdout.write("\n*** No\n")
                _write_prompt(depth)

        except HaltException:
            break
        except KeyboardInterrupt:
            sys.stdout.write("\nType Ctrl-D to exit.\n")
            _write_prompt(depth)
            continue
        except Exception as exc:
            sys.stderr.write(f"Unexpected error: {exc}\n")
            if getattr(engine, 'verbose', False):
                traceback.print_exc()
            continue

    return exit_code


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Parse command-line arguments and start the REPL."""
    parser = argparse.ArgumentParser(
        prog="wild_life",
        description="Wild Life LIFE language interpreter (Python port)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress banner and informational output",
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help=".lf files to load before entering the REPL",
    )
    args = parser.parse_args(argv)

    interactive = sys.stdin.isatty()

    return run_repl(
        quiet=args.quiet,
        load_files=args.files,
        interactive=interactive,
    )


if __name__ == "__main__":
    sys.exit(main())
