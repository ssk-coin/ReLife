"""
wild_life/main.py — Main REPL loop for the Wild Life interpreter.

Corresponds to Life.c (and Info.c) from the original C source.
Implements the Read-Evaluate-Print loop for the LIFE language.
"""

from __future__ import annotations

import sys
import os
import argparse
import traceback
from typing import Optional

# ---------------------------------------------------------------------------
# Title / version banner
# ---------------------------------------------------------------------------

_VERSION = "1.02"
_BANNER = (
    f"Wild_Life Interpreter Version {_VERSION} (Python port)\n"
    "Copyright (C) 1991-93 DEC Paris Research Laboratory\n"
    "Extensions, Copyright (C) 1994-1995 Intelligent Software Group, SFU\n"
    "Python port — 2024\n"
)

_PROMPT = "?- "


def title(quiet: bool = False) -> None:
    """Print the startup banner (mirrors C's title() in Info.c)."""
    if not quiet:
        print(_BANNER)


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
    quiet       : suppress banner and prompts
    load_files  : extra .lf files to load from command line
    interactive : True when stdin is a terminal
    """
    # Deferred imports to avoid circular-import issues at module level
    from wild_life.runtime import WL
    from wild_life.built_ins import register_all
    from wild_life.inference import Engine
    from wild_life.parser_ import parse_term_string, Parser
    from wild_life.tokenizer import tokenizer_from_string, tokenizer_from_file
    from wild_life.data_structures import GoalType
    from wild_life.unification import HaltException, AbortException

    # ---- Initialise runtime (modules, types, operators) --------------------
    if not WL._initialized:
        WL.initialize()

    # ---- Register built-ins ------------------------------------------------
    register_all(WL)

    # ---- Create the inference engine ---------------------------------------
    engine = Engine(WL)
    engine.noisy = not quiet

    # ---- Print banner -------------------------------------------------------
    title(quiet)

    # ---- Load system initialisation file (.set_up / built_ins.lf) ----------
    setup_loaded = False
    setup_candidates = [
        os.path.join(os.path.dirname(__file__), ".set_up"),
        os.path.join(os.path.dirname(__file__), "built_ins.lf"),
        "/usr/local/lib/life/Source/.set_up",
    ]
    for candidate in setup_candidates:
        if os.path.isfile(candidate):
            if not quiet:
                print(f"Loading {candidate} ...")
            try:
                engine.load_file(candidate)
                setup_loaded = True
            except Exception as exc:
                print(f"Warning: could not load {candidate}: {exc}", file=sys.stderr)
            break

    # ---- Load files given on command line -----------------------------------
    for path in (load_files or []):
        if not quiet:
            print(f"Loading {path} ...")
        try:
            engine.load_file(path)
        except HaltException:
            return 0
        except Exception as exc:
            print(f"Error loading {path}: {exc}", file=sys.stderr)

    # ---- Main REPL ---------------------------------------------------------
    exit_code = 0
    while True:
        try:
            # Show prompt on interactive terminals
            if interactive and not quiet:
                print(_PROMPT, end="", flush=True)

            # Read a line (or EOF)
            try:
                line = input()
            except EOFError:
                # Ctrl-D / end of pipe — exit cleanly
                if not quiet:
                    print()
                break

            line = line.strip()
            if not line or line.startswith("%"):
                # blank line or comment — skip
                continue

            # NOTE: '.'' はパーサが FACT の終端として必要なので削除しない。
            # '?' はクエリの終端。どちらも parse_string に渡す。

            # ---- Parse the input -------------------------------------------
            try:
                from wild_life.parser_ import parse_string
                term, sort, var_tree = parse_string(line)
            except Exception as exc:
                print(f"Parse error: {exc}", file=sys.stderr)
                continue

            if term is None:
                continue

            # eof sentinel from parser
            if term is not None and term.type is not None and term.type is WL.eof:
                break

            # ---- Dispatch on QUERY vs FACT ---------------------------------
            from wild_life.data_structures import QUERY, FACT, ERROR

            if sort == ERROR:
                # Parse error was already reported
                continue

            if sort == QUERY:
                # It's a query: ?- Goal
                # Store named variable map so _print_bindings can use it
                engine._last_var_tree = var_tree
                engine.var_occurred = _has_variables(term)
                try:
                    success = engine.prove(term)
                    # ---- Multiple-solution loop --------------------------------
                    # After the first solution, the user may type ';' to ask for
                    # the next solution (backtracking).  We keep looping until:
                    #   - the user accepts (blank / non-';' line, or EOF), or
                    #   - there are no more choice points, or
                    #   - the proof fails.
                    depth = 0
                    while True:
                        if not success:
                            if not quiet:
                                print("No")
                            # Clean up any leftover state
                            engine.trail.undo_to(0)
                            engine.goal_stack = None
                            engine.choice_stack = None
                            break

                        # Print current solution
                        if engine.var_occurred:
                            _print_bindings(engine, term, quiet)
                        else:
                            if not quiet:
                                print("Yes")

                        # No more alternatives → done
                        if not engine.choice_stack:
                            break

                        # Ask the user whether to backtrack
                        depth += 1
                        prompt = "-" * (depth * 2) + "?- "
                        if not quiet:
                            print(prompt, end="", flush=True)
                        try:
                            resp = input()
                        except EOFError:
                            # End of input → accept current solution
                            engine.trail.undo_to(0)
                            engine.goal_stack = None
                            engine.choice_stack = None
                            break

                        resp = resp.strip()
                        if resp == ";":
                            # Backtrack and find next solution
                            engine.backtrack()
                            success = engine.run()
                        else:
                            # Blank line, '.', or anything else → accept
                            engine.trail.undo_to(0)
                            engine.goal_stack = None
                            engine.choice_stack = None
                            break

                except HaltException:
                    return 0
                except AbortException:
                    if not quiet:
                        print("Aborted.")
                except KeyboardInterrupt:
                    if not quiet:
                        print("\nInterrupted.")
                    engine.trail.undo_to(0)
                    engine.goal_stack = None
                    engine.choice_stack = None
                except Exception as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    if engine.verbose:
                        traceback.print_exc()

            elif sort == FACT:
                # It's a fact/rule: head :- body  or  head.
                try:
                    engine.assert_first = False
                    engine.assert_clause(term)
                    if not quiet:
                        print("Yes")
                except HaltException:
                    return 0
                except Exception as exc:
                    print(f"Assert error: {exc}", file=sys.stderr)
            else:
                # Unknown sort — treat as fact
                try:
                    engine.assert_clause(term)
                    if not quiet:
                        print("Yes")
                except Exception as exc:
                    print(f"Error: {exc}", file=sys.stderr)

        except HaltException:
            break
        except KeyboardInterrupt:
            if not quiet:
                print("\nType Ctrl-D to exit.")
            continue
        except Exception as exc:
            print(f"Unexpected error: {exc}", file=sys.stderr)
            if engine.verbose:
                traceback.print_exc()
            continue

    return exit_code


# ---------------------------------------------------------------------------
# Variable binding display helpers
# ---------------------------------------------------------------------------

def _has_variables(term) -> bool:
    """Return True if the psi-term contains any unbound variables.

    Variables in Wild Life are PsiTerms with type=WL.top, no value,
    no attrs, and no coref (unbound).
    """
    from wild_life.runtime import WL
    seen = set()

    def _walk(t):
        if t is None:
            return False
        tid = id(t)
        if tid in seen:
            return False
        seen.add(tid)
        # Dereference coref chain
        while t.coref is not None:
            t = t.coref
            tid2 = id(t)
            if tid2 in seen:
                return False
            seen.add(tid2)
        # Variables have type=WL.top, no value, no attrs, no resid
        if t.type is WL.top and t.value is None and not t.attr_list and not t.resid:
            return True  # unbound variable
        for v in t.attr_list.values():
            if _walk(v):
                return True
        return False

    return _walk(term)


def _print_bindings(engine, term, quiet: bool) -> None:
    """Print the variable bindings after a successful proof."""
    from wild_life.print_term import term_to_string

    wl = engine.wl
    # Collect top-level variables from the original term
    # (The engine's var_tree holds the variable name -> term mapping)
    var_tree = getattr(engine, "_last_var_tree", {})

    if not var_tree:
        # No named variables — just print Yes
        if not quiet:
            print("Yes")
        return

    printed_any = False
    for name, t in sorted(var_tree.items()):
        if name.startswith("_"):
            continue  # anonymous variable
        # Dereference
        curr = t
        while curr is not None and curr.coref is not None:
            curr = curr.coref
        s = term_to_string(curr, quoted=True, print_depth=10, var_tree={}, wl=wl)
        print(f"{name} = {s}")
        printed_any = True

    if not quiet:
        print("Yes" if printed_any or True else "No")


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
