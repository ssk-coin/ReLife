# ReLife — Wild Life 1.02 Python Port

Current release: **ReLife 0.01**

Python rewrite of the **Wild Life 1.02** interpreter — an implementation of the
**LIFE** (Logic, Inheritance, Functions, Equations) programming language,
originally developed at DEC Paris Research Laboratory (1991–1995). 
This porting has been done for research purpose.

## About LIFE

LIFE is an experimental language combining:
- **Logic programming** (Prolog-style unification and backtracking)
- **Inheritance** (typed feature terms / ψ-terms with multiple inheritance)
- **Functions** (functional evaluation with currying)
- **Equations** (residuation / constraint suspension)

## Structure

```
wild_life/
├── __init__.py         — package metadata
├── __main__.py         — enables: python -m wild_life
├── data_structures.py  — PsiTerm, Definition, Goal, ChoicePoint, GoalType, DefType
├── runtime.py          — WildLifeRuntime singleton (symbol table, type hierarchy, operators)
├── tokenizer.py        — LIFE language tokenizer
├── parser_.py          — operator-precedence parser (two-state machine)
├── unification.py      — Trail, Unifier, LUB computation, copy_term
├── print_term.py       — operator-aware pretty-printer with variable tracking
├── inference.py        — inference engine (prove/unify/eval/match/clause, backtracking)
├── built_ins.py        — 80+ built-in predicates (I/O, arithmetic, type tests, …)
└── main.py             — Read-Evaluate-Print loop / CLI entry point
```

## Usage

```bash
# Interactive REPL
python -m wild_life

# Quiet mode (no banner)
python -m wild_life -q

# Load a .lf file
python -m wild_life program.lf
```

## Release Notes

### ReLife 0.01

First release.  The interpreter runs the original Wild Life 1.02 regression
suite (`tests_original/`, run with `python run_tests.py`): 308 of 324 tests
match the reference output.  The 16 remaining differences are not errors in
the LIFE semantics:

- output that shows the original implementation's internals: execution
  trace format, internal sort codes, symbol-table listings, stream records,
  library definitions shown by `listing` (`another_bug`, `babcode`, `t`,
  `children`, `test03`, `libtest`, `listing1`, `parser`);
- memory-overflow tests (`t8101`, `t8105`);
- syntax-error recovery and end-of-file handling on malformed input
  (`eoftest`, `eoftest2`), and an input-echo artifact (`glob_test2`);
- `testslc`, where the test writes standard output and its output file into
  the same file;
- `canonical`, where the original rejects the unspaced
  `` `( `a & `b.c.`d) `` that ReLife accepts;
- `testsl` (SuperLint), whose output matches the reference except for the
  banner when the C preprocessor emits old-style line markers, but which
  takes about 17 minutes and so exceeds the test time limit.

## Original C Source

Wild Life 1.02, Copyright © 1991–1993 Digital Equipment Corporation  
Extensions, Copyright © 1994–1995 Intelligent Software Group, SFU

Python port — 2026
