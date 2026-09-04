# ReLife — Wild Life 1.02 Python Port

Python rewrite of the **Wild Life 1.02** interpreter — an implementation of the
**LIFE** (Logic, Inheritance, Functions, Equations) programming language,
originally developed at DEC Paris Research Laboratory (1991–1995).

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

## Original C Source

Wild Life 1.02, Copyright © 1991–1993 Digital Equipment Corporation  
Extensions, Copyright © 1994–1995 Intelligent Software Group, SFU

Python port — 2024
