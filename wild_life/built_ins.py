"""
built_ins.py — Built-in predicates and functions for the Wild Life interpreter.
Corresponds to built_ins.c, bi_math.c, bi_sys.c, bi_type.c in the C source.

Each built-in is a function:
    def bi_xxx(goal: PsiTerm, eng: Engine) -> bool

where `goal` is the fully-dereferenced goal psi-term and `eng` is the
running Engine.  Functions return True on success, False on failure.
Exceptions (CutException, HaltException, AbortException) may be raised.
"""

from __future__ import annotations
import sys
import os
import math
import time
import re
import io
from typing import Optional, Tuple

from wild_life.data_structures import (
    PsiTerm, Definition, GoalType, DefType, FACT, QUERY, ERROR
)
from wild_life.unification import (
    UnificationFailure, CutException, HaltException, AbortException,
    copy_term, compute_lub, term_to_string as _term_str
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_two_args(t: PsiTerm) -> Tuple[Optional[PsiTerm], Optional[PsiTerm]]:
    a1 = t.attr_list.get('1')
    a2 = t.attr_list.get('2')
    return (a1.deref() if a1 else None, a2.deref() if a2 else None)


def _get_one_arg(t: PsiTerm) -> Optional[PsiTerm]:
    a1 = t.attr_list.get('1')
    return a1.deref() if a1 else None


def _get_real(t: PsiTerm, eng) -> Tuple[bool, float]:
    """Return (ok, value) for a numeric psi-term."""
    wl = eng.wl
    if t is None:
        return False, 0.0
    if t.value is not None and t.type and t.type.is_subtype_of(wl.real):
        return True, float(t.value)
    return False, 0.0


def _make_number(eng, v: float) -> PsiTerm:
    return eng.wl.make_number(v)


def _make_int(eng, n: int) -> PsiTerm:
    return eng.wl.make_integer(n)


def _make_string(eng, s: str) -> PsiTerm:
    return eng.wl.make_string(s)


def _make_atom(eng, name: str) -> PsiTerm:
    return eng.wl.make_atom(name, eng.wl.user_module)


def _unify(eng, a: PsiTerm, b: PsiTerm) -> bool:
    from wild_life.unification import Trail
    mark = eng.trail.mark()
    ok = eng.unifier.unify(a, b)
    if not ok:
        eng.trail.undo_to(mark)
    return ok


def _psi_to_python(t: Optional[PsiTerm], eng):
    """Convert a WL psi-term to a Python value (for printing etc.)."""
    if t is None:
        return None
    t = t.deref()
    wl = eng.wl
    if t.value is not None:
        if t.type and t.type.is_subtype_of(wl.real):
            v = float(t.value)
            if v == int(v) and t.type.is_subtype_of(wl.integer):
                return int(v)
            return v
        if t.type and t.type.is_subtype_of(wl.quoted_string):
            return str(t.value)
    return t


def _write_term(t: PsiTerm, eng, stream=None, quoted=True) -> None:
    from wild_life.print_term import write_term
    write_term(t, outfile=stream or sys.stdout, quoted=quoted, wl=eng.wl)


def _term_to_str(t: PsiTerm, eng, quoted=True) -> str:
    from wild_life.print_term import term_to_string
    return term_to_string(t, quoted=quoted, wl=eng.wl)


def _is_var(t: PsiTerm, eng) -> bool:
    wl = eng.wl
    return t.type == wl.top and t.value is None and not t.attr_list and t.coref is None


# ─────────────────────────────────────────────────────────────────────────────
# I/O predicates
# ─────────────────────────────────────────────────────────────────────────────

def bi_write(goal: PsiTerm, eng) -> bool:
    """write(T) — write term T without quoting."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    _write_term(arg, eng, quoted=False)
    return True


def bi_writeq(goal: PsiTerm, eng) -> bool:
    """writeq(T) — write term T with quoting."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    _write_term(arg, eng, quoted=True)
    return True


def bi_print(goal: PsiTerm, eng) -> bool:
    """print(T) — same as write."""
    return bi_write(goal, eng)


def bi_nl(goal: PsiTerm, eng) -> bool:
    """nl — print newline."""
    print()
    return True


def bi_write_err(goal: PsiTerm, eng) -> bool:
    """write_err(T) — write to stderr."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    _write_term(arg, eng, stream=sys.stderr, quoted=False)
    return True


def bi_writeln(goal: PsiTerm, eng) -> bool:
    """writeln(T) — write then newline."""
    bi_write(goal, eng)
    print()
    return True


def bi_put_char(goal: PsiTerm, eng) -> bool:
    """put_char(C) / put(C) — write a character."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    wl = eng.wl
    if arg.value is not None and arg.type and arg.type.is_subtype_of(wl.integer):
        c = int(float(arg.value))
        sys.stdout.write(chr(c))
    elif arg.value is not None and arg.type and arg.type.is_subtype_of(wl.quoted_string):
        s = str(arg.value)
        if s:
            sys.stdout.write(s[0])
    return True


def bi_get_char(goal: PsiTerm, eng) -> bool:
    """get_char(C) — read a character."""
    arg = _get_one_arg(goal)
    wl = eng.wl
    try:
        c = sys.stdin.read(1)
    except EOFError:
        c = ''
    if c == '':
        result = wl.make_atom('end_of_file', wl.user_module)
    else:
        result = wl.make_string(c)
    return _unify(eng, arg, result) if arg else False


def bi_read(goal: PsiTerm, eng) -> bool:
    """read(T) — read and parse a term from stdin."""
    arg = _get_one_arg(goal)
    from wild_life.tokenizer import tokenizer_from_string
    from wild_life.parser_ import Parser
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        line = ''
    if not line:
        wl = eng.wl
        result = PsiTerm(type=wl.eof)
    else:
        ts = tokenizer_from_string(line)
        p = Parser(ts)
        try:
            t, _ = p.parse()
            result = t or PsiTerm(type=eng.wl.top)
        except Exception:
            result = PsiTerm(type=eng.wl.top)
    return _unify(eng, arg, result) if arg else bool(result)


def bi_read_term(goal: PsiTerm, eng) -> bool:
    """read_term(T, Opts) — simplified."""
    arg, _ = _get_two_args(goal)
    return bi_read(goal, eng)  # simplified: ignore options


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def _eval_arith(t: PsiTerm, eng) -> Tuple[bool, float]:
    """Evaluate an arithmetic expression. Returns (ok, value)."""
    if t is None:
        return False, 0.0
    t = t.deref()
    wl = eng.wl
    sym = t.type.keyword.symbol if t.type and t.type.keyword else ''

    if t.value is not None and t.type and t.type.is_subtype_of(wl.real):
        return True, float(t.value)

    # Binary operators
    arg1, arg2 = _get_two_args(t)
    ok1, v1 = _eval_arith(arg1, eng) if arg1 else (False, 0.0)
    ok2, v2 = _eval_arith(arg2, eng) if arg2 else (False, 0.0)

    ops2 = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b if b != 0 else float('inf'),
        '//': lambda a, b: float(int(a) // int(b)) if b != 0 else 0.0,
        'mod': lambda a, b: float(int(a) % int(b)) if b != 0 else 0.0,
        '**': lambda a, b: a ** b,
        '^': lambda a, b: a ** b,
        'max': lambda a, b: max(a, b),
        'min': lambda a, b: min(a, b),
    }
    if sym in ops2 and ok1 and ok2:
        try:
            return True, float(ops2[sym](v1, v2))
        except Exception:
            return False, 0.0

    ops1 = {
        '-': lambda a: -a,
        'abs': lambda a: abs(a),
        'sqrt': lambda a: math.sqrt(a),
        'sin': lambda a: math.sin(a),
        'cos': lambda a: math.cos(a),
        'tan': lambda a: math.tan(a),
        'asin': lambda a: math.asin(a),
        'acos': lambda a: math.acos(a),
        'atan': lambda a: math.atan(a),
        'exp': lambda a: math.exp(a),
        'log': lambda a: math.log(a),
        'floor': lambda a: math.floor(a),
        'ceiling': lambda a: math.ceil(a),
        'round': lambda a: round(a),
        'truncate': lambda a: math.trunc(a),
        'float': lambda a: float(a),
        'integer': lambda a: float(int(a)),
        'float_integer_part': lambda a: float(math.trunc(a)),
        'float_fractional_part': lambda a: a - math.trunc(a),
        'sign': lambda a: (1.0 if a > 0 else (-1.0 if a < 0 else 0.0)),
        'msb': lambda a: int(math.log2(max(1, int(a)))),
    }
    if sym in ops1 and ok1:
        try:
            return True, float(ops1[sym](v1))
        except Exception:
            return False, 0.0

    return False, 0.0


def bi_is(goal: PsiTerm, eng) -> bool:
    """X is Expr — evaluate arithmetic expression and unify result."""
    arg1, arg2 = _get_two_args(goal)
    if arg1 is None or arg2 is None:
        return False
    ok, val = _eval_arith(arg2, eng)
    if not ok:
        print(f"*** Error: arithmetic evaluation failed.", file=sys.stderr)
        return False
    result = _make_number(eng, val)
    return _unify(eng, arg1, result)


def bi_arith_eq(goal: PsiTerm, eng) -> bool:
    """X =:= Y — arithmetic equality."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    return oka and okb and va == vb


def bi_arith_ne(goal: PsiTerm, eng) -> bool:
    """X =\\= Y — arithmetic inequality."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    return oka and okb and va != vb


def bi_arith_lt(goal: PsiTerm, eng) -> bool:
    """X < Y."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    return oka and okb and va < vb


def bi_arith_le(goal: PsiTerm, eng) -> bool:
    """X =< Y."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    return oka and okb and va <= vb


def bi_arith_gt(goal: PsiTerm, eng) -> bool:
    """X > Y."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    return oka and okb and va > vb


def bi_arith_ge(goal: PsiTerm, eng) -> bool:
    """X >= Y."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    return oka and okb and va >= vb


# ─────────────────────────────────────────────────────────────────────────────
# Unification / comparison
# ─────────────────────────────────────────────────────────────────────────────

def bi_unify(goal: PsiTerm, eng) -> bool:
    """X = Y — standard unification."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return a is b
    return _unify(eng, a, b)


def bi_not_unify(goal: PsiTerm, eng) -> bool:
    """X \\= Y — non-unifiable."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    mark = eng.trail.mark()
    ok = eng.unifier.unify(a, b)
    eng.trail.undo_to(mark)
    return not ok


def bi_identical(goal: PsiTerm, eng) -> bool:
    """X == Y — structural identity."""
    a, b = _get_two_args(goal)
    s1 = _term_to_str(a, eng)
    s2 = _term_to_str(b, eng)
    return s1 == s2


def bi_not_identical(goal: PsiTerm, eng) -> bool:
    """X \\== Y."""
    a, b = _get_two_args(goal)
    s1 = _term_to_str(a, eng)
    s2 = _term_to_str(b, eng)
    return s1 != s2


def bi_compare(goal: PsiTerm, eng) -> bool:
    """compare(Order, X, Y) — standard order comparison."""
    arg1, rest = _get_two_args(goal)
    if rest is None:
        return False
    arg2 = rest.attr_list.get('1')
    arg3 = rest.attr_list.get('2')
    if arg2 is None or arg3 is None:
        a, b = goal.attr_list.get('2'), goal.attr_list.get('3')
        arg2 = a.deref() if a else None
        arg3 = b.deref() if b else None
    if arg2 is None or arg3 is None:
        return False
    s2 = _term_to_str(arg2, eng)
    s3 = _term_to_str(arg3, eng)
    if s2 < s3:
        order = '<'
    elif s2 > s3:
        order = '>'
    else:
        order = '='
    result = eng.wl.make_atom(order, eng.wl.user_module)
    return _unify(eng, arg1, result)


# ─────────────────────────────────────────────────────────────────────────────
# Type testing
# ─────────────────────────────────────────────────────────────────────────────

def bi_var(goal: PsiTerm, eng) -> bool:
    """var(X) — true if X is an unbound variable."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    return _is_var(arg, eng)


def bi_nonvar(goal: PsiTerm, eng) -> bool:
    """nonvar(X)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return True
    return not _is_var(arg, eng)


def bi_atom(goal: PsiTerm, eng) -> bool:
    """atom(X) — true if X is an atom (constant, not number/string)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    wl = eng.wl
    if _is_var(arg, eng):
        return False
    if arg.value is not None:
        if arg.type and (arg.type.is_subtype_of(wl.real) or
                         arg.type.is_subtype_of(wl.quoted_string)):
            return False
    return not arg.attr_list


def bi_integer(goal: PsiTerm, eng) -> bool:
    """integer(X)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    wl = eng.wl
    return (arg.type is not None and arg.type.is_subtype_of(wl.integer)
            and arg.value is not None and float(arg.value) == int(float(arg.value)))


def bi_float_check(goal: PsiTerm, eng) -> bool:
    """float(X)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    wl = eng.wl
    if arg.type is None or not arg.type.is_subtype_of(wl.real):
        return False
    if arg.value is None:
        return False
    return float(arg.value) != int(float(arg.value))


def bi_number(goal: PsiTerm, eng) -> bool:
    """number(X)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    wl = eng.wl
    return (arg.type is not None and arg.type.is_subtype_of(wl.real)
            and arg.value is not None)


def bi_string(goal: PsiTerm, eng) -> bool:
    """string(X)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    wl = eng.wl
    return (arg.type is not None and arg.type.is_subtype_of(wl.quoted_string)
            and arg.value is not None)


def bi_is_list(goal: PsiTerm, eng) -> bool:
    """is_list(X) — true if X is a proper list."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    wl = eng.wl
    t = arg
    while True:
        t = t.deref()
        if t.type == wl.nil:
            return True
        if t.type != wl.alist:
            return False
        t2 = t.attr_list.get('2')
        if t2 is None:
            return False
        t = t2


def bi_compound(goal: PsiTerm, eng) -> bool:
    """compound(X)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    return bool(arg.attr_list)


def bi_callable(goal: PsiTerm, eng) -> bool:
    """callable(X)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    return not _is_var(arg, eng)


def bi_ground(goal: PsiTerm, eng) -> bool:
    """ground(X) — true if X contains no unbound variables."""
    arg = _get_one_arg(goal)
    if arg is None:
        return True
    return _is_ground(arg, eng, set())


def _is_ground(t: PsiTerm, eng, seen: set) -> bool:
    t = t.deref()
    tid = id(t)
    if tid in seen:
        return True
    seen.add(tid)
    if _is_var(t, eng):
        return False
    for v in t.attr_list.values():
        if v and not _is_ground(v, eng, seen):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Control
# ─────────────────────────────────────────────────────────────────────────────

def bi_true(goal: PsiTerm, eng) -> bool:
    """true — always succeeds."""
    return True


def bi_fail(goal: PsiTerm, eng) -> bool:
    """fail/false — always fails."""
    return False


def bi_not(goal: PsiTerm, eng) -> bool:
    r"""not(P) / \+(P) — negation as failure."""
    arg = _get_one_arg(goal)
    if arg is None:
        return True
    # Try proving arg; if it succeeds, fail
    mark = eng.trail.mark()
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack
    eng.push_goal(GoalType.PROVE, arg, _DEFRULES_SENTINEL, None)
    old_main_loop_ok = eng.main_loop_ok
    result = eng.run()
    eng.trail.undo_to(mark)
    eng.choice_stack = cp_save
    eng.goal_stack = gs_save
    eng.main_loop_ok = old_main_loop_ok
    return not result


def bi_call(goal: PsiTerm, eng) -> bool:
    """call(P) — call a goal."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    eng.push_goal(GoalType.PROVE, arg, _DEFRULES_SENTINEL, None)
    return True  # will be continued in main loop


def bi_once(goal: PsiTerm, eng) -> bool:
    """once(P) — call P exactly once."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    mark = eng.trail.mark()
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack
    eng.push_goal(GoalType.PROVE, arg, _DEFRULES_SENTINEL, None)
    result = eng.run()
    if not result:
        eng.trail.undo_to(mark)
    eng.choice_stack = cp_save
    return result


def bi_findall(goal: PsiTerm, eng) -> bool:
    """findall(Template, Goal, Bag) — collect all solutions."""
    arg1, rest = _get_two_args(goal)
    if arg1 is None or rest is None:
        return False
    template = arg1
    g = rest.attr_list.get('1')
    bag_out = rest.attr_list.get('2')
    if g is None or bag_out is None:
        # Try 3-arg form
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        a3 = goal.attr_list.get('3')
        if not (a1 and a2 and a3):
            return False
        template = a1.deref()
        g = a2.deref()
        bag_out = a3.deref()

    solutions = []
    wl = eng.wl
    mark = eng.trail.mark()
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack

    # Run the goal, collecting template copies for each solution
    def collect_solution():
        sol = copy_term(template)
        solutions.append(sol)
        return False  # force backtracking to get all solutions

    # Temporarily push collect goal — simplified: run directly
    goal_copy = copy_term(g)
    eng.push_goal(GoalType.PROVE, goal_copy, _DEFRULES_SENTINEL, None)

    # Collect all solutions via repeated backtracking
    collected = []
    while True:
        result = eng.run()
        if result:
            collected.append(copy_term(template))
            if not eng.choice_stack or eng.choice_stack is cp_save:
                break
            eng.backtrack()
        else:
            break

    eng.trail.undo_to(mark)
    eng.choice_stack = cp_save
    eng.goal_stack = gs_save

    # Build a list of collected solutions
    result_list = wl.make_list(collected)
    return _unify(eng, bag_out, result_list)


def bi_assert(goal: PsiTerm, eng) -> bool:
    """assert(Clause) / assertz(Clause)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    eng.assert_first = False
    eng.assert_clause(arg)
    return True


def bi_asserta(goal: PsiTerm, eng) -> bool:
    """asserta(Clause) — add at front."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    eng.assert_first = True
    eng.assert_clause(arg)
    eng.assert_first = False
    return True


def bi_retract(goal: PsiTerm, eng) -> bool:
    """retract(Clause) — remove first matching clause."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    arg = arg.deref()
    wl = eng.wl
    sym = arg.type.keyword.symbol if arg.type and arg.type.keyword else ''
    if sym == ':-':
        head = arg.attr_list.get('1')
        body = arg.attr_list.get('2')
    else:
        head = arg
        body = None
    if head is None:
        return False
    head = head.deref()
    defn = head.type
    if defn is None or defn.rule is None or callable(defn.rule):
        return False
    for i, (h, b) in enumerate(defn.rule):
        if h is None:
            continue
        mark = eng.trail.mark()
        ok = eng.unifier.unify(head, copy_term(h))
        if ok and body is not None:
            ok = eng.unifier.unify(body, copy_term(b) if b else wl.make_atom('true', wl.bi_module))
        if ok:
            defn.rule[i] = (None, None)  # mark as retracted
            return True
        eng.trail.undo_to(mark)
    return False


def bi_abolish(goal: PsiTerm, eng) -> bool:
    """abolish(F/A) — remove all clauses for functor/arity."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    arg = arg.deref()
    sym = arg.type.keyword.symbol if arg.type and arg.type.keyword else ''
    if sym == '/':
        functor = arg.attr_list.get('1')
        if functor is None:
            return False
        functor = functor.deref()
        defn = functor.type
        if defn and defn.rule:
            defn.rule = []
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Term manipulation
# ─────────────────────────────────────────────────────────────────────────────

def bi_functor(goal: PsiTerm, eng) -> bool:
    """functor(Term, Name, Arity)."""
    wl = eng.wl
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    t = a1.deref()
    name_out = a2.deref()
    arity_out = a3.deref()

    if not _is_var(t, eng):
        sym = t.type.keyword.symbol if t.type and t.type.keyword else ''
        name_atom = wl.make_atom(sym, wl.user_module)
        arity = wl.make_integer(len(t.attr_list))
        return _unify(eng, name_out, name_atom) and _unify(eng, arity_out, arity)
    else:
        # Build term from name and arity
        if _is_var(name_out, eng) or _is_var(arity_out, eng):
            return False
        sym = name_out.type.keyword.symbol if name_out.type and name_out.type.keyword else ''
        try:
            ar = int(float(arity_out.value))
        except Exception:
            return False
        defn = wl.update_symbol(wl.user_module, sym)
        result = PsiTerm(type=defn)
        return _unify(eng, t, result)


def bi_arg(goal: PsiTerm, eng) -> bool:
    """arg(N, Term, Arg)."""
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    n = a1.deref()
    t = a2.deref()
    arg_out = a3.deref()
    if n.value is None:
        return False
    try:
        idx = int(float(n.value))
    except Exception:
        return False
    val = t.attr_list.get(str(idx))
    if val is None:
        return False
    return _unify(eng, arg_out, val.deref())


def bi_univ(goal: PsiTerm, eng) -> bool:
    """Term =.. List (univ)."""
    wl = eng.wl
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    t = a1
    lst = a2

    if not _is_var(t, eng):
        # Decompose t into list [functor|args]
        sym = t.type.keyword.symbol if t.type and t.type.keyword else ''
        head_atom = wl.make_atom(sym, wl.user_module)
        from wild_life.data_structures import featcmp_key
        keys = sorted(t.attr_list.keys(), key=featcmp_key)
        args = [t.attr_list[k] for k in keys]
        result = wl.make_list([head_atom] + args)
        return _unify(eng, lst, result)
    else:
        # Build t from list
        lst = lst.deref()
        items = []
        cur = lst
        while cur.type == wl.alist:
            h = cur.attr_list.get('1')
            t2 = cur.attr_list.get('2')
            if h:
                items.append(h.deref())
            cur = t2.deref() if t2 else wl.make_nil()
        if not items:
            return False
        functor = items[0]
        defn = functor.type if functor.type else None
        if defn is None:
            return False
        result = PsiTerm(type=defn)
        for i, arg in enumerate(items[1:], 1):
            result.attr_list[str(i)] = arg
        return _unify(eng, a1, result)


def bi_copy_term(goal: PsiTerm, eng) -> bool:
    """copy_term(X, Y) — copy X to Y with fresh variables."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    c = copy_term(a)
    return _unify(eng, b, c)


def bi_numbervars(goal: PsiTerm, eng) -> bool:
    """numbervars(Term, Start, End) — number variables in Term."""
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    t = a1.deref()
    start = a2.deref()
    end_out = a3.deref()
    if start.value is None:
        return False
    counter = [int(float(start.value))]

    def number_vars_rec(t: PsiTerm):
        t = t.deref()
        if _is_var(t, eng):
            n = counter[0]
            counter[0] += 1
            letter = chr(ord('A') + n % 26)
            num = n // 26
            name = letter if num == 0 else f"{letter}{num}"
            t.type = eng.wl.update_symbol(eng.wl.user_module, f"${name}")
        else:
            for v in t.attr_list.values():
                if v:
                    number_vars_rec(v)

    number_vars_rec(t)
    end = eng.wl.make_integer(counter[0])
    return _unify(eng, end_out, end)


# ─────────────────────────────────────────────────────────────────────────────
# String / atom operations
# ─────────────────────────────────────────────────────────────────────────────

def bi_atom_chars(goal: PsiTerm, eng) -> bool:
    """atom_chars(Atom, Chars)."""
    wl = eng.wl
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    if not _is_var(a1, eng):
        sym = a1.type.keyword.symbol if a1.type and a1.type.keyword else ''
        chars = [wl.make_string(c) for c in sym]
        lst = wl.make_list(chars)
        return _unify(eng, a2, lst)
    else:
        # Build atom from char list
        chars = []
        cur = a2.deref()
        while cur.type == wl.alist:
            h = cur.attr_list.get('1')
            t2 = cur.attr_list.get('2')
            if h:
                hd = h.deref()
                if hd.value:
                    chars.append(str(hd.value)[0])
            cur = t2.deref() if t2 else wl.make_nil()
        result = wl.make_atom(''.join(chars), wl.user_module)
        return _unify(eng, a1, result)


def bi_atom_string(goal: PsiTerm, eng) -> bool:
    """atom_string(Atom, String)."""
    wl = eng.wl
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    if not _is_var(a1, eng):
        sym = a1.type.keyword.symbol if a1.type and a1.type.keyword else str(a1.value) if a1.value else ''
        return _unify(eng, a2, wl.make_string(sym))
    else:
        s = str(a2.value) if a2.value else ''
        return _unify(eng, a1, wl.make_atom(s, wl.user_module))


def bi_atom_length(goal: PsiTerm, eng) -> bool:
    """atom_length(Atom, Length)."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    sym = a1.type.keyword.symbol if a1.type and a1.type.keyword else str(a1.value) if a1.value else ''
    return _unify(eng, a2, eng.wl.make_integer(len(sym)))


def bi_atom_concat(goal: PsiTerm, eng) -> bool:
    """atom_concat(A1, A2, A3)."""
    wl = eng.wl
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    a1, a2, a3 = a1.deref(), a2.deref(), a3.deref()

    def sym(t):
        if t.value is not None:
            return str(t.value)
        return t.type.keyword.symbol if t.type and t.type.keyword else ''

    if not _is_var(a1, eng) and not _is_var(a2, eng):
        result = wl.make_atom(sym(a1) + sym(a2), wl.user_module)
        return _unify(eng, a3, result)
    if not _is_var(a3, eng):
        s3 = sym(a3)
        for i in range(len(s3) + 1):
            r1 = wl.make_atom(s3[:i], wl.user_module)
            r2 = wl.make_atom(s3[i:], wl.user_module)
            mark = eng.trail.mark()
            if _unify(eng, a1, r1) and _unify(eng, a2, r2):
                return True
            eng.trail.undo_to(mark)
    return False


def bi_number_chars(goal: PsiTerm, eng) -> bool:
    """number_chars(Number, Chars)."""
    wl = eng.wl
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    if not _is_var(a1, eng):
        s = str(int(float(a1.value))) if a1.value and float(a1.value) == int(float(a1.value)) else str(float(a1.value)) if a1.value else '0'
        chars = [wl.make_string(c) for c in s]
        return _unify(eng, a2, wl.make_list(chars))
    return False


def bi_number_codes(goal: PsiTerm, eng) -> bool:
    """number_codes(Number, Codes)."""
    wl = eng.wl
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    if not _is_var(a1, eng) and a1.value is not None:
        s = str(int(float(a1.value))) if float(a1.value) == int(float(a1.value)) else str(float(a1.value))
        codes = [wl.make_integer(ord(c)) for c in s]
        return _unify(eng, a2, wl.make_list(codes))
    return False


def bi_char_code(goal: PsiTerm, eng) -> bool:
    """char_code(Char, Code)."""
    wl = eng.wl
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    if not _is_var(a1, eng):
        s = str(a1.value) if a1.value else (a1.type.keyword.symbol if a1.type and a1.type.keyword else '')
        if s:
            return _unify(eng, a2, wl.make_integer(ord(s[0])))
    elif not _is_var(a2, eng) and a2.value is not None:
        c = chr(int(float(a2.value)))
        return _unify(eng, a1, wl.make_string(c))
    return False


def bi_string_to_atom(goal: PsiTerm, eng) -> bool:
    """string_to_atom(String, Atom)."""
    return bi_atom_string(goal, eng)


def bi_term_to_atom(goal: PsiTerm, eng) -> bool:
    """term_to_atom(Term, Atom)."""
    wl = eng.wl
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    if not _is_var(a1, eng):
        s = _term_to_str(a1, eng, quoted=True)
        return _unify(eng, a2, wl.make_string(s))
    else:
        # Parse atom to term
        s = str(a2.value) if a2.value else ''
        from wild_life.parser_ import parse_term_string
        t = parse_term_string(s)
        if t:
            return _unify(eng, a1, t)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# List operations
# ─────────────────────────────────────────────────────────────────────────────

def _list_to_python(t: PsiTerm, eng):
    """Convert WL list to Python list."""
    wl = eng.wl
    items = []
    cur = t.deref()
    while cur.type == wl.alist:
        h = cur.attr_list.get('1')
        t2 = cur.attr_list.get('2')
        if h:
            items.append(h.deref())
        cur = t2.deref() if t2 else wl.make_nil()
    return items


def bi_length(goal: PsiTerm, eng) -> bool:
    """length(List, N)."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    wl = eng.wl
    if not _is_var(a1, eng):
        items = _list_to_python(a1, eng)
        return _unify(eng, a2, wl.make_integer(len(items)))
    elif not _is_var(a2, eng) and a2.value is not None:
        n = int(float(a2.value))
        # Build list of n unbound variables
        nil = wl.make_nil()
        lst = nil
        for _ in range(n):
            var = PsiTerm(type=wl.top)
            lst = wl.make_cons(var, lst)
        return _unify(eng, a1, lst)
    return False


def bi_append(goal: PsiTerm, eng) -> bool:
    """append(L1, L2, L3)."""
    wl = eng.wl
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    a1, a2, a3 = a1.deref(), a2.deref(), a3.deref()

    if not _is_var(a1, eng):
        items = _list_to_python(a1, eng)
        result = a2
        for item in reversed(items):
            result = wl.make_cons(item, result)
        return _unify(eng, a3, result)
    return False


def bi_member(goal: PsiTerm, eng) -> bool:
    """member(X, List)."""
    wl = eng.wl
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    items = _list_to_python(a2, eng)
    if not items:
        return False
    # Set up choice points for each member
    x = a1
    for i, item in enumerate(reversed(items)):
        if i < len(items) - 1:
            eng.push_choice_point(GoalType.PROVE, x, item, None)
    # Try first item
    mark = eng.trail.mark()
    ok = _unify(eng, x, items[0])
    if not ok:
        eng.trail.undo_to(mark)
        return False
    return True


def bi_reverse(goal: PsiTerm, eng) -> bool:
    """reverse(List, Rev)."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    items = _list_to_python(a1, eng)
    result = eng.wl.make_list(list(reversed(items)))
    return _unify(eng, a2, result)


def bi_msort(goal: PsiTerm, eng) -> bool:
    """msort(List, Sorted) — sort without removing duplicates."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    items = _list_to_python(a1, eng)
    sorted_items = sorted(items, key=lambda t: _term_to_str(t, eng))
    result = eng.wl.make_list(sorted_items)
    return _unify(eng, a2, result)


def bi_sort(goal: PsiTerm, eng) -> bool:
    """sort(List, Sorted) — sort removing duplicates."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    items = _list_to_python(a1, eng)
    seen = set()
    unique = []
    for item in sorted(items, key=lambda t: _term_to_str(t, eng)):
        s = _term_to_str(item, eng)
        if s not in seen:
            seen.add(s)
            unique.append(item)
    result = eng.wl.make_list(unique)
    return _unify(eng, a2, result)


def bi_last(goal: PsiTerm, eng) -> bool:
    """last(List, Elem)."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    items = _list_to_python(a1, eng)
    if not items:
        return False
    return _unify(eng, a2, items[-1])


def bi_nth(goal: PsiTerm, eng) -> bool:
    """nth0(N, List, Elem) or nth1(N, List, Elem)."""
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    n = a1.deref()
    lst = a2.deref()
    elem = a3.deref()
    if n.value is None:
        return False
    idx = int(float(n.value))
    items = _list_to_python(lst, eng)
    if idx < 0 or idx >= len(items):
        return False
    return _unify(eng, elem, items[idx])


# ─────────────────────────────────────────────────────────────────────────────
# System predicates
# ─────────────────────────────────────────────────────────────────────────────

def bi_halt(goal: PsiTerm, eng) -> bool:
    """halt / halt(N)."""
    arg = _get_one_arg(goal)
    code = 0
    if arg and arg.value is not None:
        try:
            code = int(float(arg.value))
        except Exception:
            pass
    raise HaltException(code)


def bi_abort(goal: PsiTerm, eng) -> bool:
    """abort — abort current query."""
    raise AbortException()


def bi_nl_err(goal: PsiTerm, eng) -> bool:
    """nl_err — newline to stderr."""
    print(file=sys.stderr)
    return True


def bi_assert_ok(goal: PsiTerm, eng) -> bool:
    """succeed if last assert succeeded."""
    return True


def bi_load(goal: PsiTerm, eng) -> bool:
    """load(File) — load a LIFE source file."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    filename = str(arg.value) if arg.value else (
        arg.type.keyword.symbol if arg.type and arg.type.keyword else '')
    if not filename.endswith('.lf'):
        filename += '.lf'
    return eng.load_file(filename)


def bi_op(goal: PsiTerm, eng) -> bool:
    """op(Prec, Type, Name) — declare operator."""
    wl = eng.wl
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    prec = a1.deref()
    typ = a2.deref()
    name = a3.deref()
    if prec.value is None:
        return False
    p = int(float(prec.value))
    op_name = typ.type.keyword.symbol if typ.type and typ.type.keyword else ''
    sym = name.type.keyword.symbol if name.type and name.type.keyword else ''
    from wild_life.data_structures import OperatorType
    op_map = {
        'fx': OperatorType.FX, 'fy': OperatorType.FY,
        'xf': OperatorType.XF, 'yf': OperatorType.YF,
        'xfx': OperatorType.XFX, 'xfy': OperatorType.XFY,
        'yfx': OperatorType.YFX,
    }
    if op_name not in op_map:
        return False
    wl.add_operator(p, op_map[op_name], sym)
    return True


def bi_var_name(goal: PsiTerm, eng) -> bool:
    """var_name(Var, Name) — get or set variable name."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    # simplified: just try to unify with a fresh name
    return True


def bi_succ(goal: PsiTerm, eng) -> bool:
    """succ(X, Y) — Y = X + 1."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    wl = eng.wl
    if not _is_var(a1, eng) and a1.value is not None:
        return _unify(eng, a2, wl.make_integer(int(float(a1.value)) + 1))
    if not _is_var(a2, eng) and a2.value is not None:
        v = int(float(a2.value)) - 1
        if v < 0:
            return False
        return _unify(eng, a1, wl.make_integer(v))
    return False


def bi_plus(goal: PsiTerm, eng) -> bool:
    """plus(X, Y, Z) — Z = X + Y."""
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    a1, a2, a3 = a1.deref(), a2.deref(), a3.deref()
    wl = eng.wl
    if not _is_var(a1, eng) and not _is_var(a2, eng) and a1.value is not None and a2.value is not None:
        return _unify(eng, a3, wl.make_number(float(a1.value) + float(a2.value)))
    if not _is_var(a1, eng) and not _is_var(a3, eng) and a1.value is not None and a3.value is not None:
        return _unify(eng, a2, wl.make_number(float(a3.value) - float(a1.value)))
    if not _is_var(a2, eng) and not _is_var(a3, eng) and a2.value is not None and a3.value is not None:
        return _unify(eng, a1, wl.make_number(float(a3.value) - float(a2.value)))
    return False


def bi_between(goal: PsiTerm, eng) -> bool:
    """between(Low, High, X) — X ranges from Low to High."""
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    low = a1.deref()
    high = a2.deref()
    x = a3.deref()
    if low.value is None or high.value is None:
        return False
    lo = int(float(low.value))
    hi = int(float(high.value))
    wl = eng.wl
    if not _is_var(x, eng) and x.value is not None:
        v = int(float(x.value))
        return lo <= v <= hi
    # Non-deterministic: create choice points
    values = list(range(lo, hi + 1))
    if not values:
        return False
    # Push remaining values as choice points
    for v in reversed(values[1:]):
        pt = wl.make_integer(v)
        eng.push_choice_point(GoalType.PROVE, x, pt, None)
    return _unify(eng, x, wl.make_integer(values[0]))


def bi_aggregate_all(goal: PsiTerm, eng) -> bool:
    """aggregate_all(count, Goal, Count) — simplified aggregation."""
    return bi_findall(goal, eng)  # simplified


def bi_format(goal: PsiTerm, eng) -> bool:
    """format(Fmt) or format(Fmt, Args)."""
    a1 = _get_one_arg(goal)
    a2 = goal.attr_list.get('2')
    if a1 is None:
        return False
    fmt = str(a1.value) if a1.value else (a1.type.keyword.symbol if a1.type and a1.type.keyword else '')
    # Very simplified format
    fmt = fmt.replace('~w', '{}').replace('~a', '{}').replace('~n', '\n').replace('~N', '\n')
    if a2:
        items = _list_to_python(a2.deref(), eng)
        args = [_term_to_str(i, eng, quoted=False) for i in items]
        try:
            print(fmt.format(*args), end='')
        except Exception:
            print(fmt, end='')
    else:
        print(fmt, end='')
    return True


def bi_statistics(goal: PsiTerm, eng) -> bool:
    """statistics — print memory/time stats."""
    print(f"Goal count: {eng.goal_count}")
    return True


def bi_listing(goal: PsiTerm, eng) -> bool:
    """listing(F) — list clauses for functor F."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    defn = arg.type if arg.type else None
    if defn is None:
        return False
    rules = defn.rule or []
    from wild_life.print_term import term_to_string
    for h, b in rules:
        if h is None:
            continue
        hs = term_to_string(h, wl=eng.wl)
        bs = term_to_string(b, wl=eng.wl)
        wl = eng.wl
        succeed_sym = wl.succeed.keyword.symbol if wl.succeed and wl.succeed.keyword else 'succeed'
        if b and b.type and b.type.keyword and b.type.keyword.symbol != succeed_sym:
            print(f"{hs} :- {bs}.")
        else:
            print(f"{hs}.")
    return True


def bi_current_prolog_flag(goal: PsiTerm, eng) -> bool:
    """current_prolog_flag(Flag, Value)."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    flag = a1.type.keyword.symbol if a1.type and a1.type.keyword else ''
    wl = eng.wl
    flags = {
        'bounded': 'false',
        'max_integer': str(2**62),
        'min_integer': str(-(2**62)),
        'integer_rounding_function': 'toward_zero',
        'max_arity': 'unbounded',
    }
    val_str = flags.get(flag, 'undefined')
    result = wl.make_atom(val_str, wl.user_module)
    return _unify(eng, a2, result)


def bi_set_prolog_flag(goal: PsiTerm, eng) -> bool:
    """set_prolog_flag(Flag, Value) — simplified: accept but ignore."""
    return True


def bi_succ_or_zero(goal: PsiTerm, eng) -> bool:
    return bi_succ(goal, eng)


def bi_rand(goal: PsiTerm, eng) -> bool:
    """random(X) — X is a random float [0,1)."""
    import random
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    return _unify(eng, arg, eng.wl.make_number(random.random()))


def bi_msort_key(goal: PsiTerm, eng) -> bool:
    return bi_msort(goal, eng)


def bi_with_output_to(goal: PsiTerm, eng) -> bool:
    """with_output_to(string(S), Goal) — capture output."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    a1 = a1.deref()
    sym = a1.type.keyword.symbol if a1.type and a1.type.keyword else ''
    if sym == 'string':
        out_var = a1.attr_list.get('1')
        old_stdout = sys.stdout
        buf = io.StringIO()
        sys.stdout = buf
        eng.push_goal(GoalType.PROVE, a2, _DEFRULES_SENTINEL, None)
        result = eng.run()
        sys.stdout = old_stdout
        if result and out_var:
            return _unify(eng, out_var.deref(), eng.wl.make_string(buf.getvalue()))
        return result
    return False


def bi_char_type(goal: PsiTerm, eng) -> bool:
    """char_type(Char, Type) — simplified."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    c = str(a1.value)[0] if a1.value else ''
    typ = a2.type.keyword.symbol if a2.type and a2.type.keyword else ''
    checks = {
        'alpha': c.isalpha,
        'alnum': c.isalnum,
        'digit': c.isdigit,
        'space': c.isspace,
        'upper': c.isupper,
        'lower': c.islower,
    }
    fn = checks.get(typ)
    return bool(fn and fn())


def bi_string_codes(goal: PsiTerm, eng) -> bool:
    """string_codes(String, Codes)."""
    wl = eng.wl
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    if not _is_var(a1, eng):
        s = str(a1.value) if a1.value else ''
        codes = [wl.make_integer(ord(c)) for c in s]
        return _unify(eng, a2, wl.make_list(codes))
    return False


def bi_string_length(goal: PsiTerm, eng) -> bool:
    """string_length(String, Len)."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    s = str(a1.value) if a1.value else ''
    return _unify(eng, a2, eng.wl.make_integer(len(s)))


# ─────────────────────────────────────────────────────────────────────────────
# Type hierarchy predicates
# ─────────────────────────────────────────────────────────────────────────────

def bi_sub_type(goal: PsiTerm, eng) -> bool:
    """sub_type(T1, T2) — T1 is a subtype of T2."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    d1 = a1.type
    d2 = a2.type
    if d1 is None or d2 is None:
        return False
    return d1.is_subtype_of(d2)


def bi_get_attribute(goal: PsiTerm, eng) -> bool:
    """get_attribute(Term, Key, Value)."""
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    t = a1.deref()
    key_t = a2.deref()
    val_out = a3.deref()
    key = key_t.type.keyword.symbol if key_t.type and key_t.type.keyword else str(key_t.value) if key_t.value else ''
    val = t.attr_list.get(key)
    if val is None:
        return False
    return _unify(eng, val_out, val.deref())


def bi_set_attribute(goal: PsiTerm, eng) -> bool:
    """set_attribute(Term, Key, Value)."""
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    t = a1.deref()
    key_t = a2.deref()
    val = a3.deref()
    key = key_t.type.keyword.symbol if key_t.type and key_t.type.keyword else str(key_t.value) if key_t.value else ''
    mark = eng.trail.mark()
    eng.trail.trail_psi(t, 'attr_list')
    t.attr_list[key] = val
    return True


def bi_functor_of(goal: PsiTerm, eng) -> bool:
    """functor_of(Term, Type)."""
    a1, a2 = _get_two_args(goal)
    if a1 is None or a2 is None:
        return False
    t = a1.deref()
    defn = t.type
    if defn is None:
        return False
    result = PsiTerm(type=defn)
    return _unify(eng, a2, result)


def bi_type_of(goal: PsiTerm, eng) -> bool:
    """type_of(T, Type)."""
    return bi_functor_of(goal, eng)


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel used inside inference.py
# ─────────────────────────────────────────────────────────────────────────────
from wild_life.inference import _DEFRULES
_DEFRULES_SENTINEL = _DEFRULES


# ─────────────────────────────────────────────────────────────────────────────
# Registration helper
# ─────────────────────────────────────────────────────────────────────────────

def register_all(wl) -> None:
    """Register all built-in predicates on the runtime wl."""
    _reg = wl.new_built_in

    # I/O
    _reg('write', bi_write)
    _reg('writeq', bi_writeq)
    _reg('print', bi_print)
    _reg('nl', bi_nl)
    _reg('write_err', bi_write_err)
    _reg('writeln', bi_writeln)
    _reg('put', bi_put_char)
    _reg('put_char', bi_put_char)
    _reg('get_char', bi_get_char)
    _reg('get', bi_get_char)
    _reg('read', bi_read)
    _reg('read_term', bi_read_term)
    _reg('format', bi_format)
    _reg('nl_err', bi_nl_err)
    _reg('with_output_to', bi_with_output_to)

    # Arithmetic
    _reg('is', bi_is)
    _reg('=:=', bi_arith_eq)
    _reg('=\\=', bi_arith_ne)
    _reg('<', bi_arith_lt)
    _reg('=<', bi_arith_le)
    _reg('>', bi_arith_gt)
    _reg('>=', bi_arith_ge)

    # Unification
    _reg('=', bi_unify)
    _reg('\\=', bi_not_unify)
    _reg('==', bi_identical)
    _reg('\\==', bi_not_identical)
    _reg('compare', bi_compare)

    # Type testing
    _reg('var', bi_var)
    _reg('nonvar', bi_nonvar)
    _reg('atom', bi_atom)
    _reg('integer', bi_integer)
    _reg('float', bi_float_check)
    _reg('number', bi_number)
    _reg('string', bi_string)
    _reg('is_list', bi_is_list)
    _reg('compound', bi_compound)
    _reg('callable', bi_callable)
    _reg('ground', bi_ground)

    # Control
    _reg('true', bi_true)
    _reg('fail', bi_fail)
    _reg('false', bi_fail)
    _reg('not', bi_not)
    _reg('\\+', bi_not)
    _reg('call', bi_call)
    _reg('once', bi_once)
    _reg('findall', bi_findall)
    _reg('bagof', bi_findall)   # simplified
    _reg('setof', bi_findall)   # simplified
    _reg('aggregate_all', bi_aggregate_all)

    # Assert / retract
    _reg('assert', bi_assert)
    _reg('assertz', bi_assert)
    _reg('asserta', bi_asserta)
    _reg('retract', bi_retract)
    _reg('abolish', bi_abolish)
    _reg('listing', bi_listing)

    # Term manipulation
    _reg('functor', bi_functor)
    _reg('arg', bi_arg)
    _reg('=..', bi_univ)
    _reg('copy_term', bi_copy_term)
    _reg('numbervars', bi_numbervars)

    # String / atom
    _reg('atom_chars', bi_atom_chars)
    _reg('atom_string', bi_atom_string)
    _reg('atom_length', bi_atom_length)
    _reg('atom_concat', bi_atom_concat)
    _reg('number_chars', bi_number_chars)
    _reg('number_codes', bi_number_codes)
    _reg('char_code', bi_char_code)
    _reg('string_to_atom', bi_string_to_atom)
    _reg('term_to_atom', bi_term_to_atom)
    _reg('string_codes', bi_string_codes)
    _reg('string_length', bi_string_length)
    _reg('char_type', bi_char_type)

    # Lists
    _reg('length', bi_length)
    _reg('append', bi_append)
    _reg('member', bi_member)
    _reg('memberchk', bi_member)
    _reg('reverse', bi_reverse)
    _reg('msort', bi_msort)
    _reg('sort', bi_sort)
    _reg('last', bi_last)
    _reg('nth0', bi_nth)
    _reg('nth1', bi_nth)

    # Numbers
    _reg('succ', bi_succ)
    _reg('plus', bi_plus)
    _reg('between', bi_between)
    _reg('random', bi_rand)

    # System
    _reg('halt', bi_halt)
    _reg('abort', bi_abort)
    _reg('load', bi_load)
    _reg('op', bi_op)
    _reg('statistics', bi_statistics)
    _reg('current_prolog_flag', bi_current_prolog_flag)
    _reg('set_prolog_flag', bi_set_prolog_flag)

    # Type hierarchy
    _reg('sub_type', bi_sub_type)
    _reg('get_attribute', bi_get_attribute)
    _reg('set_attribute', bi_set_attribute)
    _reg('type_of', bi_type_of)
    _reg('functor_of', bi_functor_of)
