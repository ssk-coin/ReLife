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
    PsiTerm, Definition, GoalType, DefType, FACT, QUERY, ERROR,
    int_div as _int_div, NON_STRICT_TERM as _NST_SWAP,
    QUOTED_TRUE, REDUCED, feature_key_of
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
    t = eng.wl.make_number(v)
    t._is_computed = True  # mark as arithmetic-computed, not a parsed literal
    return t


def _make_int(eng, n: int) -> PsiTerm:
    return eng.wl.make_integer(n)


def _make_string(eng, s: str) -> PsiTerm:
    return eng.wl.make_string(s)


def _make_atom(eng, name: str) -> PsiTerm:
    return eng.wl.make_atom(name, eng.wl.user_module)


def _get_sym(t: PsiTerm) -> str:
    """Get the functor symbol of a term."""
    if t is None:
        return ''
    t = t.deref()
    return t.type.keyword.symbol if (t.type and t.type.keyword) else ''


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic type-constraint propagation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Binary and unary operators that return a real number.
_ARITH_OPS_SET = frozenset((
    '+', '-', '*', '/', '//', 'mod', '^',
    'max', 'min', 'abs', 'sqrt', 'sin', 'cos', 'tan',
    'exp', 'log', 'floor', 'ceiling', 'truncate', 'round',
    # Bitwise operators (also produce numeric results)
    '/\\', '\\/', 'xor', '>>', '<<',
    # Bitwise NOT (unary)
    '\\',
    # Time functions (0-ary arithmetic; always return a number)
    'cpu_time', 'real_time',
    # Global integer counter (0-ary; increments each evaluation)
    'genint',
    # Random integer draw (unary)
    'random',
))


def _is_complete_arith_expr(t: 'PsiTerm') -> bool:
    """Return True if t is a complete (non-curried) arithmetic expression.

    Curried arithmetic terms (binary op with only 1 arg) should be treated as
    regular compound terms, not as arithmetic constraints.
    Special case: '-' is both unary (FY) and binary (YFX). With one arg '1'
    it is valid unary negation. All other binary-only ops ('+', '*', '/', etc.)
    with only 1 arg are curried.
    """
    sym = t.type.keyword.symbol if t.type and t.type.keyword else ''
    if sym not in _ARITH_OPS_SET:
        return False
    # Nullary operators are always complete
    if sym in ('cpu_time', 'real_time', 'genint'):
        return True
    # Unary-only operators: need exactly '1' arg
    # `sign` is not among them: Wild Life leaves the name to the program, and
    # preparser.lf's grammar defines `sign(-1) --> [45], !` — a head that a
    # built-in would work out to -1 before the clause was ever filed.
    _unary_only = frozenset(('abs', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                              'exp', 'log', 'floor', 'ceiling', 'round', 'truncate',
                              'float', 'integer', 'msb', 'random', '\\'))
    if sym in _unary_only:
        return '1' in t.attr_list
    # '-' is both unary and binary: valid with 1 arg (unary) or 2 args (binary)
    if sym == '-':
        return '1' in t.attr_list
    # All other ops are binary-only: need BOTH '1' and '2'
    return '1' in t.attr_list and '2' in t.attr_list


def _mark_real_sort(var: 'PsiTerm', wl, eng) -> None:
    """Mark a free variable as constrained to sort real, with no pending constraints.

    Sets type=real and SORT_VAR flag.  Sets resid=[] (empty list, not None) to
    indicate 'involved in arithmetic but constraint dissolved/solved' — this
    suppresses the tilde in display (print_term treats resid=None as 'always ~'
    but resid=[] as 'no pending').

    Used when an arithmetic constraint was immediately solved (e.g. A=A+0 → A=A)
    so the variable is real-constrained but has no suspended residuation.
    """
    from wild_life.data_structures import SORT_VAR
    var = var.deref()
    if var.value is not None or var.attr_list:
        return  # Not a free variable
    if var.type is wl.top or var.type is None:
        if eng is not None:
            eng.trail.trail_psi(var, 'type')
        var.type = wl.real
    if not (var.flags & SORT_VAR):
        if eng is not None:
            eng.trail.trail_psi(var, 'flags')
        var.flags |= SORT_VAR
    # Set resid to empty list (not None) so print_term knows: "no pending constraints"
    # (resid=None means "pure sort annotation, always show ~").
    if var.resid is None:
        if eng is not None:
            eng.trail.trail_psi(var, 'resid')
        var.resid = []


def _mark_bool_sort(var: 'PsiTerm', wl, eng) -> None:
    """Mark a free variable as constrained to sort bool, with no pending constraints.

    Sets type=boolean and SORT_VAR flag.  Sets resid=[] (empty list, not None) to
    indicate 'constrained to bool but no pending residuation' — this suppresses
    the tilde in display (print_term treats resid=None as 'always ~' but resid=[]
    as 'no pending').

    Used when a boolean constraint was immediately solved (e.g. B and false → false
    short-circuits B) so the variable is bool-constrained but has no suspended goal.

    Concrete atoms (like false/true) are silently skipped — they are not variables.
    Atoms have a specific subtype (wl.false, wl.true) without the SORT_VAR flag,
    which distinguishes them from free variables constrained to sort bool.
    """
    from wild_life.data_structures import SORT_VAR
    var = var.deref()
    if var.value is not None or var.attr_list:
        return  # Not a free variable (has a numeric value or children)
    # Distinguish free variables from concrete atoms:
    # - Free untyped variable:      type=wl.top or type=None
    # - Free bool-sort variable:    type=wl.boolean AND SORT_VAR flag set
    # - Concrete atom (e.g. false): type=wl.false (subtype of bool), NO SORT_VAR
    # Only process actual free variables.
    is_free_var = (var.type is wl.top or var.type is None or
                   (var.type is wl.boolean and bool(var.flags & SORT_VAR)))
    if not is_free_var:
        return  # Concrete atom (true/false/other) — leave it unchanged
    if var.type is wl.top or var.type is None:
        if eng is not None:
            eng.trail.trail_psi(var, 'type')
        var.type = wl.boolean
    if not (var.flags & SORT_VAR):
        if eng is not None:
            eng.trail.trail_psi(var, 'flags')
        var.flags |= SORT_VAR
    # Set resid to empty list (not None) so print_term knows: "no pending constraints"
    # (resid=None means "pure sort annotation, always show ~").
    if var.resid is None:
        if eng is not None:
            eng.trail.trail_psi(var, 'resid')
        var.resid = []


def _remove_resid_for_goal(var: 'PsiTerm', goal_psi, eng) -> None:
    """Remove from var's resid list any entry whose goal.a is goal_psi.

    Used when a boolean constraint is resolved by partial evaluation during a
    re-fire: the pending Residuation entry is no longer needed and should be
    removed so the variable does not display as bool~.

    Trails the resid modification for backtracking safety.
    """
    var = var.deref()
    if not var.resid:
        return
    new_r = [r for r in var.resid
             if not (r.goal is not None and
                     getattr(r.goal, 'a', None) is goal_psi)]
    if len(new_r) != len(var.resid):
        if eng is not None:
            eng.trail.trail_psi(var, 'resid')
        var.resid = new_r if new_r else []


def _collect_arith_vars(t: 'PsiTerm', wl, result: list, seen: set) -> None:
    """Collect all unbound variables in an arithmetic expression.

    Collects both fresh top-sort variables and sort-constrained variables
    (e.g. type=real with SORT_VAR flag set by earlier arithmetic propagation).
    """
    from wild_life.data_structures import SORT_VAR
    if t is None:
        return
    t = t.deref()
    tid = id(t)
    if tid in seen:
        return
    seen.add(tid)
    # Unbound variable: top-sort, or sort-constrained (SORT_VAR flag), no attrs, no value
    is_free = not t.attr_list and t.value is None and t.coref is None
    # A bare `int` written into a term is an integer nobody has said which
    # one of yet — magic's grid is nine of them — so it is a variable an
    # equation can wait on, the same as `X:int` is.
    _is_num_sort = (t.type is not None and wl.real is not None
                    and t.type is not wl.top and t.type.is_subtype_of(wl.real))
    if is_free and (t.type is wl.top or t.type is None
                    or bool(t.flags & SORT_VAR) or _is_num_sort):
        if t not in result:
            result.append(t)
        return
    # Numeric literal — ground, no variables inside
    if t.value is not None:
        return
    sym = t.type.keyword.symbol if t.type and t.type.keyword else ''
    if sym in _ARITH_OPS_SET:
        for val in t.attr_list.values():
            _collect_arith_vars(val, wl, result, seen)


def _can_be_a_number(t: 'PsiTerm', wl) -> bool:
    """Whether t could turn out to be a number.

    A variable could, and so could anything already numeric.  A concrete name
    such as `a` could not, whatever else the program goes on to bind.
    """
    from wild_life.data_structures import SORT_VAR as _SV_num
    if t is None:
        return True
    t = t.deref()
    if t.value is not None:
        return bool(t.type is not None and t.type.is_subtype_of(wl.real))
    if t.attr_list:
        return True         # an expression still to be worked out
    if t.type is None or t.type is wl.top:
        return True         # a variable: still waiting to be something
    if t.flags & _SV_num:
        from wild_life.unification import types_compatible as _tc_num
        return _tc_num(t.type, wl.real)
    return t.type.is_subtype_of(wl.real)


def _attach_arith_resid(var: 'PsiTerm', wl, pending_goal, eng=None) -> None:
    """Constrain var to sort real and attach a pending residuated goal.

    Sets the SORT_VAR flag so the unifier continues to treat the variable as
    bindable (even though its type is no longer WL.top).  When the variable is
    later bound, _wakeup_resid fires the pending goal.

    If eng is provided, the resid list modification is trailed so that it is
    undone on backtracking (preventing accumulation of stale resid entries).
    """
    from wild_life.data_structures import Residuation, SORT_VAR
    var = var.deref()
    # Constrain to real sort only if still totally unconstrained
    if var.type is wl.top or var.type is None:
        var.type = wl.real
    # Mark as a sort-constrained variable so the unifier still binds it and
    # calls _wakeup_resid when it gets a value.
    var.flags |= SORT_VAR
    # Attach a single pending residuation for display (~).
    # Avoid duplicating the same goal object.
    if var.resid is None:
        if eng is not None:
            eng.trail.trail_psi(var, 'resid')  # trail: resid was None
        var.resid = [Residuation(goal=pending_goal)]
    else:
        # Don't attach the same goal twice
        for r in var.resid:
            if r.goal is pending_goal:
                return
        if eng is not None:
            eng.trail.trail_copy(var, 'resid')  # trail: save copy of list
        var.resid.append(Residuation(goal=pending_goal))


def _is_proper_bool_expr(t: 'PsiTerm') -> bool:
    """Return True if *t* is a well-formed boolean expression.

    Binary operators (and, or, xor) require BOTH positional attributes '1' and '2'.
    Unary operator (not) requires attribute '1'.
    Psi-terms that happen to use 'and'/'or' as a functor name but have the wrong
    arity (e.g. and(B) with only attribute '1') are NOT boolean expressions.
    """
    sym = _get_sym(t)
    if sym in ('and', 'or', 'xor'):
        return '1' in t.attr_list and '2' in t.attr_list
    if sym == 'not':
        return '1' in t.attr_list
    return False


def _is_settled_value(t: 'PsiTerm', origin) -> bool:
    """Whether t is an answer in itself, rather than something still waiting.

    A composition of named functions is: nothing in it is unknown, and none of
    it is the very call that produced it.  A global's stored `@ + 1` is not.

    The whole term is walked, however deep it runs: a list of the seven
    hundred characters of a file is as settled as a list of one, and a depth
    limit here would call the long one unsettled and leave the global that
    holds it standing for its own name.
    """
    if t is None:
        return False
    from wild_life.runtime import WL as _WL_sv
    _td = t.deref()
    if _td.value is None and not _td.attr_list and (
            _td.type is None or _td.type is _WL_sv.top):
        return False        # a bare variable: still waiting to be something
    seen: set = set()
    stack = [_td]
    while stack:
        n = stack.pop()
        if n is None:
            return False
        n = n.deref()
        if id(n) in seen:
            continue        # a cycle is as settled as it will get
        if n.type is origin:
            return False    # it stands for itself, so it says nothing
        # A sum still waiting on its variables is not a value: a global's
        # stored `@ + 1` says nothing yet.  A psi-term with variables in it
        # is a different matter — `stu` is worth `student(roommate =>
        # employee(representative => S), advisor => don(secretary => S))`,
        # and the S it shares is part of what it is worth.
        if n.attr_list and _get_sym(n) in _ARITH_OPS_SET:
            return False
        seen.add(id(n))
        stack.extend(n.attr_list.values())
    return True


def _term_reaches_itself(t: 'PsiTerm', _seen: frozenset = frozenset(),
                        _depth: int = 0) -> bool:
    """Whether following t's features leads back to t itself.

    A cycle somewhere below t is not one: matrix builds a grid of squares
    that point at each other in all four directions, and `term_size(C)` is
    still a call to work out rather than one that would call itself.
    """
    if t is None:
        return False
    root = t.deref()
    _walk_seen = {id(root)}

    def _walk(x, depth):
        if x is None or depth > 60:
            return False
        xd = x.deref()
        if xd is root and depth > 0:
            return True
        if id(xd) in _walk_seen and depth > 0:
            return False
        _walk_seen.add(id(xd))
        for ref in xd.attr_list.values():
            if _walk(ref, depth + 1):
                return True
        return False

    return _walk(root, 0)


# The comparisons whose value is a boolean.
# Built-ins whose value is true or false, whatever they are given.
_BOOL_VALUED_BUILTINS = frozenset((
    'has_feature', 'var', 'nonvar', 'is_function', 'is_predicate',
    'is_sort', 'is_number', 'is_value', 'not', 'is_persistent',
))


_BOOL_VALUED_COMPARISONS = frozenset((
    '>', '<', '>=', '=<', '=:=', '=\\=',
    ':=<', ':>=', ':<', ':>', ':==', ':\\==',
    '===', '\\===',
))


def _eval_bool_builtin(t, eng):
    """What a built-in that answers true or false comes to, or None.

    `has_feature(B,In,InB)` written as one side of an `and` is a question
    with an answer, and asking it binds InB: accumulators.lf reads an
    accumulator out of the context that way.
    """
    if t is None or eng is None:
        return None
    t = t.deref()
    if _get_sym(t) not in _BOOL_VALUED_BUILTINS or not t.attr_list:
        return None
    _r = _try_eval_string_func(t, eng)
    if _r is not None and _get_sym(_r.deref()) in ('true', 'false'):
        return _r
    return None


def _bool_operand_ok(t: 'PsiTerm', wl, _depth: int = 0) -> bool:
    """Whether a term can stand where a boolean is wanted.

    `not(B)` is a boolean whatever B turns out to be, but only if B can be one:
    `a = not(B)` and `A = not(b)` are both refused, because neither a nor b is
    a boolean and no narrowing makes them one.
    """
    from wild_life.data_structures import SORT_VAR as _SV_bo
    if t is None or _depth > 20:
        return False
    t = t.deref()
    if _is_proper_bool_expr(t):
        return all(_bool_operand_ok(_v, wl, _depth + 1)
                   for _v in t.attr_list.values())
    # A comparison answers a boolean, so it stands where one is wanted:
    # `not A :== residuation` is a question about A, not a complaint.
    if _get_sym(t) in _BOOL_VALUED_COMPARISONS and len(t.attr_list) == 2:
        return True
    # A built-in that answers true or false stands where a boolean is wanted
    # too: structures.lf asks `Ra :> Rb or has_feature(visited,B) or …`.
    if _get_sym(t) in _BOOL_VALUED_BUILTINS:
        return True
    if t.value is not None:
        return False        # a number or a string is not a boolean
    if t.type is None or t.type is wl.top:
        return True         # a variable can still become one
    if wl.boolean is None:
        return True
    if t.attr_list:
        return False
    from wild_life.unification import types_compatible as _tc_bo
    if t.flags & _SV_bo:
        return _tc_bo(t.type, wl.boolean)
    return t.type.is_subtype_of(wl.boolean)


def _collect_bool_free_vars(t: 'PsiTerm', wl, result: list, seen: set) -> None:
    """Collect unbound variables in a boolean expression (and, or, not, xor).

    Traverses the boolean expression tree; any free variable found is added to
    *result*.  Stops at ground terms (atoms, concrete values) and at variables
    that are already bound.

    Only recurses into 'and'/'or' sub-terms that have the correct arity for a
    boolean expression (both '1' and '2' args present).  A unary 'and(B)' is a
    psi-term constructor and is treated as a ground compound, not a bool expr.
    """
    from wild_life.data_structures import SORT_VAR
    if t is None:
        return
    t = t.deref()
    tid = id(t)
    if tid in seen:
        return
    seen.add(tid)
    is_free = not t.attr_list and t.value is None and t.coref is None
    if is_free and (t.type is wl.top or t.type is None or t.type is wl.boolean
                    or bool(t.flags & SORT_VAR)):
        if t not in result:
            result.append(t)
        return
    if t.value is not None:
        return  # ground numeric/string value
    # Only recurse into properly-formed boolean operator applications.
    if _is_proper_bool_expr(t):
        for val in t.attr_list.values():
            _collect_bool_free_vars(val, wl, result, seen)


def _attach_bool_resid(var: 'PsiTerm', wl, pending_goal, eng=None) -> None:
    """Constrain *var* to sort bool and attach a pending residuated goal.

    Mirrors _attach_arith_resid but uses wl.boolean instead of wl.real.
    When *var* is later bound, _wakeup_resid fires *pending_goal*.
    """
    from wild_life.data_structures import Residuation, SORT_VAR
    var = var.deref()
    if var.value is not None or var.attr_list:
        return  # not a free variable — skip
    if var.type is wl.top or var.type is None:
        if eng is not None:
            eng.trail.trail_psi(var, 'type')
        var.type = wl.boolean
    if not (var.flags & SORT_VAR):
        if eng is not None:
            eng.trail.trail_psi(var, 'flags')
        var.flags |= SORT_VAR
    if var.resid is None:
        if eng is not None:
            eng.trail.trail_psi(var, 'resid')
        var.resid = [Residuation(goal=pending_goal)]
    else:
        for r in var.resid:
            if r.goal is pending_goal:
                return
        if eng is not None:
            eng.trail.trail_copy(var, 'resid')
        var.resid.append(Residuation(goal=pending_goal))


def _var_in_expr(target: 'PsiTerm', expr: 'PsiTerm', visited: set) -> bool:
    """Return True if *target* (by identity after deref) appears in *expr*.

    Follows coref chains via deref() and recurses through attr_list children.
    Used to detect self-referential constraints (e.g. B=A-B → A appears in expr).
    """
    if expr is None:
        return False
    expr_d = expr.deref()
    if id(expr_d) == id(target):
        return True
    eid = id(expr)
    if eid in visited:
        return False
    visited.add(eid)
    for child in expr_d.attr_list.values():
        if _var_in_expr(target, child, visited):
            return True
    return False


def _simplify_arith(t: 'PsiTerm', eng) -> 'Optional[PsiTerm]':
    """Partial arithmetic simplification using identity/annihilator rules.

    Wild Life 1.02 applies all of these (when one operand is a known constant):
        0 + X  →  X        (left-zero for +)
        X + 0  →  X        (right-zero for +)
        X - 0  →  X        (right-zero for -)
        0 * X  →  0        (left-zero / annihilator for *)
        X * 0  →  0        (right-zero / annihilator for *)
        1 * X  →  X        (left-identity for *)
        X * 1  →  X        (right-identity for *)
        X / 1  →  X        (right-identity for /)
        X // 1 →  X        (right-identity for //)

    Returns a PsiTerm on success, None if no simplification applies.
    """
    if t is None:
        return None
    t = t.deref()
    sym = t.type.keyword.symbol if t.type and t.type.keyword else ''
    if sym not in _ARITH_OPS_SET:
        return None

    arg1, arg2 = _get_two_args(t)

    ok1, v1 = _eval_arith(arg1, eng) if arg1 else (False, 0.0)
    ok2, v2 = _eval_arith(arg2, eng) if arg2 else (False, 0.0)

    # Both fully evaluable — let the normal path handle it
    if ok1 and ok2:
        return None

    wl = eng.wl

    if sym == '+':
        if ok1 and v1 == 0.0 and arg2 is not None:
            return arg2.deref()           # 0 + X = X
        if ok2 and v2 == 0.0 and arg1 is not None:
            return arg1.deref()           # X + 0 = X
    elif sym == '-':
        if ok2 and v2 == 0.0 and arg1 is not None:
            return arg1.deref()           # X - 0 = X
        # X - X = 0 when both sides dereference to the same node
        if arg1 is not None and arg2 is not None:
            if id(arg1.deref()) == id(arg2.deref()):
                return wl.make_integer(0) # X - X = 0
    elif sym == '*':
        if ok1 and v1 == 0.0:
            return wl.make_integer(0)     # 0 * X = 0
        if ok2 and v2 == 0.0:
            return wl.make_integer(0)     # X * 0 = 0
        if ok1 and v1 == 1.0 and arg2 is not None:
            return arg2.deref()           # 1 * X = X
        if ok2 and v2 == 1.0 and arg1 is not None:
            return arg1.deref()           # X * 1 = X
    elif sym in ('/', '//'):
        # Dividing by one leaves the term as it was, and that is the whole
        # of it: `A = B//1` answers `B = A` rather than waiting on B, and
        # what B turns out to be is not asked to be a whole number.
        if ok2 and v2 == 1.0 and arg1 is not None:
            return arg1.deref()           # X // 1 = X

    return None


def _eval_arith_comparison(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """What an arithmetic comparison comes to, when both sides are numbers.

    A comparison answers true or false, so it stands where a boolean is
    wanted: strleq writes `or(C1 < C2, and(C1 =:= C2, …))`, and the `or`
    can only be worked out once the two comparisons have been.
    """
    if t is None:
        return None
    t = t.deref()
    sym = _get_sym(t)
    if sym not in ('>', '<', '>=', '=<', '=:=', '=\\='):
        return None
    a1, a2 = _get_two_args(t)
    if a1 is None or a2 is None:
        return None
    ok1, v1 = _eval_arith(a1, eng)
    if not ok1:
        return None
    ok2, v2 = _eval_arith(a2, eng)
    if not ok2:
        return None
    held = {'>': v1 > v2, '<': v1 < v2, '>=': v1 >= v2, '=<': v1 <= v2,
            '=:=': v1 == v2, '=\\=': v1 != v2}[sym]
    return _make_atom(eng, 'true' if held else 'false')


def _bool_operand(t: PsiTerm, eng) -> PsiTerm:
    """An operand of a boolean operator, read for what it stands for.

    A `persistent` name stands for what was last written into it wherever it
    is read, arguments included, which is what makes term_expansion.lf's
    `load_option <<- assert_rules or expand2file` a question about two stored
    booleans rather than about two bare names.
    """
    if t is None:
        return t
    return _stored_side(t, eng) if eng is not None else t.deref()


def _try_eval_bool(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """Try to evaluate a boolean function application.

    Returns a reduced PsiTerm (true or false atom) or None if cannot reduce.
    Handles: and, or, not, xor applied to concrete bool constants.
    """
    if t is None:
        return None
    t = t.deref()
    sym = _get_sym(t)

    if sym == 'and':
        a1, a2 = _get_two_args(t)
        if a1 is None or a2 is None:
            return None
        # Recursively evaluate args
        a1 = _bool_operand(a1, eng)
        a2 = _bool_operand(a2, eng)
        a1 = (_try_eval_bool(a1, eng) or _eval_arith_comparison(a1, eng)
              or _eval_sort_comparison(a1, eng)
              or _eval_bool_builtin(a1, eng) or a1.deref())
        a2 = (_try_eval_bool(a2, eng) or _eval_arith_comparison(a2, eng)
              or _eval_sort_comparison(a2, eng)
              or _eval_bool_builtin(a2, eng) or a2.deref())
        s1, s2 = _get_sym(a1), _get_sym(a2)
        if s1 == 'false' or s2 == 'false':
            return _make_atom(eng, 'false')
        if s1 == 'true' and s2 == 'true':
            return _make_atom(eng, 'true')
        # Partial evaluation: one concrete arg
        if s1 == 'true':
            return a2   # true and X = X
        if s2 == 'true':
            return a1   # X and true = X
        # Idempotent: X and X = X (same variable by identity)
        if id(a1.deref()) == id(a2.deref()):
            return a1
        return None

    elif sym == 'or':
        a1, a2 = _get_two_args(t)
        if a1 is None or a2 is None:
            return None
        a1 = _bool_operand(a1, eng)
        a2 = _bool_operand(a2, eng)
        a1 = (_try_eval_bool(a1, eng) or _eval_arith_comparison(a1, eng)
              or _eval_sort_comparison(a1, eng)
              or _eval_bool_builtin(a1, eng) or a1.deref())
        a2 = (_try_eval_bool(a2, eng) or _eval_arith_comparison(a2, eng)
              or _eval_sort_comparison(a2, eng)
              or _eval_bool_builtin(a2, eng) or a2.deref())
        s1, s2 = _get_sym(a1), _get_sym(a2)
        if s1 == 'true' or s2 == 'true':
            return _make_atom(eng, 'true')
        if s1 == 'false' and s2 == 'false':
            return _make_atom(eng, 'false')
        # Partial evaluation: one concrete arg
        if s1 == 'false':
            return a2   # false or X = X
        if s2 == 'false':
            return a1   # X or false = X
        # Idempotent: X or X = X
        if id(a1.deref()) == id(a2.deref()):
            return a1
        return None

    elif sym == 'not':
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = _bool_operand(a1, eng)
        a1 = (_try_eval_bool(a1.deref(), eng)
              or _eval_arith_comparison(a1, eng)
              or _eval_sort_comparison(a1, eng)
              or _eval_bool_builtin(a1, eng) or a1.deref())
        s1 = _get_sym(a1)
        if s1 == 'true':
            return _make_atom(eng, 'false')
        if s1 == 'false':
            return _make_atom(eng, 'true')
        return None

    elif sym == 'xor':
        a1, a2 = _get_two_args(t)
        if a1 is None or a2 is None:
            return None
        a1 = _bool_operand(a1, eng)
        a2 = _bool_operand(a2, eng)
        a1 = (_try_eval_bool(a1, eng) or _eval_arith_comparison(a1, eng)
              or _eval_sort_comparison(a1, eng)
              or _eval_bool_builtin(a1, eng) or a1.deref())
        a2 = (_try_eval_bool(a2, eng) or _eval_arith_comparison(a2, eng)
              or _eval_sort_comparison(a2, eng)
              or _eval_bool_builtin(a2, eng) or a2.deref())
        s1, s2 = _get_sym(a1), _get_sym(a2)
        if s1 in ('true', 'false') and s2 in ('true', 'false'):
            result = (s1 == 'true') ^ (s2 == 'true')
            return _make_atom(eng, 'true' if result else 'false')
        # Partial evaluation: 'false' is the identity for xor (false xor X = X)
        if s1 == 'false':
            return a2   # false xor X = X
        if s2 == 'false':
            return a1   # X xor false = X
        # Idempotent: X xor X = false (same variable by identity)
        if id(a1.deref()) == id(a2.deref()):
            return _make_atom(eng, 'false')
        return None

    return None


def _try_eval_arith_to_term(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """Try arithmetic evaluation, returning a PsiTerm or None.

    When a compound arithmetic expression (e.g. 1+2) is evaluated to a number,
    the result is memoized back into the expression term via coref so that future
    dereferences of any variable pointing to the expression yield the concrete number.
    This mimics C Wild Life's in-place evaluation: after a strict predicate evaluates
    an argument expression, the evaluated result propagates back through the variable
    chain so the binding display shows the computed value (e.g. X = 3. after my_write2
    evaluated X's binding 1+2 → 3).
    The coref update is trailed so backtracking correctly undoes it.
    """
    ok, v = _eval_arith(t, eng)
    if not ok:
        return None
    # Only return if the original term was NOT already a number
    # (to avoid infinite recursion)
    if t is None:
        return None
    t_orig = t  # save original for memoization
    t = t.deref()
    if t.value is not None and not (t.type and t.type.keyword and
                                    t.type.keyword.symbol in ('+','-','*','/','//',
                                                               'mod','^','max','min',
                                                               'abs','sqrt','sin','cos','tan',
                                                               'exp','log','floor','ceiling')):
        return None  # already a number, no evaluation needed
    result = _make_number(eng, v)
    # The number an expression comes to is a number the program has just been
    # handed, so the sort's delay rules run on it once, here: `A = 2+2` owes
    # `:: I:int | …` its 4.  The mark keeps a later unification from running
    # them a second time.
    result._delay_fired = True
    _fire_here = (eng is not None and getattr(eng, 'unifier', None) is not None
                  and eng.wl is not None and eng.wl.delay_rules
                  and result.type is not None
                  and _get_sym(t) != '*'
                  and not getattr(eng, '_in_fire_delay', False))
    # Memoize the result back into the compound arithmetic term (t) via coref.
    # This propagates the evaluated value through the variable chain:
    # after evaluation, any variable that pointed to this expression will deref to
    # the concrete number.  We trail the old coref so backtracking can undo this.
    if eng is not None and t.coref is None and t.value is None and t.attr_list:
        eng.trail.trail_psi(t, 'coref')
        t.coref = result
    if _fire_here:
        eng.unifier._fire_delay_rules(result, result.type)
    return result


def _normalize_arith_in_term(t: PsiTerm, eng, _seen=None) -> PsiTerm:
    """Return a copy of t with arithmetic sub-expressions evaluated.

    Used by assert/asserta so that storing ``mynum(N+1)`` where N=31
    stores ``mynum(32)`` (an integer) rather than the expression tree.
    Avoids infinite loops on cyclic terms via the _seen set.
    """
    if _seen is None:
        _seen = set()
    t = t.deref()
    tid = id(t)
    if tid in _seen:
        return t
    _seen.add(tid)

    # If the whole term is an arithmetic expression, evaluate it.  An
    # expression a tag names — the `1+2` of `X:(1+2)` — is not one to work
    # out: assert stores what was written, and only a strict call asks the
    # expression for its value.
    from wild_life.data_structures import NON_STRICT_TERM as _NST_NORM
    if not (t.flags & _NST_NORM):
        arith = _try_eval_arith_to_term(t, eng)
        if arith is not None:
            return arith

    # Otherwise, walk attrs and normalize each child
    if not t.attr_list:
        return t
    # Build a shallow copy of the compound term with normalized children
    new_t = PsiTerm()
    new_t.type = t.type
    new_t.value = t.value
    new_t.flags = t.flags
    new_t.status = t.status
    for k, v in t.attr_list.items():
        new_t.attr_list[k] = _normalize_arith_in_term(v, eng, _seen)
    return new_t


def _term_to_display_string(t: PsiTerm, eng) -> str:
    """Convert a psi-term to its display string (like write/1 would produce)."""
    import io
    from wild_life.print_term import write_term
    buf = io.StringIO()
    write_term(t, outfile=buf, wl=eng.wl, quoted=False)
    return buf.getvalue()


def _is_list_term(t: PsiTerm, eng) -> bool:
    """Whether t is a list — a cons cell or the empty list.

    `nil` and `cons` are separate sorts under `list`, so a check against cons
    alone leaves out [], and `append([],L)` or `length([])` would not reduce.
    """
    if t is None or t.type is None:
        return False
    wl = eng.wl
    if wl.nil is not None and t.type.is_subtype_of(wl.nil):
        return True
    if wl.alist is None or not t.type.is_subtype_of(wl.alist):
        return False
    # A `list` that is still a variable is not a list yet: it stands for
    # whichever list it comes to be, and reading it as the empty one is how
    # `append(X:list, [])` lost what X was going to hold.
    return bool(t.attr_list) or t.value is not None


def _proper_list_elems(t: PsiTerm, eng) -> Optional[list]:
    """The elements of a list that really ends in [], or None.

    `[H|append(X,[])]` is a list still being worked out, and reading it as
    `[H]` is how insforet lost the forest it was inserting.
    """
    wl = eng.wl
    items: list = []
    cur = t.deref()
    seen: set = set()
    while True:
        if cur.type is None:
            return None
        if wl.nil is not None and cur.type.is_subtype_of(wl.nil):
            return items
        if wl.alist is None or not cur.type.is_subtype_of(wl.alist):
            return None
        if id(cur) in seen:
            return None
        seen.add(id(cur))
        h = cur.attr_list.get('1')
        t2 = cur.attr_list.get('2')
        if h is None or t2 is None:
            return None
        items.append(h.deref())
        cur = t2.deref()


def _try_eval_string_func(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """Try to evaluate string built-in functions (psi2str, str2psi, strcon).

    Returns evaluated PsiTerm or None if not applicable.
    """
    if t is None:
        return None
    t = t.deref()
    # A global variable name stands for its cell.
    cell = _global_cell(t, eng)
    if cell is None:
        cell = _persistent_cell(t, eng)
    if cell is not None:
        return cell
    # An argument written as a global name is there for what its cell holds:
    # std_expander asks `root_sort(traverse_method)` for the sort of the
    # method it was handed, not for the sort of the name.  The reading is done
    # through a stand-in so that the term the caller holds is left as it is.
    if t.attr_list and eng is not None and eng.wl.global_defs:
        _sub = None
        for _gk, _gv in t.attr_list.items():
            _gc = _global_cell(_gv.deref(), eng)
            if _gc is None:
                continue
            if _sub is None:
                _sub = PsiTerm(type_def=t.type, value=t.value)
                _sub.flags = t.flags
                _sub.attr_list = dict(t.attr_list)
            _sub.attr_list[_gk] = _gc
        if _sub is not None:
            t = _sub
    sym = _get_sym(t)

    if sym == 'psi2str':
        # psi2str(T) -> string representation of T
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = a1.deref()
        # What is written is the term's value, not the call that stands for
        # it: `psi2str(chr(116))` reads "t", not "chr(116)".
        _a1_ev = _try_eval_any_func(a1, eng)
        if _a1_ev is not None:
            a1 = _a1_ev.deref()
        s = _term_to_display_string(a1, eng)
        return _make_string(eng, s)

    elif sym == 'current_module':
        # current_module -> the name of the module being read, as a string.
        # std_expander.lf builds the name of the predicate it generates with
        # `str2psi(strcon(psi2str(Name),"_traverse"), current_module)`, so a
        # name left unanswered here files that predicate in whatever module
        # the call happens to run in rather than the one that asked for it.
        if t.attr_list or eng is None:
            return None
        _cm = eng.wl.current_module
        if _cm is None:
            return None
        return _make_string(eng, _cm.module_name)

    elif sym == 'str2psi':
        # str2psi(S[, M]) -> atom from string S, named in module M
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = a1.deref()
        # Reduce a nested string function first, as strcon does, so that
        # str2psi(strcon("a","1")) reads the string and not the call.
        a1e = _try_eval_string_func(a1, eng)
        if a1e is not None:
            a1 = a1e.deref()
        if a1.type and a1.type is eng.wl.quoted_string and a1.value is not None:
            name = str(a1.value)
        elif a1.type and a1.type.keyword:
            name = a1.type.keyword.symbol
        else:
            name = _term_to_display_string(a1, eng)
        # The second argument says which module the name belongs to: the
        # reader supplies the one the call was written in, and a program may
        # name another.
        _mod = None
        _a2 = t.attr_list.get('2')
        if _a2 is not None and eng is not None:
            _a2d = _a2.deref()
            _a2e = _try_eval_string_func(_a2d, eng)
            if _a2e is not None:
                _a2d = _a2e.deref()
            _mname = None
            if (_a2d.type is not None and _a2d.type is eng.wl.quoted_string
                    and _a2d.value is not None):
                _mname = str(_a2d.value)
            if _mname:
                _mod = eng.wl.module_table.get(_mname)
        if _mod is not None:
            return eng.wl.make_atom(name, _mod)
        return _make_atom(eng, name)

    elif sym == 'strcon':
        # strcon(A, B) -> concatenation of strings A and B
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        if a1 is None or a2 is None:
            return None
        a1, a2 = a1.deref(), a2.deref()
        # Reduce what the arguments stand for FIRST (before checking
        # is_string), so that `strcon("as", strcon(F,F))` is worked out once F
        # is bound, and `strcon(charac(116), Z)` reads what charac answers.
        a1e = _try_eval_any_func(a1, eng)
        if a1e is not None:
            a1 = a1e.deref()
        a2e = _try_eval_any_func(a2, eng)
        if a2e is not None:
            a2 = a2e.deref()
        # Only evaluate when BOTH arguments are now concrete strings
        if eng is not None:
            wl = eng.wl
            def _is_string(x):
                return (x.value is not None and x.type is not None
                        and x.type.is_subtype_of(wl.quoted_string))
            if not (_is_string(a1) and _is_string(a2)):
                return None
        s1 = str(a1.value) if (a1.value is not None) else ''
        s2 = str(a2.value) if (a2.value is not None) else ''
        return _make_string(eng, s1 + s2)

    elif sym == 'makestr':
        # makestr(T) -> string representation of term T (compact, single-line).
        # Unbound variables are represented as "@" (C Wild Life convention).
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        if eng is None:
            return None
        a1 = a1.deref()
        if _term_is_unbound(a1, eng):
            return _make_string(eng, '@')
        import io
        from wild_life.print_term import write_term
        buf = io.StringIO()
        # Use a very large max_col to suppress line-wrapping (C Wild Life produces
        # single-line strings for makestr).
        write_term(a1, outfile=buf, wl=eng.wl, quoted=False, max_col=1_000_000)
        return _make_string(eng, buf.getvalue())

    elif sym == 'strlen':
        # strlen(String) -> integer length of String
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1d = a1.deref()
        if eng is None or a1d.value is None:
            return None
        wl = eng.wl
        if a1d.type is None or not a1d.type.is_subtype_of(wl.quoted_string):
            return None
        return wl.make_integer(len(str(a1d.value)))

    elif sym == 'substr':
        # substr(String, Start, Length) -> substring (1-indexed, returns "" if out of range)
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        a3 = t.attr_list.get('3')
        if a1 is None or a2 is None or a3 is None:
            return None
        a1d = a1.deref()
        # Only evaluate when String is a concrete string
        if eng is None or a1d.value is None:
            return None
        wl = eng.wl
        if a1d.type is None or not a1d.type.is_subtype_of(wl.quoted_string):
            return None
        s = str(a1d.value)
        # Evaluate start/length via arithmetic
        ok2, start_f = _eval_arith(a2.deref(), eng)
        ok3, length_f = _eval_arith(a3.deref(), eng)
        if not ok2 or not ok3:
            return None
        start = int(start_f) - 1  # convert 1-indexed to 0-indexed
        length = int(length_f)
        if start < 0:
            start = 0
        result = s[start:start + length] if start < len(s) else ''
        return _make_string(eng, result)

    elif sym == 'project' and len(t.attr_list) == 2:
        # project(A, B) is the older way of writing B.A: the feature A holds
        # on B.  built_ins.lf keeps it as `project(A,B) -> B.A.`
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        if a1 is None or a2 is None or eng is None:
            return None
        _dot_defn = eng.wl.update_symbol(eng.wl.syntax_module, '.')
        _dot = PsiTerm(type_def=_dot_defn)
        _dot.attr_list = {'1': a2, '2': a1}
        return _resolve_dot_feat(_dot, eng, create=False)

    elif sym == 'combined_name':
        # combined_name(T) -> the atom that names T's sort with the module it
        # belongs to, `user#pp`.  The library files file a table under it, so
        # that two modules' `pp` are two entries.
        a1 = t.attr_list.get('1')
        if a1 is None or eng is None:
            return None
        a1 = a1.deref()
        defn = a1.type
        if defn is None or defn.keyword is None:
            return None
        return eng.wl.make_atom(defn.keyword.combined_name, eng.wl.user_module)

    elif sym == 'root_sort' or sym == 'sort':
        # root_sort(T) -> the root sort of T.
        # For numeric/string atoms, the root sort is the value itself.
        # For compound terms, it is the functor name as an atom.
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        # A call written where the term goes is asked for the term it makes:
        # accumulators.lf builds a context out of `strip(A) & @(AIn,Out.A)`,
        # and what that is a root sort of is the `@` strip answers.
        if (a1.deref() is a1 and a1.attr_list and eng is not None
                and a1.type is not None
                and a1.type._builtin_func is not None):
            if _is_strip_func(a1):
                _rs_ev = _eval_strip_or_copy_func(a1, eng, False)
            elif _is_copy_pointer_func(a1):
                _rs_ev = _eval_strip_or_copy_func(a1, eng, True)
            else:
                _rs_ev = _try_eval_string_func(a1, eng)
            if _rs_ev is not None and _rs_ev.deref() is not a1:
                a1 = _rs_ev
        a1 = a1.deref()
        defn = a1.type
        if defn is None or defn.keyword is None:
            return None
        # Unwrap backtick-quoted atoms: `foo → root sort is foo
        if defn.keyword.symbol == '`':
            inner = a1.attr_list.get('1')
            if inner is not None:
                a1 = inner.deref()
                defn = a1.type
                if defn is None or defn.keyword is None:
                    return None
        # For concrete numeric/string values the root sort is the value itself
        if a1.value is not None and eng is not None:
            wl = eng.wl
            if defn.is_subtype_of(wl.integer):
                return wl.make_integer(int(a1.value))
            if defn.is_subtype_of(wl.real):
                return wl.make_number(float(a1.value))
            if defn.is_subtype_of(wl.quoted_string):
                return _make_string(eng, str(a1.value))
        # The sort itself, not a name read again: c_root_sort hands back a
        # term whose type is the term's own, and building a fresh atom out
        # of the symbol looks it up in whatever module is current, which
        # turns accumulators#gram_init into an undefined user#gram_init.
        return PsiTerm(type_def=defn)

    elif sym == 'getenv':
        # getenv(Name) is what the environment says Name is worth.  A name the
        # environment says nothing about has no value, and the call fails —
        # which is how `D = getenv("SLDIR"), chdir(D)` leaves the directory
        # alone when SLDIR is not set.
        a1 = t.attr_list.get('1')
        if a1 is None or eng is None:
            return None
        _n = _get_str_val(a1.deref(), eng)
        if _n is None:
            return None
        import os as _os_ge
        _v = _os_ge.environ.get(_n)
        return _make_string(eng, _v) if _v is not None else None

    elif sym == 'ops' and not t.attr_list:
        # `ops` is the operator table as a list of op(Precedence, Type, Name),
        # which is how preparser.lf's expression grammar asks what the current
        # operators are.
        if eng is None:
            return None
        wl_ops = eng.wl
        _op_defn = wl_ops.update_symbol(wl_ops.syntax_module, 'op')
        _elems = []
        for _p, _ty, _nm in getattr(wl_ops, '_enumerable_ops', []):
            _e = PsiTerm(type_def=_op_defn)
            _e.attr_list = {'1': wl_ops.make_integer(_p),
                            '2': wl_ops.make_atom(_ty),
                            '3': wl_ops.make_atom(_nm)}
            _elems.append(_e)
        return wl_ops.make_list(_elems)

    elif sym == 'call_once':
        # `call_once(G)` is a boolean function: it proves G once and answers
        # whether it held, keeping what the proof bound.  built_ins.lf spells
        # out the same thing for the interpreters that lack it as a built-in:
        # `call_once(G) -> T | (evalin(G), T = true ; T = false), !.`
        a1 = t.attr_list.get('1')
        if a1 is None or eng is None:
            return None
        from wild_life.inference import prove_cond as _pc_co
        _mark_co = eng.trail.mark()
        if _pc_co(a1.deref(), eng):
            return _make_atom(eng, 'true')
        eng.trail.undo_to(_mark_co)
        return _make_atom(eng, 'false')

    elif sym in ('eval', 'evalin'):
        # eval(T) is T's value, and a term with no value of its own is that
        # value: `A = eval(X:a(X))` answers the very term X stands for.
        # evalin asks the same of a term a non-strict call was given, so it
        # reads through the quote that kept the call from being worked out.
        a1 = t.attr_list.get('1')
        if a1 is None or eng is None:
            return None
        arg = _strip_backtick(a1.deref())
        if sym == 'evalin' and (arg.flags & QUOTED_TRUE):
            # Asking for the value spends the quote: a term a non-strict call
            # was handed is written as it stands once, and is worth its value
            # from then on.  comp_struct writes the same `X == Y` three times
            # over — once as the comparison, then twice as what it comes to —
            # so the quote goes for good rather than coming back on
            # backtracking.
            arg.flags &= ~QUOTED_TRUE
        _ok_ev, _v_ev = _eval_arith(arg, eng)
        if _ok_ev:
            return _make_number(eng, _v_ev)
        # A sort comparison asked for its value answers true or false:
        # libstruct hands `a :== c` to a non-strict call and reads it with
        # evalin.
        _sc_ev = _eval_sort_comparison(arg, eng)
        if _sc_ev is not None:
            return _sc_ev
        if _is_user_function(arg):
            # Asked for the value, not for the term to become it: eval reduces
            # a copy, so `A = eval(X:f(X))` answers 1 and leaves X the call.
            _mark_ev = eng.trail.mark()
            try:
                _red_ev = _eval_user_func_sync(copy_term(arg, {}), eng, 0)
                if _red_ev is None:
                    # A call still waiting for arguments has no value but
                    # itself: `eval(f(1))` of `f(X,Y) -> [X,Y]` is f(1), and
                    # a copy of it, so asking twice gives two of them.
                    return copy_term(arg, {})
                _red_d_ev = _red_ev.deref()
                # A boolean operator the rule handed back is still a question
                # to answer: `X \== Y -> not(X == Y)` is false once the `==`
                # inside it has said true.
                if (_get_sym(_red_d_ev) in ('and', 'or', 'not', 'xor')
                        and _red_d_ev.attr_list):
                    _eval_embedded_user_funcs(_red_d_ev, eng, 0, set())
                    _post_ev = _try_eval_any_func(_red_d_ev, eng)
                    if _post_ev is not None:
                        _red_d_ev = _post_ev.deref()
                return copy_term(_red_d_ev, {})
            finally:
                eng.trail.undo_to(_mark_ev)
        _inner_ev = _try_eval_string_func(arg, eng)
        if _inner_ev is not None:
            return _inner_ev
        # eval hands back a value, not the term it read: `A = eval(X:a(X))`
        # gives A a cyclic term of its own rather than making A and X one.
        return copy_term(arg, {})

    elif sym in ('var', 'nonvar', 'is_function', 'is_predicate', 'is_sort'):
        # These read as functions too: `A = var(_)` answers true, not var(@).
        if eng is None or '1' not in t.attr_list or t.type is None:
            return None
        _pred = t.type._builtin_func
        if _pred is None:
            return None
        return eng.wl.make_atom('true' if _pred(t, eng) else 'false')

    elif sym == 'has_feature':
        # has_feature(F, T) also reads as a function: `A = has_feature(@,S)`
        # answers false while S has no such feature and true once it does.
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        if a1 is None or a2 is None or eng is None:
            return None
        fname = _feature_name_of(_feature_arg_term(a1, eng), eng.wl)
        term = a2.deref()
        _term_cell = _global_cell(term, eng) or _persistent_cell(term, eng)
        if _term_cell is not None:
            term = _term_cell.deref()
        holds = fname is not None and fname in term.attr_list
        if holds and '3' in t.attr_list:
            _v3 = t.attr_list['3'].deref()
            if not _unify(eng, _v3, term.attr_list[fname].deref()):
                holds = False
        return eng.wl.make_atom('true' if holds else 'false')

    elif sym == 'parents':
        # parents(Sort) -> list of immediate parent sorts.  A concrete value
        # answers with the sort it is: `parents(23)` is [int].
        a1 = t.attr_list.get('1')
        if a1 is None or eng is None:
            return None
        a1 = a1.deref()
        defn = a1.type
        if defn is None or defn.keyword is None:
            return None
        wl = eng.wl
        if a1.value is not None:
            return wl.make_list([wl.make_atom(defn.keyword.symbol, wl.bi_module)])
        parent_atoms = []
        for parent_defn in getattr(defn, 'parents', []):
            if parent_defn is None or parent_defn.keyword is None:
                continue
            parent_atoms.append(wl.make_atom(parent_defn.keyword.symbol,
                                             wl.bi_module))
        return wl.make_list(parent_atoms)

    elif sym == 'children':
        # children(Sort) -> list of immediate child sorts.
        # For concrete numeric/string values (atoms with a .value), return [].
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = a1.deref()
        # Concrete values (numbers, strings) have no child sorts.
        if a1.value is not None:
            if eng is None:
                return None
            return eng.wl.make_list([])
        defn = a1.type
        if defn is None or eng is None:
            return None
        wl = eng.wl
        # Use defn.children directly to avoid duplicates from symbol aliases.
        child_atoms = []
        for child_defn in getattr(defn, 'children', []):
            if child_defn.keyword is None:
                continue
            child_atoms.append(wl.make_atom(child_defn.keyword.symbol, wl.bi_module))
        return wl.make_list(child_atoms)

    elif sym == 'local_time':
        # local_time is a 0-ary built-in sort with attributes:
        # day, hour, minute, month, second, weekday, year
        import datetime as _datetime
        _now = _datetime.datetime.now()
        wl = eng.wl
        lt = PsiTerm()
        lt.type = t.type
        lt.status = 4
        # Insert in alphabetical order (preserved by Python dict)
        lt.attr_list = {
            'day':     wl.make_integer(_now.day),
            'hour':    wl.make_integer(_now.hour),
            'minute':  wl.make_integer(_now.minute),
            'month':   wl.make_integer(_now.month),
            'second':  wl.make_integer(_now.second),
            'weekday': wl.make_integer(_now.weekday()),
            'year':    wl.make_integer(_now.year),
        }
        return lt

    elif sym == 'length' and len(t.attr_list) == 1:
        # length(L) -> number of elements, the functional form of length/2.
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        # What the argument stands for is what is measured: `length(
        # factors(I))` counts the factors, not the call that finds them.
        _a1_len = a1.deref()
        _ev_len = _try_eval_any_func(_a1_len, eng)
        if _ev_len is not None:
            _a1_len = _ev_len.deref()
        _elems_len = _proper_list_elems(_a1_len, eng)
        if _elems_len is None:
            return None
        return eng.wl.make_integer(len(_elems_len))

    elif sym == 'append' and len(t.attr_list) == 2:
        # append(L1, L2) -> L1 with L2 appended, the functional form of
        # append/3.  L2 becomes the tail as-is, so the result shares it, and
        # the elements of L1 are shared rather than copied.
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        if a1 is None or a2 is None:
            return None
        _a1_app = a1.deref()
        _ev_app = _try_eval_any_func(_a1_app, eng)
        if _ev_app is not None:
            _a1_app = _ev_app.deref()
        _elems_app = _proper_list_elems(_a1_app, eng)
        if _elems_app is None:
            return None
        result = a2.deref()
        for item in reversed(_elems_app):
            result = eng.wl.make_cons(item, result)
        return result

    elif sym in ('features', 'feature_values'):
        # features(T[, MOD]) -> list of attribute labels (sorted: positional first, then named)
        # If MOD is given, only includes features visible from module MOD,
        # and each named feature is returned as a MOD-qualified atom.
        # feature_values names the same features and answers what they hold,
        # as make_feature_list does with its `val` flag.
        _want_values = (sym == 'feature_values')
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = a1.deref()
        # A backtick holds a term as it is written, so `` `f `` is the term f
        # and has the features f has — none of them the backtick's own.
        if (a1.type is not None and a1.type.keyword is not None
                and a1.type.keyword.symbol == '`'):
            _inner_bq = a1.attr_list.get('1')
            if _inner_bq is not None:
                a1 = _inner_bq.deref()
        # Try to evaluate a1 first (e.g. local_time built-in)
        _a1_ev = _try_eval_string_func(a1, eng)
        if _a1_ev is not None:
            a1 = _a1_ev
        a2 = t.attr_list.get('2')  # optional module name argument
        wl = eng.wl

        # Determine context module (from 2nd argument if given)
        ctx_mod = None
        if a2 is not None:
            _mod_t = a2.deref()
            _mod_name = None
            if _mod_t.value is not None and isinstance(_mod_t.value, str):
                _mod_name = _mod_t.value
            elif _mod_t.type and _mod_t.type.keyword:
                _mod_name = _mod_t.type.keyword.symbol
            if _mod_name:
                ctx_mod = wl.find_module(_mod_name)
        else:
            # Asked without a module, a term's features are the ones the
            # module doing the asking can see, written as that module
            # writes them.
            ctx_mod = wl.current_module

        # Get the term's defining module (for feature visibility checks)
        term_type_mod = None
        if a1.type and a1.type.keyword and a1.type.keyword.module:
            term_type_mod = a1.type.keyword.module

        # Feature order — the same order the printer lays attributes out in,
        # so features(@('' => A,0 => 22)) is ['',0] rather than [0,''].
        from wild_life.data_structures import featcmp_key as _featcmp_key
        sorted_keys = sorted(a1.attr_list.keys(), key=_featcmp_key)

        lst = PsiTerm()
        lst.type = wl.nil

        for key in reversed(sorted_keys):
            is_pos = key.lstrip('-').isdigit() and int(key) >= 0

            _bare_fv = key
            if ctx_mod is not None and not is_pos and '#' in key:
                # The name a feature is filed under says which module
                # keeps it to itself; no other module sees it, and the
                # one that does sees it by its plain name.
                _km_fv, _bare_fv = key.split('#', 1)
                if _km_fv != ctx_mod.module_name:
                    continue

            if _want_values:
                kterm = a1.attr_list[key]
            elif is_pos:
                n = int(key)
                kterm = wl.make_integer(n)
            else:
                # Named feature: create atom in context module if given, else user module
                target_mod = ctx_mod if ctx_mod is not None else wl.user_module
                kterm = wl.make_atom(_bare_fv, target_mod)

            pair = PsiTerm()
            pair.type = wl.alist
            pair.attr_list = {'1': kterm, '2': lst}
            lst = pair

        return lst

    elif sym == 'readf':
        # readf(File) — the file's characters, as the codes a grammar reads.
        _rf_a = t.attr_list.get('1')
        if _rf_a is None or '2' in t.attr_list:
            return None
        _rf_d = _rf_a.deref()
        if _rf_d.value is not None:
            _rf_name = str(_rf_d.value)
        elif _rf_d.type is not None and _rf_d.type.keyword is not None:
            _rf_name = _rf_d.type.keyword.symbol
        else:
            return None
        try:
            with open(_rf_name, 'r') as _rf_f:
                _rf_text = _rf_f.read()
        except OSError:
            import sys as _sys_rf
            _sys_rf.stderr.write(
                "*** Error: cannot open file %s\n" % _rf_name)
            return None
        return eng.wl.make_list([_make_number(eng, float(ord(_c)))
                                 for _c in _rf_text])

    elif sym == '.':
        # T.F — feature access: get feature F of term T
        a1 = t.attr_list.get('1')  # T
        a2 = t.attr_list.get('2')  # F (feature label)
        if a1 is None or a2 is None:
            return None
        term = a1.deref()
        feat = a2.deref()
        # `P.a.d` parses as `(P.a).d`, so the host is itself a dot term: read
        # it left to right, or the feature is looked for on the unresolved
        # call rather than on the term it stands for.
        if (term.type is not None and term.type.keyword is not None
                and term.type.keyword.symbol == '.'):
            _host_ev = _try_eval_string_func(term, eng)
            if _host_ev is None:
                return None
            term = _host_ev.deref()
        # Determine the feature key string
        # Note: the integer sort has keyword.symbol == 'int' (not 'integer'),
        # and the real sort has keyword.symbol == 'real'.  When a numeric
        # feature label is given (e.g. C.1 or C.2), feat.value holds the
        # number and we convert it to an integer string for attr_list lookup.
        if feat.value is not None and feat.type and feat.type.keyword:
            fsym = feat.type.keyword.symbol
            if fsym in ('integer', 'real', 'int', 'float', 'number'):
                fkey = str(int(feat.value))
            else:
                # For string types and other value-bearing non-numeric types,
                # use the actual value as the key (e.g. "" -> '', not 'string')
                fkey = str(feat.value)
        elif feat.type and feat.type.keyword:
            fkey = feature_key_of(feat.type)
        else:
            return None
        val = term.attr_list.get(fkey)
        if val is None:
            return None
        return val.deref()

    elif sym == 'chr':
        # chr(N) → character string for ASCII code N (uses N mod 256)
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1d = a1.deref()
        ok, v = _eval_arith(a1d, eng)
        if not ok:
            return None
        code = int(v) % 256
        return _make_string(eng, chr(code))

    elif sym == 'int2str':
        # int2str(N) → string representation of number N
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1d = a1.deref()
        ok, v = _eval_arith(a1d, eng)
        if not ok:
            return None
        iv = int(v)
        return _make_string(eng, str(iv) if float(iv) == v else str(v))

    return None


def _unify(eng, a: PsiTerm, b: PsiTerm) -> bool:
    from wild_life.unification import Trail
    mark = eng.trail.mark()
    ok = eng.unifier.unify(a, b)
    if not ok:
        eng.trail.undo_to(mark)
    return ok


class _WriteFailure(Exception):
    """Internal signal: write/writeq should return False (*** No)."""
    pass


def _eval_and_conjunction(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """Evaluate an and_sym (&) psi-term conjunction.

    Returns the merged psi-term if the conjunction succeeds, or None if it fails.
    Used by _write_term to handle writeq(`X & Y) before printing.
    """
    wl = eng.wl

    def _strip_bq(x: PsiTerm) -> PsiTerm:
        """Strip one level of backtick quotation."""
        x = x.deref()
        if (x.type is not None and x.type.keyword is not None
                and x.type.keyword.symbol == '`'):
            inner = x.attr_list.get('1')
            if inner is not None:
                return inner.deref()
        return x

    t1_ref = t.attr_list.get('1')
    t2_ref = t.attr_list.get('2')
    if t1_ref is None or t2_ref is None:
        return None

    t1 = t1_ref.deref()
    t2 = t2_ref.deref()

    def _eval_side(s: PsiTerm) -> Optional[PsiTerm]:
        """Evaluate one side of & before conjunction: user func, cond, nested &."""
        s = s.deref()
        if s.type is not None and s.type is wl.and_sym:
            return _eval_and_conjunction(s, eng)
        s = _strip_bq(s)
        # A name declared with `global` stands for its cell, and it is the
        # cell the meet narrows: eratosthenes reads its limit with
        # `read_token(limit & int)`.
        _g_side = _global_cell(s, eng)
        if _g_side is not None:
            return _g_side.deref()
        # Evaluate user-defined function calls (e.g. posint_stream_to(5))
        if _is_user_function(s):
            _mark = eng.trail.mark()
            try:
                ev = _eval_user_func_sync(s, eng, 0)
                if ev is not None:
                    ev = copy_term(ev.deref(), {})
            finally:
                eng.trail.undo_to(_mark)
            if ev is not None:
                return _evaluate_result_for_display(ev, eng, 1)
            return None
        # Evaluate built-in cond() functionally
        if _is_cond_builtin_local(s):
            ev = _eval_body_sync(s, eng, 0)
            if ev is not None:
                return _evaluate_result_for_display(ev.deref(), eng, 1)
            return None
        # A built-in written for its value is asked for it here too:
        # tri_ins meets `strip(L)` with `root_sort(L)` to make a fresh head
        # for the list, and neither side is the term it stands for.
        if _is_strip_func(s):
            return _eval_strip_or_copy_func(s, eng, False)
        if _is_copy_pointer_func(s):
            return _eval_strip_or_copy_func(s, eng, True)
        if s.attr_list:
            ev = _try_eval_any_func(s, eng)
            if ev is not None and ev.deref() is not s:
                return ev.deref()
        return s

    t1 = _eval_side(t1)
    if t1 is None:
        return None
    t2 = _eval_side(t2)
    if t2 is None:
        return None

    # A composition waiting for its argument is a function, not a term with
    # room for one: `add3 & @(23)` is refused, and says which function it was.
    def _is_apply_side(x):
        sym_ap = _get_sym(x)
        return ((sym_ap == '@' or (wl.apply is not None and x.type is wl.apply))
                and bool(x.attr_list))

    for _curried, _other in ((t1, t2), (t2, t1)):
        if (_is_user_function(_curried) and _curried.attr_list
                and not _has_applicable_rule(_curried)
                and _is_apply_side(_other)):
            import io as _io_cf
            from wild_life.print_term import write_term as _wt_cf
            _buf_cf = _io_cf.StringIO()
            _wt_cf(_curried, outfile=_buf_cf, quoted=True, wl=wl,
                   max_col=1_000_000)   # the message is one line
            sys.stderr.write(
                f"*** Error: attempt to unify with curried function "
                f"{_buf_cf.getvalue()}\n")
            return None

    def _check_sort_member(elem: PsiTerm, sort_t: PsiTerm) -> bool:
        """Check if elem satisfies the sort sort_t.

        For a conditional sort (sort_t.type has sort-membership rules stored as
        [(pattern, condition), ...]), unify elem with the pattern and prove the
        condition.  Falls back to direct unification for simple sorts.
        """
        _ct = copy_term  # copy_term imported at module level from wild_life.unification
        sort_d = sort_t.deref()
        sort_def = sort_d.type
        if sort_def is None:
            return False
        rules = sort_def.rule
        if rules and isinstance(rules, list) and rules:
            # Conditional sort rules: [(pattern, condition), ...]
            for pat, cond in rules:
                _mark2 = eng.trail.mark()
                _vm2: dict = {}
                pat_copy = _ct(pat, _vm2)
                cond_copy = _ct(cond, _vm2) if cond is not None else None
                ok_pat = eng.unifier.unify(elem, pat_copy)
                if ok_pat:
                    if cond_copy is None:
                        eng.trail.undo_to(_mark2)
                        return True
                    # Prove condition synchronously
                    from wild_life.inference import GoalType as _GT2, _DEFRULES as _DR2, _INNER_RUN_BARRIER as _IRB2
                    _cp2 = eng.choice_stack
                    _gs2 = eng.goal_stack
                    eng.goal_stack = None
                    eng.push_goal(_GT2.PROVE, cond_copy.deref(), _DR2, None)
                    _old_ok2 = eng.main_loop_ok
                    _bar2 = _cp2 if _cp2 is not None else _IRB2
                    ok_cond = eng.run(cs_barrier=_bar2)
                    eng.main_loop_ok = _old_ok2
                    eng.choice_stack = _cp2
                    eng.goal_stack = _gs2
                    eng.trail.undo_to(_mark2)
                    if ok_cond:
                        return True
                else:
                    eng.trail.undo_to(_mark2)
            return False
        # No conditional rules: plain sort — try direct unification
        _mark3 = eng.trail.mark()
        fresh3 = PsiTerm(); fresh3.type = wl.top
        ok_a = eng.unifier.unify(fresh3, elem)
        ok_b = ok_a and eng.unifier.unify(fresh3.deref(), sort_d)
        eng.trail.undo_to(_mark3)
        return ok_b

    # If one side is a disjunction, distribute & over elements and filter
    t1_is_disj = t1.type is not None and (t1.type is wl.disjunction or t1.type is wl.disj_nil)
    t2_is_disj = t2.type is not None and (t2.type is wl.disjunction or t2.type is wl.disj_nil)

    if t1_is_disj or t2_is_disj:
        # Determine the disjunction and the filter term
        if t1_is_disj and not t2_is_disj:
            disj_side, filter_side = t1, t2
        elif t2_is_disj and not t1_is_disj:
            disj_side, filter_side = t2, t1
        else:
            # Both are disjunctions: cross-product (keep pairs that unify)
            elems1 = _collect_disjunction(t1, eng)
            elems2 = _collect_disjunction(t2, eng)
            surviving: list = []
            for e1 in elems1:
                for e2 in elems2:
                    _mark4 = eng.trail.mark()
                    fresh4 = PsiTerm(); fresh4.type = wl.top
                    ok_a = eng.unifier.unify(fresh4, e1.deref())
                    ok_b = ok_a and eng.unifier.unify(fresh4.deref(), e2.deref())
                    if ok_b:
                        surviving.append(copy_term(fresh4.deref(), {}))
                    eng.trail.undo_to(_mark4)
            if not surviving:
                return None
            return _make_disjunction_psi(surviving, wl)

        elems = _collect_disjunction(disj_side, eng)
        surviving = []
        for e in elems:
            e_d = e.deref()
            if not _check_sort_member(e_d, filter_side):
                continue
            # What survives is the meet, not the element it was made from:
            # `2 & prime` is the 2 that is a prime, written `2: prime`.
            _m_meet = eng.trail.mark()
            _fresh_meet = PsiTerm()
            _fresh_meet.type = wl.top
            try:
                _ok_meet = (eng.unifier.unify(_fresh_meet, e_d)
                            and eng.unifier.unify(_fresh_meet.deref(),
                                                  filter_side))
            except Exception:
                _ok_meet = False
            surviving.append(copy_term(_fresh_meet.deref(), {})
                             if _ok_meet else e_d)
            eng.trail.undo_to(_m_meet)
        if not surviving:
            return None  # empty disjunction = fail (No)
        return _make_disjunction_psi(surviving, wl)

    # Unify t1 and t2 through a fresh variable to find their meet.
    # The order the two sides reach the fresh variable matters for a sort
    # carrying a membership condition (`posint := X:int | X>=0`): the condition
    # is proven the moment the variable takes that sort, so a still-uninstantiated
    # variable would be judged against it.  `posint & 2` therefore feeds the
    # concrete side in first, exactly as `2 & posint` already did.
    def _meet(first: PsiTerm, second: PsiTerm) -> Optional[PsiTerm]:
        fresh = PsiTerm()
        fresh.type = wl.top
        mark = eng.trail.mark()
        if not eng.unifier.unify(fresh, first):
            eng.trail.undo_to(mark)
            return None
        if not eng.unifier.unify(fresh.deref(), second):
            eng.trail.undo_to(mark)
            return None
        return fresh.deref()

    _r = _meet(t1, t2)
    if _r is None:
        _r = _meet(t2, t1)
    return _r


def _has_concrete_non_numeric_arg(t: PsiTerm, eng) -> bool:
    """Return True if any immediate argument of arithmetic term t is a
    concrete non-numeric atom (i.e. not an unbound variable, not a number).

    Used to decide whether an arithmetic-evaluation failure should be a hard
    failure (with warning) vs. a soft failure (term printed as-is).
    """
    from wild_life.data_structures import SORT_VAR
    wl = eng.wl

    def _is_concrete_non_numeric(a: PsiTerm) -> bool:
        a = a.deref()
        # Strip one backtick level
        if (a.type is not None and a.type.keyword is not None
                and a.type.keyword.symbol == '`'):
            inner = a.attr_list.get('1')
            if inner is not None:
                a = inner.deref()
        # Unbound top-type variable
        if a.type is wl.top:
            return False
        # Sort-constrained variable (X:sort syntax)
        if a.flags & SORT_VAR:
            return False
        # Symbol '@' is the top-type marker for unbound vars
        sym_a = a.type.keyword.symbol if (a.type and a.type.keyword) else ''
        if sym_a == '@':
            return False
        # Numeric value (integer or real)
        if (a.value is not None and a.type and wl.real is not None
                and a.type.is_subtype_of(wl.real)):
            return False
        # A term narrowed no further than a number sort is a number nobody
        # has said yet: the `int` a grid square holds is one of these, and
        # the sum of a row waits on it rather than refusing it.
        if _may_yet_be_a_number(a, eng):
            return False
        # A sum written inside a sum is not an argument of its own: what is
        # wrong with `int + (int + (int + 0))`, if anything, is in there.
        if a.attr_list and sym_a in _ARITH_OPS_SET:
            return any(_is_concrete_non_numeric(_sub)
                       for _k, _sub in a.attr_list.items()
                       if _k in ('1', '2'))
        # Everything else: a concrete atom / compound that is not numeric
        return True

    for key in ('1', '2'):
        ref = t.attr_list.get(key) if t.attr_list else None
        if ref is not None and _is_concrete_non_numeric(ref):
            return True
    return False


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


def _write_term(t: PsiTerm, eng, stream=None, quoted=True, compact=False) -> None:
    from wild_life.print_term import write_term
    var_tree = getattr(eng, '_last_var_tree', None)
    wl = eng.wl if eng else None
    # Writing out a global reads a cell the query does not own.
    note_persistent_use(t.type, eng)

    # ── backtick-quoted term: strip ONE outer backtick ──────────────────────
    # write(`expr) prints the inner expr without the backtick.
    # Inner backtick-quoted subterms keep their backtick during recursive printing.
    # When backtick is stripped, pass directly to the printer with no_arith_eval=True,
    # bypassing the disj_nil/bottom-type check (write(`{}) must print "{}", not fail).
    if (t.type is not None and t.type.keyword is not None
            and t.type.keyword.symbol == '`'):
        inner = t.attr_list.get('1')
        if inner is not None:
            from wild_life.print_term import write_term as _wt
            _pd = getattr(eng.wl, 'print_depth', None) if eng and eng.wl else None
            from wild_life.print_term import PRINT_DEPTH as _PD

            # Track this backtick as "directly written" so print_variables can strip it.
            _wl = eng.wl if eng else None
            if _wl is not None:
                if not hasattr(_wl, '_written_backtick_ids'):
                    _wl._written_backtick_ids = set()
                _wl._written_backtick_ids.add(id(t))

            # Temporarily redirect vars pointing to this backtick → inner.
            # This makes go_through treat the inner as self-referential (SHARED),
            # so insert_variables correctly assigns the var's name to the inner term.
            # Without this, the inner is only seen once → gets a generated name '_A'.
            inner_deref = inner.deref()
            tid = id(t)
            redirect_pairs = []
            for _vname, _vref in (var_tree or {}).items():
                if _vref is not None and id(_vref.deref()) == tid:
                    _old_coref = _vref.coref
                    _vref.coref = inner_deref
                    redirect_pairs.append((_vref, _old_coref))

            try:
                _wt(inner_deref, outfile=stream or __import__('sys').stdout,
                    quoted=quoted, wl=eng.wl, var_tree=var_tree,
                    print_depth=_pd if _pd is not None else _PD,
                    no_arith_eval=True)
            finally:
                for _vref, _old_coref in redirect_pairs:
                    _vref.coref = _old_coref
            return

    # ── psi-term conjunction (&): evaluate before printing ──────────────────
    # writeq(`X & Y) should evaluate the conjunction and print the result.
    # If the conjunction fails, the predicate fails.
    if wl and t.type is not None and t.type is wl.and_sym:
        evaluated = _eval_and_conjunction(t, eng)
        if evaluated is None:
            raise _WriteFailure("conjunction failed")
        t = evaluated

    # ── eval(Expr): force evaluation even in non-strict (write) context ─────
    # write/writeq are non-strict predicates: their arguments carry the
    # NON_STRICT_TERM flag which normally suppresses arithmetic evaluation.
    # eval(Expr) explicitly requests evaluation regardless of that flag.
    # We handle it here, before the NON_STRICT_TERM check below.
    _sym_early = t.type.keyword.symbol if (t.type and t.type.keyword) else ''
    if _sym_early == 'eval':
        _a1_eval = t.attr_list.get('1')
        if _a1_eval is not None:
            _a1d_eval = _a1_eval.deref()
            # Unwrap backtick if present (eval(`Expr) evaluates Expr)
            _a1_sym = (_a1d_eval.type.keyword.symbol
                       if _a1d_eval.type and _a1d_eval.type.keyword else '')
            if _a1_sym == '`':
                _inner_eval = _a1d_eval.attr_list.get('1')
                if _inner_eval is not None:
                    _a1d_eval = _inner_eval.deref()
            _eval_ok, _eval_v = _eval_arith(_a1d_eval, eng)
            if _eval_ok:
                t = _make_number(eng, _eval_v)
            # else: evaluation failed → fall through and print term as-is

    # ── copy_term(X) functional use at top level ────────────────────────────
    if _is_copy_term_func(t):
        t = _eval_copy_term_func(t)

    # ── bottom type ({} / disj_nil): cannot be written ──────────────────────
    sym = t.type.keyword.symbol if (t.type and t.type.keyword) else ''
    if sym == '{}':
        raise _WriteFailure("bottom type '{}'")
    if wl and wl.disj_nil is not None and t.type is wl.disj_nil:
        raise _WriteFailure("bottom type disj_nil")

    # ── arithmetic evaluation (e.g. 23+23 → 46) ─────────────────────────────
    # Guard against cyclic terms (e.g. X = s(X)) causing infinite recursion.
    # Skip arithmetic evaluation for terms with NON_STRICT_TERM flag: these are
    # expressions passed to non-strict predicates (e.g. `write(1+2)` where the
    # argument was labeled in a non-strict context); they must be printed as-is.
    _arith_binary_ops = frozenset(('+', '-', '*', '/', '//', 'mod', '^',
                                   'max', 'min'))
    _arith_unary_ops = frozenset(('abs', 'sqrt', 'sin', 'cos', 'tan', 'exp',
                                  'log', 'floor', 'ceiling', 'round',
                                  'truncate'))
    from wild_life.data_structures import NON_STRICT_TERM as _WT_NST
    _is_nst = bool(t.flags & _WT_NST)
    # A call that reaches itself is written as the call it is: `X : f(X)` has
    # no value to show that is not itself.  What asks for its value — `=`, or
    # an argument position — still reduces it.
    _self_call = _is_user_function(t) and _term_reaches_itself(t)
    # A call no rule of its function applies to is a partial application, and
    # writes as itself: `fact` alone is `fact`, not the 1 that `fact(0) -> 1`
    # would answer if the missing argument were filled in.
    _partial_call = _is_user_function(t) and not _has_applicable_rule(t)
    try:
        t_eval = (_try_eval_arith_to_term(t, eng)
                  if not _is_nst and not _self_call and not _partial_call
                  else None)
        if t_eval is not None:
            t = t_eval
        else:
            # Evaluate '.' (feature access) in write context.
            # write(C.1) evaluates C.1 and prints the result (e.g. 'a' for a cons head).
            t_str = _try_eval_string_func(t, eng)
            if t_str is not None:
                t = t_str
            elif sym == '.':
                # T.F where T hasn't got F yet.  c_get_feature gives the term
                # the feature and answers the fresh variable it put there, so
                # the write shows `@` and the term keeps the feature after.
                _dot_cell = (_resolve_dot_feat(t, eng)
                             if eng is not None else None)
                if _dot_cell is not None:
                    t = _dot_cell.deref()
            elif (_is_user_function(t) and eng is not None
                    and not _term_reaches_itself(t)
                    and _has_applicable_rule(t)):
                # A call that reaches itself is written as the call it is:
                # `X : f(X)` has no value to show that is not itself.
                # User-defined function call: evaluate synchronously for display.
                # Use a trail mark so pattern-matching bindings don't leak.
                # NOTE: 0-arity user functions (global variables declared with
                # setq/persistent) are handled carefully: we only substitute the
                # evaluated result if it is a concrete numeric value.  If the
                # stored body is an unevaluated expression with unbound variables
                # (e.g. after backtracking), we keep the original atom name so
                # that write(result) prints "result" rather than "@ + 1".
                _is_zero_arity_fn = not t.attr_list
                _evaled = None
                _mark = eng.trail.mark()
                try:
                    _evaled = _eval_user_func_sync(t, eng, 0)
                    if _evaled is not None:
                        # Deep-copy while trail bindings are still active so that
                        # trail-bound variables (e.g. H' → d) are resolved into
                        # concrete values before we roll back the trail.
                        _evaled = copy_term(_evaled.deref(), {})
                finally:
                    eng.trail.undo_to(_mark)
                if _evaled is not None:
                    # Fully evaluate the result: walk disjunction elements and
                    # compute any arithmetic ops that involve disjunction operands
                    # (e.g. {1; 1+posint_stream_to(N-1)} → {1;2;3}).
                    _evaled = _evaluate_result_for_display(_evaled, eng, 1)
                    if _is_zero_arity_fn:
                        # For 0-arity globals: substitute only a result that
                        # stands on its own.  `quadruple -> *(2 => 4)` is worth
                        # that partial application and writes as it, while a
                        # global whose stored body still has variables in it —
                        # `@ + 1` after backtracking — has nothing to show, so
                        # the name is written instead.
                        _evd = _evaled.deref()
                        if _evd.value is not None or _is_ground_term(_evd):
                            t = _evaled
                        # else: keep original t (print the atom name as-is)
                    else:
                        t = _evaled
            elif _is_cond_builtin_local(t) and eng is not None:
                # Built-in cond(C, T, E) as a functional expression: evaluate
                # synchronously so write(cond(3<2,{},f(3))) prints the result,
                # not the unevaluated cond term.
                _evaled = _eval_body_sync(t, eng, 0)
                if _evaled is not None:
                    _evaled = _evaluate_result_for_display(_evaled.deref(), eng, 1)
                    t = _evaled
            elif wl and sym in (_arith_binary_ops | _arith_unary_ops):
                # The top-level operator is arithmetic but evaluation failed.
                # If any immediate arg is a concrete non-numeric atom, this is
                # a hard failure (emit warning + fail); otherwise just print.
                if _has_concrete_non_numeric_arg(t, eng):
                    expr_str = _term_to_str(t, eng, quoted=True)
                    print(f"*** Warning: non-numeric argument(s) in '{expr_str}'.",
                          file=sys.stderr)
                    raise _WriteFailure("non-numeric argument")
    except RecursionError:
        pass

    from wild_life.print_term import PRINT_DEPTH as _PRINT_DEPTH, MAX_COL as _MAX_COL
    _pd = getattr(eng.wl, 'print_depth', _PRINT_DEPTH) if eng and eng.wl else _PRINT_DEPTH
    # C Wild Life's write/1 does NOT pretty-print (no line-wrapping). Only
    # pretty_write/1 produces multi-line indented output.  compact=True disables
    # line-wrapping (max_col=1_000_000); compact=False uses the default 79-char limit.
    _mc = (1_000_000 if compact
           else max(1, getattr(eng.wl, 'page_width', 80) - 1))
    write_term(t, outfile=stream or sys.stdout, quoted=quoted, wl=eng.wl,
               var_tree=var_tree, print_depth=_pd, max_col=_mc)


def _note_global_used(eng, defn) -> None:
    """Record that a global which predates this query was read or assigned.

    A query that only declares new globals leaves nothing behind worth
    keeping; one that uses a global already in scope does, since that cell's
    value would be undone along with the query.
    """
    pre = getattr(eng, 'pre_query_globals', None)
    if pre is not None and id(defn) in pre:
        eng.used_existing_global = True


def note_persistent_use(defn, eng) -> None:
    """Record that a `persistent` global was written, or read out at top level.

    Its cell is not the query's to undo, so such a query opens a level and
    keeps what it did — which is why `write(a)` answers at `--1>`.  A global
    a predicate reads on its way to an answer is not that: power_4 reads
    `result` throughout and still answers at the top level.
    """
    if defn is not None and getattr(defn, 'is_persistent', False):
        _note_global_used(eng, defn)


_IUF_SKIP_FLAGS = QUOTED_TRUE | REDUCED


def _global_cell(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """The cell a global variable name stands for, or None for anything else.

    Every reference reads the same psi-term, so what one name binds is visible
    through the others — that is what makes a global assignable.
    """
    if t is None:
        return None
    while t.coref is not None:
        t = t.coref
    if t.attr_list or t.type is None or t.type.type is not DefType.GLOBAL:
        return None
    _note_global_used(eng, t.type)
    return t.type.global_value


def _stored_side(t: PsiTerm, eng) -> PsiTerm:
    """The term a name stands for when something has been stored in it."""
    if t is None or eng is None:
        return t
    _d = t.deref()
    _defn = _d.type
    if (_defn is not None and getattr(_defn, 'is_persistent', False)
            and _defn.rule):
        _cell = _persistent_cell(_d, eng)
        if _cell is not None:
            return _cell.deref()
    return _d


def _persistent_cell(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """The one term a `persistent` name stands for, or None for anything else.

    A persistent name keeps what is written into it, features and all:
    acc_declarations.lf files what it knows about a predicate under
    `predicates_info.combined_name(X).A <<- true`, and every later reading of
    predicates_info has to find it there.  A name nothing has been written
    into yet is given the term the writing will go into.
    """
    if t is None or eng is None:
        return None
    while t.coref is not None:
        t = t.coref
    defn = t.type
    if (t.attr_list or t.value is not None or defn is None
            or not getattr(defn, 'is_persistent', False)
            or defn.type != DefType.FUNCTION):
        return None
    if defn.rule:
        for _h, _b in defn.rule:
            if _h is None or _b is None:
                continue
            if _h.deref().attr_list:
                return None          # a function of the program's own
            _bd = _b.deref()
            _bd._wl_persistent_cell = True
            return _bd
        return None
    from wild_life.unification import copy_term as _ct_pc
    _cell = PsiTerm(type_def=eng.wl.top)
    _cell._wl_persistent_cell = True
    defn.rule = [(_ct_pc(t, {}), _cell)]
    return _cell


def _term_to_str(t: PsiTerm, eng, quoted=True) -> str:
    from wild_life.print_term import term_to_string
    # Pass the query's variables so a diagnostic names them as the user wrote
    # them (`B`) rather than by an internal label (`_A`).
    return term_to_string(t, quoted=quoted, wl=eng.wl,
                          var_tree=getattr(eng, '_last_var_tree', None))


def _is_var(t: PsiTerm, eng) -> bool:
    wl = eng.wl
    return t.type is wl.top and t.value is None and not t.attr_list and t.coref is None


# ─────────────────────────────────────────────────────────────────────────────
# I/O predicates
# ─────────────────────────────────────────────────────────────────────────────

def _write_all_args(goal: PsiTerm, eng, quoted: bool, stream=None,
                    compact: bool = False) -> bool:
    """Write all positional arguments of goal, concatenated (no separator).

    In LIFE, write(a,b,c) writes each argument in order without separator.
    If the goal has no positional args, write the goal's sort name.
    compact=True disables line-wrapping (for write/1 which is always single-line).
    compact=False uses pretty-printing (for pretty_write/1).
    Returns False if any argument fails to write (e.g., bottom type, bad arithmetic).
    """
    attrs = goal.attr_list
    # Collect positional arguments '1', '2', '3', ...
    i = 1
    written_any = False
    while True:
        key = str(i)
        if key not in attrs:
            break
        arg = attrs[key].deref()
        try:
            _write_term(arg, eng, stream=stream, quoted=quoted, compact=compact)
        except _WriteFailure:
            return False
        written_any = True
        i += 1
    if not written_any:
        # No positional args: treat as write of goal itself
        try:
            _write_term(goal, eng, stream=stream, quoted=quoted, compact=compact)
        except _WriteFailure:
            return False
    return True


def bi_write(goal: PsiTerm, eng) -> bool:
    """write(T) — write term T (or all positional args) without quoting.
    C Wild Life's write/1 does NOT pretty-print — always single-line compact output.
    Use pretty_write/1 for multi-line indented output.
    """
    return _write_all_args(goal, eng, quoted=False, compact=True)


def bi_writeq(goal: PsiTerm, eng) -> bool:
    """writeq(T) — write term T (or all positional args) with quoting.
    Like write/1, writeq/1 does NOT pretty-print — always single-line compact output.
    """
    return _write_all_args(goal, eng, quoted=True, compact=True)


def bi_pretty_write(goal: PsiTerm, eng) -> bool:
    """pretty_write(T) — write term T with pretty-printing (multi-line indented output).
    Unlike write/1 which is always compact, pretty_write/1 wraps at 79 columns.
    """
    return _write_all_args(goal, eng, quoted=False, compact=False)


def bi_pretty_writeq(goal: PsiTerm, eng) -> bool:
    """pretty_writeq(T) — writeq with pretty-printing (multi-line indented output)."""
    return _write_all_args(goal, eng, quoted=True, compact=False)


def bi_write_canonical(goal: PsiTerm, eng) -> bool:
    """write_canonical(T) — write term T in canonical (non-operator) form.

    write_canonical writes all positional args concatenated in canonical form.
    The canonical form uses functor(arg1,arg2,...) notation instead of
    operator-sugar forms.
    """
    from wild_life.print_term import write_term
    attrs = goal.attr_list
    var_tree = getattr(eng, '_last_var_tree', None)
    i = 1
    written_any = False
    while True:
        key = str(i)
        if key not in attrs:
            break
        arg = attrs[key].deref()
        # Backtick-quoted terms `(X): pass directly to write_term.
        # _pretty_tag_or_psi_term already handles backtick by setting
        # no_arith_eval=True before printing the inner expression, so
        # the arithmetic structure is preserved correctly.
        # For non-backtick terms, evaluate arithmetic first so that
        # e.g. write_canonical(1+2) prints 3 as expected.
        # A term that came in under a backquote is written as it stands.
        # The backquote itself is gone once the term has been through a
        # variable -- `B = `(1+2)` leaves B the sum, marked as one not to be
        # worked out -- so the mark counts for as much as the backquote:
        # `write_canonical(B)` writes `+(1,2)`, not 3.
        from wild_life.data_structures import NON_STRICT_TERM as _NST_wc
        is_backtick = ((arg.type is not None and arg.type.keyword is not None
                        and arg.type.keyword.symbol == '`')
                       or bool(arg.flags & (_NST_wc | QUOTED_TRUE)))
        if not is_backtick:
            try:
                t_eval = _try_eval_arith_to_term(arg, eng)
                if t_eval is not None:
                    arg = t_eval
            except (RecursionError, Exception):
                pass
        write_term(arg, outfile=sys.stdout, quoted=True, wl=eng.wl,
                   var_tree=var_tree, canonical=True)
        written_any = True
        i += 1
    if not written_any:
        # No positional args: write the goal itself canonically
        try:
            t_eval = _try_eval_arith_to_term(goal, eng)
            if t_eval is not None:
                goal = t_eval
        except (RecursionError, Exception):
            pass
        write_term(goal, outfile=sys.stdout, quoted=True, wl=eng.wl,
                   var_tree=var_tree, canonical=True)
    return True


def bi_print(goal: PsiTerm, eng) -> bool:
    """print(T) — same as write."""
    return bi_write(goal, eng)


def bi_nl(goal: PsiTerm, eng) -> bool:
    """nl — print newline."""
    print()
    return True


def bi_write_err(goal: PsiTerm, eng) -> bool:
    """write_err(T) — write to stderr (compact, no pretty-printing).

    Like write/1, it writes every positional argument in turn:
    `write_err("*** Profile : ", Type, " '", What, "'")` is one message.
    """
    return _write_all_args(goal, eng, quoted=False, stream=sys.stderr,
                           compact=True)


def bi_writeln(goal: PsiTerm, eng) -> bool:
    """writeln(T) — write then newline."""
    bi_write(goal, eng)
    print()
    return True


def bi_page_width(goal: PsiTerm, eng) -> bool:
    """page_width / page_width(N) — get or set the line width used when
    a term is written out over several lines.

    0-arity resets the width to its 80-column default.
    """
    wl = eng.wl
    arg = _get_one_arg(goal)
    if arg is None:
        if not goal.deref().attr_list:
            wl.page_width = 80
            return True
        return False
    arg = arg.deref()
    if arg.value is not None and arg.type and arg.type.is_subtype_of(wl.real):
        n = int(float(arg.value))
        if n <= 0:
            return False
        wl.page_width = n
        return True
    # Unbound argument: report the width in force.
    if _term_is_unbound(arg, eng):
        return _unify(eng, arg, wl.make_integer(getattr(wl, 'page_width', 80)))
    return False


def bi_print_depth(goal: PsiTerm, eng) -> bool:
    """print_depth / print_depth(N) — get/set the global print depth limit.

    0-arity form (print_depth):
      Resets the print depth to unlimited (wl.print_depth = 0) and succeeds.

    1-arity form (print_depth(N)), C Wild Life semantics:
      N < 0  → unlimited depth (no truncation); error message is printed.
      N = 0  → show only the root functor, arguments shown as '...'.
      N > 0  → show N levels of arguments (N+1 levels total including root).

    Internal mapping: wl.print_depth = 0 means unlimited; wl.print_depth = K > 0
    means truncate at K levels (write_term convention).  So C Wild Life's N maps
    to wl.print_depth = N + 1 for N >= 0, and 0 for N < 0.
    """
    wl = eng.wl
    # 0-arity: print_depth? — reset to unlimited
    arg = _get_one_arg(goal)
    if arg is None:
        # Check if there truly are no args (arity 0), not just a parsing failure.
        # _get_one_arg returns None if arity != 1; for arity 0, treat as reset.
        goal_d = goal.deref()
        if not goal_d.attr_list:  # no arguments = arity 0
            wl.print_depth = 0  # reset to unlimited
            return True
        return False
    arg = arg.deref()
    if arg.value is not None and arg.type and arg.type.is_subtype_of(wl.real):
        n = int(float(arg.value))
        if n < 0:
            # Negative argument: print error, reset to unlimited.
            pd = wl.print_depth
            if pd <= 1:
                # pd=1 means output would be truncated at top level → show "..."
                arg_str = "..."
            else:
                import io
                from wild_life.print_term import write_term
                buf = io.StringIO()
                write_term(arg, outfile=buf, quoted=False, print_depth=pd, wl=wl)
                arg_str = buf.getvalue()
            sys.stderr.write(
                f"*** Error: argument in print_depth({arg_str}) must be positive or zero.\n"
            )
            wl.print_depth = 4  # reset to C Wild Life default (4)
            return True
        else:
            # N >= 0: show N levels of args (root + N levels = N+1 levels total).
            # Our internal convention: wl.print_depth = 0 means unlimited;
            # wl.print_depth = K means show K levels (K=1 → just root functor).
            wl.print_depth = n + 1
        return True
    return False


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


def _read_one_char(eng, as_code: bool) -> PsiTerm:
    """Read one character from the current input, as a code or as a string."""
    wl = eng.wl
    try:
        c = sys.stdin.read(1)
    except EOFError:
        c = ''
    if c == '':
        return wl.make_atom('end_of_file', wl.user_module)
    return wl.make_number(float(ord(c))) if as_code else wl.make_string(c)


def bi_get_char(goal: PsiTerm, eng) -> bool:
    """get_char(C) — read a character, as a one-character string."""
    arg = _get_one_arg(goal)
    return _unify(eng, arg, _read_one_char(eng, False)) if arg else False


def bi_get_code(goal: PsiTerm, eng) -> bool:
    """get(C) — read a character, as its code.

    A program that reads a file reads numbers: the tokenizer in preparser.lf
    asks whether a character is `>= 48 and =< 57`, and `charac(Z) ->
    psi2str(chr(Z))` turns one back into text.  End of input still answers
    `end_of_file`, which is what `X = end_of_file` in copyfile.lf looks for.
    """
    arg = _get_one_arg(goal)
    return _unify(eng, arg, _read_one_char(eng, True)) if arg else False


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
        result = PsiTerm(type_def=wl.eof)
    else:
        ts = tokenizer_from_string(line)
        p = Parser(ts)
        try:
            t, _ = p.parse()
            result = t or PsiTerm(type_def=eng.wl.top)
        except Exception:
            result = PsiTerm(type_def=eng.wl.top)
    return _unify(eng, arg, result) if arg else bool(result)


def bi_read_term(goal: PsiTerm, eng) -> bool:
    """read_term(T, Opts) — simplified."""
    arg, _ = _get_two_args(goal)
    return bi_read(goal, eng)  # simplified: ignore options


def bi_parse(goal: PsiTerm, eng) -> bool:
    """parse([Result,] String[, Status[, Vars]]) — parse a LIFE string.

    Called when parse(...) appears as a PREDICATE goal (not in functional
    position).  The 'result' is the first argument of the goal term itself
    in that case, so the arity varies:

      parse(String)?            → just parse the string; no binding (rare)
      parse(String, Status)?    → parse + Status
      Result = parse(String)?   → handled by bi_unify + _eval_parse_func

    As a predicate the common forms are:
      parse(S)      — 1-arg: S is parsed; this form rarely makes sense alone
      parse(S, St)  — 2-arg: S is parsed, St = status
      parse(S, St, V) — 3-arg: also binds V = true
    """
    a1 = goal.attr_list.get('1')
    if a1 is None:
        return False
    a1d = a1.deref()

    # Evaluate the parse function (which also binds Status/Vars side-args)
    result = _eval_parse_func(goal, eng)
    return result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic
# ─────────────────────────────────────────────────────────────────────────────

_ARITH_DEBUG = False  # Set True to debug arithmetic evaluation

def _apply_to_call(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """Rebuild F(Args) from an apply(Args, functor => F) term.

    The parser turns a call through a functor variable, `F(A)`, into
    apply(A, functor => F).  Once F is bound this gives back the call it
    stands for; it returns None while F is still unknown.
    """
    wl = eng.wl if eng is not None else None
    if wl is None or getattr(wl, 'apply', None) is None or t.type is not wl.apply:
        return None
    key = (wl.functor.symbol
           if (getattr(wl, 'functor', None) and wl.functor and wl.functor.keyword)
           else 'functor')
    fa = t.attr_list.get(key)
    if fa is None:
        return None
    fv = fa.deref()
    if fv.type is None or _term_is_unbound(fv, eng):
        return None
    call = PsiTerm()
    call.type = fv.type
    for k, v in t.attr_list.items():
        if k != key:
            call.attr_list[k] = v
    # Features the functor already carries (a partial application such as *(23))
    for k, v in fv.attr_list.items():
        if k not in call.attr_list:
            call.attr_list[k] = v
    from wild_life.data_structures import NON_STRICT_TERM as _NST_AP
    if fv.flags & _NST_AP:
        call.flags |= _NST_AP
    return call


def _eval_arith(t: PsiTerm, eng, _depth: int = 0) -> Tuple[bool, float]:
    """Evaluate an arithmetic expression. Returns (ok, value)."""
    if t is None or _depth > 40:
        return False, 0.0
    t = t.deref()
    wl = eng.wl

    # A number is its own value, and that is what most of these calls are
    # asked about, so it is answered before anything else is looked at.  A
    # number is neither a global's name nor a call through a functor.
    _t_val = t.value
    _t_type = t.type
    if (_t_val is not None and _t_type is not None
            and (_t_type is wl.integer or _t_type is wl.real
                 or _t_type.is_subtype_of(wl.real))):
        # Fire int/real delay rule for parsed literal integers (not computed by _make_number).
        # In C Wild Life, literal integers in expressions act like narrowed sort-vars
        # and fire the :: I:int | ... delay when they are "evaluated".
        from wild_life.runtime import WL as _WL_ea
        if (_WL_ea.delay_rules and eng is not None
                and not getattr(eng, '_in_fire_delay', False)
                and not t.__dict__.get('_delay_fired')):
            t._delay_fired = True
            eng.unifier._fire_delay_rules(t, t.type)
        return True, float(_t_val)

    # A global variable name stands for its cell, so `b <- a+a` reads a's
    # value rather than treating a as a non-numeric atom.
    if not t.attr_list and _t_type is not None and _t_type.type is DefType.GLOBAL:
        cell = _global_cell(t, eng)
        if cell is not None:
            return _eval_arith(cell, eng, _depth + 1)
    sym = _t_type.keyword.symbol if (_t_type and _t_type.keyword) else ''

    # A call written through a functor variable evaluates once the functor is
    # known, so that `F(A)*F(C) > 0` can be computed at all.
    if getattr(wl, 'apply', None) is not None and _t_type is wl.apply:
        _ap_call = _apply_to_call(t, eng)
        if _ap_call is None:
            return False, 0.0
        return _eval_arith(_ap_call, eng, _depth + 1)

    # User-defined function: try to evaluate it inline (no condition case).
    # A call under a backtick is the call, not what it answers, so it is left
    # alone here the same way `_is_user_function` leaves it alone elsewhere;
    # so is a call that reaches itself, which has no value but itself.
    if _is_user_function(t):
        # A call to a function declared non_strict is not an arithmetic
        # expression: check_func reduces a call's arguments only when
        # evaluate_args says so, and reducing `transLifeCode({…})` here to
        # see whether it comes to a number walks it and point_virgule round
        # each other until the depth runs out.
        _ns_ea = getattr(eng, 'non_strict_set', None)
        if _ns_ea and t.type in _ns_ea:
            return False, 0.0
        note_persistent_use(t.type, eng)
        active = [(h, b) for (h, b) in t.type.rule if h is not None and b is not None]
        from wild_life.unification import copy_term
        # Pre-evaluate built-in function calls in args (e.g. features(X)) so
        # that they are resolved before we try to unify with rule heads.
        # Create a shallow copy of t with evaluated args.
        t_pre = PsiTerm()
        t_pre.type = t.type
        t_pre.value = t.value
        t_pre.attr_list = {}
        for _k, _v in t.attr_list.items():
            _vd = _v.deref()
            # Try arithmetic evaluation first (handles N-1, N*2, etc. in recursive calls)
            _ok_arith, _arith_val = _eval_arith(_vd, eng, _depth + 1)
            if _ok_arith:
                # Reuse the original literal term when it's already a concrete number
                # with the same value.  This preserves _delay_fired=True and avoids
                # creating a fresh _make_number term that would re-fire delay rules
                # during unification.
                if (_vd.value is not None and not _vd.attr_list and _vd.type is not None
                        and float(_vd.value) == _arith_val):
                    t_pre.attr_list[_k] = _vd
                else:
                    _computed = _make_number(eng, _arith_val)
                    # Fire delay for computed arithmetic results (e.g. N-1=0).
                    # In C Wild Life, arithmetic results on int/real sort-vars trigger
                    # the :: I:int (or :: R:real) global delay rule.
                    from wild_life.runtime import WL as _WL_pre
                    if (_WL_pre.delay_rules and eng is not None
                            and not getattr(eng, '_in_fire_delay', False)
                            and not getattr(_computed, '_delay_fired', False)):
                        _computed._delay_fired = True
                        eng.unifier._fire_delay_rules(_computed, _computed.type)
                    t_pre.attr_list[_k] = _computed
            else:
                _evaled_arg = _try_eval_string_func(_vd, eng)
                if (_evaled_arg is None and _vd.attr_list
                        and not _is_user_function(_vd)):
                    # An argument that is a built-in call of its own stands
                    # for what it answers: `sum(map(F, L))` reads the list
                    # map makes.  A call to a rule of the program's own is
                    # left to the rules below, which put back what does not
                    # match rather than reducing it where it stands.
                    _evaled_arg = _try_eval_any_func(_vd, eng)
                    if _evaled_arg is _vd:
                        _evaled_arg = None
                t_pre.attr_list[_k] = _evaled_arg if _evaled_arg is not None else _vd
        _cp_save = eng.choice_stack  # Save choice stack before user-func unification
        for _ri, (h0, b0) in enumerate(active):
            # A head that asks for a number the call does not carry cannot
            # fit, and saying so costs a comparison where copying the rule
            # out to find the same thing costs two terms: `gcd(I,0)` is
            # asked of `gcd(48,18)` once per step of every division.
            _h0_d = h0.deref()
            _mismatch = False
            for _hk, _hv in _h0_d.attr_list.items():
                _hv_d = _hv.deref()
                if _hv_d.value is None or _hv_d.attr_list:
                    continue
                _cv = t_pre.attr_list.get(_hk)
                if _cv is None:
                    continue
                _cv_d = _cv.deref()
                if (_cv_d.value is not None and not _cv_d.attr_list
                        and _cv_d.value != _hv_d.value):
                    _mismatch = True
                    break
            if _mismatch:
                continue
            _vm: dict = {}
            head = copy_term(h0, _vm)
            body = copy_term(b0, _vm)
            body_d = body.deref()
            # Skip sort-constrained rules: head is a bare variable (no attrs)
            # AND the call term itself has arguments. These rules are
            # X:sort -> body, designed for eval_aim not direct arithmetic.
            # Evaluating them causes infinite recursion because the body
            # typically calls features(X) which can't be resolved here.
            # EXCEPTION: 0-arity functions like `result -> 4` also have no
            # head attrs, but they should still be evaluated — they differ
            # from sort-constrained rules in that the call term has no args.
            head_d = head.deref()
            if not head_d.attr_list and t.attr_list:
                continue  # Sort-constrained rule — skip
            # Handle conditional: body = (value | condition) — skip if conditioned
            if body_d.type is not None and body_d.type is wl.such_that:
                continue  # Can't evaluate conditionals without engine; skip
            # Unify head with pre-evaluated copy of t to bind arguments.
            # Unifying with disjunction terms may create orphaned choice points;
            # restore the choice stack afterward to discard them.
            mark = eng.trail.mark()
            ok = eng.unifier.unify(t_pre, head)
            if ok:
                result = _eval_arith(body_d, eng, _depth + 1)
                eng.trail.undo_to(mark)
                eng.choice_stack = _cp_save  # Discard any orphaned choice points
                if result[0]:
                    return result
                # Evaluation failed; try next rule
            eng.trail.undo_to(mark)
            eng.choice_stack = _cp_save  # Discard any orphaned choice points

    # Feature access: T.F → evaluate as arithmetic if possible
    if sym == '.':
        arg1, arg2 = _get_two_args(t)
        feat_val = _try_eval_string_func(t, eng)
        if feat_val is not None:
            return _eval_arith(feat_val, eng, _depth + 1)
        return False, 0.0

    # eval(Expr) — evaluate arithmetic expression (also unwraps backtick-quoted terms)
    if sym == 'eval':
        a1 = t.attr_list.get('1')
        if a1 is None:
            return False, 0.0
        a1d = a1.deref()
        # If arg is a backtick-quoted term `(Expr), unwrap it before evaluating
        a1_sym = a1d.type.keyword.symbol if a1d.type and a1d.type.keyword else ''
        if a1_sym == '`':
            inner = a1d.attr_list.get('1')
            if inner is not None:
                a1d = inner.deref()
        # A call is reduced for its value, not into its value: that is left to
        # the eval branch of _try_eval_string_func, which reduces a copy, so
        # `A = eval(X:f(X))` answers 1 and leaves X the call it was.
        if _is_user_function(a1d):
            return False, 0.0
        return _eval_arith(a1d, eng, _depth + 1)

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
        'msb': lambda a: int(math.log2(max(1, int(a)))),
        # Bitwise NOT
        '\\': lambda a: float(~int(a)),
    }
    # Unary arithmetic functions are applied here, ahead of the binary-operator
    # early exit below: that exit rejects every symbol outside the binary set,
    # which is all of floor, sqrt, abs and the rest.
    if sym in ops1:
        _un1, _un2 = _get_two_args(t)
        if _un2 is None and _un1 is not None:
            _un_ok, _un_v = _eval_arith(_un1, eng, _depth + 1)
            if not _un_ok:
                return False, 0.0
            try:
                return True, float(ops1[sym](_un_v))
            except OverflowError:
                # Past what a double holds: `exp(1e300)` is Infinity, the way
                # the C library answers it.
                return True, (math.inf if _un_v >= 0 else -math.inf)
            except ValueError:
                # Undefined there: `cos(Infinity)` is NaN.
                if not math.isfinite(_un_v):
                    return True, math.nan
                return False, 0.0
            except Exception:
                return False, 0.0

    # Binary operators — early exit if sym is not a known arithmetic binary op.
    # This prevents infinite recursion on cyclic terms like cons(A,A) where
    # the 'cons' symbol is not arithmetic but the pre-check would recurse forever.
    _arith_binary_syms = frozenset(('+', '-', '*', '/', '//', 'mod', '^',
                                     'max', 'min', '/\\', '\\/', 'xor', '>>', '<<'))
    # The handlers further down (strlen, asc, int, real and the 0-ary
    # cpu_time / real_time / genint) sit past this exit, so they are named
    # here — otherwise none of them would ever be reached.
    _arith_late_syms = frozenset(('strlen', 'asc', 'int', 'real', 'random',
                                  'cpu_time', 'real_time', 'genint'))
    if sym not in _arith_binary_syms and sym not in _arith_late_syms:
        return False, 0.0
    arg1, arg2 = _get_two_args(t)
    ok1, v1 = _eval_arith(arg1, eng, _depth + 1) if arg1 else (False, 0.0)
    ok2, v2 = _eval_arith(arg2, eng, _depth + 1) if arg2 else (False, 0.0)

    ops2 = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        'mod': lambda a, b: float(int(a) % int(b)) if b != 0 else 0.0,
        '^': lambda a, b: a ** b,
        'max': lambda a, b: max(a, b),
        'min': lambda a, b: min(a, b),
        # Bitwise operators
        '/\\': lambda a, b: float(int(a) & int(b)),
        '\\/': lambda a, b: float(int(a) | int(b)),
        'xor': lambda a, b: float(int(a) ^ int(b)),
        '>>': lambda a, b: float(int(a) >> int(b)),
        '<<': lambda a, b: float(int(a) << int(b)),
    }
    # `mod` counts whole remainders, so a side with a fraction is not
    # something it can be asked about: C answers No to `2 mod 1.5`.
    if sym == 'mod' and ((ok1 and v1 != int(v1)) or (ok2 and v2 != int(v2))):
        return False, 0.0
    if sym in ('//', '/') and (ok1 or ok2):
        # Dividing by zero, and for integer division a non-integer argument,
        # leave the expression unevaluated and are reported by the caller (see
        # _report_division_problem), so that a goal proved several times over
        # does not repeat the diagnostic.
        if sym == '//' and ((ok1 and v1 != int(v1)) or (ok2 and v2 != int(v2))):
            return False, 0.0
        if ok2 and v2 == 0:
            return False, 0.0
        if ok1 and ok2:
            return True, _int_div(v1, v2) if sym == '//' else v1 / v2

    if sym in ops2 and ok1 and ok2:
        try:
            _result_val = float(ops2[sym](v1, v2))
            # A product is a number the program has been handed: `N*fact(N-1)`
            # owes the int rule each partial result.  _try_eval_arith_to_term
            # leaves products alone for the same reason.
            if sym == '*':
                from wild_life.runtime import WL as _WL_mul
                if (_WL_mul.delay_rules and eng is not None
                        and not getattr(eng, '_in_fire_delay', False)):
                    _prod_term = _make_number(eng, _result_val)
                    _prod_term._delay_fired = True
                    eng.unifier._fire_delay_rules(_prod_term, _prod_term.type)
            return True, _result_val
        except Exception:
            return False, 0.0

    if sym in ops1 and ok1 and arg2 is None:
        try:
            return True, float(ops1[sym](v1))
        except Exception:
            return False, 0.0

    # strlen(String) — length of string as integer
    if sym == 'strlen':
        a1 = t.attr_list.get('1')
        if a1 is None:
            return False, 0.0
        a1d = a1.deref()
        # Only evaluate when the argument is a concrete string value
        if a1d.value is not None and eng is not None:
            wl = eng.wl
            if a1d.type and a1d.type.is_subtype_of(wl.quoted_string):
                return True, float(len(str(a1d.value)))
        # Fall through: cannot evaluate (unbound variable etc.)
        return False, 0.0

    # asc(C) — ASCII code of character C (first character, mod 256)
    if sym == 'asc':
        _state, _val = _asc_argument(t, eng)
        return (True, _val) if _state == 'code' else (False, 0.0)

    # int(X) — integer part of X (truncate towards zero)
    if sym == 'int':
        a1 = t.attr_list.get('1')
        if a1 is None:
            return False, 0.0
        a1d = a1.deref()
        ok, v = _eval_arith(a1d, eng)
        if ok:
            return True, float(int(v))
        return False, 0.0

    # real(X) / float(X) — convert X to floating-point
    if sym == 'real':
        a1 = t.attr_list.get('1')
        if a1 is None:
            return False, 0.0
        a1d = a1.deref()
        ok, v = _eval_arith(a1d, eng)
        if ok:
            return True, float(v)
        return False, 0.0

    # random(N) — a random integer in [0,N), drawn from the generator that
    # initrandom(Seed) seeds, so that the same seed replays the same sequence.
    if sym == 'random':
        a1 = t.attr_list.get('1')
        if a1 is None:
            return False, 0.0
        ok, v = _eval_arith(a1.deref(), eng)
        if not ok or v <= 0:
            return False, 0.0
        return True, float(_random_gen(eng).randrange(int(v)))

    # cpu_time — 0-ary function returning process CPU time in seconds
    if sym == 'cpu_time' and not t.attr_list:
        return True, float(time.process_time())

    # real_time — 0-ary function returning wall-clock time in seconds
    if sym == 'real_time' and not t.attr_list:
        return True, float(time.time())

    # genint — 0-ary global integer counter; increments on each evaluation
    if sym == 'genint' and not t.attr_list:
        wl = getattr(eng, 'wl', None) if eng is not None else None
        if wl is not None:
            current = getattr(wl, '_genint_counter', 0) + 1
            wl._genint_counter = current
            return True, float(current)
        return False, 0.0

    return False, 0.0


def _get_linear_coeff(expr, x_var, eng):
    """Return (a, b) such that expr = a*x_var + b, or None if not linear in x_var.

    x_var is the PsiTerm node (already deref'd) for the single free variable.
    a and b are floats.  Works recursively on +, -, *, unary -.
    """
    if expr is None:
        return None
    expr = expr.deref()
    # Is this node the variable itself?
    if id(expr) == id(x_var):
        return (1.0, 0.0)
    # Is it a concrete number?
    ok, val = _eval_arith(expr, eng)
    if ok:
        return (0.0, val)
    # Is it an arithmetic expression?
    sym = expr.type.keyword.symbol if expr.type and expr.type.keyword else ''
    if sym not in _ARITH_OPS_SET:
        return None
    arg1, arg2 = _get_two_args(expr)
    if sym == '+':
        if arg1 is None or arg2 is None:
            return None
        c1 = _get_linear_coeff(arg1, x_var, eng)
        c2 = _get_linear_coeff(arg2, x_var, eng)
        if c1 is None or c2 is None:
            return None
        return (c1[0] + c2[0], c1[1] + c2[1])
    elif sym == '-':
        if arg1 is None:
            return None
        c1 = _get_linear_coeff(arg1, x_var, eng)
        if c1 is None:
            return None
        if arg2 is None:
            # unary minus
            return (-c1[0], -c1[1])
        c2 = _get_linear_coeff(arg2, x_var, eng)
        if c2 is None:
            return None
        return (c1[0] - c2[0], c1[1] - c2[1])
    elif sym == '*':
        if arg1 is None or arg2 is None:
            return None
        ok1, v1 = _eval_arith(arg1, eng)
        ok2, v2 = _eval_arith(arg2, eng)
        if ok1:
            c2 = _get_linear_coeff(arg2, x_var, eng)
            if c2 is None:
                return None
            return (v1 * c2[0], v1 * c2[1])
        if ok2:
            c1 = _get_linear_coeff(arg1, x_var, eng)
            if c1 is None:
                return None
            return (c1[0] * v2, c1[1] * v2)
        return None
    elif sym in ('/', '//'):
        # An integer division reads as a division here; what keeps `3 = A//2`
        # from answering A = 6 is the caller, which takes the answer only
        # where it comes to nothing at all.
        if arg1 is None or arg2 is None:
            return None
        ok2, v2 = _eval_arith(arg2, eng)
        if ok2 and v2 != 0.0:
            c1 = _get_linear_coeff(arg1, x_var, eng)
            if c1 is None:
                return None
            return (c1[0] / v2, c1[1] / v2)
        return None
    return None


# Returned by a solver that has shown the equation has no solution at all,
# as opposed to None, which only says this solver could not find one.
_NO_SOLUTION = object()


def _report_division_problem(t: 'PsiTerm', eng, _depth: int = 0) -> bool:
    """Report the first arithmetic fault in a sub-term of t, if there is one.

    No division takes a zero divisor, integer division additionally needs
    integer arguments, and neither a square root of a negative number nor a
    logarithm of zero or less has a value.  Each fault is decidable as soon as
    the offending argument is known, with any other one still free, so
    reporting it here lets the caller fail a goal rather than suspend on a
    constraint that can never hold.  Returns True when something was reported.
    """
    if t is None or _depth > 10:
        return False
    t = t.deref()
    if t.type is None:
        return False
    _sym_rp = _get_sym(t)
    if _sym_rp in ('sqrt', 'log'):
        import sys as _sys_rp
        _a1_rp = t.attr_list.get('1')
        _ok_rp, _v_rp = (_eval_arith(_a1_rp, eng) if _a1_rp is not None
                         else (False, 0.0))
        if _ok_rp:
            _msg_rp = None
            if _sym_rp == 'sqrt' and _v_rp < 0:
                _msg_rp = 'square root of negative number'
            elif _sym_rp == 'log' and _v_rp == 0:
                _msg_rp = 'logarithm of zero'
            elif _sym_rp == 'log' and _v_rp < 0:
                _msg_rp = 'logarithm of negative number'
            if _msg_rp is not None:
                _sys_rp.stderr.write(
                    f"*** Error: {_msg_rp} in {_term_to_str(t, eng)}.\n")
                eng._arith_error = True
                return True
    if _sym_rp in ('//', '/'):
        import sys as _sys_div
        a1, a2 = t.attr_list.get('1'), t.attr_list.get('2')
        ok1, v1 = _eval_arith(a1, eng) if a1 is not None else (False, 0.0)
        ok2, v2 = _eval_arith(a2, eng) if a2 is not None else (False, 0.0)
        if _sym_rp == '//':
            for arg, ok, val in ((a1, ok1, v1), (a2, ok2, v2)):
                if ok and val != int(val):
                    _sys_div.stderr.write(
                        f"*** Warning: argument '{_term_to_str(arg.deref(), eng)}' "
                        f"of integer division is not an integer.\n")
                    eng._arith_error = True
                    return True
        if ok2 and v2 == 0:
            _sys_div.stderr.write(
                f"*** Error: division by zero in {_term_to_str(t, eng)}.\n")
            eng._arith_error = True
            return True
    for sub in t.attr_list.values():
        if _report_division_problem(sub, eng, _depth + 1):
            return True
    return False


def _solve_int_div_divisor(dividend: float, quotient: float):
    """Solve `dividend // x == quotient` for x.

    `//` truncates toward zero, so |x| ranges over (|a|/(|v|+1), |a|/|v|] and x
    takes the sign of a*v.  Exactly one integer in that range is the solution;
    an empty range means the equation has none (_NO_SOLUTION); a wider one
    leaves several divisors, so the constraint residuates instead (None).
    A zero dividend is the degenerate case C Wild Life answers with 0.
    """
    if dividend != int(dividend) or quotient != int(quotient):
        return None
    a, v = abs(int(dividend)), abs(int(quotient))
    if a == 0:
        return 0.0
    if v == 0:
        return _NO_SOLUTION
    lo = a // (v + 1) + 1
    hi = a // v
    if lo > hi:
        return _NO_SOLUTION
    if lo != hi:
        return None
    return float(lo if (dividend > 0) == (quotient > 0) else -lo)


def _try_solve_nonlinear(expr, x_var, v_lhs, eng):
    """Try to solve expr = v_lhs for x_var when the expression is not linear.

    Handles simple inversion patterns:
      a / x = v  →  x = a / v    (denominator is the unknown)
      x ^ n = v  →  x = v^(1/n)  (x to an integer power, v ≥ 0)
      \(x) = v   →  x = ~v = -(v+1)  (bitwise NOT inversion)

    Returns the solution as a float, or None if no pattern matched.
    """
    if expr is None:
        return None
    expr = expr.deref()
    sym = expr.type.keyword.symbol if expr.type and expr.type.keyword else ''
    if sym not in _ARITH_OPS_SET:
        return None
    arg1, arg2 = _get_two_args(expr)

    # ── Unary operators (arg2 is None) ────────────────────────────────────────
    if sym == '\\' and arg1 is not None and arg2 is None:
        # \ (bitwise NOT): \(x) = v  →  x = ~v = -(v+1)
        arg1_d = arg1.deref()
        if id(arg1_d) == id(x_var):
            return float(~int(round(v_lhs)))
        return None

    if arg1 is None or arg2 is None:
        return None

    if sym == '/':
        arg1_d = arg1.deref()
        arg2_d = arg2.deref()
        # x / x = v → (v-1)*x = 0. If v≠1: x=0
        if id(arg1_d) == id(x_var) and id(arg2_d) == id(x_var):
            if abs(v_lhs - 1.0) > 1e-12:
                return 0.0  # x=0
        # a / x = v  →  x = a / v  (only when arg2 contains x_var and is x_var itself)
        if id(arg2_d) == id(x_var):
            ok1, v1 = _eval_arith(arg1, eng)
            if ok1 and v_lhs != 0.0:
                return v1 / v_lhs
    if sym == '//':
        # x // x = v → (v-1)*x = 0, so v≠1 leaves x nothing but 0, the same
        # as for `/`: `24 = B//B` answers B = 0.
        if (id(arg1.deref()) == id(x_var) and id(arg2.deref()) == id(x_var)
                and abs(v_lhs - 1.0) > 1e-12):
            return 0.0
        # a // x = v  →  x, when exactly one integer divisor gives v
        if id(arg2.deref()) == id(x_var):
            ok1, v1 = _eval_arith(arg1, eng)
            if ok1:
                return _solve_int_div_divisor(v1, v_lhs)
    if sym == '*':
        arg1_d = arg1.deref()
        arg2_d = arg2.deref()
        if id(arg1_d) == id(x_var) and id(arg2_d) == id(x_var):
            # x * x = v
            if v_lhs < 0:
                return _NO_SOLUTION   # no real number squares to a negative
            if abs(v_lhs) < 1e-12:
                return 0.0
            # A positive v has two roots, so leave it for the constraint.
    # Could extend with x^n etc., but division covers the main arith cases
    return None


def _linear_decompose_psi(expr, x_var, eng, wl):
    """Decompose expr into (a_coeff, b_psi) where expr = a_coeff * x_var + b_psi.

    a_coeff is a float (coefficient of x_var in expr).
    b_psi is a PsiTerm for the remainder (may not be evaluable to a concrete number).
    Returns None if expr is not linear in x_var.

    This extends _get_linear_coeff to return a symbolic remainder PsiTerm
    so we can solve cases like A = A+C → 0 = C even when C is free.
    """
    if expr is None:
        return None
    expr = expr.deref()
    zero_term = wl.make_integer(0)
    # Is this the variable itself?
    if id(expr) == id(x_var):
        return (1.0, zero_term)
    # Is it a concrete number?
    ok, val = _eval_arith(expr, eng)
    if ok:
        return (0.0, _make_number(eng, val))
    # Is it an arithmetic expression?
    sym = expr.type.keyword.symbol if expr.type and expr.type.keyword else ''
    if sym not in _ARITH_OPS_SET:
        # Another free variable (not x_var) — treat as constant remainder.
        return (0.0, expr)
    arg1, arg2 = _get_two_args(expr)
    if sym == '+':
        if arg1 is None or arg2 is None:
            return None
        r1 = _linear_decompose_psi(arg1, x_var, eng, wl)
        r2 = _linear_decompose_psi(arg2, x_var, eng, wl)
        if r1 is None or r2 is None:
            return None
        # b_psi = r1[1] + r2[1]
        a_coeff = r1[0] + r2[0]
        b1, b2 = r1[1], r2[1]
        ok1, v1 = _eval_arith(b1, eng)
        ok2, v2 = _eval_arith(b2, eng)
        if ok1 and v1 == 0.0:
            b_psi = b2
        elif ok2 and v2 == 0.0:
            b_psi = b1
        else:
            # Build b1 + b2 PsiTerm
            plus_sym = expr.type  # reuse the same + type
            b_psi = PsiTerm(type_def=plus_sym)
            b_psi.attr_list['1'] = b1
            b_psi.attr_list['2'] = b2
        return (a_coeff, b_psi)
    elif sym == '-':
        if arg1 is None:
            return None
        r1 = _linear_decompose_psi(arg1, x_var, eng, wl)
        if r1 is None:
            return None
        if arg2 is None:
            # Unary minus
            ok_b, v_b = _eval_arith(r1[1], eng)
            if ok_b:
                return (-r1[0], _make_number(eng, -v_b))
            # Negate b_psi symbolically
            minus_sym = expr.type
            neg_b = PsiTerm(type_def=minus_sym)
            neg_b.attr_list['1'] = r1[1]
            return (-r1[0], neg_b)
        r2 = _linear_decompose_psi(arg2, x_var, eng, wl)
        if r2 is None:
            return None
        a_coeff = r1[0] - r2[0]
        b1, b2 = r1[1], r2[1]
        ok1, v1 = _eval_arith(b1, eng)
        ok2, v2 = _eval_arith(b2, eng)
        if ok2 and v2 == 0.0:
            b_psi = b1
        elif ok1 and v1 == 0.0:
            # 0 - b2
            minus_sym = expr.type
            b_psi = PsiTerm(type_def=minus_sym)
            b_psi.attr_list['1'] = zero_term
            b_psi.attr_list['2'] = b2
        else:
            minus_sym = expr.type
            b_psi = PsiTerm(type_def=minus_sym)
            b_psi.attr_list['1'] = b1
            b_psi.attr_list['2'] = b2
        return (a_coeff, b_psi)
    elif sym == '*':
        if arg1 is None or arg2 is None:
            return None
        ok1, v1 = _eval_arith(arg1, eng)
        ok2, v2 = _eval_arith(arg2, eng)
        if ok1:
            r2 = _linear_decompose_psi(arg2, x_var, eng, wl)
            if r2 is None:
                return None
            ok_b, v_b = _eval_arith(r2[1], eng)
            b_psi = _make_number(eng, v1 * v_b) if ok_b else r2[1]
            return (v1 * r2[0], b_psi)
        if ok2:
            r1 = _linear_decompose_psi(arg1, x_var, eng, wl)
            if r1 is None:
                return None
            ok_b, v_b = _eval_arith(r1[1], eng)
            b_psi = _make_number(eng, v_b * v2) if ok_b else r1[1]
            return (r1[0] * v2, b_psi)
        return None
    elif sym in ('/', '//'):
        if arg1 is None or arg2 is None:
            return None
        ok2, v2 = _eval_arith(arg2, eng)
        if ok2 and v2 != 0.0:
            r1 = _linear_decompose_psi(arg1, x_var, eng, wl)
            if r1 is None:
                return None
            ok_b, v_b = _eval_arith(r1[1], eng)
            b_psi = _make_number(eng, v_b / v2) if ok_b else r1[1]
            return (r1[0] / v2, b_psi)
        return None
    return None


def _has_int_div(t: 'PsiTerm', _depth: int = 0) -> bool:
    """Whether a `//` is written anywhere inside the expression.

    An integer division only pins its dividend down where the answer is
    nothing at all: `0 = A//2` says A is 0, while `3 = A//2` leaves A
    waiting, since 6 and 7 both divide to 3.
    """
    if t is None or _depth > 12:
        return False
    t = t.deref()
    if t.type is not None and t.type.keyword is not None \
            and t.type.keyword.symbol == '//':
        return True
    for _sub in t.attr_list.values():
        if _has_int_div(_sub, _depth + 1):
            return True
    return False


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


def _feature_arg_term(feat: PsiTerm, eng):
    """The term a feature-name argument names, calls worked out.

    `has_feature(combined_name(X),predicates_info)` asks about the label
    combined_name answers, not about one called combined_name.
    """
    if feat is None:
        return feat
    # Only a call written where the label goes is worked out.  A variable's
    # value is the term it already is: `has_feature(Leaf,table,Pred)` asks
    # about the sort of the leaf it was handed, `a + in`, and not about what
    # adding a to in would come to.
    _written = feat.deref() is feat
    feat = feat.deref()
    if (_written and feat.attr_list and eng is not None
            and feat.value is None
            and _get_sym(feat) not in _ARITH_OPS_SET):
        _ev = _try_eval_string_func(feat, eng)
        if _ev is not None:
            _ev = _ev.deref()
            if _ev is not feat and (_ev.value is not None
                                    or (_ev.type is not None
                                        and _ev.type.keyword is not None)):
                return _ev
    return feat


def _feature_name_of(feat: PsiTerm, wl):
    """The feature key a term names, or None if it names none.

    `1` names the first positional feature, and so do the string "1" and the
    atom `'1'` that str2psi("1") builds.
    """
    if feat is None:
        return None
    if feat.value is not None:
        v = feat.value
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    if feat.type is not None and feat.type.keyword is not None:
        return feat.type.keyword.symbol
    return None


def _find_embedded_user_func(t: PsiTerm, _seen: set = None, _depth: int = 0):
    """The first user-defined function call inside t, below t itself."""
    if t is None or _depth > 20:
        return None
    if _seen is None:
        _seen = set()
    t = t.deref()
    if id(t) in _seen:
        return None
    _seen.add(id(t))
    for ref in t.attr_list.values():
        sub = ref.deref()
        if _is_user_function(sub):
            return sub
        found = _find_embedded_user_func(sub, _seen, _depth + 1)
        if found is not None:
            return found
    return None


def _push_deferred_cmp(goal: PsiTerm, eng, a, b, oka, okb) -> bool:
    """If one arg is an unevaluated user function, defer via EVAL + PROVE.

    Returns True if goals were pushed (evaluation deferred), False otherwise.
    After EVAL binds R to the function's value, also unifies the original arg
    with R so that subsequent uses of that variable see the computed value.
    """
    from wild_life.inference import _DEFRULES
    wl = eng.wl

    def _defer(func_arg, other_arg, func_is_first: bool) -> bool:
        """Push EVAL(func) + UNIFY(func_arg, R) + PROVE(cmp(R, other))."""
        func_d = func_arg.deref()
        if not _is_user_function(func_d):
            return False
        R = wl.make_var()
        # Build new comparison goal term with R substituted for func_arg
        new_goal = PsiTerm(type_def=goal.type)
        if func_is_first:
            new_goal.attr_list = {'1': R, '2': other_arg}
        else:
            new_goal.attr_list = {'1': other_arg, '2': R}
        # Push in LIFO order (goals execute in reverse push order):
        #   1. EVAL(func → R)     — evaluate the function, binding R
        #   2. UNIFY(func_arg, R) — bind the original arg to R so it's shared
        #   3. PROVE(cmp(R, b))   — run the comparison with the now-known value
        eng.push_goal(GoalType.PROVE, new_goal, _DEFRULES, None)
        eng.push_goal(GoalType.UNIFY, func_d, R, None)
        eng.push_goal(GoalType.EVAL, func_d, R, func_d.type.rule)
        return True

    if not oka and a is not None:
        if _defer(a, b, True):
            return True
    if not okb and b is not None:
        if _defer(b, a, False):
            return True

    # A call buried inside the expression holds the comparison up just as much:
    # `A*A =:= B*B+C*C` over A:digit, B:digit, C:digit asks each digit for its
    # value.  One is reduced and the comparison is put back, so the next one is
    # found on the way round.
    for _side in (a, b):
        if _side is None:
            continue
        _sub = _find_embedded_user_func(_side)
        if _sub is None:
            continue
        _R = wl.make_var()
        eng.push_goal(GoalType.PROVE, goal, _DEFRULES, None)
        eng.push_goal(GoalType.UNIFY, _sub, _R, None)
        eng.push_goal(GoalType.EVAL, _sub, _R, _sub.type.rule)
        return True

    # Neither side is a call waiting to be made, so what is missing is a value.
    # The comparison suspends on the variables that hold it up and is proven
    # again when one of them is bound, which is how `pyth(A,B,C)` can state
    # `A*A =:= B*B+C*C` before A, B and C are known.
    _cmp_vars: list = []
    _cmp_seen: set = set()
    for _side in (a, b):
        if _side is not None:
            _collect_arith_vars(_side, wl, _cmp_vars, _cmp_seen)
    if _cmp_vars:
        from wild_life.data_structures import Goal as _CmpPredGoal
        _pend = _CmpPredGoal(GoalType.PROVE, goal, _DEFRULES, None, pending=True)
        for _cv in _cmp_vars:
            _attach_arith_resid(_cv, wl, _pend, eng)
        return True
    # Nothing is missing, so the operands are simply not computable — a
    # division by zero among them is reported here rather than passed off as a
    # comparison that merely did not hold.
    for _side in (a, b):
        if _side is not None and _report_division_problem(_side, eng):
            break
    return False


def bi_arith_eq(goal: PsiTerm, eng) -> bool:
    """X =:= Y — arithmetic equality."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    if oka and okb:
        return va == vb
    return _push_deferred_cmp(goal, eng, a, b, oka, okb)


def bi_arith_ne(goal: PsiTerm, eng) -> bool:
    """X =\\= Y — arithmetic inequality."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    if oka and okb:
        return va != vb
    return _push_deferred_cmp(goal, eng, a, b, oka, okb)


def bi_arith_lt(goal: PsiTerm, eng) -> bool:
    """X < Y."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    if oka and okb:
        return va < vb
    return _push_deferred_cmp(goal, eng, a, b, oka, okb)


def bi_arith_le(goal: PsiTerm, eng) -> bool:
    """X =< Y."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    if oka and okb:
        return va <= vb
    return _push_deferred_cmp(goal, eng, a, b, oka, okb)


def bi_arith_gt(goal: PsiTerm, eng) -> bool:
    """X > Y."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    if oka and okb:
        return va > vb
    return _push_deferred_cmp(goal, eng, a, b, oka, okb)


def bi_arith_ge(goal: PsiTerm, eng) -> bool:
    """X >= Y."""
    a, b = _get_two_args(goal)
    oka, va = _eval_arith(a, eng)
    okb, vb = _eval_arith(b, eng)
    if oka and okb:
        return va >= vb
    return _push_deferred_cmp(goal, eng, a, b, oka, okb)


# ─────────────────────────────────────────────────────────────────────────────
# String comparison operators:  A$>B  A$>=B  A$<B  A$=<B  A$==B  A$\==B
# ─────────────────────────────────────────────────────────────────────────────

def _get_str_val(t: PsiTerm, eng) -> Optional[str]:
    """Return the string comparison key for a psi-term, or None on failure.

    Wild Life string comparisons ($>, $<, etc.) compare the *print name*
    of atoms and strings.

    Rules:
      - Quoted string (backtick literal): return t.value (the raw string)
      - Atom:                             return t.type.keyword.symbol
      - Number (int/float):               return str(int(v)) or str(v)
      - Anything else (variable, compound): return None → predicate fails
    """
    if t is None:
        return None
    t = t.deref()
    wl = eng.wl
    # Quoted string — value holds the raw string content
    if t.type is not None and t.type.is_subtype_of(wl.quoted_string):
        return str(t.value) if t.value is not None else ''
    # Number
    if t.value is not None:
        v = t.value
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    # Atom: a plain atom has a keyword symbol and no children/value
    if t.type is not None and t.type.keyword is not None and not t.attr_list:
        return t.type.keyword.symbol
    # Anything else (variable, compound term): cannot compare
    return None


def bi_str_gt(goal: PsiTerm, eng) -> bool:
    """A $> B — string greater-than."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    sa, sb = _get_str_val(a, eng), _get_str_val(b, eng)
    if sa is None or sb is None:
        return False
    return sa > sb


def bi_str_ge(goal: PsiTerm, eng) -> bool:
    """A $>= B — string greater-than-or-equal."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    sa, sb = _get_str_val(a, eng), _get_str_val(b, eng)
    if sa is None or sb is None:
        return False
    return sa >= sb


def bi_str_lt(goal: PsiTerm, eng) -> bool:
    """A $< B — string less-than."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    sa, sb = _get_str_val(a, eng), _get_str_val(b, eng)
    if sa is None or sb is None:
        return False
    return sa < sb


def bi_str_le(goal: PsiTerm, eng) -> bool:
    """A $=< B — string less-than-or-equal."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    sa, sb = _get_str_val(a, eng), _get_str_val(b, eng)
    if sa is None or sb is None:
        return False
    return sa <= sb


def bi_str_eq(goal: PsiTerm, eng) -> bool:
    """A $== B — string equality."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    sa, sb = _get_str_val(a, eng), _get_str_val(b, eng)
    if sa is None or sb is None:
        return False
    return sa == sb


def bi_str_ne(goal: PsiTerm, eng) -> bool:
    r"""A $\== B — string inequality."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    sa, sb = _get_str_val(a, eng), _get_str_val(b, eng)
    if sa is None or sb is None:
        return False
    return sa != sb


# ─────────────────────────────────────────────────────────────────────────────
# Unification / comparison
# ─────────────────────────────────────────────────────────────────────────────

def _collect_disjunction(t: PsiTerm, eng) -> list:
    """Collect all leaf elements from a disjunction linked-list into a flat list.

    {1;2;3} is stored as disj(1, disj(2, disj(3, disj_nil))).
    Returns [term1, term2, term3].
    """
    elems = []
    node = t
    disj_nil = eng.wl.disj_nil
    while node is not None:
        node = node.deref()
        if node.type is None:
            break
        if node.type is disj_nil or node.type is eng.wl.disj_nil:
            break
        if node.type is eng.wl.disjunction:
            head = node.attr_list.get('1')
            tail = node.attr_list.get('2')
            if head is not None:
                elems.append(head.deref())
            node = tail.deref() if tail else None
        else:
            # Not a disjunction node — treat as leaf
            elems.append(node)
            break
    return elems


def _term_contains_disjunction(t: PsiTerm, eng, depth: int = 0,
                               visited: set = None) -> bool:
    """Return True if t (or any subterm up to depth 10) is a disjunction.

    A term whose parts point at one another is walked once: matrix builds a
    grid of squares that reach each other by many paths, and asking each of
    them the same question again for every path is what made it slow.
    """
    if depth > 10:
        return False
    t = t.deref()
    if t.type is None:
        return False
    # A term held as it is written is not evaluated, so a choice inside it
    # is part of what is written: the `{40;41;…}` of a grammar rule reaches
    # the clause the rule compiles to whole.
    if t.flags & QUOTED_TRUE:
        return False
    if t.type is eng.wl.disjunction:
        return True
    if not t.attr_list:
        return False
    if visited is None:
        visited = set()
    elif id(t) in visited:
        return False
    visited.add(id(t))
    for v in t.attr_list.values():
        if _term_contains_disjunction(v, eng, depth + 1, visited):
            return True
    return False


def _disjunction_nodes(t: PsiTerm, eng, _seen: set = None,
                       _depth: int = 0) -> list:
    """The disjunction nodes written inside a term, in the order they read."""
    out: list = []
    if t is None or _depth > 40:
        return out
    if _seen is None:
        _seen = set()
    t = t.deref()
    if id(t) in _seen or (t.flags & QUOTED_TRUE):
        return out
    _seen.add(id(t))
    for _k in sorted(t.attr_list.keys()):
        _sub = t.attr_list[_k].deref()
        if _sub.flags & QUOTED_TRUE:
            continue
        if (_sub.type is eng.wl.disjunction and _sub.attr_list
                and id(_sub) not in _seen):
            _seen.add(id(_sub))
            out.append(_sub)
            continue
        out.extend(_disjunction_nodes(_sub, eng, _seen, _depth + 1))
    return out


def _expand_term_disjunctions(t: PsiTerm, eng) -> list:
    """Return a list of all alternative terms obtained by expanding embedded disjunctions.

    E.g., [{1;2;3}|T] → [[1|T], [2|T], [3|T]]
         f({a;b}, {c;d}) → [f(a,c), f(a,d), f(b,c), f(b,d)]
    """
    t = t.deref()
    if t.type is None:
        return [t]  # unbound variable

    if t.type is eng.wl.disjunction:
        return _collect_disjunction(t, eng)

    # Collect alternatives for each attribute
    attr_keys = list(t.attr_list.keys())
    if not attr_keys:
        return [t]

    # Build cartesian product of attribute alternatives
    # Start with a single combo (empty)
    combos = [{}]
    has_disj = False
    for key in attr_keys:
        val_d = t.attr_list[key].deref()
        alts = _expand_term_disjunctions(val_d, eng)
        if len(alts) > 1:
            has_disj = True
        new_combos = []
        for combo in combos:
            for alt in alts:
                new_combo = dict(combo)
                new_combo[key] = alt
                new_combos.append(new_combo)
        combos = new_combos

    if not has_disj:
        return [t]

    # Build new terms for each attribute combination
    result = []
    for attrs in combos:
        new_term = PsiTerm(type_def=t.type, value=t.value)
        new_term.attr_list = attrs
        result.append(new_term)
    return result


def _eval_user_function_deferred(t: PsiTerm, eng, result: PsiTerm) -> bool:
    """Push EVAL + deferred goals when t is a user function.

    Returns True if goals were pushed (t is a user function that needs
    evaluation). The caller should push additional continuation goals
    AFTER this call (they will execute after EVAL produces result).
    """
    t_d = t.deref()
    if not _is_user_function(t_d):
        return False
    from wild_life.data_structures import GoalType
    eng.push_goal(GoalType.EVAL, t_d, result, t_d.type.rule)
    return True


def _is_user_function(t: PsiTerm) -> bool:
    """Return True if t is a user-defined function call (has -> rules)."""
    if t is None:
        return False
    while t.coref is not None:
        t = t.coref
    defn = t.type
    if defn is None or defn.type != DefType.FUNCTION:
        return False
    if defn._builtin_func is not None or not defn.rule:
        return False
    # Backtick-quoted terms (QUOTED_TRUE) are sort references, not function
    # calls.  A call being reduced is the value its own rule body sees:
    # `X:sum -> f(X)` binds X to the very sum term under evaluation, and
    # reducing it again there would restart the rule instead of reading the
    # term's features.
    return not (t.flags & _IUF_SKIP_FLAGS)


def _settle_disj_body_sync(body_d, eng, _depth):
    """The alternative a body of alternatives comes to, read here and now.

    eval_aim unifies the body with the result, and a body of alternatives
    settles there to the first one that holds, keeping the rest as choice
    points.  Read where a value is wanted there is no goal to come back to,
    so the first alternative whose guard holds is the answer: transequ's
    `{ ([get_const(V,T)] | consta(T),!) ; … }` is the get_const list when the
    term is a constant.  Returns None when none of them holds.
    """
    if body_d is None or eng is None:
        return None
    body_d = body_d.deref()
    if body_d.type is not eng.wl.disjunction:
        return None
    _alts = _collect_disjunction(body_d, eng)
    if not _alts:
        return None
    from wild_life.inference import prove_cond as _pc_dj
    for _i_dj, _alt_dj in enumerate(_alts):
        _last_dj = (_i_dj == len(_alts) - 1)
        _a_d = _alt_dj.deref()
        _mark_dj = eng.trail.mark()
        if _a_d.type is eng.wl.such_that:
            _val_dj = _a_d.attr_list.get('1')
            _grd_dj = _a_d.attr_list.get('2')
            if _val_dj is None or _grd_dj is None:
                eng.trail.undo_to(_mark_dj)
                continue
            if not _pc_dj(_grd_dj.deref(), eng):
                eng.trail.undo_to(_mark_dj)
                continue
            # Only a guard that commits settles the body here.  The
            # alternatives after it are choice points in eval_aim, and
            # there is nowhere to keep them when the value is read out.
            # A guard ending in a cut takes them away itself, and the last
            # alternative has none after it to lose, so for those nothing
            # is lost.  Otherwise the call is left to the engine's own
            # EVAL goal, which can hold the alternatives.
            from wild_life.inference import _body_has_cut as _bhc_dj
            if not (_last_dj or _bhc_dj(_grd_dj.deref(), eng.wl)):
                eng.trail.undo_to(_mark_dj)
                return None
            _v_dj = _val_dj.deref()
        elif _last_dj:
            _v_dj = _a_d
        else:
            eng.trail.undo_to(_mark_dj)
            return None
        _ev_dj = _eval_body_sync(_v_dj, eng, _depth + 1)
        return _ev_dj if _ev_dj is not None else _v_dj
    return None


def _eval_user_func_sync(t: PsiTerm, eng, _depth: int = 0) -> Optional[PsiTerm]:
    """Synchronously evaluate a user-defined function call, cycles included.

    This is used to eagerly evaluate function-call arguments before pattern
    matching (e.g., reverse([1,2,3,4]) in rev(reverse([1,2,3,4]),[])).
    Only evaluates simple (unconditional, deterministic first-rule) cases.

    Returns the result PsiTerm, or None if evaluation can't proceed.
    The engine trail is NOT rolled back — bindings persist on the trail.

    How deep this may go is a guard against a call that reduces for ever, not
    a limit on what a program may ask for: factorize looks for a factor of
    44449 by trying every number up to 211, which is a chain of some six
    hundred reductions and a perfectly ordinary thing to ask.
    """
    if _depth > 2000:
        return None
    if t is None:
        return None
    t = t.deref()
    if not _is_user_function(t):
        return None

    from wild_life.unification import copy_term
    from wild_life.data_structures import QUOTED_TRUE

    # A call that reaches itself is not reduced through its own argument:
    # `X : f(X)` is the very call being evaluated, so it stands as it is while
    # the rule is matched against it.
    _active_sync = getattr(eng, '_sync_eval_active', None)
    if _active_sync is None:
        _active_sync = eng._sync_eval_active = set()
    if id(t) in _active_sync:
        return None
    _active_sync.add(id(t))
    # Working a call out here answers what it is worth; the alternatives met
    # on the way — a disjunction settling to its first element and keeping
    # the rest — belong to that working out, not to any goal, and a
    # backtrack into one would take the engine on from a goal it never
    # proved.
    _cs_sync = eng.choice_stack
    try:
        return _eval_user_func_sync_inner(t, eng, _depth)
    except RecursionError:
        # A chain of reductions longer than the Python stack holds is one
        # this cannot work out here; the call is left as it stands.
        return None
    finally:
        _active_sync.discard(id(t))
        eng.choice_stack = _cs_sync


def _eval_user_func_sync_inner(t: PsiTerm, eng, _depth: int) -> Optional[PsiTerm]:
    """The body of _eval_user_func_sync, once the call is known to be new."""
    from wild_life.unification import copy_term
    from wild_life.data_structures import QUOTED_TRUE

    # Try each rule in order (no backtracking support here)
    rules = t.type.rule or []
    active = [(h, b) for (h, b) in rules if h is not None and b is not None]

    # Pre-evaluate any user-defined or built-in functional sub-terms in
    # the input term's arguments before trying to unify with the head.
    # This mirrors the EVAL goal handler in inference.py (lines ~843-857)
    # and is necessary so that e.g. app([1], rev([2,3])) can match
    # app(L, [H|T]) after rev([2,3]) is reduced to [3,2].
    # Asked once for the call, not once per rule: a rule that does not match
    # puts the call's arguments back as they were, and an argument whose
    # reduction was a side effect — cb's create_vvars handing out numbered
    # variables — would hand out fresh ones on the next rule.
    # A function declared non_strict is handed its arguments as they are
    # written: check_func reduces them only when evaluate_args says so.
    _ns_sync = getattr(eng, 'non_strict_set', None)
    _strict_args = (() if (_ns_sync and t.type in _ns_sync)
                    else list(t.attr_list.keys()))
    for _key in _strict_args:
        _attr = t.attr_list[_key].deref()
        # An argument held as it is written stays as it is: the `{ … }` a
        # grammar rule hands the expander is the goal it was written as.
        if _attr.flags & QUOTED_TRUE:
            continue
        # A backquote's work is done once the term is handed over: what the
        # call is given is the term itself, held as it is written.
        # std_expander.lf's `X comma Y` compares X with `succeed`, and a
        # quote left standing in front of it makes that comparison false
        # however the code a grammar rule carries came out.
        if (_attr.type is not None and _attr.type.keyword is not None
                and _attr.type.keyword.symbol == '`'
                and list(_attr.attr_list.keys()) == ['1']):
            from wild_life.inference import (
                _mark_arith_non_strict as _mans_bq,
                _freeze_calls_deep as _fcd_bq)
            _inner_bq = _attr.attr_list['1'].deref()
            _mans_bq(_inner_bq)
            _fcd_bq(_inner_bq, QUOTED_TRUE)
            eng.unifier.set_attr(t, _key, _inner_bq)
            continue
        _ev = _try_eval_any_func(_attr, eng, _depth + 1)
        if _ev is None and _attr.attr_list:
            # Compound arg (e.g. `(CX, NT) & memo_copy(X, Table)` or
            # `(B, NT) & copy_body(...)`) — use _eval_body_sync so that
            # `&` conjunction semantics are handled (evaluate RHS and
            # unify with LHS), rather than just evaluating sub-functions
            # in-place without the conjunction unification step.
            _ev = _eval_body_sync(_attr, eng, _depth + 1)
        # A number that carries features is already its own value, and the
        # bare number is less than the term is: `term_explore(2(2), Seen)`
        # has a feature to count.
        if (_ev is not None and _attr.value is not None and _attr.attr_list
                and not _ev.deref().attr_list):
            _ev = None
        if _ev is not None and _ev is not _attr:
            # Trailed: the value was worked out under bindings that a
            # later backtrack may undo, and a call left holding a stale
            # one would go on reducing against variables nothing binds
            # any more.
            eng.unifier.set_attr(t, _key, _ev)
    # What a rule that does not match puts back is the call as it now
    # stands, with its arguments worked out.
    t_copy_attrs = dict(t.attr_list)

    for h0, b0 in active:
        _vm: dict = {}
        head = copy_term(h0, _vm)
        body = copy_term(b0, _vm)
        body_d = body.deref()

        # A rule whose head asks for features the call does not supply belongs
        # to a partial application: `e5(F,A) -> F(A)` passes `p` as a value, and
        # reducing it against `p(X) -> …` here would both give `p` an argument
        # it was never called with and hand back p's body as the value of F.
        if set(head.deref().attr_list.keys()) - set(t.attr_list.keys()):
            continue

        # Handle conditional rule: body = val_part | guard
        # Run the guard with an inner proof and return the value part.
        if body_d.type is not None and body_d.type is eng.wl.such_that:
            val_part = body_d.attr_list.get('1')
            cond_part = body_d.attr_list.get('2')
            if val_part is None or cond_part is None:
                continue
            # Bind head arguments first (for arity > 0 functions)
            mark = eng.trail.mark()
            if head.attr_list:
                ok_h = eng.unifier.unify(t, head)
                if not ok_h:
                    eng.trail.undo_to(mark)
                    continue
            # When the value is a function call, point the rule's value
            # variable at a fresh node: the guard constrains the call's
            # *result*, so `Y.1` in
            #   bodify_list([(A,X)|T]) -> Y : bodify_list(T) | X = Y.A.
            # must not collide with the call's own first argument.  This has to
            # happen before the guard is touched at all, since
            # _eval_embedded_user_funcs already resolves `Y.A` in place.
            _st_call = None
            _vp_d = val_part.deref()
            if _is_user_function(_vp_d):
                _st_call = PsiTerm(type_def=_vp_d.type)
                _st_call.attr_list = dict(_vp_d.attr_list)
                _st_call.flags = _vp_d.flags
                eng.trail.trail_psi(_vp_d, 'coref')
                _vp_d.coref = PsiTerm(type_def=eng.wl.top)
            # Evaluate built-in / user-defined functional sub-terms inside
            # the guard goal (e.g. genChildren(children(X), A) → the
            # children(X) arg must be reduced before the predicate is called).
            # A conjunction is proven left to right, so only its leftmost goal
            # is ready: a later one is still waiting on what the goals before
            # it will bind or change, and reading `F = g` before the goals in
            # front of it have set the global g is how fact's factorial came
            # back with the number it started from.
            from wild_life.inference import _leftmost_goal as _lmg_sync
            _cond_d = cond_part.deref()
            _eval_embedded_user_funcs(_lmg_sync(_cond_d, eng.wl), eng,
                                      _depth + 1, set())
            # Run the guard in an inner proof loop.
            # IMPORTANT: clear goal_stack so only the guard is proved;
            # the outer continuation must not run inside this inner loop.
            from wild_life.inference import GoalType as _GoalType, _DEFRULES as _DR, _INNER_RUN_BARRIER as _IRB
            cp_save = eng.choice_stack
            gs_save = eng.goal_stack
            eng.goal_stack = None
            eng.push_goal(_GoalType.PROVE, _cond_d, _DR, None)
            old_ok = eng.main_loop_ok
            _barrier = cp_save if cp_save is not None else _IRB
            cond_ok = eng.run(cs_barrier=_barrier)
            eng.main_loop_ok = old_ok
            eng.choice_stack = cp_save
            eng.goal_stack = gs_save
            if cond_ok:
                if _st_call is not None:
                    # Reduce the call now, and merge it with the features the
                    # guard attached to the value node.  A call that cannot be
                    # reduced here stands as its own value.
                    _evaled = _eval_user_func_sync(_st_call, eng, _depth + 1) or _st_call
                    if not _unify(eng, val_part, _evaled):
                        eng.trail.undo_to(mark)
                        continue
                    return val_part.deref()
                val_d = val_part.deref()
                ok_a, val = _eval_arith(val_d, eng)
                if ok_a:
                    return _make_number(eng, val)
                _eval_embedded_user_funcs(val_d, eng, _depth + 1, set())
                return val_d
            else:
                eng.trail.undo_to(mark)
                continue

        # Matching is one-way here as well: `mult_list(2,6,X)` with X still a
        # variable does not match `mult_list(U,N,[H|T])`, and narrowing X to
        # [H|T] to make it fit would answer a question the call has not
        # settled.  A rule no narrowing could ever fit is passed over; one the
        # call is not specific enough for is left to the engine's own EVAL
        # goal, which suspends the call until a variable is bound.
        from wild_life.inference import (
            _rule_match_status as _rms_sync, _call_is_curried as _cic_sync)
        if _cic_sync(head.deref(), t):
            t.attr_list = t_copy_attrs
            return t
        _sync_match = _rms_sync(head.deref(), t, eng)
        if _sync_match == 'never':
            t.attr_list = t_copy_attrs
            continue
        if _sync_match != 'ready':
            t.attr_list = t_copy_attrs
            return None

        mark = eng.trail.mark()
        ok = eng.unifier.unify(t, head)
        if not ok:
            # Restore original attrs in case we modified them
            t.attr_list = t_copy_attrs
            eng.trail.undo_to(mark)
            continue

        body_d2 = body_d.deref()
        if body_d2.type is eng.wl.disjunction:
            _dj_val = _settle_disj_body_sync(body_d2, eng, _depth)
            if _dj_val is not None:
                return _dj_val
        result = _eval_body_sync(body_d2, eng, _depth + 1)
        if (result is None and _is_user_function(body_d2)
                and _has_applicable_rule(body_d2)):
            # Body is a user function that can't eval synchronously
            return None
        if result is None and _is_cond_builtin_local(body_d2):
            # A condition nothing has settled yet: the call has no value to
            # hand back, and handing back the cond itself would pass a cond
            # where the caller expects what the branch produces.
            return None
        return result if result is not None else body_d2

    return None


def _is_ground_term(t: 'PsiTerm', _seen=None) -> bool:
    """Whether t holds no unbound variable anywhere under it."""
    if _seen is None:
        _seen = set()
    t = t.deref()
    if id(t) in _seen:
        return True
    _seen.add(id(t))
    if t.value is None and not t.attr_list:
        from wild_life.runtime import WL as _WL_gt
        if t.type is None or t.type is _WL_gt.top:
            return False
    return all(_is_ground_term(v, _seen) for v in t.attr_list.values())


def _has_applicable_rule(t: 'PsiTerm') -> bool:
    """Whether any rule of t's function asks only for features t supplies.

    `comp(func1 => succ, func2 => succ)` is a composition waiting for its
    argument, not a call that could still be reduced: its one rule wants a
    third feature, so the term itself is the value.
    """
    defn = t.type
    if defn is None or not defn.rule:
        return False
    supplied = set(t.attr_list.keys())
    for _h, _b in defn.rule:
        if _h is None or _b is None:
            continue
        if not (set(_h.deref().attr_list.keys()) - supplied):
            return True
    return False


# The comparisons that read numbers, and so cannot be settled while one of
# their sides is still a variable.
_ARITH_COMPARISONS = frozenset(('>', '<', '>=', '=<', '=:=', '=\\='))

# The comparisons that read sorts, and so cannot be settled while one of
# their sides is still a variable.
_SORT_COMPARISONS = frozenset((
    ':==', ':\\==', ':<', ':>', ':=<', ':>=',
    ':\\<', ':\\>', ':\\=<', ':\\>=', ':\\><',
))


def _may_yet_be_a_number(t: 'PsiTerm', eng) -> bool:
    """Whether a term with nothing in it yet could still come out a number.

    `P:posint` is a number nobody has said yet, so a comparison on it is an
    open question; `a` is not a number at all, and asking is a mistake.
    """
    if t is None or eng is None:
        return False
    t = t.deref()
    if t.value is not None or t.attr_list:
        return False
    _d = t.type
    _real = eng.wl.real
    if _d is None or _real is None:
        return False
    return (_d is _real or _d is eng.wl.top
            or (getattr(_d, 'is_subtype_of', None) is not None
                and (_d.is_subtype_of(_real) or _real.is_subtype_of(_d))))


def _cond_is_undecided(c: 'PsiTerm', eng, _depth: int = 0) -> bool:
    """Whether a condition made of number comparisons cannot be decided yet.

    `cond(Y >= 33, …)` with Y still a variable has no answer, so the term is
    worth itself: that is how a grammar's `#( cond(Y >= 33, …), … )` is filed
    as the clause's code instead of being worked out at compile time against a
    Y that nothing has bound.
    """
    if c is None or _depth > 20:
        return False
    c = c.deref()
    sym = _get_sym(c)
    if sym in ('and', 'or', 'not', 'xor'):
        return any(_cond_is_undecided(_v, eng, _depth + 1)
                   for _v in c.attr_list.values())
    if sym in _ARITH_COMPARISONS and len(c.attr_list) == 2:
        for _v in c.attr_list.values():
            _ok, _ = _eval_arith(_v, eng)
            # A side that will not come out because nothing has bound its
            # variables is an open question.  One that will not come out
            # although everything in it is known — `3 / 0` — is a wrong
            # question, and the caller reports it rather than waiting.
            if _ok:
                continue
            if not _is_ground_term(_v):
                return True
            # A term narrowed no further than a number sort is a number
            # nothing has said yet: `number_of_factors(P:posint)` asks
            # `P < 2` of a P that has still to arrive.
            if _may_yet_be_a_number(_v, eng):
                return True
        return False
    # A comparison of sorts always has an answer, whether or not either side
    # has been said yet: `T :== xfx` on a T nothing has bound is false, since
    # what T is so far is not xfx.  std_expander.lf's
    # `X comma Y -> cond(X :== succeed, Y, cond(Y :== succeed, X, (X,Y)))`
    # is built on that -- it joins two goals before either is worked out --
    # and a question left open there leaves the cond itself in the clause.
    return False


def _cond_args(t: 'PsiTerm'):
    """cond's three arguments, read by name rather than by position.

    `cond(Y >= 58, 3 => cond(…))` names its else branch and leaves the then
    branch out, which is not the two-argument form: what is missing is a goal
    nothing constrains, and a goal nothing constrains holds.  Reading the
    arguments in the order they happen to be stored would take the else
    branch for the then branch and read the whole test backwards.
    """
    a1 = t.attr_list.get('1')
    a2 = t.attr_list.get('2')
    a3 = t.attr_list.get('3')
    return (a1.deref() if a1 is not None else None,
            a2.deref() if a2 is not None else None,
            a3.deref() if a3 is not None else None)


def _is_cond_builtin_local(t: 'PsiTerm') -> bool:
    """Return True if t is the built-in cond(…) call."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    if t.type.keyword.symbol != 'cond':
        return False
    return getattr(t.type, '_builtin_func', None) is not None


def _is_copy_term_func(t: 'PsiTerm') -> bool:
    """Return True if t is copy_term(X) or copy(X) with exactly 1 argument (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    sym = t.type.keyword.symbol
    if sym not in ('copy_term', 'copy'):
        return False
    # 1-arg form only (2-arg is the predicate form copy_term(X, Y))
    return '1' in t.attr_list and '2' not in t.attr_list


def _eval_copy_term_func(t: 'PsiTerm') -> 'PsiTerm':
    """Evaluate copy_term(X) → fresh copy of X."""
    arg = t.attr_list['1'].deref()
    return copy_term(arg)


def _is_glb_func(t: 'PsiTerm') -> bool:
    """Return True if t is glb(X, Y) with exactly 2 arguments (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    return (t.type.keyword.symbol == 'glb' and
            '1' in t.attr_list and '2' in t.attr_list and '3' not in t.attr_list)


def _is_children_func(t: 'PsiTerm') -> bool:
    """Return True if t is children(X) with exactly 1 argument (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    return (t.type.keyword.symbol == 'children' and
            '1' in t.attr_list and '2' not in t.attr_list)


def _eval_children_func(t: 'PsiTerm', eng) -> 'PsiTerm':
    """Evaluate children(X) → WL list of direct subsorts of X's sort."""
    arg = t.attr_list['1'].deref()
    defn = arg.type
    wl = eng.wl
    nil_term = PsiTerm(type_def=wl.nil)
    if defn is None:
        return nil_term
    # A number or a string is a sort with nothing under it: `children(23.3)`
    # asks about that one real, not about real, whose child is int.
    if arg.value is not None:
        return nil_term
    child_defs = getattr(defn, 'children', [])
    lst = nil_term
    for cd in reversed(child_defs):
        child_term = PsiTerm(type_def=cd)
        pair = PsiTerm()
        pair.type = wl.alist
        pair.attr_list = {'1': child_term, '2': lst}
        lst = pair
    return lst


def _is_lub_func(t: 'PsiTerm') -> bool:
    """Return True if t is lub(X, Y) with exactly 2 arguments (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    return (t.type.keyword.symbol == 'lub' and
            '1' in t.attr_list and '2' in t.attr_list and '3' not in t.attr_list)


def _is_chr_func(t: 'PsiTerm') -> bool:
    """Return True if t is chr(N) with exactly 1 argument (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    return (t.type.keyword.symbol == 'chr' and
            '1' in t.attr_list and '2' not in t.attr_list)


def _asc_argument(t: 'PsiTerm', eng):
    """What asc(X) can make of its argument.

    Answers ('code', n) for a character it can read, ('wait', None) for a
    string whose characters nothing has said yet — `asc(string)` waits rather
    than failing — and ('error', None) for an argument that is no string at
    all, which is reported as C Wild Life reports it.
    """
    a1 = t.attr_list.get('1')
    if a1 is None:
        return ('error', None)
    a1d = a1.deref()
    wl = eng.wl if eng is not None else None
    # A meet is the term it comes to: `asc(thingy & "hello")` reads the h.
    if (wl is not None and a1d.type is not None and a1d.type is wl.and_sym
            and a1d.attr_list):
        _merged = _eval_and_conjunction(a1d, eng)
        if _merged is not None:
            a1d = _merged.deref()
    # A string function first: `asc(chr(65))` reads what chr answers.
    char_term = _try_eval_string_func(a1d, eng)
    if char_term is not None:
        a1d = char_term.deref()
    if a1d.value is not None and wl is not None:
        if a1d.type is not None and a1d.type.is_subtype_of(wl.quoted_string):
            _s = str(a1d.value)
            return ('code', float(ord(_s[0]) % 256)) if _s else ('wait', None)
        return ('error', None)     # a number is no string
    # A bare name of one character is read as that character.  A compound is
    # not: `asc(2 * X)` is not the 42 that `*` would give.
    if (not a1d.attr_list and a1d.type is not None
            and a1d.type.keyword is not None):
        _s = a1d.type.keyword.symbol
        if len(_s) == 1:
            return ('code', float(ord(_s[0]) % 256))
    # A string with nothing in it yet, or a variable that may still be one.
    if not a1d.attr_list and a1d.value is None and wl is not None and (
            a1d.type is None or a1d.type is wl.top
            or a1d.type.is_subtype_of(wl.quoted_string)):
        return ('wait', None)
    return ('error', None)


def _report_asc_error(t: 'PsiTerm', eng) -> None:
    """Say that asc was given something that is no string.

    The argument is named by what it is worth: an expression still waiting on
    its variables is a `real~`, which is how C Wild Life names it.
    """
    a1 = t.attr_list.get('1')
    a1d = a1.deref() if a1 is not None else None
    if a1d is None:
        _shown = '@'
    else:
        _ok, _v = _eval_arith(a1d, eng)
        if _ok:
            _shown = _term_to_str(_make_number(eng, _v), eng)
        elif _get_sym(a1d) in _ARITH_OPS_SET:
            _shown = 'real~'
        else:
            _shown = _term_to_str(a1d, eng)
    sys.stderr.write(
        f"*** Error: String argument expected in 'asc({_shown})'\n")


def _is_asc_func(t: 'PsiTerm') -> bool:
    """Return True if t is asc(C) with exactly 1 argument (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    return (t.type.keyword.symbol == 'asc' and
            '1' in t.attr_list and '2' not in t.attr_list)


def _is_strip_func(t: 'PsiTerm') -> bool:
    """Return True if t is strip(S) with exactly 1 positional arg (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    return (t.type.keyword.symbol == 'strip' and
            '1' in t.attr_list and '2' not in t.attr_list)


def _is_copy_pointer_func(t: 'PsiTerm') -> bool:
    """Return True if t is copy_pointer(S) with exactly 1 positional arg (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    return (t.type.keyword.symbol == 'copy_pointer' and
            '1' in t.attr_list and '2' not in t.attr_list)


def _eval_strip_or_copy_func(t: 'PsiTerm', eng, use_src_type: bool) -> 'PsiTerm':
    """Evaluate strip(S) or copy_pointer(S) → result PsiTerm.

    Replaces each positional attr of src with a fresh var bound to the old
    value (via coref).  Both src and the returned result share those fresh
    vars so the printer emits '_A: q' for src and '_A' for the result.
    """
    src = t.attr_list['1'].deref()
    wl = eng.wl
    new_src_attrs: dict = {}
    new_res_attrs: dict = {}

    for k, v in src.attr_list.items():
        try:
            int(k)
            is_pos = True
        except (ValueError, TypeError):
            is_pos = False

        if is_pos:
            v_d = v.deref()
            # If already an unbound variable share it; otherwise wrap in fresh var
            if (v_d.value is None and not v_d.attr_list and
                    (v_d.type is None or v_d.type is wl.top)):
                fresh = v_d
            else:
                fresh = PsiTerm()
                fresh.type = wl.top  # proper unbound variable (type=top)
                fresh.coref = v  # fresh.deref() == v.deref() == the atom/value
            new_src_attrs[k] = fresh
            new_res_attrs[k] = fresh
        else:
            # A named feature is the stripped term's as much as a
            # positional one, and it is the same cell in both: binding
            # `strip(X).a` binds X's a.
            new_src_attrs[k] = v
            new_res_attrs[k] = v

    # Trail src.attr_list so backtracking restores the raw values
    eng.trail.trail_psi(src, 'attr_list')
    src.attr_list = new_src_attrs

    res = PsiTerm()
    res.type = src.type if use_src_type else wl.top
    res.attr_list = new_res_attrs
    return res


def _is_parse_func(t: 'PsiTerm') -> bool:
    """Return True if t is parse(String[, Status[, Vars]]) (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    return t.type.keyword.symbol == 'parse' and '1' in t.attr_list


def _eval_parse_func(t: 'PsiTerm', eng) -> Optional['PsiTerm']:
    """Evaluate parse(String[, Status[, Vars]]) → parsed psi-term.

    In C Wild Life, parse(S) parses the LIFE string S and returns the
    resulting psi-term.  Variables in S are shared by name with the current
    query scope (eng._last_var_tree).

    Two- and three-argument forms:
      parse(String, Status)       — also unify Status with declaration/query/error
      parse(String, Status, Vars) — Status same; Vars is unified with true (stub)

    Returns the parsed term, or None if the string argument is not yet
    concrete (delay evaluation until the string is available).
    """
    from wild_life.parser_ import parse_string
    from wild_life.data_structures import FACT, QUERY, ERROR

    a1 = t.attr_list.get('1')
    if a1 is None:
        return None
    a1d = a1.deref()
    wl = eng.wl if eng is not None else None

    # Reject unbound variables (type=top, no value, no attrs) — can't parse yet.
    # Without this check, an unbound var with type=top would be misread as the
    # atom '@' (wl.top.keyword.symbol) and parsed incorrectly.
    if a1d.value is None and not a1d.attr_list:
        if a1d.type is None or (wl is not None and a1d.type is wl.top):
            return None  # unbound variable — delay until bound

    # Get the string content
    if wl is not None and a1d.type is not None and a1d.type.is_subtype_of(wl.quoted_string):
        if a1d.value is None:
            return None  # string not yet concrete
        s = str(a1d.value)
    elif a1d.type is not None and a1d.type.keyword is not None:
        # Atom used as string
        s = a1d.type.keyword.symbol
    else:
        return None  # not a string — can't parse yet

    # Add terminator if missing so the parser can classify it
    s_for_parse = s
    stripped = s.rstrip()
    has_terminator = stripped.endswith('.') or stripped.endswith('?')
    if not has_terminator:
        s_for_parse = s + '.'  # treat as declaration for partial parse

    # Parse using the current query's variable scope so names are shared.
    # Variables in the string whose names exist in the current scope are
    # unified with the existing psi-terms; new names get fresh psi-terms.
    inherited = getattr(eng, '_last_var_tree', None) or {}
    try:
        term, kind, new_vt = parse_string(s_for_parse, inherited_vars=inherited or None)
    except Exception:
        term, kind, new_vt = None, ERROR, {}

    # Merge newly created variables back into the engine's var_tree so that
    # subsequent queries (at depth+1) can inherit them by name.  In C Wild
    # Life the interactive shell keeps a global variable table; this replicates
    # that behaviour by extending eng._last_var_tree.
    if eng is not None and new_vt:
        lv = getattr(eng, '_last_var_tree', None)
        if lv is not None:
            _added = [k for k in new_vt if k not in lv]
            if _added:
                eng.trail.trail_dict(lv)
                for k in _added:
                    lv[k] = new_vt[k]

    # Determine status atom
    if not has_terminator:
        # No proper terminator → error status, but term may still be returned
        status_sym = 'error'
    elif kind == FACT:
        status_sym = 'declaration'
    elif kind == QUERY:
        status_sym = 'query'
    else:
        status_sym = 'error'

    # Bind Status argument if present (arg 2)
    a2 = t.attr_list.get('2')
    if a2 is not None and eng is not None:
        status_term = _make_atom(eng, status_sym)
        _unify(eng, a2.deref(), status_term)

    # Bind Vars argument if present (arg 3) — stub: unify with 'true'
    a3 = t.attr_list.get('3')
    if a3 is not None and eng is not None:
        true_term = _make_atom(eng, 'true')
        _unify(eng, a3.deref(), true_term)

    if term is None:
        # Parse failed; return a top-sort unbound variable so the caller can
        # still see the result (consistent with C Wild Life partial parse)
        fresh = PsiTerm()
        if wl is not None:
            fresh.type = wl.top
        return fresh

    # Mark all compound arithmetic nodes in the parse result as NON_STRICT_TERM
    # so they are displayed as structure (e.g. 1+2) rather than evaluated to a
    # number (3) during printing.  This matches C Wild Life behaviour where
    # parse() returns structural terms, not computed values.
    from wild_life.data_structures import NON_STRICT_TERM as _NST_P
    _arith_syms_p = frozenset(('+', '-', '*', '/', '//', 'mod', '^',
                               'max', 'min', 'abs', 'sqrt', 'floor', 'ceiling',
                               'round', 'truncate', 'exp', 'log', 'sin', 'cos', 'tan'))

    def _mark_nst(node, _visited=None):
        if node is None:
            return
        if _visited is None:
            _visited = set()
        nd = node.deref()
        nid = id(nd)
        if nid in _visited:
            return
        _visited.add(nid)
        sym = nd.type.keyword.symbol if nd.type and nd.type.keyword else ''
        if sym in _arith_syms_p and nd.attr_list:
            nd.flags |= _NST_P
        for v in nd.attr_list.values():
            _mark_nst(v, _visited)

    _mark_nst(term)

    return term


def _eval_glb_func(t: 'PsiTerm', eng) -> Optional['PsiTerm']:
    """Evaluate glb(X, Y) → GLB (unification) of X and Y, or None on failure.

    Returns only the FIRST GLB; multiple-GLB non-determinism is handled by
    _apply_glb_to_var (called from bi_unify) which pushes choice points.
    """
    t1 = t.attr_list['1'].deref()
    t2 = t.attr_list['2'].deref()
    # Compute GLB non-destructively via copy + unify + copy-result
    c1 = copy_term(t1)
    c2 = copy_term(t2)
    mark = eng.trail.mark()
    ok = eng.unifier.unify(c1, c2)
    if not ok:
        eng.trail.undo_to(mark)
        return None
    result = copy_term(c1.deref())
    eng.trail.undo_to(mark)
    return result


def _apply_and_conjunction_to_var(conj_t: 'PsiTerm', target: 'PsiTerm', eng) -> bool:
    """Unify target with (sort1 & sort2), creating choice points for multiple GLBs.

    Unlike calling _eval_and_conjunction + _unify(target, result), this
    function creates choice points that directly bind *target* (not an
    internal fresh variable), so that backtracking correctly yields each
    alternative GLB bound to the original target variable.

    Falls back to _eval_and_conjunction for disjunction/user-function cases.
    """
    from wild_life.unification import compute_all_glbs as _all_glbs
    wl = eng.wl

    t1_r = conj_t.attr_list.get('1')
    t2_r = conj_t.attr_list.get('2')
    if t1_r is None or t2_r is None:
        return False

    t1 = t1_r.deref()
    t2 = t2_r.deref()

    # Resolve nested conjunctions on each side
    if t1.type is not None and t1.type is wl.and_sym:
        t1 = _eval_and_conjunction(t1, eng)
        if t1 is None:
            return False
        t1 = t1.deref()
    if t2.type is not None and t2.type is wl.and_sym:
        t2 = _eval_and_conjunction(t2, eng)
        if t2 is None:
            return False
        t2 = t2.deref()

    # For disjunction cases or user-function sides, fall back to old approach
    t1_is_disj = t1.type is not None and (t1.type is wl.disjunction or t1.type is wl.disj_nil)
    t2_is_disj = t2.type is not None and (t2.type is wl.disjunction or t2.type is wl.disj_nil)
    if t1_is_disj or t2_is_disj or _is_user_function(t1) or _is_user_function(t2):
        result = _eval_and_conjunction(conj_t, eng)
        if result is None:
            return False
        return _unify(eng, target, result)

    # Both sides are concrete sorts with no concrete value: use GLB enumeration
    d1 = t1.type
    d2 = t2.type
    if (d1 is None or d2 is None or t1.value is not None or t2.value is not None
            or t1.attr_list or t2.attr_list):
        # Has a concrete value (e.g. a number or string), no type def, or
        # has attributes that must be unified (e.g. a(10) & @(1=>11)):
        # use the old approach which handles glb(1, int) and attribute merging.
        result = _eval_and_conjunction(conj_t, eng)
        if result is None:
            return False
        return _unify(eng, target, result)

    glbs = _all_glbs(d1, d2)
    if not glbs:
        return False

    # Push choice points for alternatives (last to first so first fires next)
    for alt_def in reversed(glbs[1:]):
        alt_psi = PsiTerm(type_def=alt_def)
        eng.push_choice_point(GoalType.UNIFY, target, alt_psi, None)

    # Unify target with first GLB
    first_psi = PsiTerm(type_def=glbs[0])
    return _unify(eng, target, first_psi)


def _apply_glb_to_var(t: 'PsiTerm', target: 'PsiTerm', eng) -> bool:
    """Unify target with glb(X,Y), creating choice points when multiple GLBs exist.

    When `glb(k,l)` has minimal common subtypes a and b (both are GLBs),
    this pushes a choice point for b and returns target=a first; backtracking
    yields target=b.

    When either arg carries a concrete value (integer, float, string), we use
    the copy+unify approach so that glb(1, int) → 1 (not int).
    """
    from wild_life.unification import compute_all_glbs as _all_glbs
    t1 = t.attr_list['1'].deref()
    t2 = t.attr_list['2'].deref()
    d1 = t1.type
    d2 = t2.type

    # If either arg has a concrete value or no type, use copy+unify approach
    # (handles glb(1, int) → 1, glb(3.14, real) → 3.14, etc.)
    if d1 is None or d2 is None or t1.value is not None or t2.value is not None:
        r = _eval_glb_func(t, eng)
        return _unify(eng, target, r) if r is not None else False

    # Normalize backtick '`' syntax type to the disj sort for lattice operations.
    wl = eng.wl
    d1 = _normalize_backtick_type(d1, wl)
    d2 = _normalize_backtick_type(d2, wl)

    glbs = _all_glbs(d1, d2)
    if not glbs:
        return False

    # Push choice points for alternatives (last to first so first fires next)
    from wild_life.data_structures import GoalType as _GT
    for alt_def in reversed(glbs[1:]):
        alt_psi = PsiTerm(type_def=alt_def)
        eng.push_choice_point(_GT.UNIFY, target, alt_psi, None)

    # Unify target with first GLB
    first_psi = PsiTerm(type_def=glbs[0])
    return _unify(eng, target, first_psi)


def _compute_all_lubs(d1, d2, wl):
    """Compute all minimal common supertypes (LUBs) of d1 and d2.

    Returns a list of Definition objects. When multiple incomparable minimal
    common supertypes exist (e.g. k and l both supertype of a and b),
    all are returned so that lub(a,b) can non-deterministically yield each.
    """
    if d1 is d2:
        return [d1]
    if d1 is wl.top:
        return [wl.top]
    if d2 is wl.top:
        return [wl.top]

    # Fast paths: subtype relationship
    if d1.is_subtype_of(d2):
        return [d2]   # d2 is more general; LUB = d2
    if d2.is_subtype_of(d1):
        return [d1]

    # BFS upward from d to collect all ancestors (self included), closest first
    def _ancestors(start):
        seen = set()
        result = []
        queue = [start]
        while queue:
            d = queue.pop(0)
            if d is None or d in seen:
                continue
            seen.add(d)
            result.append(d)
            for p in getattr(d, 'parents', []):
                if p not in seen:
                    queue.append(p)
        return result

    anc1 = _ancestors(d1)
    set1 = set(anc1)
    anc2 = _ancestors(d2)
    set2 = set(anc2)

    # Common ancestors (excluding self)
    common = [a for a in anc1 if a in set2 and a is not d1]
    # Also include ancestors of d2 that are in set1
    for a in anc2:
        if a in set1 and a is not d2 and a not in common:
            common.append(a)

    if not common:
        return [wl.top]

    # Keep only minimal common ancestors (those not subsumed by a more specific one)
    minimal = []
    for a in common:
        if not any(b is not a and b.is_subtype_of(a) and b in common for b in common):
            if a not in minimal:
                minimal.append(a)

    return minimal if minimal else [wl.top]


def _lub_concrete_equal(t1, t2):
    """Return copy_term(t1) if t1 and t2 are equal concrete values, else None.

    Two terms are "equal concrete values" when both have the same non-None value
    and the same type definition, with no attribute sub-terms that would need
    their own LUB treatment.  This captures: lub(1,1)→1, lub("a","a")→"a",
    lub(12.2,12.2)→12.2.
    """
    if (t1.value is not None and t2.value is not None
            and t1.value == t2.value
            and t1.type is t2.type
            and not t1.attr_list and not t2.attr_list):
        return copy_term(t1)
    return None


def _normalize_backtick_type(d, wl):
    """Map the syntax backtick '`' type to wl.disjunction for LUB/GLB purposes.

    In C Wild Life, `{a;b} is a disjunction term (type disj).  Our parser
    gives it the syntax '`' type, but for sort-lattice operations we must
    treat it the same as disj.
    """
    if (wl.disjunction and d is not wl.disjunction
            and d.keyword and d.keyword.symbol == '`'):
        return wl.disjunction
    return d


def _eval_lub_func(t: 'PsiTerm', eng) -> Optional['PsiTerm']:
    """Evaluate lub(X, Y) → first LUB; non-determinism handled by _apply_lub_to_var."""
    t1 = t.attr_list['1'].deref()
    t2 = t.attr_list['2'].deref()
    # lub(V, V) → V when both sides carry the same concrete scalar value.
    eq = _lub_concrete_equal(t1, t2)
    if eq is not None:
        return eq
    lubs = _compute_all_lubs_from_t(t, eng)
    return PsiTerm(type_def=lubs[0]) if lubs else None


def _compute_all_lubs_from_t(t: 'PsiTerm', eng):
    t1 = t.attr_list['1'].deref()
    t2 = t.attr_list['2'].deref()
    d1 = t1.type
    d2 = t2.type
    wl = eng.wl
    if d1 is None or d2 is None:
        return [wl.top]
    # Normalize backtick '`' syntax type to the disj sort for lattice operations.
    d1 = _normalize_backtick_type(d1, wl)
    d2 = _normalize_backtick_type(d2, wl)
    return _compute_all_lubs(d1, d2, wl)


def _apply_lub_to_var(t: 'PsiTerm', target: 'PsiTerm', eng) -> bool:
    """Unify target with lub(X,Y), creating choice points for multiple LUBs."""
    from wild_life.data_structures import GoalType as _GT
    t1 = t.attr_list['1'].deref()
    t2 = t.attr_list['2'].deref()
    # lub(V, V) → V when both sides carry the same concrete scalar value.
    eq = _lub_concrete_equal(t1, t2)
    if eq is not None:
        return _unify(eng, target, eq)
    lubs = _compute_all_lubs_from_t(t, eng)
    if not lubs:
        return False
    # Push choice points for alternatives (reversed so first fires next)
    for alt_def in reversed(lubs[1:]):
        alt_psi = PsiTerm(type_def=alt_def)
        eng.push_choice_point(_GT.UNIFY, target, alt_psi, None)
    first_psi = PsiTerm(type_def=lubs[0])
    return _unify(eng, target, first_psi)


def _eval_suchthat_sync(st_d: 'PsiTerm', eng, _depth: int) -> Optional['PsiTerm']:
    """Work out a `Value | Goal` standing where a value belongs.

    A such-that is a function, so a cond that hands one back as its value
    has it checked out on the spot: the goal runs and the value is what the
    cond is worth.  Answers None when the goal does not hold, leaving the
    bindings it made undone.
    """
    val_part = st_d.attr_list.get('1')
    cond_part = st_d.attr_list.get('2')
    if val_part is None or cond_part is None:
        return None
    from wild_life.inference import (
        GoalType as _GT_st, _DEFRULES as _DR_st,
        _INNER_RUN_BARRIER as _IRB_st, _leftmost_goal as _lmg_st)
    mark = eng.trail.mark()
    _cond_d = cond_part.deref()
    _eval_embedded_user_funcs(_lmg_st(_cond_d, eng.wl), eng, _depth + 1, set())
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack
    eng.goal_stack = None
    eng.push_goal(_GT_st.PROVE, _cond_d, _DR_st, None)
    old_ok = eng.main_loop_ok
    cond_ok = eng.run(cs_barrier=cp_save if cp_save is not None else _IRB_st)
    eng.main_loop_ok = old_ok
    eng.choice_stack = cp_save
    eng.goal_stack = gs_save
    if not cond_ok:
        eng.trail.undo_to(mark)
        return None
    val_d = val_part.deref()
    _ok_a, _v = _eval_arith(val_d, eng)
    if _ok_a:
        return _make_number(eng, _v)
    _eval_embedded_user_funcs(val_d, eng, _depth + 1, set())
    return val_d


def _eval_body_sync(body_d: 'PsiTerm', eng, _depth: int) -> Optional['PsiTerm']:
    """Synchronously evaluate a function body expression.

    Handles: arithmetic, user-defined function calls, built-in cond(C,T,E),
    and compound terms with embedded user-function sub-terms.
    Returns the evaluated PsiTerm or None if evaluation cannot proceed.
    """
    if _depth > 2000:
        return None

    # A term held as it is written is worth itself: nothing inside it is
    # asked for a value, however much of it reads like a call.
    if body_d.flags & QUOTED_TRUE:
        return body_d

    # Arithmetic expression?
    ok_a, val = _eval_arith(body_d, eng)
    if ok_a:
        return _make_number(eng, val)

    # Special: where(goal1[, goal2, ...]) — in LIFE, where(G) in a function
    # body executes G as a side-effect goal, then returns @ (top).
    # `where -> @.` makes it a zero-arity sort but positional attributes are
    # goals to execute in index order before returning @.
    _wh_kw = body_d.type.keyword if body_d.type else None
    if (_wh_kw is not None and _wh_kw.symbol == 'where'
            and body_d.attr_list):
        for _wk in sorted(
                (k for k in body_d.attr_list if isinstance(k, str) and k.isdigit()),
                key=int):
            _warg = body_d.attr_list[_wk].deref()
            _eval_body_sync(_warg, eng, _depth + 1)
        # Return @ (top) so intersecting with another sort gives that sort unchanged
        return PsiTerm(type_def=eng.wl.top)

    # `Value | Goal` standing where a value belongs: run the goal, answer
    # the value.  A cond hands its chosen branch back as its value and
    # checks it out, and a such-that is a function like any other.
    if (body_d.type is not None and body_d.type is eng.wl.such_that
            and body_d.attr_list):
        return _eval_suchthat_sync(body_d, eng, _depth)

    # User-defined function call?
    if _is_user_function(body_d):
        return _eval_user_func_sync(body_d, eng, _depth)

    # Built-in copy_term(X) functional use — return a fresh copy
    if _is_copy_term_func(body_d):
        return _eval_copy_term_func(body_d)

    # Built-in cond(C, T, E) — evaluate functionally
    if _is_cond_builtin_local(body_d):
        cond_g, then_g, else_g = _cond_args(body_d)
        if cond_g is None or (then_g is None and else_g is None):
            return None

        if _cond_is_undecided(cond_g, eng):
            return None

        from wild_life.inference import prove_cond as _prove_cond
        mark_c = eng.trail.mark()
        cond_ok = _prove_cond(cond_g, eng)

        if cond_ok:
            branch = then_g.deref()
        else:
            eng.trail.undo_to(mark_c)
            if else_g is None:
                return None
            branch = else_g.deref()

        return _eval_body_sync(branch, eng, _depth + 1)

    # Disjunction body {a; b; ...}: evaluate each element recursively.
    # This handles function bodies like {1; 1+posint_stream_to(N-1)} where
    # arithmetic ops inside the disjunction need to be fully evaluated.
    wl = eng.wl
    if body_d.type is not None and body_d.type is wl.disjunction:
        elems = _collect_disjunction(body_d, eng)
        new_elems: list = []
        for e in elems:
            ev_r = _eval_body_sync(e.deref(), eng, _depth + 1)
            ev = (ev_r if ev_r is not None else e).deref()
            if ev.type is not None and ev.type is wl.disjunction:
                new_elems.extend(_collect_disjunction(ev, eng))
            elif ev.type is None or ev.type is not wl.disj_nil:
                new_elems.append(ev)
            # disj_nil (empty branch) → drop
        if not new_elems:
            nil = PsiTerm(); nil.type = wl.disj_nil; return nil
        return _make_disjunction_psi(new_elems, wl)

    # Arithmetic op that may have disjunction operands (e.g. 1 + f(N) where
    # f(N) returns a disjunction): use _eval_arith_psi for distribution.
    psi_r = _eval_arith_psi(body_d, eng, _depth)
    if psi_r is not None:
        return psi_r

    # Built-in map(F, List) functional use — evaluate the mapped list
    if (body_d.type is not None and body_d.type.keyword is not None
            and body_d.type.keyword.symbol == 'map'
            and '1' in body_d.attr_list and '2' in body_d.attr_list
            and '3' not in body_d.attr_list):
        return _eval_map_func(body_d, eng)

    # Sort-conjunction `&` — LIFE call-by-value semantics: evaluate both sides
    # and unify to produce the sort intersection. This handles:
    #   `Copy & root_sort(X) & bodify_list(B)` — sort intersection
    #   `(CX, NT) & memo_copy(X, Table)` — binds CX/NT from memo_copy result
    #   `result_term & where(side_effects)` — where() runs side effects, returns @
    _and_kw = body_d.type.keyword if body_d.type else None
    if (_and_kw is not None and _and_kw.symbol == '&'
            and '1' in body_d.attr_list and '2' in body_d.attr_list):
        _lhs = body_d.attr_list['1'].deref()
        _rhs = body_d.attr_list['2'].deref()
        # Evaluate RHS first — side effects (e.g. where()) may bind vars
        _rhs_val = _eval_body_sync(_rhs, eng, _depth + 1)
        _rhs_ev = (_rhs_val if _rhs_val is not None else _rhs).deref()
        # Evaluate LHS independently (now with any vars bound by RHS)
        _lhs_val = _eval_body_sync(_lhs, eng, _depth + 1)
        _lhs_ev = (_lhs_val if _lhs_val is not None else _lhs).deref()
        # Sort intersection: unify both evaluated results.
        # For `root_sort(X) & bodify_list(B)`: unify(ww, @(a=>1,b=>2)) → ww(a=>1,b=>2)
        # For `(CX,NT) & memo_copy(...)`: unify((CX,NT), pair) → binds CX, NT
        _unify(eng, _lhs_ev, _rhs_ev)
        return _lhs_ev.deref()

    # `strip(S)` and `copy_pointer(S)` answer a term, and a rule body is
    # where accumulators.lf asks them for one: `strip(A) & @(AIn,Out.A)` is
    # the `@` strip makes with two features on it.
    if _is_strip_func(body_d):
        _st_v = _eval_strip_or_copy_func(body_d, eng, False)
        if _st_v is not None:
            return _st_v
    if _is_copy_pointer_func(body_d):
        _st_v = _eval_strip_or_copy_func(body_d, eng, True)
        if _st_v is not None:
            return _st_v

    # Try evaluating the whole body as a pure built-in (root_sort, features, etc.)
    _sv = _try_eval_string_func(body_d, eng)
    if _sv is not None:
        return _sv

    # Compound term: evaluate embedded user-function and cond sub-terms in-place
    _eval_embedded_user_funcs(body_d, eng, _depth, set())
    return body_d


def _try_eval_any_func(t: PsiTerm, eng,
                       _depth: int = 0) -> Optional[PsiTerm]:
    """Try to evaluate t as any functional form (user-defined or built-in).

    Returns the evaluated PsiTerm, or None if t is not a functional form
    (or evaluation fails).  Used to eagerly reduce function sub-terms that
    appear in predicate-argument position inside function bodies.

    How far the reduction has already gone is carried through: this is one
    step of the same chain the caller is in, and starting it again from
    nothing let a call reduce its own arguments for ever without the depth
    limit ever coming near.
    """
    if t is None or eng is None:
        return None
    td = t.deref()
    if td.type is None:
        return None

    # A literal is its own value: there is nothing here to work out, and
    # handing back a fresh term for it would lose the one the term already
    # holds — `B = filter([2|L],3)` shares its 7 with L's, and it stops
    # sharing it the moment a copy takes its place.
    if td.value is not None and not td.attr_list:
        return None

    # A list cell is a term, not a call: none of the readings below apply to
    # one, and a walk over a list asks this question of every cell it has.
    _td_type = td.type
    if _td_type is eng.wl.alist or _td_type is eng.wl.nil:
        return None

    # A call written through a functor variable is the call the functor
    # names, once something has named it: `[F(E)|L]` in lrmap's value is
    # transequ of E, and reading it as the apply term itself puts the
    # unread call into the answer.
    if (getattr(eng.wl, 'apply', None) is not None
            and td.type is eng.wl.apply and td.coref is None):
        _ap_t = _apply_to_call(td, eng)
        if _ap_t is None:
            return None
        # Let the call stand where the apply term stood, so that whoever
        # reads it next works it out the way any other call is worked out
        # -- a body of alternatives among them.
        eng.trail.trail_psi(td, 'coref')
        td.coref = _ap_t
        return _try_eval_any_func(_ap_t, eng, _depth)

    # User-defined function
    if _is_user_function(td):
        return _eval_user_func_sync(td, eng, _depth)

    # Built-in copy_term
    if _is_copy_term_func(td):
        return _eval_copy_term_func(td)

    # `strip(T)` is T's features under the top sort, and copy_pointer(T)
    # is T's features under T's own.  Read where a value belongs -- the
    # `Y : strip(X)` of feature_module's test2 -- they are worth that,
    # not the call.
    if _is_strip_func(td):
        return _eval_strip_or_copy_func(td, eng, False)
    if _is_copy_pointer_func(td):
        return _eval_strip_or_copy_func(td, eng, True)

    # Built-in cond(C,T,E)
    if _is_cond_builtin_local(td):
        return _eval_body_sync(td, eng, _depth)

    # children(X) — returns list of direct sub-sorts
    if _is_children_func(td):
        return _eval_children_func(td, eng)

    # chr(N) — returns character string for ASCII code N
    if _is_chr_func(td):
        return _try_eval_string_func(td, eng)

    # asc(C) — returns ASCII code of character C
    if _is_asc_func(td):
        ok, v = _eval_arith(td, eng)
        return _make_number(eng, v) if ok else None

    # glb(X, Y) — greatest lower bound in sort hierarchy
    if _is_glb_func(td):
        return _eval_glb_func(td, eng)

    # lub(X, Y) — least upper bound in sort hierarchy
    if _is_lub_func(td):
        return _eval_lub_func(td, eng)

    # Built-in map(F, List) functional use
    if (td.type is not None and td.type.keyword is not None
            and td.type.keyword.symbol == 'map'
            and '1' in td.attr_list and '2' in td.attr_list and '3' not in td.attr_list):
        return _eval_map_func(td, eng)

    # Built-in reduce(F, E, List) functional use
    if (td.type is not None and td.type.keyword is not None
            and td.type.keyword.symbol == 'reduce'
            and '1' in td.attr_list and '2' in td.attr_list
            and '3' in td.attr_list and '4' not in td.attr_list):
        return _eval_reduce_func(td, eng)

    # General string function (strcon, substr, strlen, int2str, …)
    r = _try_eval_string_func(td, eng)
    if r is not None:
        return r

    # Arithmetic expression (+, -, *, /, abs, sqrt, …)
    # _eval_arith returns (False, 0.0) quickly for non-arithmetic terms,
    # so calling it unconditionally is safe.
    ok, v = _eval_arith(td, eng)
    if ok:
        # A number that carries features is already its own value, and the
        # number on its own is less than the term is: `2(2)` handed to
        # term_explore has a feature to count.
        if td.value is not None and td.attr_list:
            return None
        return _make_number(eng, v)

    # A boolean operator asked for its value answers true or false:
    # structures.lf writes `X \\== Y -> not(X == Y)`, and what that hands
    # back is false, not `not true`.  Only a value that is settled counts
    # here; a partly-known expression is left as it stands.
    if _get_sym(td) in ('and', 'or', 'not', 'xor'):
        _r_bool = _try_eval_bool(td, eng)
        if _r_bool is not None and _get_sym(_r_bool.deref()) in ('true', 'false'):
            return _r_bool

    # So does a comparison between two numbers: bruno_disj's
    # `cond_funct(X =:= 9, nl, …)` picks its branch by the answer, and the
    # rule head it is matched against is written `true` or `false`.
    from wild_life.data_structures import (QUOTED_TRUE as _QT_ceq,
                                           NON_STRICT_TERM as _NST_ceq)
    if (_get_sym(td) in _ARITH_COMPARISONS
            and not (td.flags & (_QT_ceq | _NST_ceq))):
        _r_cmp = _eval_arith_comparison(td, eng)
        if _r_cmp is not None:
            return _r_cmp

    return None



# Built-in predicates whose first argument ('1') should NOT be eagerly evaluated
# by _eval_embedded_user_funcs.  These predicates treat their first argument as a
# FUNCTION/PREDICATE NAME (a symbol to look up), not as a value to evaluate.
# For example, `setq(seed, 99)` should treat `seed` as a name, not call the
# function `seed` to get 1 and then set the integer-1 definition to 99.
_NON_STRICT_ARG1_BUILTINS: frozenset = frozenset({
    'setq', 'dynamic', 'static', 'assert', 'asserta', 'retract',
    'clause', 'abolish', 'listing',
    # `X <- V` and `X <<- V` write to the place X names, so X is where the
    # value goes rather than a value itself: `res <<- false` must not read
    # res first and write to what it was.
    '<-', '<<-',
    # `call_once(G)` is handed a goal to prove, not a value to work out.
    'call_once',
    # Dot feature-access `T.F`: arg '1' is the HOST subject of feature access
    # or creation, NOT a function-value to reduce in isolation.  Pre-evaluating
    # it (e.g. bodify_list(T) → @) replaces the shared reference and causes
    # conditions like `1 = Y.A` (where Y = bodify_list(T)) to add attr 'A' to
    # the fresh evaluated result rather than to val_part (bodify_list_copy).
    '.',
})


def _eval_embedded_user_funcs(
        t: PsiTerm, eng, _depth: int, visited: set) -> None:
    """Walk t's attribute tree and evaluate any user-function sub-terms.

    Also evaluates built-in functional sub-terms (children, chr, asc,
    glb, lub, arithmetic, string functions) so that predicate arguments
    that contain functional calls are fully reduced before the predicate
    is called.

    Modifies t's attr_list in-place (replacing function calls with their
    evaluated results).  t must be a fresh copy (not a stored rule term).

    The depth here counts the term's own nesting, not a chain of reductions:
    every node is visited once, so what the limit guards against is running
    out of Python stack on a very deep term.  It has to leave room for an
    ordinary one — fact writes 102! as a list of 41 cells, and 150! as 66 —
    because giving up part way through leaves calls unreduced and the answer
    wrong rather than merely incomplete.
    """
    if _depth > 100:
        return
    td = t.deref()
    if id(td) in visited:
        return
    visited.add(id(td))
    # A backtick holds its term as it is written: the `1 + 2` of
    # ``write(X:`(1+2), eval(X))`` is printed as the sum, not as 3.
    if (td.type is not None and td.type.keyword is not None
            and td.type.keyword.symbol == '`'):
        return
    # A choice held as it is written stays whole: the `{ X = strcon(…),
    # cond(…) }` a grammar rule hands the expander is the goal it was
    # written as, and reading it here would ask a question the rule has
    # not come to yet.
    if (td.flags & QUOTED_TRUE) and td.type is eng.wl.disjunction:
        return
    # Check if this term is a non-strict-first-arg built-in (e.g. setq, assert).
    # For these, skip evaluating argument '1' — it is a function/predicate NAME
    # that should be looked up, not evaluated as a value.
    _skip_arg1 = (
        td.type is not None and
        td.type.keyword is not None and
        td.type.keyword.symbol in _NON_STRICT_ARG1_BUILTINS
    )
    for key in list(td.attr_list.keys()):
        if _skip_arg1 and key == '1':
            continue  # do not eagerly evaluate function/predicate name arguments
        child = td.attr_list[key].deref()
        # A `T.F` written into a term stands for the feature, not for the
        # reading of it: `f(A,X) -> @(A, X.A)` hands back the feature X has
        # at A, and waits on A while it is still a variable.
        if (child.type is not None and child.type.keyword is not None
                and child.type.keyword.symbol == '.'):
            _cell_d = _resolve_dot_feat(child, eng)
            if _cell_d is not None and _cell_d.deref() is not child:
                eng.unifier.set_attr(td, key, _cell_d)
                _eval_embedded_user_funcs(_cell_d, eng, _depth + 1, visited)
                continue
        evaled = _try_eval_any_func(child, eng)
        if evaled is not None and evaled is not child:
            # An expression asked for its value is worth that value from then
            # on, wherever else it is written: `nvar -> vr(X:(varcount+1)) |
            # setq(varcount,X)` asks X for a number to file under varcount,
            # and the `vr(X)` it hands back is that same number rather than a
            # fresh count of a varcount that has since moved on.
            if (_get_sym(child) in _ARITH_OPS_SET and child.attr_list
                    and child.value is None and child.coref is None):
                eng.trail.trail_psi(child, 'coref')
                child.coref = evaled.deref()
            # Trailed: what the call worked out holds only under the bindings
            # in force now, and a backtrack that takes those away has to take
            # the value with them — or the term is left holding a call whose
            # arguments have gone back to being variables.
            eng.unifier.set_attr(td, key, evaled)
            _eval_embedded_user_funcs(evaled, eng, _depth + 1, visited)
            # What the call handed back can itself be worked out once the
            # calls inside it have been: `X \== Y` answers `not(X == Y)`,
            # and that is false once the `==` has said true.
            _re_ev1 = _try_eval_any_func(evaled, eng)
            if _re_ev1 is not None and _re_ev1.deref() is not evaled.deref():
                eng.unifier.set_attr(td, key, _re_ev1)
        elif child.attr_list:
            # If child is a sort-conjunction `A & B`, evaluate it via
            # _eval_body_sync to get the sort intersection (e.g.
            # `Copy & root_sort(X) & bodify_list(B)` → ww(a=>1,b=>2)).
            # Only do this when at least one side contains an evaluable
            # sub-term (a user or built-in function, or a nested `&`),
            # to avoid accidentally unifying goal-position conjunctions.
            _child_kw = child.type.keyword if child.type else None
            if _child_kw is not None and _child_kw.symbol == '&':
                _c1 = child.attr_list.get('1')
                _c2 = child.attr_list.get('2')
                _c1d = _c1.deref() if _c1 is not None else None
                _c2d = _c2.deref() if _c2 is not None else None
                _c1_kw = _c1d.type.keyword if (_c1d is not None and _c1d.type) else None
                _c2_kw = _c2d.type.keyword if (_c2d is not None and _c2d.type) else None
                _has_evaluable = (
                    (_c1d is not None and _try_eval_any_func(_c1d, eng) is not None) or
                    (_c2d is not None and _try_eval_any_func(_c2d, eng) is not None) or
                    (_c1_kw is not None and _c1_kw.symbol == '&') or
                    (_c2_kw is not None and _c2_kw.symbol == '&')
                )
                if _has_evaluable:
                    _ev_conj = _eval_body_sync(child, eng, _depth + 1)
                    if _ev_conj is not None and _ev_conj is not child:
                        eng.unifier.set_attr(td, key, _ev_conj)
                        _eval_embedded_user_funcs(_ev_conj, eng, _depth + 1, visited)
                        continue
            _eval_embedded_user_funcs(child, eng, _depth + 1, visited)
            # What the child stands for can become clear once the calls under
            # it have been worked out: `append([H|append(L,[])],[])` is a list
            # to append to only after the inner append has made one.
            _re_ev = _try_eval_any_func(child, eng)
            if _re_ev is not None and _re_ev is not child:
                eng.unifier.set_attr(td, key, _re_ev)


def _reduce_embedded_calls(t: PsiTerm, eng, _depth: int, visited: set) -> None:
    """Reduce the user-function calls written inside a term, and nothing else.

    `X = pair(foo_b(Y), s_b(B))` hands X the term foo_b builds.  Unlike
    _eval_embedded_user_funcs this leaves everything else alone: the term is
    the one the goal was written with, and its arithmetic, its features and
    its built-in calls are part of what is being said, not questions to ask.
    """
    if _depth > 100:
        return
    td = t.deref()
    if id(td) in visited:
        return
    visited.add(id(td))
    if (td.type is not None and td.type.keyword is not None
            and td.type.keyword.symbol == '`'):
        return
    from wild_life.data_structures import NON_STRICT_TERM as _NST_rec
    if td.flags & _NST_rec:
        return
    for key in list(td.attr_list.keys()):
        child = td.attr_list[key].deref()
        if not child.attr_list:
            continue
        if (_is_user_function(child) and _has_applicable_rule(child)
                and not _term_reaches_itself(child)):
            evaled = _eval_user_func_sync(child, eng, _depth)
            if evaled is not None and evaled.deref() is not child:
                eng.unifier.set_attr(td, key, evaled)
                _reduce_embedded_calls(evaled, eng, _depth + 1, visited)
                continue
        # `&` written inside a term is the meet of its two sides, and the
        # term holds what they meet at: structures.lf marks where it has
        # been with `A = @(visited => B&@(visited => A))`, which is a mark
        # on B as much as on A.
        if (eng.wl.and_sym is not None and child.type is eng.wl.and_sym
                and '1' in child.attr_list and '2' in child.attr_list):
            _met = _eval_and_conjunction(child, eng)
            if _met is not None and _met.deref() is not child:
                eng.unifier.set_attr(td, key, _met)
                _reduce_embedded_calls(_met, eng, _depth + 1, visited)
                continue
        _reduce_embedded_calls(child, eng, _depth + 1, visited)


def _make_disjunction_psi(elems: list, wl) -> PsiTerm:
    """Build {e1;e2;...} from a list of PsiTerms.  Empty list → disj_nil ({})."""
    tail = PsiTerm()
    tail.type = wl.disj_nil
    for e in reversed(elems):
        node = PsiTerm()
        node.type = wl.disjunction
        node.attr_list = {'1': e, '2': tail}
        tail = node
    return tail


def _eval_arith_psi(t: PsiTerm, eng, _depth: int = 0) -> Optional[PsiTerm]:
    """Evaluate an arithmetic expression that may contain disjunctions.

    Returns a PsiTerm (a concrete number *or* a disjunction of numbers) when
    evaluation succeeds, or None on failure.

    Binary/unary ops distribute over disjunction operands:
        1 + {a; b}  →  {1+a; 1+b}
    User-defined function calls are evaluated synchronously; if they return
    a disjunction the distribution continues recursively.
    """
    if t is None or _depth > 40:
        return None
    t = t.deref()
    wl = eng.wl
    sym = t.type.keyword.symbol if t.type and t.type.keyword else ''

    # ── concrete number ──────────────────────────────────────────────────────
    if t.value is not None and t.type and t.type.is_subtype_of(wl.real):
        return t

    # ── disjunction / disj_nil leaf ─────────────────────────────────────────
    if t.type is not None and (t.type is wl.disjunction or t.type is wl.disj_nil):
        return t

    # ── user-defined function ────────────────────────────────────────────────
    if _is_user_function(t):
        _mark = eng.trail.mark()
        try:
            result = _eval_user_func_sync(t, eng, _depth)
            if result is not None:
                result = copy_term(result.deref(), {})
        finally:
            eng.trail.undo_to(_mark)
        if result is not None:
            return _eval_arith_psi(result, eng, _depth + 1)
        return None

    # ── binary operators ─────────────────────────────────────────────────────
    _ops2 = frozenset(('+', '-', '*', '/', '//', 'mod', '^',
                       'max', 'min', '/\\', '\\/', 'xor', '>>', '<<'))
    _ops1 = frozenset(('-', 'abs', 'sqrt', 'sin', 'cos', 'tan',
                       'asin', 'acos', 'atan', 'exp', 'log',
                       'floor', 'ceiling', 'round', 'truncate',
                       'float', 'integer', '\\'))
    arg1_r = t.attr_list.get('1')
    arg2_r = t.attr_list.get('2')

    if sym in _ops2 and arg1_r is not None and arg2_r is not None:
        r1 = _eval_arith_psi(arg1_r.deref(), eng, _depth + 1)
        r2 = _eval_arith_psi(arg2_r.deref(), eng, _depth + 1)
        if r1 is None or r2 is None:
            return None
        is_d1 = r1.type is not None and (r1.type is wl.disjunction or r1.type is wl.disj_nil)
        is_d2 = r2.type is not None and (r2.type is wl.disjunction or r2.type is wl.disj_nil)
        if is_d1 or is_d2:
            elems1 = _collect_disjunction(r1, eng) if is_d1 else [r1]
            elems2 = _collect_disjunction(r2, eng) if is_d2 else [r2]
            result_elems: list = []
            for e1 in elems1:
                for e2 in elems2:
                    op_t = PsiTerm()
                    op_t.type = t.type
                    op_t.attr_list = {'1': e1, '2': e2}
                    elem_r = _eval_arith_psi(op_t, eng, _depth + 1)
                    if elem_r is not None:
                        elem_r = elem_r.deref()
                        if elem_r.type is wl.disj_nil:
                            pass  # empty branch → drop
                        elif elem_r.type is not None and elem_r.type is wl.disjunction:
                            result_elems.extend(_collect_disjunction(elem_r, eng))
                        else:
                            result_elems.append(elem_r)
            if not result_elems:
                nil = PsiTerm(); nil.type = wl.disj_nil; return nil
            return _make_disjunction_psi(result_elems, wl)
        # Both concrete — scalar arithmetic
        ok1, v1 = _eval_arith(r1, eng, _depth + 1)
        ok2, v2 = _eval_arith(r2, eng, _depth + 1)
        if ok1 and ok2:
            _op_f = {
                '+': lambda a, b: a + b, '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a / b if b != 0 else float('inf'),
                '//': lambda a, b: _int_div(a, b) if b != 0 else 0.0,
                'mod': lambda a, b: float(int(a) % int(b)) if b != 0 else 0.0,
                '^': lambda a, b: a ** b,
                'max': lambda a, b: max(a, b), 'min': lambda a, b: min(a, b),
                '/\\': lambda a, b: float(int(a) & int(b)),
                '\\/': lambda a, b: float(int(a) | int(b)),
                'xor': lambda a, b: float(int(a) ^ int(b)),
                '>>': lambda a, b: float(int(a) >> int(b)),
                '<<': lambda a, b: float(int(a) << int(b)),
            }
            if sym in _op_f:
                try:
                    return _make_number(eng, float(_op_f[sym](v1, v2)))
                except Exception:
                    return None
        return None

    # ── unary operators ──────────────────────────────────────────────────────
    if sym in _ops1 and arg1_r is not None and arg2_r is None:
        r1 = _eval_arith_psi(arg1_r.deref(), eng, _depth + 1)
        if r1 is None:
            return None
        is_d1 = r1.type is not None and (r1.type is wl.disjunction or r1.type is wl.disj_nil)
        if is_d1:
            elems = _collect_disjunction(r1, eng)
            result_elems = []
            for e in elems:
                op_t = PsiTerm()
                op_t.type = t.type
                op_t.attr_list = {'1': e}
                elem_r = _eval_arith_psi(op_t, eng, _depth + 1)
                if elem_r is not None:
                    elem_r = elem_r.deref()
                    if elem_r.type is wl.disj_nil:
                        pass
                    elif elem_r.type is not None and elem_r.type is wl.disjunction:
                        result_elems.extend(_collect_disjunction(elem_r, eng))
                    else:
                        result_elems.append(elem_r)
            if not result_elems:
                nil = PsiTerm(); nil.type = wl.disj_nil; return nil
            return _make_disjunction_psi(result_elems, wl)
        ok1, v1 = _eval_arith(r1, eng, _depth + 1)
        if ok1:
            _op_f = {
                '-': lambda a: -a, 'abs': lambda a: abs(a),
                'sqrt': lambda a: math.sqrt(a), 'sin': lambda a: math.sin(a),
                'cos': lambda a: math.cos(a), 'tan': lambda a: math.tan(a),
                'asin': lambda a: math.asin(a), 'acos': lambda a: math.acos(a),
                'atan': lambda a: math.atan(a), 'exp': lambda a: math.exp(a),
                'log': lambda a: math.log(a), 'floor': lambda a: math.floor(a),
                'ceiling': lambda a: math.ceil(a), 'round': lambda a: round(a),
                'truncate': lambda a: math.trunc(a),
                'float': lambda a: float(a), 'integer': lambda a: float(int(a)),
                '\\': lambda a: float(~int(a)),
            }
            if sym in _op_f:
                try:
                    return _make_number(eng, float(_op_f[sym](v1)))
                except Exception:
                    return None
        return None

    # ── fallback: standard scalar arithmetic ─────────────────────────────────
    ok, v = _eval_arith(t, eng, _depth)
    if ok:
        return _make_number(eng, v)
    return None


def _evaluate_result_for_display(t: PsiTerm, eng, _depth: int = 0) -> PsiTerm:
    """Fully evaluate a function result for display (used by _write_term).

    Walks disjunction elements and recursively evaluates arithmetic ops and
    user-function calls within them, distributing ops over disjunctions so
    that e.g. {1; 1+posint_stream_to(2)} becomes {1;2;3}.
    """
    if t is None or _depth > 40:
        return t
    t = t.deref()
    wl = eng.wl

    # ── disjunction: evaluate each element, then flatten ────────────────────
    if t.type is not None and t.type is wl.disjunction:
        elems = _collect_disjunction(t, eng)
        new_elems: list = []
        for e in elems:
            ev = _evaluate_result_for_display(e, eng, _depth + 1)
            ev = ev.deref()
            if ev.type is not None and ev.type is wl.disjunction:
                new_elems.extend(_collect_disjunction(ev, eng))
            elif ev.type is None or ev.type is not wl.disj_nil:
                new_elems.append(ev)
            # disj_nil branches are empty — drop them
        if not new_elems:
            nil = PsiTerm(); nil.type = wl.disj_nil; return nil
        return _make_disjunction_psi(new_elems, wl)

    # ── arithmetic op (possibly with disjunction operands) ───────────────────
    sym = t.type.keyword.symbol if t.type and t.type.keyword else ''
    _ops = frozenset(('+', '-', '*', '/', '//', 'mod', '^',
                      'max', 'min', '/\\', '\\/', 'xor', '>>', '<<',
                      'abs', 'sqrt', 'sin', 'cos', 'tan',
                      'asin', 'acos', 'atan', 'exp', 'log',
                      'floor', 'ceiling', 'round', 'truncate', 'float', 'integer'))
    if sym in _ops:
        r = _eval_arith_psi(t, eng, _depth)
        if r is not None:
            return _evaluate_result_for_display(r, eng, _depth + 1)

    # ── user-defined function call ───────────────────────────────────────────
    if _is_user_function(t):
        _mark = eng.trail.mark()
        try:
            evaled = _eval_user_func_sync(t, eng, _depth)
            if evaled is not None:
                evaled = copy_term(evaled.deref(), {})
        finally:
            eng.trail.undo_to(_mark)
        if evaled is not None:
            return _evaluate_result_for_display(evaled, eng, _depth + 1)

    return t


def _resolve_dot_feat(dot_term: 'PsiTerm', eng,
                      create: bool = True) -> 'Optional[PsiTerm]':
    """Get (or create) the attribute cell for a T.F dot-access term.

    Returns the PsiTerm stored at attr fkey of T's host (creating a fresh
    variable and inserting it into host.attr_list if the key is absent).
    Returns None if the dot-term is malformed or F cannot be resolved.
    """
    if dot_term.type is None or dot_term.type.keyword is None:
        return None
    if dot_term.type.keyword.symbol != '.':
        return None
    a1 = dot_term.attr_list.get('1')  # T
    a2 = dot_term.attr_list.get('2')  # F (feature label)
    if a1 is None or a2 is None:
        return None
    host = a1.deref()
    feat = a2.deref()
    # A name declared with `global` stands for a cell, and the feature
    # belongs to the cell: `sieve.M` reads and writes what sieve holds.
    _host_cell = _global_cell(host, eng)
    if _host_cell is None:
        _host_cell = _persistent_cell(host, eng)
    if _host_cell is not None:
        host = _host_cell.deref()
    # If the host is itself a dot-access expression (nested chain like A.a.b.c.d),
    # resolve it recursively to get the actual psi-term that holds the feature.
    if (host.type is not None and host.type.keyword is not None
            and host.type.keyword.symbol == '.'):
        host = _resolve_dot_feat(host, eng)
        if host is None:
            return None
        host = host.deref()
    # A name with rules of its own stands for what those rules make of it,
    # and it is that term the feature belongs to: `X:a1` with `a1 -> t(A,B)`
    # reads `X.1` off the t, not off the name.
    if (eng is not None and _is_user_function(host)
            and _has_applicable_rule(host)):
        _hv = _eval_user_func_sync(host, eng, 0)
        if _hv is not None and _hv.deref() is not host:
            _hv_d = _hv.deref()
            # The name becomes the term its rules make of it, so everything
            # else pointing at the name reads that term too.
            try:
                if eng.unifier.unify(host, _hv_d):
                    host = host.deref()
                else:
                    # The name and the term its rules make of it are sorts
                    # that have no meet, so there is nothing to unify — but
                    # the name is still worth that term, and a second reading
                    # of it has to find the same one: `Z:a1.1/Z.2` reads both
                    # features off the one t the rules made.
                    _hd_nm = host.deref()
                    if (_hd_nm is not _hv_d and _hd_nm.coref is None
                            and not _hd_nm.attr_list and _hd_nm.value is None):
                        eng.trail.trail_psi(_hd_nm, 'coref')
                        _hd_nm.coref = _hv_d
                    host = _hv_d
            except Exception:
                host = _hv_d
    # If the feature label is an unbound variable, we cannot eagerly create a
    # feature with key '@'.  Suspend: register a pending residuated goal so
    # that when the label is later bound the dot-access is re-evaluated.
    _feat_is_free = (eng is not None and
                     (feat.type is None or feat.type is eng.wl.top) and
                     feat.value is None and not feat.attr_list)
    if _feat_is_free:
        from wild_life.data_structures import Goal as _DotGoal, Residuation as _DotResid, SORT_VAR as _DOT_SV
        wl_dr = eng.wl
        _fresh_dr = PsiTerm()
        _fresh_dr.type = wl_dr.top
        _eq_defn_dr = getattr(wl_dr, 'eqsym', None)
        if _eq_defn_dr is None and hasattr(wl_dr, 'syntax_module'):
            _eq_defn_dr = wl_dr.syntax_module.symbol_table.get('=')
        _eq_term_dr = PsiTerm(type_def=_eq_defn_dr)
        _eq_term_dr.attr_list['1'] = _fresh_dr
        _eq_term_dr.attr_list['2'] = dot_term
        _eq_term_dr._resid_marker = True
        _pg_dr = _DotGoal(GoalType.PROVE, _eq_term_dr, None, None, pending=True)
        if feat.resid is None:
            eng.trail.trail_psi(feat, 'resid')
            feat.resid = [_DotResid(goal=_pg_dr)]
        else:
            if not any(_r.goal is _pg_dr for _r in feat.resid):
                eng.trail.trail_copy(feat, 'resid')
                feat.resid.append(_DotResid(goal=_pg_dr))
        if not (feat.flags & _DOT_SV):
            eng.trail.trail_psi(feat, 'flags')
            feat.flags |= _DOT_SV
        return _fresh_dr
    # Compute feature key string
    if feat.value is not None and feat.type and feat.type.keyword:
        fsym = feat.type.keyword.symbol
        if fsym in ('integer', 'real', 'int', 'float', 'number'):
            fkey = str(int(feat.value))
        else:
            # For string types and other value-bearing non-numeric types,
            # use the actual value as the key (e.g. "" -> '', not 'string')
            fkey = str(feat.value)
    elif feat.type and feat.type.keyword:
        # Try to evaluate arithmetic expressions like -N, N-1, etc. as feature keys.
        # This handles cases like Y.(-N) when N is bound to a number, so -N evaluates
        # to a negative integer atom key like '-3'.
        _ok, _v = _eval_arith(feat, eng)
        if _ok:
            fkey = str(int(_v))
        else:
            # A call written as the label is there for the label it answers:
            # acc_declarations.lf files what it knows under
            # `predicates_info.combined_name(X)`, which is one entry per
            # module's X rather than one called combined_name.
            if feat.attr_list:
                _feat_ev = _try_eval_string_func(feat, eng)
                if _feat_ev is not None:
                    _feat_ev = _feat_ev.deref()
                    if (_feat_ev is not feat and _feat_ev.type is not None
                            and _feat_ev.type.keyword is not None):
                        feat = _feat_ev
            if feat.value is not None:
                fkey = str(feat.value)
            else:
                fkey = feature_key_of(feat.type)
    else:
        return None
    existing = host.attr_list.get(fkey)
    if existing is not None:
        return existing  # caller will deref as needed
    if not create:
        # Asked only for what the term already holds.  A feature it may still
        # be given — by the prototype of its sort, say — is not read as an
        # empty one here: the call stands until the term has it.
        return None
    # A call still waiting for arguments is a function, not a term with room
    # for another feature: `X.2 = 2` on the `f(1)` of `f(X,Y) -> [X,Y]` is
    # refused, and says which function it was.
    if (_is_user_function(host) and host.attr_list
            and not _has_applicable_rule(host)):
        import io as _io_dot
        from wild_life.print_term import write_term as _wt_dot
        _buf_dot = _io_dot.StringIO()
        _wt_dot(host, outfile=_buf_dot, quoted=True, wl=eng.wl,
                max_col=1_000_000)   # the message is one line
        sys.stderr.write(
            f"*** Error: attempt to add a feature to curried function "
            f"{_buf_dot.getvalue()}\n")
        return None
    # Attr absent — create a fresh variable (type=top = unbound), insert it (trailed)
    wl_rd = eng.wl
    fresh = PsiTerm()
    fresh.type = wl_rd.top  # must be WL.top so unification recognises it as a free var
    # What a `persistent` name holds is not the query's to undo: a table the
    # library files write into has to still be there on the next query, and
    # on the one after a failure.
    _host_keeps = host.__dict__.get('_wl_persistent_cell', False)
    if _host_keeps:
        fresh._wl_persistent_cell = True
        # A feature opened on a term that lives in persistent store lives
        # there too, so it is read rather than narrowed.
        if host.__dict__.get('_wl_persistent_written', False):
            fresh._wl_persistent_written = True
        eng.persistent_store_touched = True
    else:
        eng.trail.trail_psi(host, 'attr_list')
    new_attrs = dict(host.attr_list)
    new_attrs[fkey] = fresh
    host.attr_list = new_attrs
    # Fix D: fire pending daemon resids on host after a new attribute is added.
    # e.g. X.set = true? fires the daemon write(X) that was set by such_that.
    if host.resid and eng is not None and getattr(eng, 'unifier', None) is not None:
        eng.unifier._wakeup_resid(host, fresh)
    return fresh


def _has_disjunctive_body(t: PsiTerm, wl, _seen: set = None,
                          _depth: int = 0) -> bool:
    """True when t is a function whose rule reduces to a disjunction.

    `sgn -> {1;-1}.` is 0-arity, so the synchronous path that normally
    evaluates such functions would have to pick one alternative and keep it;
    only an EVAL goal gives each alternative its own choice point.  What a
    body hands the question on to counts as the body's own answer: magic's
    `number -> number_to(size*size)` is nine numbers by way of number_to.
    """
    rules = t.type.rule if t.type is not None else None
    if not rules or _depth > 6:
        return False
    if _seen is None:
        _seen = set()
    if id(t.type) in _seen:
        return False
    _seen.add(id(t.type))

    def _reaches_disj(_b, _d):
        if _b is None or _d > 6:
            return False
        _b = _b.deref()
        if _b.type is None:
            return False
        if _b.type in (wl.disjunction, wl.life_or):
            return True
        _sym_b = _b.type.keyword.symbol if _b.type.keyword else ''
        if _sym_b == 'cond' and _b.attr_list:
            return any(_reaches_disj(_v, _d + 1)
                       for _k, _v in _b.attr_list.items() if _k in ('2', '3'))
        if _is_user_function(_b):
            return _has_disjunctive_body(_b, wl, _seen, _depth + 1)
        return False

    for _head, body in rules:
        if body is None:
            continue
        if _reaches_disj(body, 0):
            return True
    return False


def _expand_disjunctions_in_place(lhs: PsiTerm, rhs: PsiTerm, eng,
                                  bind_nodes: bool = False):
    """Prove `lhs = rhs` once per combination of the disjunctions inside them.

    Each disjunction node is swapped (trailed) for a fresh variable, so an
    alternative can be picked by an ordinary unification goal.  That keeps the
    choice visible through the terms themselves: `X:(3*sgn)` IS the expression
    node, so X reads as 3 for one alternative and -3 for the next, where
    unifying a copy of the expression would have left X as the disjunction.

    Returns True once the goals are pushed, or None when neither side holds a
    disjunction and the caller should carry on.
    """
    from wild_life.inference import _DEFRULES as _DR
    wl = eng.wl
    slots = []   # [(fresh var standing in for a disjunction, its alternatives)]

    def collect(t, depth=0):
        if depth > 10 or not t.attr_list:
            return
        for key in list(t.attr_list.keys()):
            sub = t.attr_list[key].deref()
            if sub.type is not None and sub.type is wl.disjunction:
                elems = _collect_disjunction(sub, eng)
                if len(elems) > 1:
                    fresh = PsiTerm(type_def=wl.top)
                    if bind_nodes:
                        # The node itself stands for the choice, so a name
                        # written on it — the Y of `Y:{a;Z}` — reads the
                        # alternative rather than the whole disjunction.
                        eng.unifier.bind(sub, fresh)
                    else:
                        eng.unifier.set_attr(t, key, fresh)
                    slots.append((fresh, elems))
                    continue
            collect(sub, depth + 1)

    collect(lhs)
    collect(rhs)
    if not slots:
        return None

    eq_defn = getattr(wl, 'eqsym', None) or wl.syntax_module.symbol_table.get('=')

    def conjoin(goals):
        joined = goals[-1]
        for goal in reversed(goals[:-1]):
            conj = PsiTerm(type_def=wl.commasym)
            conj.attr_list = {'1': goal, '2': joined}
            joined = conj
        return joined

    def equation(lhs, rhs):
        eq = PsiTerm(type_def=eq_defn)
        eq.attr_list = {'1': lhs, '2': rhs}
        return eq

    import itertools
    combos = [
        conjoin([equation(slots[i][0], choice) for i, choice in enumerate(combo)]
                + [equation(lhs, rhs)])
        for combo in itertools.product(*[elems for _, elems in slots])
    ]
    for alt in reversed(combos[1:]):
        eng.push_choice_point(GoalType.PROVE, alt, _DR, None)
    eng.push_goal(GoalType.PROVE, combos[0], _DR, None)
    return True


def _inline_disjunctive_funcs(t: PsiTerm, eng, depth: int = 0,
                             visited: set = None) -> bool:
    """Replace sub-terms of t that are functions reducing to a disjunction.

    `3 * sgn` with `sgn -> {1;-1}.` has to read as `3 * {1;-1}` before the
    disjunction expansion can give each alternative its own choice point.
    The reduction's bindings are undone again: only the value is wanted, not
    the coref linking the function atom to its rule-head copy.
    """
    if depth > 20 or not t.attr_list:
        return False
    # A term whose parts point at one another — matrix's grid of squares —
    # is walked once, not once per path that reaches each square.
    if visited is None:
        visited = set()
    if id(t) in visited:
        return False
    visited.add(id(t))
    wl = eng.wl
    changed = False
    for key in list(t.attr_list.keys()):
        sub = t.attr_list[key].deref()
        if _is_user_function(sub):
            # Reduce a copy: _eval_user_func_sync rewrites its argument's
            # features in place, which is not undone by the trail.
            probe = PsiTerm(type_def=sub.type, value=sub.value,
                            attr_list=dict(sub.attr_list))
            probe.flags = sub.flags
            mark = eng.trail.mark()
            evaled = _eval_user_func_sync(probe, eng, 0)
            eng.trail.undo_to(mark)
            evaled = evaled.deref() if evaled is not None else None
            if (evaled is not None and evaled.type is not None
                    and evaled.type in (wl.disjunction, wl.life_or)):
                t.attr_list[key] = evaled
                changed = True
                continue
        if _inline_disjunctive_funcs(sub, eng, depth + 1, visited):
            changed = True
    return changed


def _unify_through_eq(eng, a: PsiTerm, b: PsiTerm) -> bool:
    """Unify two terms the way `=` does, calls worked out and all."""
    wl = eng.wl
    _eq_defn = (getattr(wl, 'eqsym', None)
                or wl.syntax_module.symbol_table.get('='))
    if _eq_defn is None:
        return _unify(eng, a, b)
    _eq = PsiTerm(type_def=_eq_defn)
    _eq.attr_list = {'1': a, '2': b}
    return bi_unify(_eq, eng)


def bi_unify(goal: PsiTerm, eng) -> bool:
    """X = Y — LIFE sort unification (with functional evaluation).

    What `<<-` has written stays written, so an equation may read it but
    not narrow it: pers2 asks each of its terms against a persistent X and
    counts the ones X already is, not the ones X could be made into.
    """
    _prot = _protected_snapshot(goal, eng)
    if _prot is None:
        return _bi_unify_inner(goal, eng)
    _prot_mark = eng.trail.mark()
    _prot_cs = eng.choice_stack
    _ok_prot = _bi_unify_inner(goal, eng)
    if _ok_prot and _snapshot_moved(_prot):
        # The alternatives the equation opened go with it: a narrowing it
        # is not allowed to make is not one to come back to either.
        eng.trail.undo_to(_prot_mark)
        eng.choice_stack = _prot_cs
        return False
    return _ok_prot


def _protected_snapshot(goal: PsiTerm, eng):
    """What the equation may not change, as it stands before it runs."""
    _shot = None
    for _k in ("1", "2"):
        _s = goal.attr_list.get(_k)
        if _s is None:
            continue
        _s = _s.deref()
        if not _s.__dict__.get("_wl_persistent_written", False):
            continue
        if _shot is None:
            _shot = []
        _shot.append((_s, _s.type, _s.value, frozenset(_s.attr_list)))
    return _shot


def _snapshot_moved(shot) -> bool:
    for _s, _ty, _v, _keys in shot:
        _now = _s.deref()
        if (_now.type is not _ty or _now.value != _v
                or frozenset(_now.attr_list) != _keys):
            return True
    return False


def _bi_unify_inner(goal: PsiTerm, eng) -> bool:
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return a is b

    a_d = a.deref()
    b_d = b.deref()

    # A name declared with `global` stands for the cell every reference to it
    # reads, on either side of the equation: `traverse_method =
    # str2psi(strcon(psi2str(Name),"_traverse"), current_module)` writes what
    # the call answers into the cell rather than asking the name to be it.
    _a_cell = _global_cell(a_d, eng)
    if _a_cell is not None:
        a_d = _a_cell.deref()
    _b_cell = _global_cell(b_d, eng)
    if _b_cell is not None:
        b_d = _b_cell.deref()

    # A cond standing where a value belongs is worked out here and now:
    # built_ins.c registers it as a function, so `X = cond(C,T,E)` gives X
    # the branch the condition picks, and a condition that reads an unbound
    # variable simply picks the other branch rather than waiting.  The
    # accumulator expander is built on this -- `X comma Y` joins two goals
    # through a pair of conds, and left unread they end up in the clause.
    for _n_cv in range(2):
        _side_cv = a_d if _n_cv == 0 else b_d
        if (_is_cond_builtin_local(_side_cv) and _side_cv.coref is None
                and '1' in _side_cv.attr_list
                and ('2' in _side_cv.attr_list or '3' in _side_cv.attr_list)):
            _val_cv = _eval_body_sync(_side_cv, eng, 0)
            if _val_cv is not None and _val_cv.deref() is not _side_cv:
                _val_cv = _keep_call_value(_side_cv, _val_cv, eng)
                if _n_cv == 0:
                    a_d = _val_cv.deref()
                else:
                    b_d = _val_cv.deref()

    # Handle T.F = V and V = T.F (dot feature access / creation).
    # When T.F does not yet exist as an attribute, a fresh variable is
    # inserted into T's attr_list (trailed) and unified with V.
    # `.` names a feature only when it is written with a term and a label:
    # the operator table holds `.` itself as a functor, and that atom is a
    # value like any other.
    _dot_sym_check = (lambda td: td.type is not None and td.type.keyword is not None
                      and td.type.keyword.symbol == '.'
                      and '1' in td.attr_list and '2' in td.attr_list)
    if _dot_sym_check(a_d):
        _attr_cell = _resolve_dot_feat(a_d, eng)
        if _attr_cell is None:
            return False
        # If the right side is also a dot-term, resolve it too so that an
        # unbound label on the right also gets its pending residuation set up.
        if _dot_sym_check(b_d):
            _attr_cell_b = _resolve_dot_feat(b_d, eng)
            if _attr_cell_b is None:
                return False
            return _unify(eng, _attr_cell.deref(), _attr_cell_b.deref())
        # What a feature is given is a value, the same as anywhere else `=`
        # puts one: `C.1 = root_sort(app)` gives the feature `app`, not the
        # call.  The cell goes back through `=` for that.
        return _unify_through_eq(eng, _attr_cell.deref(), b_d)
    if _dot_sym_check(b_d):
        _attr_cell = _resolve_dot_feat(b_d, eng)
        if _attr_cell is None:
            return False
        return _unify_through_eq(eng, a_d, _attr_cell.deref())

    # Unwrap backtick-quoted terms: `Expr = X → bind X to inner Expr (marked NON_STRICT_TERM).
    # In Wild Life, `Expr (backtick-quoted) "freezes" the expression to prevent evaluation.
    # We unwrap here so that subsequent feature unification (e.g. A=@(2=>val)) operates
    # on the actual arithmetic term rather than the backtick wrapper.
    from wild_life.data_structures import NON_STRICT_TERM as _BI_BQ_NST  # noqa: F811
    from wild_life.inference import _mark_arith_non_strict as _BI_MANS  # noqa: F811
    _bq_sym_check = (lambda td: td.type is not None and td.type.keyword is not None
                     and td.type.keyword.symbol == '`')
    def _freeze_call(td):
        """Keep a backticked call as the call it is, not the value it has.

        `` `(X:f(X)) `` is the term f(X), so it is not reduced to what f
        answers, the way the arithmetic under a backtick is not reduced.
        A cond is a call like any other: `` `(cond(a,b,c)) `` is that term,
        and working it out would ask `a` as a goal.
        """
        from wild_life.data_structures import QUOTED_TRUE as _QT_bq
        if _is_user_function(td) or _is_cond_builtin_local(td):
            eng.trail.trail_psi(td, 'flags')
            td.flags |= _QT_bq

    _b_was_backtick = False
    if _bq_sym_check(b_d):
        _bq_inner = b_d.attr_list.get('1')
        if _bq_inner is not None:
            _bq_inner_d = _bq_inner.deref()
            _BI_MANS(_bq_inner_d)  # recursively mark arithmetic sub-terms as NON_STRICT
            _freeze_call(_bq_inner_d)
            b_d = _bq_inner_d
            _b_was_backtick = True
    _a_was_backtick = False
    if _bq_sym_check(a_d):
        _bq_inner = a_d.attr_list.get('1')
        if _bq_inner is not None:
            _bq_inner_d = _bq_inner.deref()
            _BI_MANS(_bq_inner_d)
            _freeze_call(_bq_inner_d)
            a_d = _bq_inner_d
            _a_was_backtick = True

    # Detect non-frozen arithmetic operator being applied via @(1,2)-style term.
    # Example: A=(+), A=@(1,2) — without backtick-freeze, `+` is an eager operator,
    # not a curriable function value; attempting to add args via apply merging is an error.
    from wild_life.data_structures import NON_STRICT_TERM as _BI_UNI_NST
    _wl_uni = eng.wl
    def _is_bare_arith_op(td):
        """A two-argument operator that has not been given both arguments.

        `and(B)` is a function waiting for its second argument, not a term
        with room for one, so `A = and(B), A = @(2 => C)` is refused.
        """
        sym = td.type.keyword.symbol if (td.type and td.type.keyword) else ''
        if sym not in _CURRIABLE_BINARY_OPS:
            return False
        if td.flags & _BI_UNI_NST:   # frozen by a backtick: a term, not a call
            return False
        return not ('1' in td.attr_list and '2' in td.attr_list)
    # Use symbol-based check for apply type — the parsed @(1,2) may use the '@' symbol
    # definition rather than wl.apply which is set up later during boot.
    _b_sym_apply = b_d.type.keyword.symbol if (b_d.type and b_d.type.keyword) else ''
    _a_sym_apply = a_d.type.keyword.symbol if (a_d.type and a_d.type.keyword) else ''
    _b_is_apply_type = (b_d.type is not None and
                        (_b_sym_apply == '@' or b_d.type is _wl_uni.apply))
    _a_is_apply_type = (a_d.type is not None and
                        (_a_sym_apply == '@' or a_d.type is _wl_uni.apply))
    if _is_bare_arith_op(a_d) and (_b_is_apply_type or _is_bare_arith_op(b_d)) and b_d.attr_list:
        import sys as _sys_uni
        _sym_uni = _term_to_str(a_d, eng)
        _sys_uni.stderr.write(f'*** Error: attempt to unify with curried function {_sym_uni}\n')
        return False
    if _is_bare_arith_op(b_d) and (_a_is_apply_type or _is_bare_arith_op(a_d)) and a_d.attr_list:
        import sys as _sys_uni2
        _sym_uni2 = _term_to_str(b_d, eng)
        _sys_uni2.stderr.write(f'*** Error: attempt to unify with curried function {_sym_uni2}\n')
        return False

    # Evaluate === (triple equals) identity function when both args are present.
    # ===(X, Y) → true  if X and Y are the same pointer after deref,
    #             false otherwise (even if either is free).
    def _eval_triple_eq(t_d):
        sym_te = t_d.type.keyword.symbol if (t_d.type and t_d.type.keyword) else ''
        if sym_te != '===':
            return None
        te1 = t_d.attr_list.get('1')
        te2 = t_d.attr_list.get('2')
        if te1 is None or te2 is None:
            return None  # Partially applied — not ready to evaluate
        te1d = te1.deref()
        te2d = te2.deref()
        if id(te1d) == id(te2d):
            return eng.wl.make_atom('true')
        else:
            return eng.wl.make_atom('false')

    _te_b = _eval_triple_eq(b_d)
    if _te_b is not None:
        return _unify(eng, a_d, _te_b)
    _te_a = _eval_triple_eq(a_d)
    if _te_a is not None:
        return _unify(eng, b_d, _te_a)

    # Fail if either side contains {} (bottom / empty disjunction = disj_nil).
    # A term containing bottom has no solutions, so the unification fails.
    # This handles cases like A=g(s,{}) or A=s(g(t,{})) where {} makes the
    # whole term undefined/bottom.
    _wl_bi_dn = eng.wl
    def _has_disj_nil(t, _depth=0):
        if _depth > 8:
            return False
        td = t.deref()
        if td.type is _wl_bi_dn.disj_nil:
            return True
        # Do NOT descend into non-empty disjunction nodes — their tail IS a
        # disj_nil sentinel (the linked-list end), which is normal and should
        # not trigger failure.  Only a standalone {} at argument level fails.
        if td.type is _wl_bi_dn.disjunction:
            return False
        # A term held as it is written is not worked out, so a `{}` inside it
        # is part of what is written rather than a call with no answer:
        # accumulators.lf reads a grammar rule's braces with
        # `non_strict(transLifeCode)` and `transLifeCode({}) -> fail`.
        if td.flags & QUOTED_TRUE:
            return False
        _ns_dn = getattr(eng, 'non_strict_set', None)
        if _ns_dn and td.type in _ns_dn:
            return False
        for _v in td.attr_list.values():
            if _has_disj_nil(_v, _depth + 1):
                return True
        return False
    if (a_d.type is not None and _has_disj_nil(a_d)) or (b_d.type is not None and _has_disj_nil(b_d)):
        return False

    # Try to evaluate b as a user-defined function call (f -> result style).
    # EXCEPTION: 0-arity user functions (global variables like `result` declared
    # with `persistent` or `setq`) are handled later by direct synchronous
    # evaluation (line ~4061), NOT via an EVAL goal.  Using EVAL goals for them
    # would create arithmetic constraints when the stored value is an arithmetic
    # expression with unbound variables, causing spurious `real~` display.
    # A 0-arity function whose value is a disjunction is the one exception to
    # that exception: `sgn -> {1;-1}.` needs a choice point per alternative,
    # which only the EVAL goal sets up.
    if _is_user_function(b_d) and (b_d.attr_list or _has_disjunctive_body(b_d, eng.wl)):
        result = PsiTerm(type_def=eng.wl.top)
        if _is_user_function(a_d) and (a_d.attr_list
                                       or _has_disjunctive_body(a_d, eng.wl)):
            # Two calls meeting each other: `tata(10) = tata(10)` asks tata
            # twice, and what the two answer is what is compared.
            result_a = PsiTerm(type_def=eng.wl.top)
            eng.push_goal(GoalType.UNIFY, result_a, result, None)
            eng.push_goal(GoalType.EVAL, a_d, result_a, a_d.type.rule)
            eng.push_goal(GoalType.EVAL, b_d, result, b_d.type.rule)
            return True
        # LIFO: push UNIFY first, then EVAL on top (EVAL executes first)
        eng.push_goal(GoalType.UNIFY, a_d, result, None)
        eng.push_goal(GoalType.EVAL, b_d, result, b_d.type.rule)
        return True

    # Try to evaluate a as a user-defined function call
    if _is_user_function(a_d) and (a_d.attr_list or _has_disjunctive_body(a_d, eng.wl)):
        result = PsiTerm(type_def=eng.wl.top)
        eng.push_goal(GoalType.UNIFY, result, b_d, None)
        eng.push_goal(GoalType.EVAL, a_d, result, a_d.type.rule)
        return True

    # Handle cond(C, T, E) in functional position:
    #   X = cond(3 < 2, {}, f(N))  →  evaluate cond, unify result with X
    for _cond_side, _other_side in ((b_d, a_d), (a_d, b_d)):
        if not _is_cond_builtin_local(_cond_side):
            continue
        # A cond written down rather than asked is worth itself:
        # `Z = `(cond(a,b,c))` hands Z the cond, and working it out would
        # ask `a` as a goal, which is not what a backquote is for.
        if _cond_side.flags & QUOTED_TRUE:
            return _unify(eng, _other_side, _cond_side)
        _cond_arg = (list(_cond_side.attr_list.values()) or [None])[0]
        if _cond_arg is not None and _cond_is_undecided(_cond_arg.deref(), eng):
            # Nothing has said which way it goes, so it is worth itself: the
            # `cond(Y >= 33, …)` a grammar rule carries as its code is stored
            # as written rather than settled against an unbound Y.
            return _unify(eng, _other_side, _cond_side)
        evaled = _eval_body_sync(_cond_side, eng, 0)
        if evaled is None:
            return False
        evaled = _evaluate_result_for_display(evaled.deref(), eng, 1)
        return _unify(eng, _other_side, evaled)

    # Handle copy_term(X) functional use: Y = copy_term(X) → Y = fresh copy of X
    if _is_copy_term_func(b_d):
        c = _eval_copy_term_func(b_d)
        return _unify(eng, _stored_side(a_d, eng), c)
    if _is_copy_term_func(a_d):
        c = _eval_copy_term_func(a_d)
        return _unify(eng, _stored_side(b_d, eng), c)

    # Handle glb(X,Y) functional use: B = glb(X,Y) → B = GLB of X and Y
    # Uses _apply_glb_to_var to create choice points for multiple GLBs.
    if _is_glb_func(b_d):
        return _apply_glb_to_var(b_d, a_d, eng)
    if _is_glb_func(a_d):
        return _apply_glb_to_var(a_d, b_d, eng)

    # Handle lub(X,Y) functional use: B = lub(X,Y) → B = LUB of X and Y
    if _is_lub_func(b_d):
        return _apply_lub_to_var(b_d, a_d, eng)
    if _is_lub_func(a_d):
        return _apply_lub_to_var(a_d, b_d, eng)

    # Handle children(X) functional use: L = children(X) → list of direct subsorts
    if _is_children_func(b_d):
        return _unify(eng, a_d, _eval_children_func(b_d, eng))
    if _is_children_func(a_d):
        return _unify(eng, b_d, _eval_children_func(a_d, eng))

    # Handle strip(S) / copy_pointer(S) functional use:
    #   R = strip(S)          → R has type @, shares S's positional args as vars
    #   R = copy_pointer(S)   → R has S's type, shares S's positional args as vars
    if _is_strip_func(b_d):
        return _unify(eng, a_d, _eval_strip_or_copy_func(b_d, eng, False))
    if _is_strip_func(a_d):
        return _unify(eng, b_d, _eval_strip_or_copy_func(a_d, eng, False))
    if _is_copy_pointer_func(b_d):
        return _unify(eng, a_d, _eval_strip_or_copy_func(b_d, eng, True))
    if _is_copy_pointer_func(a_d):
        return _unify(eng, b_d, _eval_strip_or_copy_func(a_d, eng, True))

    # Handle apply(Args, functor=>F) functional use: X = F(Args).
    # When the parser sees a variable F used as a functor (e.g. A(Q)), it creates
    # an apply term: apply{1: Q, functor: A}.  We need to intercept this in
    # bi_unify so that once A is bound we can reconstruct F(Args) and proceed.
    def _handle_apply_term(lhs, rhs):
        """Try to handle lhs = apply{..., functor: F} by reconstructing F(Args).
        Returns True/False on success/failure, or None if rhs is not an apply term."""
        wl_a = eng.wl
        if not (hasattr(wl_a, 'apply') and wl_a.apply is not None):
            return None
        if rhs.type is not wl_a.apply:
            return None
        _functor_key = wl_a.functor.symbol if (hasattr(wl_a, 'functor') and wl_a.functor and wl_a.functor.keyword) else 'functor'
        _functor_arg = rhs.attr_list.get(_functor_key)
        if _functor_arg is None:
            return None
        _functor_val = _functor_arg.deref()
        if _term_is_unbound(_functor_val, eng):
            # Functor is unbound — residuate on functor_val so that when A=parse
            # fires, the pending goal X=A(Q) is re-evaluated.
            from wild_life.data_structures import Goal, Residuation, SORT_VAR as _SV2
            wl_p2 = eng.wl
            eq_defn2 = getattr(wl_p2, 'eqsym', None)
            if eq_defn2 is None and hasattr(wl_p2, 'syntax_module'):
                eq_defn2 = wl_p2.syntax_module.symbol_table.get('=')
            eq_term2 = PsiTerm(type_def=eq_defn2)
            eq_term2.attr_list['1'] = lhs
            eq_term2.attr_list['2'] = rhs
            eq_term2._resid_marker = True
            pending_goal2 = Goal(GoalType.PROVE, eq_term2, None, None, pending=True)
            if _functor_val.resid is None:
                eng.trail.trail_psi(_functor_val, 'resid')
                _functor_val.resid = [Residuation(goal=pending_goal2)]
            else:
                if not any(rv.goal is pending_goal2 for rv in _functor_val.resid):
                    eng.trail.trail_copy(_functor_val, 'resid')
                    _functor_val.resid.append(Residuation(goal=pending_goal2))
            if not (_functor_val.flags & _SV2):
                eng.trail.trail_psi(_functor_val, 'flags')
                _functor_val.flags |= _SV2
            return True  # lhs stays unbound (@); functor var shows as @~
        # Functor is bound to an atom — reconstruct call_psi with type=functor_type
        _ftype = _functor_val.type
        if _ftype is None:
            return None
        # If functor is a non-frozen arithmetic operator (no NON_STRICT_TERM), allow
        # full application (where the apply node supplies all required args) but refuse
        # zero-arg invocations.  We count the non-functor keys in rhs.attr_list to decide:
        # if there is at least one arg being passed, proceed and let the arithmetic
        # evaluator handle the result (e.g. +(1,2)→3).  If no args at all, refuse.
        # This allows `F={(+);(-)}, C=F(A,B)` to work while still rejecting bare `(+)`
        # when applied to nothing.
        from wild_life.data_structures import NON_STRICT_TERM as _BI_APPLY_NST_CHK  # noqa: F811
        _fval_sym = _ftype.keyword.symbol if _ftype.keyword else ''
        if (_fval_sym in _ARITH_OPS_SET
                and not (_functor_val.flags & _BI_APPLY_NST_CHK)
                and not _functor_val.attr_list):  # bare arithmetic operator (no args yet)
            # Count args being supplied by the apply node (excluding the functor slot)
            _supplied_args = sum(1 for k in rhs.attr_list if k != _functor_key)
            if _supplied_args == 0:
                import sys as _sys_apply
                _sys_apply.stderr.write(f'*** Error: attempt to unify with curried function {_fval_sym}\n')
                return False
            # else: fall through — at least one arg supplied, allow full application
        call_psi = PsiTerm()
        call_psi.type = _ftype
        call_psi.value = None
        call_psi.coref = None
        call_psi.resid = None
        for k, v in rhs.attr_list.items():
            if k != _functor_key:
                call_psi.attr_list[k] = v
        # Merge existing features from the bound functor term (curried partial application).
        # e.g. A=*(23), then B=A(2=>13) → call_psi gets feature '1'=23 from *(23).
        for _fk, _fv in _functor_val.attr_list.items():
            if _fk not in call_psi.attr_list:
                call_psi.attr_list[_fk] = _fv
        # Propagate NON_STRICT_TERM flag from the functor (preserve non-eval semantics).
        from wild_life.data_structures import NON_STRICT_TERM as _BI_APPLY_NST  # noqa: F811
        if _functor_val.flags & _BI_APPLY_NST:
            call_psi.flags |= _BI_APPLY_NST
        # Now treat lhs = call_psi — first try parse special case
        if _is_parse_func(call_psi):
            r2 = _eval_parse_func(call_psi, eng)
            if r2 is not None:
                return _unify(eng, lhs, r2)
            # String arg is unbound — residuate on it
            s_arg2 = call_psi.attr_list.get('1')
            if s_arg2 is not None:
                s_var2 = s_arg2.deref()
                if _term_is_unbound(s_var2, eng):
                    from wild_life.data_structures import Goal, Residuation, SORT_VAR as _SV3
                    wl_p3 = eng.wl
                    eq_defn3 = getattr(wl_p3, 'eqsym', None)
                    if eq_defn3 is None and hasattr(wl_p3, 'syntax_module'):
                        eq_defn3 = wl_p3.syntax_module.symbol_table.get('=')
                    eq_term3 = PsiTerm(type_def=eq_defn3)
                    eq_term3.attr_list['1'] = lhs
                    eq_term3.attr_list['2'] = call_psi
                    eq_term3._resid_marker = True
                    pending_goal3 = Goal(GoalType.PROVE, eq_term3, None, None, pending=True)
                    if s_var2.resid is None:
                        eng.trail.trail_psi(s_var2, 'resid')
                        s_var2.resid = [Residuation(goal=pending_goal3)]
                    else:
                        if not any(rv.goal is pending_goal3 for rv in s_var2.resid):
                            eng.trail.trail_copy(s_var2, 'resid')
                            s_var2.resid.append(Residuation(goal=pending_goal3))
                    if not (s_var2.flags & _SV3):
                        eng.trail.trail_psi(s_var2, 'flags')
                        s_var2.flags |= _SV3
                    return True  # lhs stays @
            return False
        # Push a PROVE goal for (lhs = call_psi) so that bi_unify handles it
        # with full arithmetic constraint logic.  This is needed so that when
        # call_psi is a complete arithmetic expression (e.g. *(10, X) after
        # merging functor features), the arithmetic residuation machinery fires
        # (marking B and X as real~) rather than binding B directly without any
        # constraint.
        _wl_ap = eng.wl
        _eq_defn_ap = getattr(_wl_ap, 'eqsym', None)
        if _eq_defn_ap is None and hasattr(_wl_ap, 'syntax_module'):
            _eq_defn_ap = _wl_ap.syntax_module.symbol_table.get('=')
        if _eq_defn_ap is not None:
            _eq_term_ap = PsiTerm(type_def=_eq_defn_ap)
            _eq_term_ap.attr_list['1'] = lhs
            _eq_term_ap.attr_list['2'] = call_psi
            eng.push_goal(GoalType.PROVE, _eq_term_ap, None, None)
            return True
        # Fallback: direct unification if = symbol not found
        return _unify(eng, lhs, call_psi)

    _apply_b = _handle_apply_term(a_d, b_d)
    if _apply_b is not None:
        return _apply_b
    _apply_a = _handle_apply_term(b_d, a_d)
    if _apply_a is not None:
        return _apply_a

    # Handle parse(String[, Status[, Vars]]) functional use.
    # parse is EAGER in LIFE: evaluate immediately if the string is bound.
    # If the string argument is unbound, residuate: attach a pending PROVE goal
    # on the string variable so that once it gets bound, the parse is triggered
    # and the result is unified with the LHS.
    if _is_parse_func(b_d):
        r = _eval_parse_func(b_d, eng)
        if r is not None:
            return _unify(eng, a_d, r)
        # String is unbound — attach a residuated goal on the string variable.
        s_arg = b_d.attr_list.get('1')
        if s_arg is not None:
            s_var = s_arg.deref()
            if _term_is_unbound(s_var, eng):
                from wild_life.data_structures import Goal, Residuation, SORT_VAR
                wl_p = eng.wl
                eq_defn = getattr(wl_p, 'eqsym', None)
                if eq_defn is None and hasattr(wl_p, 'syntax_module'):
                    eq_defn = wl_p.syntax_module.symbol_table.get('=')
                eq_term = PsiTerm(type_def=eq_defn)
                eq_term.attr_list['1'] = a_d
                eq_term.attr_list['2'] = b_d
                eq_term._resid_marker = True
                pending_goal = Goal(GoalType.PROVE, eq_term, None, None, pending=True)
                if s_var.resid is None:
                    eng.trail.trail_psi(s_var, 'resid')
                    s_var.resid = [Residuation(goal=pending_goal)]
                else:
                    if not any(rv.goal is pending_goal for rv in s_var.resid):
                        eng.trail.trail_copy(s_var, 'resid')
                        s_var.resid.append(Residuation(goal=pending_goal))
                # Mark as constrained so it shows as @~
                if not (s_var.flags & SORT_VAR):
                    eng.trail.trail_psi(s_var, 'flags')
                    s_var.flags |= SORT_VAR
                return True  # a_d stays unbound (shown as @)
        return False
    if _is_parse_func(a_d):
        r = _eval_parse_func(a_d, eng)
        if r is not None:
            return _unify(eng, b_d, r)
        # String is unbound — attach a residuated goal on the string variable.
        s_arg = a_d.attr_list.get('1')
        if s_arg is not None:
            s_var = s_arg.deref()
            if _term_is_unbound(s_var, eng):
                from wild_life.data_structures import Goal, Residuation, SORT_VAR
                wl_p = eng.wl
                eq_defn = getattr(wl_p, 'eqsym', None)
                if eq_defn is None and hasattr(wl_p, 'syntax_module'):
                    eq_defn = wl_p.syntax_module.symbol_table.get('=')
                eq_term = PsiTerm(type_def=eq_defn)
                eq_term.attr_list['1'] = b_d
                eq_term.attr_list['2'] = a_d
                eq_term._resid_marker = True
                pending_goal = Goal(GoalType.PROVE, eq_term, None, None, pending=True)
                if s_var.resid is None:
                    eng.trail.trail_psi(s_var, 'resid')
                    s_var.resid = [Residuation(goal=pending_goal)]
                else:
                    if not any(rv.goal is pending_goal for rv in s_var.resid):
                        eng.trail.trail_copy(s_var, 'resid')
                        s_var.resid.append(Residuation(goal=pending_goal))
                from wild_life.data_structures import SORT_VAR
                if not (s_var.flags & SORT_VAR):
                    eng.trail.trail_psi(s_var, 'flags')
                    s_var.flags |= SORT_VAR
                return True
        return False

    # Handle chr(N) functional use: C = chr(N) → character string for ASCII code N
    if _is_chr_func(b_d):
        r = _try_eval_string_func(b_d, eng)
        return _unify(eng, a_d, r) if r is not None else False
    if _is_chr_func(a_d):
        r = _try_eval_string_func(a_d, eng)
        return _unify(eng, b_d, r) if r is not None else False

    # Handle asc(C) functional use: N = asc(C) → ASCII code of character C.
    # A string nothing has filled in yet leaves the answer open rather than
    # failing, and an argument that is no string at all is reported.
    for _asc_side, _asc_other in ((b_d, a_d), (a_d, b_d)):
        if not _is_asc_func(_asc_side):
            continue
        _asc_state, _asc_val = _asc_argument(_asc_side, eng)
        if _asc_state == 'code':
            return _unify(eng, _asc_other, _make_number(eng, _asc_val))
        if _asc_state == 'wait':
            return True
        _report_asc_error(_asc_side, eng)
        return False

    # No number is its own bitwise negation, so `A = \\(A)` fails — and so
    # does `A = B` once `A = \\(B)` is waiting on them, which is the same
    # equation with A and B made one.
    for _bn_side, _bn_other in ((b_d, a_d), (a_d, b_d)):
        if _get_sym(_bn_side) != '\\' or len(_bn_side.attr_list) != 1:
            continue
        _bn_arg = _bn_side.attr_list.get('1')
        if _bn_arg is not None and _bn_arg.deref() is _bn_other:
            return False

    # Handle bagof/findall/setof in functional position:
    #   L = bagof(Template, Goal)  →  collect all solutions and unify with L
    _BAGOF_NAMES = frozenset(('bagof', 'findall', 'setof'))
    if (b_d.type is not None and
            b_d.type._builtin_func is not None and
            b_d.type.keyword and
            b_d.type.keyword.symbol in _BAGOF_NAMES):
        tmpl = b_d.attr_list.get('1')
        g_   = b_d.attr_list.get('2')
        if tmpl is not None and g_ is not None and '3' not in b_d.attr_list:
            collected = _collect_solutions(tmpl.deref(), g_.deref(), eng)
            result_list = eng.wl.make_list(collected)
            return _unify(eng, a_d, result_list)

    if (a_d.type is not None and
            a_d.type._builtin_func is not None and
            a_d.type.keyword and
            a_d.type.keyword.symbol in _BAGOF_NAMES):
        tmpl = a_d.attr_list.get('1')
        g_   = a_d.attr_list.get('2')
        if tmpl is not None and g_ is not None and '3' not in a_d.attr_list:
            collected = _collect_solutions(tmpl.deref(), g_.deref(), eng)
            result_list = eng.wl.make_list(collected)
            return _unify(eng, b_d, result_list)

    # Handle conjunction (& / psi-term meet): A = t1 & t2
    # Use _apply_and_conjunction_to_var which creates choice points that bind
    # the TARGET variable (a_d/b_d) directly — unlike the old approach of
    # computing the result in a local fresh variable and then unifying, which
    # caused choice points to bind the local fresh var rather than the target,
    # so backtracking would leave the target unbound.
    if b_d.type is not None and b_d.type is eng.wl.and_sym:
        t1_r = b_d.attr_list.get('1')
        t2_r = b_d.attr_list.get('2')
        if t1_r is not None and t2_r is not None:
            return _apply_and_conjunction_to_var(b_d, a_d, eng)

    if a_d.type is not None and a_d.type is eng.wl.and_sym:
        t1_r = a_d.attr_list.get('1')
        t2_r = a_d.attr_list.get('2')
        if t1_r is not None and t2_r is not None:
            return _apply_and_conjunction_to_var(a_d, b_d, eng)

    # Handle disjunction on RHS: A = {b1;b2;...} → try A=b1, choice for rest
    if b_d.type is not None and b_d.type is eng.wl.disjunction:
        elems = _collect_disjunction(b_d, eng)
        if elems:
            # Push choice points in reverse order (last alternative first)
            for alt in reversed(elems[1:]):
                eng.push_choice_point(GoalType.UNIFY, a_d, alt, None)
            return _unify(eng, a_d, elems[0])

    # Handle disjunction on LHS: {a1;a2} = B → try a1=B with choice for rest
    if a_d.type is not None and a_d.type is eng.wl.disjunction:
        elems = _collect_disjunction(a_d, eng)
        if elems:
            for alt in reversed(elems[1:]):
                eng.push_choice_point(GoalType.UNIFY, alt, b_d, None)
            return _unify(eng, elems[0], b_d)

    # Handle disjunctions embedded in either side (e.g. [{1;2;3}|T] → [1|T],
    # [2|T], [3|T]): solve the equation once per combination.
    a_is_var = (a_d.type is None or (a_d.type is eng.wl.top and not a_d.attr_list))
    b_is_var = (b_d.type is None or (b_d.type is eng.wl.top and not b_d.attr_list))
    _disj_sides = [_t for _t in (a_d, b_d) if _t.type is not None and _t.attr_list]
    for _side in _disj_sides:
        # Only inside arithmetic, where a disjunction has to surface before the
        # expression can distribute over it.  Elsewhere the function call is
        # left for the ordinary evaluation to reduce.
        if _get_sym(_side) in _ARITH_OPS_SET:
            _inline_disjunctive_funcs(_side, eng)
    if not a_is_var and not b_is_var:
        # Two arithmetic terms, so there is no variable to bind a rebuilt copy
        # to: pick the alternatives inside the terms themselves, which is also
        # what lets `X:(3*sgn)` read as 3 rather than as the whole disjunction.
        _disj_arith = [_side for _side in _disj_sides
                       if _term_contains_disjunction(_side, eng)]
        if _disj_arith and all(_get_sym(_side) in _ARITH_OPS_SET
                               for _side in _disj_arith):
            if _expand_disjunctions_in_place(a_d, b_d, eng):
                return True
    else:
        _expr, _var = (b_d, a_d) if a_is_var else (a_d, b_d)
        if _expr.type is not None and _term_contains_disjunction(_expr, eng):
            # A disjunction written inside a term is the term's own: `X =
            # f(Y:{a;Z},Z:{b;Y})` leaves Y worth a and comes back for Z,
            # where a rebuilt copy would leave Y the whole disjunction it
            # was.  Arithmetic is a different matter — `1 + {1;2}` is
            # `{2;3}` — and is left to the distribution below.
            if _get_sym(_expr) not in _ARITH_OPS_SET:
                _nodes_dj = _disjunction_nodes(_expr, eng)
                if _nodes_dj:
                    if _expand_disjunctions_in_place(_var, _expr, eng,
                                                     bind_nodes=True):
                        return True
            alts = _expand_term_disjunctions(_expr, eng)
            if len(alts) > 1:
                # Evaluate embedded user function calls in each alternative
                # in-place, so that s(f(1)) reduces to s(1).
                for _alt_ev in alts:
                    _eval_embedded_user_funcs(_alt_ev, eng, 0, set())
                for alt in reversed(alts[1:]):
                    eng.push_choice_point(GoalType.UNIFY, _var, alt, None)
                return _unify(eng, _var, alts[0])

    # Pre-check: detect concrete non-boolean arguments in and/or expressions.
    # Wild Life emits "Non-boolean argument or result in '...'." when any direct
    # argument of an and/or operator is a concrete atom that is neither true nor
    # false (free variables and nested bool expressions are OK).
    # This check must run BEFORE _try_eval_bool so that 'true and c' shows
    # 'true and c' in the error message (not just 'c' after simplification).
    def _check_nonbool_bool_arg(expr_t):
        """Report a concrete non-boolean argument of a boolean operator."""
        _sym_nb = _get_sym(expr_t)
        if _sym_nb not in ('and', 'or', 'not', 'xor'):
            return False
        _a1_nb = expr_t.attr_list.get('1')
        _a2_nb = expr_t.attr_list.get('2')
        if _a1_nb is None:
            return False
        if _sym_nb == 'not':
            # `not` takes one argument, and reads as `not b` when it is wrong.
            if _a2_nb is not None:
                return False
            _a1_nb = _a1_nb.deref()
            if _bool_operand_ok(_a1_nb, eng.wl):
                return False
            import sys as _sys_n1
            print(f"*** Error: Non-boolean argument or result in "
                  f"'not {_get_sym(_a1_nb) or '@'}'.", file=_sys_n1.stderr)
            return True
        if _a2_nb is None:
            return False
        _a1_nb = _a1_nb.deref()
        _a2_nb = _a2_nb.deref()
        _wl_nb = eng.wl
        from wild_life.data_structures import SORT_VAR as _SV_NB
        def _bool_arg_ok(t_ok):
            t_ok = _bool_operand(t_ok, eng)
            s_ok = _get_sym(t_ok)
            if s_ok in ('true', 'false'):
                return True
            if _is_proper_bool_expr(t_ok):
                return True
            # A comparison answers a boolean, so it stands where one is
            # wanted: `A =< 57 or A >= 97` asks two questions about A.
            if s_ok in _BOOL_VALUED_COMPARISONS and len(t_ok.attr_list) == 2:
                return True
            # Free variable: no attrs, no value, and top/None/bool/sort-var type
            _fr = not t_ok.attr_list and t_ok.value is None and t_ok.coref is None
            return _fr and (t_ok.type is None or t_ok.type is _wl_nb.top
                            or t_ok.type is _wl_nb.boolean
                            or bool(t_ok.flags & _SV_NB))
        if not _bool_arg_ok(_a1_nb) or not _bool_arg_ok(_a2_nb):
            _ds1 = _get_sym(_a1_nb) or '@'
            _ds2 = _get_sym(_a2_nb) or '@'
            import sys as _sys_nb
            print(f"*** Error: Non-boolean argument or result in "
                  f"'{_ds1} {_sym_nb} {_ds2}'.",
                  file=_sys_nb.stderr)
            return True
        return False
    if _check_nonbool_bool_arg(b_d) or _check_nonbool_bool_arg(a_d):
        return False

    # Try to evaluate functional terms before unifying (boolean ops)
    _b_orig_for_bool = b_d   # save original so we can collect vars to mark after eval
    b_evaled = _try_eval_bool(b_d, eng)
    if b_evaled is not None:
        # Boolean expression was fully or partially evaluated.
        # Mark all free variables that appeared in the original expression as
        # bool-constrained with resid=[] (no pending constraint — tilde suppressed).
        # This handles short-circuit cases (e.g. B and false → false consumes B)
        # and idempotent cases (e.g. B and B → B keeps sort bool, resid=[]).
        _bool_free_evaled: list = []
        _collect_bool_free_vars(_b_orig_for_bool, eng.wl, _bool_free_evaled, set())
        # Check if this is a re-fire of a pending residuated goal. If so,
        # clear the specific pending resid entry from all affected variables
        # (constraint resolved by partial eval — no watchdog needed anymore).
        _bool_is_refire = getattr(goal, '_resid_marker', False)
        for _bv_evaled in _bool_free_evaled:
            _mark_bool_sort(_bv_evaled, eng.wl, eng)
            if _bool_is_refire:
                _remove_resid_for_goal(_bv_evaled, goal, eng)
        # Also mark the result itself if it is a free variable
        # (e.g. and(B,B)→B or and(true,X)→X; A = result gives A bool sort)
        _result_evaled = b_evaled.deref()
        _mark_bool_sort(_result_evaled, eng.wl, eng)
        if _bool_is_refire:
            _remove_resid_for_goal(_result_evaled, goal, eng)
        # Also mark a_d (LHS) if it is a free variable: when we are about to
        # unify it with b_evaled, a_d should have resid=[] too so it does not
        # display as bool~ if it ends up as the canonical representative.
        _a_d_cur_for_mark = a_d.deref()
        _mark_bool_sort(_a_d_cur_for_mark, eng.wl, eng)
        if _bool_is_refire:
            _remove_resid_for_goal(_a_d_cur_for_mark, goal, eng)
        b_d = b_evaled
    else:
        _a_orig_for_bool = a_d
        a_evaled = _try_eval_bool(a_d, eng)
        if a_evaled is not None:
            _bool_free_evaled_a: list = []
            _collect_bool_free_vars(_a_orig_for_bool, eng.wl, _bool_free_evaled_a, set())
            for _bv_evaled_a in _bool_free_evaled_a:
                _mark_bool_sort(_bv_evaled_a, eng.wl, eng)
            _result_evaled_a = a_evaled.deref()
            _mark_bool_sort(_result_evaled_a, eng.wl, eng)
            # Also mark b_d (other side) if it's a free variable.
            _b_d_cur_for_mark = b_d.deref()
            _mark_bool_sort(_b_d_cur_for_mark, eng.wl, eng)
            a_d = a_evaled

    # Boolean residuation: if one side is an unevaluated boolean expression
    # with free variables, propagate bool-sort constraints and set up a
    # suspended goal that re-fires when any free variable is bound.
    #
    # Design mirrors arithmetic residuation:
    #  - true = and(B,C)  → force B=true, C=true  (unique back-propagation)
    #  - false = or(B,C)  → force B=false, C=false (unique)
    #  - true = not(B)    → force B=false
    #  - false = not(B)   → force B=true
    #  - other cases      → suspend (non-deterministic or no-op if LHS is free)
    # Use _is_proper_bool_expr to distinguish genuine boolean operator applications
    # (and(X,Y), or(X,Y), not(X), xor(X,Y)) from psi-terms that happen to use
    # 'and'/'or' as a constructor name with the wrong arity (e.g. and(B) with
    # only 1 argument, which should be treated as a regular psi-term).
    # A such-that term standing as a value — `A = (X | call(p(X)))` — proves
    # its guard and takes the value part, the same as a such-that rule body.
    _st_side = None
    for _cand in (b_d, a_d):
        if (_cand.type is not None and _cand.type is eng.wl.such_that
                and '1' in _cand.attr_list and '2' in _cand.attr_list):
            _st_side = _cand
            break
    if _st_side is not None:
        _st_other = a_d if _st_side is b_d else b_d
        _st_val = _st_side.attr_list['1']
        _st_cond = _st_side.attr_list['2'].deref()
        # Pushed in LIFO order: the guard runs first, then the value is taken.
        eng.push_goal(GoalType.UNIFY, _st_other, _st_val, None)
        eng.push_goal(GoalType.PROVE, _st_cond, _DEFRULES_SENTINEL, None)
        return True

    # An arithmetic comparison in functional position has a boolean value:
    # `X = (1 =< 7)` answers true, and part.lf writes the test as
    # `(1 =< X) = true`.  While an operand is still unknown the comparison
    # suspends on it, so `leq(7,N)` can constrain N before N is known.
    _CMP_FUNC_SYMS = frozenset(('>', '<', '>=', '=<', '=:=', '=\\='))

    def _is_cmp_expr(t):
        from wild_life.data_structures import NON_STRICT_TERM as _NST_CMP
        return (_get_sym(t) in _CMP_FUNC_SYMS and '1' in t.attr_list
                and '2' in t.attr_list and not (t.flags & _NST_CMP))

    _b_is_cmp = _is_cmp_expr(b_d)
    _a_is_cmp = (not _b_is_cmp) and _is_cmp_expr(a_d)
    if _b_is_cmp or _a_is_cmp:
        _cmp_expr, _cmp_other = (b_d, a_d) if _b_is_cmp else (a_d, b_d)
        _ok1_cmp, _v1_cmp = _eval_arith(_cmp_expr.attr_list['1'], eng)
        _ok2_cmp, _v2_cmp = _eval_arith(_cmp_expr.attr_list['2'], eng)
        if _ok1_cmp and _ok2_cmp:
            _sym_cmp = _get_sym(_cmp_expr)
            _truth_cmp = {
                '>': _v1_cmp > _v2_cmp, '<': _v1_cmp < _v2_cmp,
                '>=': _v1_cmp >= _v2_cmp, '=<': _v1_cmp <= _v2_cmp,
                '=:=': _v1_cmp == _v2_cmp, '=\\=': _v1_cmp != _v2_cmp,
            }[_sym_cmp]
            return _unify(eng, _cmp_other,
                          _make_atom(eng, 'true' if _truth_cmp else 'false'))
        # The comparison itself is not an arithmetic operator, so its two
        # operands are walked rather than the term as a whole.
        _cmp_vars: list = []
        _cmp_seen: set = set()
        for _ck in ('1', '2'):
            _collect_arith_vars(_cmp_expr.attr_list[_ck], eng.wl,
                                _cmp_vars, _cmp_seen)
        if _cmp_vars:
            from wild_life.data_structures import Goal as _CmpGoal, SORT_VAR as _SV_CMP
            _eq_defn_cmp = (getattr(eng.wl, 'eqsym', None) or
                            eng.wl.syntax_module.symbol_table.get('='))
            _eq_cmp = PsiTerm(type_def=_eq_defn_cmp)
            _eq_cmp.attr_list['1'] = _cmp_other
            _eq_cmp.attr_list['2'] = _cmp_expr
            _eq_cmp._resid_marker = True
            _pend_cmp = _CmpGoal(GoalType.PROVE, _eq_cmp, None, None, pending=True)
            for _cv in _cmp_vars:
                _attach_arith_resid(_cv, eng.wl, _pend_cmp, eng)
            _other_cur_cmp = _cmp_other.deref()
            if (_other_cur_cmp.value is None and not _other_cur_cmp.attr_list
                    and (_other_cur_cmp.type is eng.wl.top
                         or _other_cur_cmp.type is None
                         or bool(_other_cur_cmp.flags & _SV_CMP))):
                _attach_bool_resid(_other_cur_cmp, eng.wl, _pend_cmp, eng)
            return True
        return False

    _b_bool_unevaluated = (b_evaled is None) and _is_proper_bool_expr(b_d)
    # Also handle: bool expr on the LHS (e.g. and(B,C) = true)
    # We check a_d only if b_d is not already a bool expr (to avoid double-handling).
    _a_bool_unevaluated = (not _b_bool_unevaluated) and _is_proper_bool_expr(a_d)

    if _b_bool_unevaluated or _a_bool_unevaluated:
        # Normalise: bool_expr is the expression side, other_side is the other side.
        if _b_bool_unevaluated:
            _bool_expr_br, _other_br = b_d, a_d
        else:
            _bool_expr_br, _other_br = a_d, b_d
        _bool_sym_br = _get_sym(_bool_expr_br)

        # Both what the expression is made of and what it is being equated
        # with have to be able to be booleans at all.
        if not (_bool_operand_ok(_bool_expr_br, eng.wl)
                and _bool_operand_ok(_other_br, eng.wl)):
            return False

        # Nothing is its own negation, so `A = not(A)` fails — and so does
        # `A = B` once `A = not(B)` is waiting on them, which is the same
        # equation with A and B made one.
        if _bool_sym_br == 'not':
            _not_arg_br = _bool_expr_br.attr_list.get('1')
            if _not_arg_br is not None and _not_arg_br.deref() is _other_br:
                return False

        # Collect free variables inside the boolean expression.
        _bool_vars_br: list = []
        _collect_bool_free_vars(_bool_expr_br, eng.wl, _bool_vars_br, set())

        if _bool_vars_br:
            # --- Deterministic backward propagation ---
            _other_sym_br = _get_sym(_other_br)
            if _other_sym_br == 'true' and _bool_sym_br == 'and':
                # true = and(B, C)  →  B = true, C = true
                for _bv_br in _bool_vars_br:
                    if not _unify(eng, _bv_br, _make_atom(eng, 'true')):
                        return False
                return True
            elif _other_sym_br == 'false' and _bool_sym_br == 'or':
                # false = or(B, C)  →  B = false, C = false
                for _bv_br in _bool_vars_br:
                    if not _unify(eng, _bv_br, _make_atom(eng, 'false')):
                        return False
                return True
            elif _other_sym_br == 'false' and _bool_sym_br == 'not':
                # false = not(B)  →  B = true
                for _bv_br in _bool_vars_br:
                    if not _unify(eng, _bv_br, _make_atom(eng, 'true')):
                        return False
                return True
            elif _other_sym_br == 'true' and _bool_sym_br == 'not':
                # true = not(B)  →  B = false
                for _bv_br in _bool_vars_br:
                    if not _unify(eng, _bv_br, _make_atom(eng, 'false')):
                        return False
                return True
            elif _bool_sym_br == 'xor':
                # xor backward propagation.
                # Evaluate each arg of the top-level xor to get their effective
                # simplified values (e.g. xor(xor(false,false), D) → D). This
                # lets us detect propagatable constraints even when the xor args
                # are nested sub-expressions whose concrete sub-terms have cancelled.
                _xor_a1_raw_br = _bool_expr_br.attr_list.get('1')
                _xor_a2_raw_br = _bool_expr_br.attr_list.get('2')
                _xor_a1_raw_d = _xor_a1_raw_br.deref() if _xor_a1_raw_br is not None else None
                _xor_a2_raw_d = _xor_a2_raw_br.deref() if _xor_a2_raw_br is not None else None
                # Effective arg = simplified via _try_eval_bool, or original deref
                _xor_a1_eff = ((_try_eval_bool(_xor_a1_raw_d, eng) or _xor_a1_raw_d)
                               if _xor_a1_raw_d is not None else None)
                _xor_a2_eff = ((_try_eval_bool(_xor_a2_raw_d, eng) or _xor_a2_raw_d)
                               if _xor_a2_raw_d is not None else None)
                _xor_s1_eff = _get_sym(_xor_a1_eff) if _xor_a1_eff is not None else ''
                _xor_s2_eff = _get_sym(_xor_a2_eff) if _xor_a2_eff is not None else ''

                def _xor_arg_is_free(t):
                    """True iff t (already deref'd) is an unbound free variable."""
                    if t is None:
                        return False
                    t = t.deref()
                    return (not t.attr_list and t.coref is None and t.value is None
                            and _get_sym(t) not in ('true', 'false'))

                _a1_eff_free = _xor_arg_is_free(_xor_a1_eff)
                _a2_eff_free = _xor_arg_is_free(_xor_a2_eff)

                if _other_sym_br == 'false':
                    if _a1_eff_free and _a2_eff_free:
                        # false = xor(D, E) with both D and E free → D = E
                        if not _unify(eng, _xor_a1_eff.deref(), _xor_a2_eff.deref()):
                            return False
                        _canon_xor_br = _xor_a1_eff.deref()
                        _mark_bool_sort(_canon_xor_br, eng.wl, eng)
                        if _canon_xor_br.resid:
                            eng.trail.trail_psi(_canon_xor_br, 'resid')
                            _canon_xor_br.resid = []
                        return True
                    elif _a1_eff_free and _xor_s2_eff in ('true', 'false'):
                        # false = xor(D, concrete) → D = concrete
                        if not _unify(eng, _xor_a1_eff.deref(),
                                      _make_atom(eng, _xor_s2_eff)):
                            return False
                        return True
                    elif _a2_eff_free and _xor_s1_eff in ('true', 'false'):
                        # false = xor(concrete, E) → E = concrete
                        if not _unify(eng, _xor_a2_eff.deref(),
                                      _make_atom(eng, _xor_s1_eff)):
                            return False
                        return True
                elif _other_sym_br == 'true':
                    # true = xor(B, C) → B and C must differ (complement)
                    if _a1_eff_free and _xor_s2_eff in ('true', 'false'):
                        _comp = 'false' if _xor_s2_eff == 'true' else 'true'
                        if not _unify(eng, _xor_a1_eff.deref(), _make_atom(eng, _comp)):
                            return False
                        return True
                    elif _a2_eff_free and _xor_s1_eff in ('true', 'false'):
                        _comp = 'false' if _xor_s1_eff == 'true' else 'true'
                        if not _unify(eng, _xor_a2_eff.deref(), _make_atom(eng, _comp)):
                            return False
                        return True
                # For true=xor(both free), fall through to suspend

                # Self-referential xor: LHS free var appears in xor args.
                # E.g. C = B xor C → (C xor C) = B → false = B → B = false.
                # Also handles symmetric: C = C xor B, B = B xor C, etc.
                # Check if LHS is a free/unbound variable (possibly bool-sorted).
                # NOTE: _get_sym returns 'bool' for bool-sorted free vars (type=wl.boolean),
                # so we cannot use `_other_sym_br is None`; instead check the actual state.
                _other_deref_sr = _other_br.deref()
                _other_is_free_sr = (
                    not _other_deref_sr.attr_list and
                    _other_deref_sr.coref is None and
                    _other_deref_sr.value is None and
                    _other_sym_br not in ('true', 'false')
                )
                if _other_is_free_sr:
                    _other_d_selfref = _other_deref_sr
                    # Check using both raw and effective args for self-reference
                    _xor_arg_pairs = [
                        (_xor_a1_raw_d, _xor_a2_eff),
                        (_xor_a2_raw_d, _xor_a1_eff),
                    ]
                    for _xarg_self, _xarg_other_self in _xor_arg_pairs:
                        if (_xarg_self is not None and
                                id(_xarg_self.deref()) == id(_other_d_selfref)):
                            # Self-reference: LHS = LHS xor other → other = false
                            if _xarg_other_self is not None:
                                _xother_d = _xarg_other_self.deref()
                                if not _unify(eng, _xother_d,
                                              _make_atom(eng, 'false')):
                                    return False
                            # Mark LHS as bool, clear its pending goals
                            _mark_bool_sort(_other_d_selfref, eng.wl, eng)
                            if _other_d_selfref.resid:
                                eng.trail.trail_psi(_other_d_selfref, 'resid')
                                _other_d_selfref.resid = []
                            return True

            # --- Non-deterministic or free-LHS case: suspend ---
            from wild_life.data_structures import Goal as _BoolGoal
            _wl_br = eng.wl
            _eq_defn_br = (getattr(_wl_br, 'eqsym', None) or
                           _wl_br.syntax_module.symbol_table.get('='))
            _bool_eq_br = PsiTerm(type_def=_eq_defn_br)
            # Store as (other = bool_expr) so re-firing reads the right
            # sides as a and b respectively.
            _bool_eq_br.attr_list['1'] = _other_br
            _bool_eq_br.attr_list['2'] = _bool_expr_br
            _bool_eq_br._resid_marker = True
            _bool_pend_br = _BoolGoal(GoalType.PROVE, _bool_eq_br,
                                      None, None, pending=True)
            for _bv_br in _bool_vars_br:
                _attach_bool_resid(_bv_br, _wl_br, _bool_pend_br, eng)
            # Also attach to _other_br if it is a genuine free variable
            # (not a concrete atom like 'false'), so it displays as bool~
            # and wakes the goal when it gets a value.
            from wild_life.data_structures import SORT_VAR as _SORT_VAR_BR
            _other_cur_br = _other_br.deref()
            _other_is_free_br = (
                _other_cur_br.value is None and not _other_cur_br.attr_list
                and (_other_cur_br.type is _wl_br.top
                     or _other_cur_br.type is None
                     or bool(_other_cur_br.flags & _SORT_VAR_BR))
            )
            if _other_is_free_br:
                _attach_bool_resid(_other_cur_br, _wl_br, _bool_pend_br, eng)
            return True

    # Detect whether this call is a re-fire of a suspended residuated goal
    # (as opposed to the initial constraint setup).  The eq_term created during
    # residuation is tagged with _resid_marker=True; when _wakeup_resid fires
    # the pending goal it passes the tagged eq_term as *goal*, so we can detect
    # re-fires here without any extra bookkeeping.
    is_resid_refiring: bool = getattr(goal, '_resid_marker', False)

    # An equation is read the same either way round: `A*B = 20` states what
    # the product is, just as `20 = A*B` does, and suspends on A and B rather
    # than failing for having the expression on the left.
    if (_get_sym(a_d) in _ARITH_OPS_SET and a_d.attr_list
            and _get_sym(b_d) not in _ARITH_OPS_SET
            and not (a_d.flags & _NST_SWAP) and not (b_d.flags & _NST_SWAP)
            and _try_eval_arith_to_term(a_d, eng) is None):
        _swapped = PsiTerm(type_def=goal.type)
        _swapped.attr_list = {'1': b_d, '2': a_d}
        return bi_unify(_swapped, eng)

    # Try arithmetic evaluation on the RHS (for A = 1+2 style).
    # Skip user-defined function calls here — they are handled by eval_aim,
    # and evaluating them twice creates separate Python objects that each fire
    # delay rules independently, producing double output.
    # EXCEPTION: 0-arity user functions (e.g. `result` after `result<<-4`)
    # are NOT handled by eval_aim in this context (no EVAL goal is set up for
    # them), so we evaluate them directly here.
    # Also skip if the term is tagged NON_STRICT_TERM (bound inside a non-strict
    # predicate — the expression should remain as data, not be evaluated).
    from wild_life.data_structures import NON_STRICT_TERM as _BI_NST
    _b_is_user_fn = (b_d.type is not None and b_d.type.type == DefType.FUNCTION)
    _b_is_non_strict = bool(b_d.flags & _BI_NST)
    # Evaluate 0-arity user functions (global variables like `result`) directly.
    # _eval_user_func_sync makes trailed side-effects (unifying the function atom
    # with its rule head copy).  We save a trail mark, call eval, undo the side
    # effects, and only keep the VALUE if it turned out to be concrete.
    # This avoids corrupting `result`'s coref and prevents spurious arithmetic
    # constraints from unevaluated or self-referential rule bodies.
    # A name on the left stands for what it answers too: `A:emp = stu` puts
    # school's two terms together, and reading only the right-hand one leaves
    # the answer half made.
    _a_is_user_fn_0 = (a_d.type is not None and a_d.type.type == DefType.FUNCTION
                       and a_d.type.rule and not callable(a_d.type.rule))
    if (_a_is_user_fn_0 and not (a_d.flags & _BI_NST) and not a_d.attr_list
            and not _a_was_backtick):
        _a_evaled = _eval_user_func_sync(a_d, eng, 0)
        if _a_evaled is not None:
            _a_ev_d = _a_evaled.deref()
            if _a_ev_d is not a_d and _is_settled_value(_a_ev_d, a_d.type):
                # The name's own node becomes what it answers, so a tag on it
                # — the A of `A:emp` — reads the term and not the name.
                if a_d.coref is None:
                    eng.trail.trail_psi(a_d, 'coref')
                    a_d.coref = _a_ev_d
                a_d = _a_ev_d

    if _b_is_user_fn and not _b_is_non_strict and not b_d.attr_list and not _b_was_backtick:
        _0a_mark = eng.trail.mark()
        _b_evaled = _eval_user_func_sync(b_d, eng, 0)
        # Copied while the bindings that made it are still standing, since the
        # trail is wound back next.
        _b_evaled_d = (copy_term(_b_evaled.deref(), {})
                       if _b_evaled is not None else None)
        eng.trail.undo_to(_0a_mark)  # undo coref-linking of atom with rule-head copy
        if _b_evaled_d is not None:
            # A concrete value is taken as the answer.  A compound one is too,
            # so long as it is settled — `add3` answers the composition it
            # stands for — where an expression still waiting on something, as a
            # global's stored `@ + 1` is, leaves the name standing for itself.
            if _b_evaled_d.value is not None:
                # Concrete numeric result → create a fresh number term (the
                # original _b_evaled object may reference now-undone bindings).
                b_d = _make_number(eng, float(_b_evaled_d.value))
                _b_is_user_fn = False
            else:
                # Compound result — try arithmetic evaluation
                _b_arith2 = _try_eval_arith_to_term(_b_evaled_d, eng)
                if _b_arith2 is not None:
                    b_d = _b_arith2
                    _b_is_user_fn = False
                elif (_is_settled_value(_b_evaled_d, b_d.type)
                      or _b_evaled_d.type is eng.wl.disjunction):
                    # A name that answers a disjunction has answered: what
                    # `number` is worth is one of nine numbers, and the
                    # equation takes them one at a time.
                    b_d = _b_evaled_d
                    _b_is_user_fn = False
                # else: keep original b_d (the 0-arity function atom) so that
                # normal unification treats `result` as a sort variable.
    b_arith = _try_eval_arith_to_term(b_d, eng) if (not _b_is_user_fn and not _b_is_non_strict) else None
    if b_arith is not None:
        # Expression fully evaluated — proceed to unify LHS with result.
        b_d = b_arith
        # A left side that is still an expression states a constraint, not a
        # shape to match: `A + B = 0 + 1` says what the sum is, the same as
        # `A + B = 1` does, and waits on A and B rather than failing to match
        # a sum against a number.
        if (_get_sym(a_d) in _ARITH_OPS_SET and a_d.attr_list
                and not (a_d.flags & _BI_NST)
                and _try_eval_arith_to_term(a_d, eng) is None):
            _eq_rhs = PsiTerm(type_def=goal.type)
            _eq_rhs.attr_list = {'1': a_d, '2': b_d}
            return bi_unify(_eq_rhs, eng)
    else:
        # RHS not fully evaluated. Try evaluating the LHS if it looks like an
        # arithmetic expression (handles  eval(A) = B  or  3+4 = X  style).
        if not _b_is_user_fn:
            _a_sym_lhs = a_d.type.keyword.symbol if (a_d.type and a_d.type.keyword) else ''
            _a_is_nst_lhs = bool(a_d.flags & _BI_NST)
            _a_is_ufn_lhs = (a_d.type is not None and a_d.type.type == DefType.FUNCTION)
            _a_could_eval_lhs = (_a_sym_lhs == 'eval' or
                                 (_a_sym_lhs in _ARITH_OPS_SET
                                  and not _a_is_nst_lhs
                                  and _is_complete_arith_expr(a_d)))
            if _a_could_eval_lhs and not _a_is_ufn_lhs and not _a_is_nst_lhs:
                _a_arith_lhs = _try_eval_arith_to_term(a_d, eng)
                if _a_arith_lhs is not None:
                    # Re-enter as an equation: the RHS may still be a
                    # constraint to solve (`3*1 = 10//A`), which plain
                    # unification against a number could only fail on.
                    _eq_lhs = PsiTerm(type_def=goal.type)
                    _eq_lhs.attr_list = {'1': _a_arith_lhs, '2': b_d}
                    return bi_unify(_eq_lhs, eng)
        # Arithmetic expression that couldn't be fully evaluated (has variables).
        wl = eng.wl
        b_sym = b_d.type.keyword.symbol if b_d.type and b_d.type.keyword else ''
        # `A = +(B)` is a plus waiting for its second operand.  It cannot be
        # evaluated and so leaves no residuation, but the operand it does have
        # still has to be a number, which is what makes B display as real.
        if (b_sym == '+' and not _b_is_non_strict
                and set(b_d.attr_list.keys()) == {'1'}):
            _plus_arg = b_d.attr_list['1'].deref()
            if _plus_arg.value is None and not _plus_arg.attr_list:
                _mark_real_sort(_plus_arg, eng.wl, eng)

        if b_sym in _ARITH_OPS_SET and not _b_is_non_strict and _is_complete_arith_expr(b_d):
            # Mark all free variables in the arithmetic expression (and the LHS
            # if free) as constrained to sort real.  This ensures that even when
            # the constraint is solved immediately (e.g. A=A+0 → trivial) the
            # variable still displays as 'real' rather than '@'.
            _arith_mark_vars: list = []
            _collect_arith_vars(b_d, eng.wl, _arith_mark_vars, set())
            for _amv in _arith_mark_vars:
                _mark_real_sort(_amv, eng.wl, eng)
            _lhs_chk = a_d.deref()
            if _lhs_chk.value is None and not _lhs_chk.attr_list:
                _mark_real_sort(_lhs_chk, eng.wl, eng)

            # Check whether LHS is currently free (determines which rules apply).
            a_d_cur = a_d.deref()
            a_d_cur_is_free = (a_d_cur.value is None and not a_d_cur.attr_list)

            # Try simplification first (before self-ref check).
            b_simplified = _simplify_arith(b_d, eng)
            if b_simplified is not None and not a_d_cur_is_free:
                # What is left of the expression once the other side is
                # known is solved for its number, not made one term with
                # the side that solved it.  `A = B*A` is a product while
                # A is free; once A is worth 1 the product comes down to
                # B, and B is worth 1 in its own right -- login.c reaches
                # it by dividing, and never corefs the two.  An expression
                # that came down to a name while the other side was still
                # free -- `A = B+0` -- did make the two one term, and that
                # stands.
                _bs_d = b_simplified.deref()
                _a_num = a_d.deref()
                if (_a_num.value is not None and _bs_d.value is None
                        and not _bs_d.attr_list and _bs_d.coref is None):
                    return _unify(eng, _bs_d, _make_number(eng, _a_num.value))
            if b_simplified is not None:
                # Simplification succeeded — start the equation again with the
                # simpler form, so that everything an equation gets is applied
                # to it.  `C2 + E = N + 0` is `C2 + E = N`, which states what
                # the sum is; carrying on from here instead would try to match
                # a sum against N and fail.
                _eq_simp = PsiTerm(type_def=goal.type)
                _eq_simp.attr_list = {'1': a_d, '2': b_simplified}
                return bi_unify(_eq_simp, eng)
            else:
                # Gather free variables in the expression.
                vars_in_expr: list = []
                _collect_arith_vars(b_d, wl, vars_in_expr, set())

                a_d_final = a_d.deref()
                a_d_is_free = (a_d_final.value is None and not a_d_final.attr_list)

                # --- Algebraic solving ---
                # Case 1: LHS (a_d_final) is BOUND and expression has exactly
                #         one free variable → solve  a_d_val = a*x + b  for x.
                # Case 2: LHS is FREE and x_var appears in expression (self-ref):
                #         a = a_coeff*a + b_psi  →  if a_coeff=1: unify b_psi with 0;
                #         else: a = b_psi / (1 - a_coeff).
                solved = False
                if vars_in_expr:
                    ok_lhs, v_lhs = _eval_arith(a_d_final, eng)
                    # Integer division answers a whole number, so a side with
                    # a fraction is not something it can come to, whatever
                    # its arguments: `A = B//B` refuses `A = 24.332` rather
                    # than solving it for B.
                    if (ok_lhs and v_lhs != int(v_lhs)
                            and _get_sym(b_d) == '//'):
                        return False
                    if ok_lhs and len(vars_in_expr) == 1:
                        x_var = vars_in_expr[0].deref()
                        coeffs = _get_linear_coeff(b_d, x_var, eng)
                        if coeffs is not None:
                            a_coeff, b_const = coeffs
                            # v_lhs = a_coeff * x + b_const  →  x = (v_lhs - b_const) / a_coeff
                            if a_coeff != 0.0:
                                x_val = (v_lhs - b_const) / a_coeff
                                if x_val != 0.0 and _has_int_div(b_d):
                                    coeffs = None   # leave it waiting
                            if coeffs is not None and a_coeff != 0.0:
                                x_term = _make_number(eng, x_val)
                                solved = True
                                result = _unify(eng, x_var, x_term)
                                if not result:
                                    return False
                                return True
                        else:
                            # Special pre-check: 0 = k/x  (numerator is concrete k)
                            _tsnl_sym = b_d.type.keyword.symbol if b_d.type and b_d.type.keyword else ''
                            if _tsnl_sym == '/' and ok_lhs and abs(v_lhs) < 1e-12:
                                _tsnl_a1, _tsnl_a2 = _get_two_args(b_d)
                                if _tsnl_a1 is not None and _tsnl_a2 is not None:
                                    _tsnl_a2_d = _tsnl_a2.deref()
                                    if id(_tsnl_a2_d) == id(x_var):
                                        ok_ka, v_ka = _eval_arith(_tsnl_a1, eng)
                                        if ok_ka:
                                            if abs(v_ka) < 1e-12:
                                                # 0 = 0/x: trivially satisfied → mark x as real (no tilde)
                                                solved = True
                                                _mark_real_sort(x_var, wl, eng)
                                                return True
                                            else:
                                                # 0 = k/x where k≠0: impossible → fail
                                                return False
                            # Try non-linear inversion (e.g. a/x = v → x = a/v)
                            x_val = _try_solve_nonlinear(b_d, x_var, v_lhs, eng)
                            if x_val is _NO_SOLUTION:
                                return False   # the equation has no solution
                            if x_val is not None:
                                x_term = _make_number(eng, x_val)
                                solved = True
                                result = _unify(eng, x_var, x_term)
                                if not result:
                                    return False
                                return True
                    # Case 1b: LHS=0 and expression is X-Y with two free vars →
                    #           0 = X - Y  →  X = Y  (unify both vars).
                    if ok_lhs and v_lhs == 0.0 and len(vars_in_expr) == 2:
                        b_d_sym = b_d.type.keyword.symbol if b_d.type and b_d.type.keyword else ''
                        if b_d_sym == '-':
                            ba1, ba2 = _get_two_args(b_d)
                            if ba1 is not None and ba2 is not None:
                                ok_ba1, _ = _eval_arith(ba1, eng)
                                ok_ba2, _ = _eval_arith(ba2, eng)
                                if not ok_ba1 and not ok_ba2:
                                    solved = True
                                    result = _unify(eng, ba1, ba2)
                                    if not result:
                                        return False
                                    return True
                    # Case 1c: v = A/B with both A,B free vars.  An integer
                    # division is read the same way where the answer it
                    # forces is nothing at all: `0 = A//B` says A is 0
                    # whatever B is, while `3 = A//2` leaves A waiting,
                    # since 6 and 7 both divide to 3.
                    if ok_lhs and len(vars_in_expr) == 2:
                        b_d_sym2 = b_d.type.keyword.symbol if b_d.type and b_d.type.keyword else ''
                        if b_d_sym2 in ('/', '//'):
                            ba1, ba2 = _get_two_args(b_d)
                            if ba1 is not None and ba2 is not None:
                                ba1_d = ba1.deref()
                                ba2_d = ba2.deref()
                                ba1_free = (ba1_d.value is None and not ba1_d.attr_list)
                                ba2_free = (ba2_d.value is None and not ba2_d.attr_list)
                                if ba1_free and ba2_free:
                                    same_var = (id(ba1_d) == id(ba2_d))
                                    if same_var:
                                        # v = A/A → (v-1)*A=0. If v≠1: A=0
                                        if abs(v_lhs - 1.0) > 1e-12:
                                            solved = True
                                            x_term = _make_number(eng, 0.0)
                                            result = _unify(eng, ba1_d, x_term)
                                            if not result:
                                                return False
                                            return True
                                        else:  # v=1: A/A=1 trivially true for any A≠0, mark real
                                            solved = True
                                            _mark_real_sort(ba1_d, wl, eng)
                                            return True
                                    elif v_lhs == 0.0:
                                        # 0 = A/B → A=0, B=real
                                        solved = True
                                        _mark_real_sort(ba2_d, wl, eng)
                                        x_term = _make_number(eng, 0.0)
                                        result = _unify(eng, ba1_d, x_term)
                                        if not result:
                                            return False
                                        return True
                                    elif v_lhs == 1.0:
                                        # 1 = A/B → A=B (unify)
                                        solved = True
                                        _mark_real_sort(ba1_d, wl, eng)
                                        _mark_real_sort(ba2_d, wl, eng)
                                        result = _unify(eng, ba1_d, ba2_d)
                                        if not result:
                                            return False
                                        return True
                                    # else v≠0,1 and different vars → suspend (fall through)

                    if a_d_is_free and _var_in_expr(a_d_final, b_d, set()):
                        # Self-referential: LHS appears in RHS.
                        # a = a_coeff * a + b_psi
                        decomp = _linear_decompose_psi(b_d, a_d_final, eng, wl)
                        if decomp is not None:
                            a_coeff, b_psi = decomp
                            if abs(a_coeff - 1.0) < 1e-12:
                                # a = a + b_psi → 0 = b_psi
                                # b_psi may itself be an arithmetic expression
                                # (e.g. 0-C_var), so push a new prove goal
                                # "0 = b_psi" for the engine to solve rather
                                # than attempting direct structural unification.
                                zero_t = wl.make_integer(0)
                                solved = True
                                ok_bpsi, _ = _eval_arith(b_psi, eng)
                                if ok_bpsi:
                                    # b_psi is already concrete — just check it's 0
                                    result = _unify(eng, b_psi, zero_t)
                                    if not result:
                                        return False
                                    return True
                                elif (b_psi.value is None and
                                        not b_psi.attr_list and
                                        (b_psi.type is wl.top or
                                         getattr(b_psi, 'flags', 0) & __import__('wild_life.data_structures', fromlist=['SORT_VAR']).SORT_VAR)):
                                    # b_psi is a free variable — unify directly with 0
                                    result = _unify(eng, b_psi, zero_t)
                                    if not result:
                                        return False
                                    return True
                                else:
                                    # b_psi is a compound expression — push 0 = b_psi
                                    # as a new prove goal for the engine to handle
                                    eq_defn = getattr(wl, 'eqsym', None) or (
                                        wl.syntax_module.symbol_table.get('=')
                                        if hasattr(wl, 'syntax_module') else None)
                                    if eq_defn is not None:
                                        new_eq = PsiTerm(type_def=eq_defn)
                                        new_eq.attr_list['1'] = zero_t
                                        new_eq.attr_list['2'] = b_psi
                                        eng.push_goal(GoalType.PROVE, new_eq, None, None)
                                    return True
                            else:
                                # a = a_coeff * a + b_psi → a = b_psi/(1-a_coeff)
                                ok_b, v_b = _eval_arith(b_psi, eng)
                                if ok_b:
                                    x_val = v_b / (1.0 - a_coeff)
                                    x_term = _make_number(eng, x_val)
                                    solved = True
                                    result = _unify(eng, a_d_final, x_term)
                                    if not result:
                                        return False
                                    return True
                        else:
                            # decomp is None: try non-linear self-referential patterns
                            _b_sym_nl = b_d.type.keyword.symbol if b_d.type and b_d.type.keyword else ''
                            if _b_sym_nl == '/':
                                _nls_a1, _nls_a2 = _get_two_args(b_d)
                                if _nls_a1 is not None and _nls_a2 is not None:
                                    _nls_a2_d = _nls_a2.deref()
                                    if id(_nls_a2_d) == id(a_d_final):
                                        # A = k/A → A² = k. If k=0: A=0.
                                        ok_k, v_k = _eval_arith(_nls_a1, eng)
                                        if ok_k and abs(v_k) < 1e-12:
                                            solved = True
                                            x_term = _make_number(eng, 0.0)
                                            result = _unify(eng, a_d_final, x_term)
                                            if not result:
                                                return False
                                            return True

                    # Case 3: LHS is free, RHS has exactly one free var, expr = 1*B+0 → unify
                    if (not solved and a_d_is_free
                            and not _var_in_expr(a_d_final, b_d, set())
                            and len(vars_in_expr) == 1):
                        x_b = vars_in_expr[0].deref()
                        coeffs_b = _get_linear_coeff(b_d, x_b, eng)
                        if coeffs_b is not None:
                            a_coeff_b, b_const_b = coeffs_b
                            if a_coeff_b == 1.0 and abs(b_const_b) < 1e-12:
                                # A = 1*B + 0 = B → unify A and B
                                solved = True
                                _mark_real_sort(a_d_final, wl, eng)
                                _mark_real_sort(x_b, wl, eng)
                                result = _unify(eng, a_d_final, x_b)
                                if not result:
                                    return False
                                return True

                if not solved:
                    # Can't solve now — suspend (re-suspend with tildes).
                    # Re-suspension is correct even for is_resid_refiring cases:
                    # drop only when truly cyclic (a_coeff==1 with no const solution).
                    if not vars_in_expr:
                        # Concrete but unevaluable (e.g. division by zero,
                        # non-numeric atom argument).  Check if any immediate
                        # arg is a concrete non-numeric atom — if so, emit the
                        # standard Wild Life warning.
                        if _report_division_problem(b_d, eng):
                            return False
                        _b_sym_fail = (b_d.type.keyword.symbol
                                       if (b_d.type and b_d.type.keyword) else '')
                        if _b_sym_fail in _ARITH_OPS_SET:
                            _fa1, _fa2 = _get_two_args(b_d)
                            # Evaluate each arg to its concrete form (resolving dot
                            # accesses, feature lookups, etc.) for the display message.
                            def _eval_arg_for_warn(a_ref):
                                if a_ref is None:
                                    return None
                                a_d_w = a_ref.deref()
                                _ev_w = _try_eval_string_func(a_d_w, eng)
                                return _ev_w if _ev_w is not None else a_d_w
                            _fa1_ev = _eval_arg_for_warn(_fa1)
                            _fa2_ev = _eval_arg_for_warn(_fa2)
                            # Build a normalised copy of b_d with evaluated args.
                            _b_norm_w = PsiTerm()
                            _b_norm_w.type = b_d.type
                            _b_norm_w.attr_list = {}
                            if _fa1_ev is not None:
                                _b_norm_w.attr_list['1'] = _fa1_ev
                            if _fa2_ev is not None:
                                _b_norm_w.attr_list['2'] = _fa2_ev
                            if _has_concrete_non_numeric_arg(_b_norm_w, eng):
                                import sys as _sys_w
                                _expr_str_w = _term_to_str(_b_norm_w, eng, quoted=True)
                                print(f"*** Warning: non-numeric argument(s) in "
                                      f"'{_expr_str_w}'.", file=_sys_w.stderr)
                        return False
                    else:
                        # A constraint that can never hold, such as a division
                        # by a divisor already known to be zero, fails now
                        # instead of suspending on its remaining free vars.
                        if _report_division_problem(b_d, eng):
                            return False
                        # Integer division answers an integer, so a side
                        # that is a number with a fraction cannot be what it
                        # comes to: `A = B//C` refuses `A = 24.332` rather
                        # than waiting on B and C for ever.
                        if _get_sym(b_d) == '//':
                            _lhs_num = a_d.deref()
                            if (_lhs_num.value is not None
                                    and not _lhs_num.attr_list):
                                try:
                                    _lv = float(_lhs_num.value)
                                except (TypeError, ValueError):
                                    _lv = None
                                if _lv is not None and _lv != int(_lv):
                                    return False
                        # `A = 0 // A` answers 0: a division whose divisor is
                        # the very term it is equated with has nothing left to
                        # wait for, and no other divisor would make it another
                        # number.  `A = 0 // B` is a different matter and does
                        # wait, since B may yet be a zero.
                        if _get_sym(b_d) == '//':
                            _zd = b_d.attr_list.get('1')
                            _zv = b_d.attr_list.get('2')
                            _zd_ok, _zd_v = (_eval_arith(_zd, eng) if _zd is not None
                                             else (False, 0.0))
                            if (_zd_ok and _zd_v == 0 and _zv is not None
                                    and _zv.deref() is a_d_final.deref()):
                                return _unify(eng, a_d_final,
                                              _make_number(eng, 0.0))
                        # `0.5 = sin(B)` says what B is: a function with one
                        # free argument and a known result is read backwards.
                        _inv = _invert_unary_call(a_d, b_d, eng)
                        if _inv is not None:
                            _inv_arg, _inv_val = _inv
                            return _unify(eng, _inv_arg,
                                          _make_number(eng, _inv_val))
                        # An expression answers a number, so the side it is
                        # equated with has to be able to be one: `a = \\(Z)`
                        # is refused rather than left waiting on Z, because no
                        # Z makes an atom a number.
                        if not _can_be_a_number(a_d, wl):
                            return False
                        from wild_life.data_structures import Goal, Residuation
                        eq_defn = getattr(wl, 'eqsym', None) or wl.syntax_module.symbol_table.get('=')
                        eq_term = PsiTerm(type_def=eq_defn)
                        eq_term.attr_list['1'] = a_d
                        eq_term.attr_list['2'] = b_d
                        eq_term._resid_marker = True
                        pending_goal = Goal(GoalType.PROVE, eq_term, None, None, pending=True)
                        for v in vars_in_expr:
                            _attach_arith_resid(v, wl, pending_goal, eng)
                        if a_d_final.attr_list:
                            # The left side may be an expression of its own,
                            # and what it waits on is what will settle the
                            # equation: `I + 7 = J + 1` is solved for J the
                            # moment I is known, so the goal waits on I too.
                            _lhs_vars: list = []
                            _collect_arith_vars(a_d_final, wl, _lhs_vars, set())
                            for _lv in _lhs_vars:
                                _attach_arith_resid(_lv, wl, pending_goal, eng)
                        if not a_d_is_free:
                            # LHS is bound: also attach to original LHS var so tilde shows.
                            a_orig = a
                            if a_orig.resid is None:
                                if eng is not None:
                                    eng.trail.trail_psi(a_orig, 'resid')
                                a_orig.resid = [Residuation(goal=pending_goal)]
                            else:
                                if not any(r.goal is pending_goal for r in a_orig.resid):
                                    if eng is not None:
                                        eng.trail.trail_copy(a_orig, 'resid')
                                    a_orig.resid.append(Residuation(goal=pending_goal))
                        else:
                            _attach_arith_resid(a_d_final, wl, pending_goal, eng)
                        return True
    # Try string function evaluation on RHS (psi2str, str2psi, strcon, substr, strlen)
    # String functions that should delay when key arguments are unbound:
    _DELAY_STRING_FUNCS = frozenset(('strcon', 'substr', 'strlen', 'str2psi'))
    _b_sym_str = _get_sym(b_d)
    _a_sym_str = _get_sym(a_d)

    def _str_func_delay(func_t, target_t, eng):
        """Try to evaluate a string function; delay on blocking free vars if needed.

        Returns True when successfully evaluated or suspended (delay registered).
        Returns None when this is not a delayable string function situation.
        """
        sym_f = _get_sym(func_t)
        if sym_f not in _DELAY_STRING_FUNCS:
            return None
        # Try to evaluate immediately
        result = _try_eval_string_func(func_t, eng)
        if result is not None:
            return _unify(eng, target_t, result)
        # Evaluation failed — find blocking unbound variables
        blocking = []
        for val in func_t.attr_list.values():
            v = val.deref()
            if _term_is_unbound(v, eng):
                blocking.append(v)
        if not blocking:
            return None  # no unbound vars — fall through to structural unify
        # Register a pending PROVE goal on each blocking variable
        from wild_life.data_structures import Goal, Residuation, SORT_VAR
        wl_sf = eng.wl
        eq_defn = getattr(wl_sf, 'eqsym', None)
        if eq_defn is None and hasattr(wl_sf, 'syntax_module'):
            eq_defn = wl_sf.syntax_module.symbol_table.get('=')
        eq_term = PsiTerm(type_def=eq_defn)
        eq_term.attr_list['1'] = target_t
        eq_term.attr_list['2'] = func_t
        eq_term._resid_marker = True
        pending_goal = Goal(GoalType.PROVE, eq_term, None, None, pending=True)
        for s_var in blocking:
            if s_var.resid is None:
                eng.trail.trail_psi(s_var, 'resid')
                s_var.resid = [Residuation(goal=pending_goal)]
            else:
                if not any(rv.goal is pending_goal for rv in s_var.resid):
                    eng.trail.trail_copy(s_var, 'resid')
                    s_var.resid.append(Residuation(goal=pending_goal))
            if not (s_var.flags & SORT_VAR):
                eng.trail.trail_psi(s_var, 'flags')
                s_var.flags |= SORT_VAR
        return True  # successfully suspended

    # Check RHS first
    if _b_sym_str in _DELAY_STRING_FUNCS and eng is not None:
        _r = _str_func_delay(b_d, a_d, eng)
        if _r is not None:
            return _r
    # Check LHS
    if _a_sym_str in _DELAY_STRING_FUNCS and eng is not None:
        _r = _str_func_delay(a_d, b_d, eng)
        if _r is not None:
            return _r

    # A call written inside a term stands for what it answers, however deep
    # it sits: `X = pair(foo_b(Y), s_b(B))` hands X the term foo_b builds,
    # not the call.  Only a term written into this goal is read that way —
    # a variable's value was built once already, and reading it again would
    # ask `random(1000)` for a second number — and only calls are reduced:
    # the arithmetic of `s(a(X),b(X:(1+2)))` is the term's, not this goal's.
    from wild_life.data_structures import NON_STRICT_TERM as _NST_eq
    for _eq_side, _eq_raw in ((b_d, b), (a_d, a)):
        if (_eq_side is _eq_raw and _eq_side.attr_list
                and _eq_side.type is not None
                and _eq_side.type._builtin_func is None
                and not (_eq_side.flags & _NST_eq)
                and not _is_user_function(_eq_side)):
            _reduce_embedded_calls(_eq_side, eng, 0, set())

    # A sort comparison standing where a value belongs answers true or false:
    # `A = (1 :=< 1.1)` is false, not the comparison written out again.
    for _sc_side, _sc_other in ((b_d, a_d), (a_d, b_d)):
        _sc_val = _eval_sort_comparison(_sc_side, eng)
        if _sc_val is not None:
            return _unify(eng, _sc_other, _sc_val)

    # Non-delaying string functions (psi2str, root_sort, children, chr evaluated already above)
    # Both sides, not just one: `features(A) = features(B)` compares the two
    # feature lists, which is how structures.lf asks whether two terms carry
    # the same features.
    # `map(F,L)` and `reduce(F,E,L)` standing where a value belongs are
    # asked for one: magic compares the row sums `map(sum_up,Square)` makes,
    # not the call that would make them.
    _b_mr = _eval_map_or_reduce(b_d, eng)
    if _b_mr is not None:
        b_d = _b_mr.deref()
    _a_mr = _eval_map_or_reduce(a_d, eng)
    if _a_mr is not None:
        a_d = _a_mr.deref()

    b_str = _try_eval_string_func(b_d, eng)
    a_str = _try_eval_string_func(a_d, eng)
    if b_str is not None:
        b_d = b_str
    if a_str is not None:
        a_d = a_str

    return _unify(eng, a_d, b_d)


def bi_not_unify(goal: PsiTerm, eng) -> bool:
    """X \\= Y — non-unifiable."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    mark = eng.trail.mark()
    ok = eng.unifier.unify(a, b)
    eng.trail.undo_to(mark)
    return not ok


def _term_is_unbound(t: Optional[PsiTerm], eng) -> bool:
    """Return True if t dereferences to an unbound (top/free) variable."""
    if t is None:
        return True
    t = t.deref()
    wl = eng.wl
    return (t.value is None and not t.attr_list
            and (t.type is None or t.type is wl.top))


def _sort_compare_args(goal, eng):
    """The two sorts a sort-comparison predicate compares, or None.

    An argument that is a call is reduced first: `features(X) :== []` asks
    about the sort of the feature list, not of the call.
    """
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return None
    sorts = []
    for _arg in (a, b):
        _d = _strip_backtick(_arg.deref())
        _ev = _try_eval_any_func(_d, eng)
        if _ev is None:
            _ev = _try_eval_string_func(_d, eng)
        if _ev is not None:
            _d = _strip_backtick(_ev.deref())
        if _d.type is None:
            return None
        # A number or string is a sort of its own, so 3 and 4 are no more the
        # same sort than a and b are, though both are integers.  What a
        # number is is read from the number: `1.0` is the integer 1, and is
        # the same sort as `1` and under int, which is what isatest asks.
        _ty = _d.type
        if (_d.value is not None and not _d.attr_list
                and isinstance(_d.value, (int, float))
                and eng.wl.real is not None
                and _ty is not None and _ty.is_subtype_of(eng.wl.real)):
            _ty = (eng.wl.integer if float(_d.value).is_integer()
                   else eng.wl.real)
        sorts.append((_ty, _d.value))
    return sorts[0], sorts[1]


def _strip_backtick(t: PsiTerm) -> PsiTerm:
    """What a backtick holds, or the term itself."""
    while (t is not None and t.type is not None and t.type.keyword is not None
           and t.type.keyword.symbol == '`' and '1' in t.attr_list):
        t = t.attr_list['1'].deref()
    return t


def _sort_key_under(lower, upper) -> bool:
    """Whether the first sort key is the second or lies under it.

    Every sort lies under `@`, whether or not the program ever said so:
    accumulators.lf asks `AccPred :< @` of a term to mean that there is one.
    """
    (ld, lv), (ud, uv) = lower, upper
    if uv is not None:
        return ld is ud and lv == uv
    from wild_life.runtime import WL as _WL_sku
    if ud is _WL_sku.top:
        return True
    return ld.is_subtype_of(ud)


def bi_sort_eq(goal: PsiTerm, eng) -> bool:
    """X :== Y — X and Y have the same sort."""
    sorts = _sort_compare_args(goal, eng)
    return sorts is not None and sorts[0] == sorts[1]


def bi_sort_ne(goal: PsiTerm, eng) -> bool:
    """X :\\== Y — X and Y have different sorts."""
    sorts = _sort_compare_args(goal, eng)
    return sorts is not None and sorts[0] != sorts[1]


def bi_sort_le(goal: PsiTerm, eng) -> bool:
    """X :=< Y — X's sort is Y's sort or lies under it."""
    sorts = _sort_compare_args(goal, eng)
    return sorts is not None and _sort_key_under(sorts[0], sorts[1])


def bi_sort_lt(goal: PsiTerm, eng) -> bool:
    """X :< Y — X's sort lies strictly under Y's."""
    sorts = _sort_compare_args(goal, eng)
    return (sorts is not None and sorts[0] != sorts[1]
            and _sort_key_under(sorts[0], sorts[1]))


def bi_sort_ge(goal: PsiTerm, eng) -> bool:
    """X :>= Y — X's sort is Y's sort or lies above it."""
    sorts = _sort_compare_args(goal, eng)
    return sorts is not None and _sort_key_under(sorts[1], sorts[0])


def bi_sort_gt(goal: PsiTerm, eng) -> bool:
    """X :> Y — X's sort lies strictly above Y's."""
    sorts = _sort_compare_args(goal, eng)
    return (sorts is not None and sorts[0] != sorts[1]
            and _sort_key_under(sorts[1], sorts[0]))


def bi_sort_not_lt(goal: PsiTerm, eng) -> bool:
    """X :\\< Y — X's sort does not lie strictly under Y's."""
    return not bi_sort_lt(goal, eng)


def bi_sort_not_le(goal: PsiTerm, eng) -> bool:
    """X :\\=< Y — X's sort neither is Y's nor lies under it."""
    return not bi_sort_le(goal, eng)


def bi_sort_not_gt(goal: PsiTerm, eng) -> bool:
    """X :\\> Y — X's sort does not lie strictly above Y's."""
    return not bi_sort_gt(goal, eng)


def bi_sort_not_ge(goal: PsiTerm, eng) -> bool:
    """X :\\>= Y — X's sort neither is Y's nor lies above it."""
    return not bi_sort_ge(goal, eng)


def bi_sort_comparable(goal: PsiTerm, eng) -> bool:
    """X :>< Y — the two sorts lie on one chain, either way round."""
    sorts = _sort_compare_args(goal, eng)
    return (sorts is not None
            and (sorts[0].is_subtype_of(sorts[1])
                 or sorts[1].is_subtype_of(sorts[0])))


def bi_sort_incomparable(goal: PsiTerm, eng) -> bool:
    """X :\\>< Y — neither sort lies under the other."""
    sorts = _sort_compare_args(goal, eng)
    return (sorts is not None
            and not sorts[0].is_subtype_of(sorts[1])
            and not sorts[1].is_subtype_of(sorts[0]))


# A sort comparison answers true or false, so it stands where a value is
# wanted as well as where a goal is: isatest writes `1 :=< 1.1` and expects
# to read false.
_SORT_CMP_FUNCS = {
    ':==': bi_sort_eq, ':\\==': bi_sort_ne,
    ':=<': bi_sort_le, ':<': bi_sort_lt,
    ':>=': bi_sort_ge, ':>': bi_sort_gt,
    ':\\=<': bi_sort_not_le, ':\\<': bi_sort_not_lt,
    ':\\>=': bi_sort_not_ge, ':\\>': bi_sort_not_gt,
    ':><': bi_sort_comparable, ':\\><': bi_sort_incomparable,
}


def _eval_sort_comparison(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """What a sort comparison comes to, where a value rather than a goal is
    wanted: isatest writes `1 :=< 1.1` and expects to read false.

    Asked only where the answer is to be written or assigned, not of every
    term walked past: `cond(T :== xfx, …)` is a question a goal asks, and
    settling it early would answer it before T is what it will be.
    """
    if t is None:
        return None
    t = t.deref()
    sym = t.type.keyword.symbol if (t.type and t.type.keyword) else ''
    fn = _SORT_CMP_FUNCS.get(sym)
    if fn is None or '1' not in t.attr_list or '2' not in t.attr_list:
        return None
    if _sort_compare_args(t, eng) is None:
        return None
    return _make_atom(eng, 'true' if fn(t, eng) else 'false')


def bi_identical(goal: PsiTerm, eng) -> bool:
    """X == Y — structural identity.

    In Wild Life, == requires both sides to be ground (no free variables).
    If either side is unbound, the predicate fails — you cannot assert
    identity of unknowns in a constraint-logic setting.
    """
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    if _term_is_unbound(a, eng) or _term_is_unbound(b, eng):
        return False
    s1 = _term_to_str(a, eng)
    s2 = _term_to_str(b, eng)
    return s1 == s2


def bi_not_identical(goal: PsiTerm, eng) -> bool:
    """X \\== Y — structural non-identity.

    In Wild Life, \\== requires both sides to be ground (no free variables).
    If either side is unbound, the predicate fails — you cannot assert
    definitive non-identity of unknowns in a constraint-logic setting.
    For example, if Flag is unbound, Flag:\\==error fails because Flag
    could potentially be unified with error.
    """
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    if _term_is_unbound(a, eng) or _term_is_unbound(b, eng):
        return False
    s1 = _term_to_str(a, eng)
    s2 = _term_to_str(b, eng)
    return s1 != s2


def c_same_address(goal: PsiTerm, eng) -> bool:
    """X === Y — address (identity) equality.

    Succeeds if X and Y are the same psi-term node after dereferencing
    (pointer equality).  Fails otherwise.  Used by deja_vu/copy in LIFE.
    """
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    return a is b


def c_diff_address(goal: PsiTerm, eng) -> bool:
    r"""X \=== Y — address (identity) inequality.

    Succeeds if X and Y are different psi-term nodes after dereferencing.
    """
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return False
    return a is not b


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
        if t.type is wl.nil:
            return True
        if t.type is not wl.alist:
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


def bi_repeat(goal: PsiTerm, eng) -> bool:
    """repeat — always succeeds, creates an infinite choice point on backtrack.

    Equivalent to the Prolog definition:
        repeat.
        repeat :- repeat.
    """
    # Push a choice point that re-enters repeat on backtracking.
    # goal is the repeat term itself; proving it again creates another choice
    # point, giving infinite backtracking.
    eng.push_choice_point(GoalType.PROVE, goal, _DEFRULES_SENTINEL, None)
    return True


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
    # Use _INNER_RUN_BARRIER so run() does not undo trail to position 0 on failure;
    # that would destroy outer bindings (e.g. N=1 set before the not() call).
    _barrier = cp_save if cp_save is not None else _INNER_RUN_BARRIER
    result = eng.run(cs_barrier=_barrier)
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


def _subsumes(pattern: PsiTerm, term: PsiTerm, eng, depth: int = 0) -> bool:
    """True when pattern is at least as general as term.

    The call's sort must be a sub-sort of the pattern's, its value must be the
    one the pattern asks for if it asks for one, and every feature the pattern
    names must be present and covered in turn.
    """
    if depth > 20:
        return True
    pattern = pattern.deref()
    term = term.deref()
    if pattern is term:
        return True
    if pattern.type is not None and pattern.type is not eng.wl.top:
        if term.type is None or not term.type.is_subtype_of(pattern.type):
            return False
    if pattern.value is not None and pattern.value != term.value:
        return False
    for key, sub in pattern.attr_list.items():
        other = term.attr_list.get(key)
        if other is None or not _subsumes(sub, other, eng, depth + 1):
            return False
    return True


def bi_implies(goal: PsiTerm, eng) -> bool:
    """implies(Goal) — prove Goal by matching instead of unification.

    A clause applies only where its head covers the call: the head may be more
    general than the call, never the other way round.  So `implies(a(int))`
    skips `a(X:0)`, whose head asks for something narrower than int, and runs
    `a(X:int)` and `a(X:real)`, which cover it.
    """
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    call = arg.deref()
    rules = getattr(call.type, 'rule', None) if call.type is not None else None
    if not rules:
        return bi_call(goal, eng)   # not a user predicate — an ordinary call
    covering = [(h, b) for (h, b) in rules
                if h is not None and _subsumes(h, call, eng)]
    if not covering:
        return False
    eng.push_goal(GoalType.PROVE, call, covering, None)
    return True


def bi_and(goal: PsiTerm, eng) -> bool:
    """and(A, B) — Boolean conjunction as a goal: succeed iff both A and B hold.

    Wild Life uses 'and' as both a boolean function sort and as a conjunction
    predicate.  When proved as a goal, and(A,B) tries to prove both A and B
    (like Prolog's ','(A,B)).  It first tries to evaluate the boolean value of
    the expression; if the result is a definite true/false atom it acts on that;
    otherwise it falls back to proving A as a goal and then B.
    """
    arg1, arg2 = _get_two_args(goal)
    if arg1 is None or arg2 is None:
        return True  # degenerate: succeed
    arg1d = arg1.deref()
    arg2d = arg2.deref()
    # Try evaluating as booleans first
    b_result = _try_eval_bool(goal, eng)
    if b_result is not None:
        sym = _get_sym(b_result)
        return sym == 'true'
    # Fall back: prove both as goals
    from wild_life.unification import GoalType as _GoalType
    _GoalType = GoalType  # use the imported GoalType
    # Push in reverse order (goal_stack is a stack, so last pushed = first proved)
    eng.push_goal(GoalType.PROVE, arg2d, _DEFRULES_SENTINEL, None)
    eng.push_goal(GoalType.PROVE, arg1d, _DEFRULES_SENTINEL, None)
    return True


def bi_or(goal: PsiTerm, eng) -> bool:
    """or(A, B) — Boolean disjunction as a goal: succeed iff A or B holds.

    Tries to evaluate as a boolean first; if definite, acts on it.
    Otherwise creates a choice point: try A, or on failure try B.
    """
    arg1, arg2 = _get_two_args(goal)
    if arg1 is None or arg2 is None:
        return True
    arg1d = arg1.deref()
    arg2d = arg2.deref()
    # Try evaluating as booleans first
    b_result = _try_eval_bool(goal, eng)
    if b_result is not None:
        sym = _get_sym(b_result)
        return sym == 'true'
    # Fall back: choice between arg1 and arg2 (simplified: try arg1; if fails, try arg2)
    mark = eng.trail.mark()
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack
    # The inner run is this one's own loop re-entered, and it leaves
    # main_loop_ok saying whether *it* ran out of goals.  Handing that back to
    # the loop that called us would end it there and drop everything after the
    # `or`: `(A =< 57 or A =:= 111), D = A` would answer yes without ever
    # proving `D = A`.
    ok_save = eng.main_loop_ok
    count_save = eng.goal_count
    eng.push_goal(GoalType.PROVE, arg1d, _DEFRULES_SENTINEL, None)
    _barrier = cp_save if cp_save is not None else _INNER_RUN_BARRIER
    try:
        result1 = eng.run(cs_barrier=_barrier)
    finally:
        eng.main_loop_ok = ok_save
        eng.goal_count = count_save
    if result1:
        eng.choice_stack = cp_save
        return True
    eng.trail.undo_to(mark)
    eng.choice_stack = cp_save
    eng.goal_stack = gs_save
    eng.push_goal(GoalType.PROVE, arg2d, _DEFRULES_SENTINEL, None)
    return True


def bi_once(goal: PsiTerm, eng) -> bool:
    """once(P) — call P exactly once."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    mark = eng.trail.mark()
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack
    eng.push_goal(GoalType.PROVE, arg, _DEFRULES_SENTINEL, None)
    _barrier = cp_save if cp_save is not None else _INNER_RUN_BARRIER
    result = eng.run(cs_barrier=_barrier)
    if not result:
        eng.trail.undo_to(mark)
        eng.choice_stack = cp_save
        return result
    # What the goal cut reaches past this call: `call_once(B)` with B the
    # cut atom takes the query's own alternatives with it.  Only the
    # alternatives the goal itself left behind are the ones dropped here.
    _cp_seen = eng.choice_stack
    while _cp_seen is not None and _cp_seen is not cp_save:
        _cp_seen = _cp_seen.next
    if cp_save is None or _cp_seen is cp_save:
        eng.choice_stack = cp_save
    return result


def bi_call_once(goal: PsiTerm, eng) -> bool:
    """call_once(P) — prove P once, waiting while P is still unknown.

    `call_once(X)` with X free has nothing to prove yet, so it suspends on X
    and runs once X says what it is.
    """
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    if _is_var(arg, eng):
        from wild_life.data_structures import (Goal as _G_co, Residuation as _R_co,
                                               SORT_VAR as _SV_co)
        _pending = _G_co(GoalType.PROVE, goal, _DEFRULES_SENTINEL, None,
                         next=None, pending=True)
        eng.trail.trail_psi(arg, 'resid')
        arg.resid = list(arg.resid or []) + [_R_co(goal=_pending)]
        if not (arg.flags & _SV_co):
            eng.trail.trail_psi(arg, 'flags')
            arg.flags |= _SV_co
        return True
    return bi_once(goal, eng)


def _all_builtin_goals(t: 'PsiTerm', eng, _depth: int = 0) -> bool:
    """Whether t is a goal that can be proven rather than worked out.

    cond/2 reads its branch as a function where it can, but `(write(X),nl)`
    has no value to read — it is the printing cond/2 is there to do — and
    neither has eratosthenes's `(sieve.M <- multiple_of(P),
    remove_multiples(P,M+P))`, which is the sieving.  A branch made of goals,
    whether built in or the program's own, is proven.
    """
    if t is None or _depth > 20:
        return False
    t = t.deref()
    defn = t.type
    if defn is None:
        return False
    if defn is eng.wl.commasym or defn is eng.wl.life_or:
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        return (a1 is not None and a2 is not None
                and _all_builtin_goals(a1, eng, _depth + 1)
                and _all_builtin_goals(a2, eng, _depth + 1))
    return (defn._builtin_func is not None
            or defn.type == DefType.PREDICATE)


def _eval_as_bool_func(t: 'PsiTerm', eng, _depth: int = 0) -> 'Optional[bool]':
    """Evaluate *t* as a boolean function expression.

    Returns True, False, or None (cannot reduce to a boolean).
    This is the Wild Life functional-evaluation mode: only FUNCTION definitions
    (defined with '->') and built-in comparisons are evaluated.  PREDICATE
    definitions (defined with ':-') are *not* callable in this mode and yield
    None (unresolvable).

    Used by the 2-argument form of cond/2.
    """
    if t is None or _depth > 20:
        return None
    t = t.deref()
    defn = t.type
    if defn is None:
        return None

    sym = defn.keyword.symbol if defn.keyword else ''

    # ── Literal boolean atoms ──
    if sym == 'true':
        return True
    if sym in ('false', 'fail'):
        return False

    wl = eng.wl

    # ── Conjunction (, or 'and') ──
    if defn is wl.commasym or sym == 'and':
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        r1 = _eval_as_bool_func(a1.deref() if a1 else None, eng, _depth + 1)
        if r1 is None:
            return None   # can't determine → propagate failure
        if r1 is False:
            return False
        # r1 is True
        r2 = _eval_as_bool_func(a2.deref() if a2 else None, eng, _depth + 1)
        return r2

    # ── Negation (not) ──
    if sym == 'not':
        a1 = t.attr_list.get('1') if t.attr_list else None
        r1 = _eval_as_bool_func(a1.deref() if a1 else None, eng, _depth + 1)
        return None if r1 is None else (not r1)

    # ── Disjunction (; or 'or') ──
    if defn is wl.life_or or defn is wl.disjunction or sym == 'or':
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        r1 = _eval_as_bool_func(a1.deref() if a1 else None, eng, _depth + 1)
        if r1 is True:
            return True
        r2 = _eval_as_bool_func(a2.deref() if a2 else None, eng, _depth + 1)
        if r2 is True:
            return True
        if r1 is False and r2 is False:
            return False
        return None

    # ── Arithmetic comparisons ──
    # A comparison reads as a boolean wherever one is wanted, whether it is
    # registered as a function or as a predicate: `X > 0.698 and X < 0.702`
    # is the condition cond/2 is given.
    if sym in ('>', '<', '>=', '=<', '=:=', '=\\='):
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        if a1 and a2:
            ok1, v1 = _eval_arith(a1, eng)
            ok2, v2 = _eval_arith(a2, eng)
            if ok1 and ok2:
                cmp_map: dict = {
                    '>': v1 > v2, '<': v1 < v2,
                    '>=': v1 >= v2, '=<': v1 <= v2,
                    '=:=': v1 == v2, '=\\=': v1 != v2,
                }
                return cmp_map.get(sym)
        return None

    # ── Sort comparisons ──
    # `Level :\== fail` reads as a boolean wherever one is wanted, the same
    # as an arithmetic comparison: it is the condition that
    # `cond(Level :\== fail, call_pred(Pred,Verbose))` asks.
    if sym in _SORT_COMPARISONS:
        if defn._builtin_func is None or _cond_is_undecided(t, eng):
            return None
        _m_sc = eng.trail.mark()
        try:
            _r_sc = bool(defn._builtin_func(t, eng))
        except Exception:
            _r_sc = None
        eng.trail.undo_to(_m_sc)
        return _r_sc

    # ── A feature read for its value ──
    # `project(1,Bool)` is the condition structures.lf asks it as, because
    # the feature it reads holds true or false.
    if sym == 'project':
        _m_pj = eng.trail.mark()
        _s_pj = ''
        try:
            _r_pj = _try_eval_any_func(t, eng)
            if _r_pj is not None:
                _s_pj = _get_sym(_r_pj.deref())
        except Exception:
            pass
        eng.trail.undo_to(_m_pj)
        if _s_pj == 'true':
            return True
        if _s_pj in ('false', 'fail'):
            return False
        return None

    # ── Built-in FUNCTION whose value is a boolean ──
    # `has_feature(visited,B)` answers true or false, so it reads as the
    # condition structures.lf writes it as.
    if sym in _BOOL_VALUED_BUILTINS:
        _m_bv = eng.trail.mark()
        try:
            _r_bv = _try_eval_string_func(t, eng)
        except Exception:
            _r_bv = None
        eng.trail.undo_to(_m_bv)
        if _r_bv is not None:
            _s_bv = _get_sym(_r_bv.deref())
            if _s_bv == 'true':
                return True
            if _s_bv in ('false', 'fail'):
                return False
        return None

    # ── Built-in FUNCTION ──
    if defn._builtin_func is not None and defn.type == DefType.FUNCTION:
        # Other built-in functions cannot be evaluated without engine machinery
        return None

    # ── User-defined FUNCTION (defined with '->') ──
    if defn.type == DefType.FUNCTION and defn.rule:
        from wild_life.unification import copy_term as _copy_term
        active = [(h, b) for (h, b) in defn.rule if h is not None and b is not None]
        for (h0, b0) in active:
            _vm: dict = {}
            head = _copy_term(h0, _vm)
            body = _copy_term(b0, _vm)
            body_d = body.deref()
            # Skip guarded rules (value | condition) — need engine machinery
            if body_d.type is not None and body_d.type is wl.such_that:
                continue
            mark = eng.trail.mark()
            ok = eng.unifier.unify(t, head)
            if ok:
                result = _eval_as_bool_func(body_d, eng, _depth + 1)
                eng.trail.undo_to(mark)
                if result is not None:
                    return result
            else:
                eng.trail.undo_to(mark)
        return None

    # ── PREDICATE — cannot evaluate functionally ──
    if defn.type == DefType.PREDICATE:
        return None

    return None


def _settle_conds_in_branch(branch, eng, _depth: int = 0) -> None:
    """Settle a cond written inside the branch another cond just chose.

    c_cond hands the chosen branch back as the cond's value and checks it out
    on the spot (built_ins.c, c_cond: push_goal(unify,result,arg2) followed by
    i_check_out(arg2)), so a cond standing inside that branch is settled before
    the branch is ever proved -- on what its condition says at that moment,
    not on what it would say once the branch has run.  arnaud_bug turns on
    this: its inner test reads a feature the branch has yet to fill in.
    """
    if branch is None or eng is None or _depth > 16:
        return
    b = branch.deref()
    if b.type is None or b.type.keyword is None or not b.attr_list:
        return
    _sym_cb = b.type.keyword.symbol
    if _sym_cb in (',', 'and'):
        for _k_cb in ('1', '2'):
            _a_cb = b.attr_list.get(_k_cb)
            if _a_cb is not None:
                _settle_conds_in_branch(_a_cb, eng, _depth + 1)
        return
    if _sym_cb != 'cond' or b.coref is not None:
        return
    _c_cb, _t_cb, _e_cb = _cond_args(b)
    if _c_cb is None or (_t_cb is None and _e_cb is None):
        return
    from wild_life.inference import prove_cond as _pc_cb
    _mark_cb = eng.trail.mark()
    _ok_cb = _pc_cb(_c_cb, eng)
    if not _ok_cb:
        eng.trail.undo_to(_mark_cb)
    _chosen_cb = _t_cb if _ok_cb else _e_cb
    if _chosen_cb is None:
        # A branch the call leaves out is a goal nothing constrains.
        _chosen_cb = PsiTerm(type_def=eng.wl.succeed)
    eng.trail.trail_psi(b, 'coref')
    b.coref = _chosen_cb.deref()
    _settle_conds_in_branch(_chosen_cb, eng, _depth + 1)


def bi_cond(goal: PsiTerm, eng) -> bool:
    """cond(Cond, Then[, Else]) — Wild Life conditional.

    2-argument form  cond(Cond, Then):
        Evaluate Cond and Then as *boolean functions* (Wild Life functional
        semantics).  PREDICATE definitions cannot be called in this mode.
        - If Cond evaluates to true and Then evaluates to true → succeed.
        - If Cond evaluates to false or unknown → succeed silently.
        - If Cond is true but Then evaluates to false/unknown → FAIL.

    3-argument form  cond(Cond, Then, Else):
        Prove Cond as a predicate goal (with inner run, cutting alternatives).
        - If Cond succeeds → push Then as predicate goal.
        - If Cond fails    → undo Cond's bindings and push Else as predicate goal.
    """
    cond_g, then_g, else_g = _cond_args(goal)
    if cond_g is None or (then_g is None and else_g is None):
        return True   # degenerate: succeed

    if else_g is None:
        # ── 2-arg form: fully functional evaluation ──
        cond_result = _eval_as_bool_func(cond_g, eng)
        if cond_result is not True:
            # Cond is false or unresolvable → succeed silently (no Then)
            return True
        # Cond is true → evaluate Then as a boolean function
        then_result = _eval_as_bool_func(then_g, eng)
        if then_result is not None:
            return then_result
        # A plain predicate call has no boolean value to read; it is proved,
        # which is how `cond(true, foo(X))` binds X at all.  Anything else
        # unresolvable stays a failure, as a conjunction of goals would be.
        _then_defn = then_g.type
        _provable = (
            _then_defn is not None
            and (_then_defn.type == DefType.PREDICATE
                 or _all_builtin_goals(then_g, eng)))
        if _provable:
            eng.push_goal(GoalType.PROVE, then_g, _DEFRULES_SENTINEL, None)
            return True
        return False

    # ── 3-arg form: predicate if-then-else ──
    from wild_life.inference import prove_cond as _prove_cond
    mark = eng.trail.mark()
    cond_ok = _prove_cond(cond_g, eng)

    if cond_ok:
        # Cond succeeded → push Then.  A branch the call leaves out is a goal
        # nothing constrains, and holds.
        if then_g is not None:
            _settle_conds_in_branch(then_g, eng)
            eng.push_goal(GoalType.PROVE, then_g, _DEFRULES_SENTINEL, None)
        return True
    else:
        # Cond failed → undo its bindings, push Else
        eng.trail.undo_to(mark)
        _settle_conds_in_branch(else_g, eng)
        eng.push_goal(GoalType.PROVE, else_g, _DEFRULES_SENTINEL, None)
        return True


def _collect_solutions(template: PsiTerm, g: PsiTerm, eng) -> list:
    """Collect all solutions of goal g, returning copies of template.

    Helper shared by findall/bagof/setof.
    Saves/restores trail, choice_stack, goal_stack.

    Template and goal are copied together (shared var_map) so that variables
    in the template are bound when the goal is solved.
    """
    wl = eng.wl
    mark = eng.trail.mark()
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack

    # Copy template and goal with shared variable mapping so they share variables.
    shared_map: dict = {}
    template_copy = copy_term(template, shared_map)
    goal_copy = copy_term(g, shared_map)
    # IMPORTANT: clear goal_stack so the inner runs only prove goal_copy;
    # the outer continuation must NOT run inside the inner loop.
    eng.goal_stack = None
    eng.push_goal(GoalType.PROVE, goal_copy, _DEFRULES_SENTINEL, None)

    collected = []
    while True:
        result = eng.run()
        if result:
            # Copy template_copy with current bindings resolved.  A template
            # written as a call through a functor variable — `bagof(F(5),q(F))`
            # parses as apply(5,functor => F) — is evaluated once F is known,
            # because built_ins.lf collects `evalin(A)` rather than A.
            _elem = template_copy.deref()
            _mark_ev = eng.trail.mark()
            # A template that names a feature of what the goal bound — the
            # `X.functor` of `bagof(X.functor,X:op)` — is read out the same
            # way, once the solution has given X something to read it from.
            _ev_wanted = (
                (getattr(wl, 'apply', None) is not None
                 and _elem.type is wl.apply)
                or (_elem.type is not None and _elem.type.keyword is not None
                    and _elem.type.keyword.symbol == '.'))
            if _ev_wanted:
                _gs_ev, _cs_ev = eng.goal_stack, eng.choice_stack
                _ok_ev = eng.main_loop_ok
                try:
                    _eq_defn = (getattr(wl, 'eqsym', None)
                                or wl.syntax_module.symbol_table.get('='))
                    _fresh_ev = PsiTerm(type_def=wl.top)
                    _eq_ev = PsiTerm(type_def=_eq_defn)
                    _eq_ev.attr_list = {'1': _fresh_ev, '2': _elem}
                    eng.goal_stack = None
                    if bi_unify(_eq_ev, eng):
                        # The unification may leave goals (the EVAL of the
                        # reconstructed call) to run before the value is there.
                        if eng.goal_stack is not None:
                            from wild_life.inference import (
                                _INNER_RUN_BARRIER as _IRB_ev)
                            eng.run(cs_barrier=_cs_ev if _cs_ev is not None
                                    else _IRB_ev)
                        _elem = _fresh_ev.deref()
                except Exception:
                    pass
                finally:
                    eng.goal_stack, eng.choice_stack = _gs_ev, _cs_ev
                    eng.main_loop_ok = _ok_ev
            collected.append(copy_term(_elem))
            eng.trail.undo_to(_mark_ev)
            if not eng.choice_stack or eng.choice_stack is cp_save:
                break
            eng.backtrack()
        else:
            break

    eng.trail.undo_to(mark)
    eng.choice_stack = cp_save
    eng.goal_stack = gs_save
    # C Wild Life collects solutions via LIFO (last-in-first-out) internally,
    # yielding results in reverse exploration order.  Reverse here to match.
    collected.reverse()
    return collected


def bi_findall(goal: PsiTerm, eng) -> bool:
    """findall(Template, Goal, Bag) — collect all solutions.

    Supports both:
      findall(Template, Goal, Bag)   — 3-arg predicate form
      _A = findall(Template, Goal)   — 2-arg functional form (handled via bi_unify)
    """
    wl = eng.wl
    # Try 3-arg form first
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if a1 and a2 and a3:
        template = a1.deref()
        g = a2.deref()
        bag_out = a3.deref()
        collected = _collect_solutions(template, g, eng)
        result_list = wl.make_list(collected)
        return _unify(eng, bag_out, result_list)

    # 2-arg functional form: result must be provided separately
    # This path is normally reached via bi_unify's bagof-as-function handling.
    # Called as predicate with 2 args → fail (use = form instead)
    return False


def _normalize_clause_for_assert(arg: PsiTerm, eng) -> PsiTerm:
    """Evaluate the arithmetic a clause carries, leaving its head alone.

    assert(mynum(N+1)) with N=31 stores mynum(32) rather than the expression
    tree.  The head of a rule is a pattern, though, not something to work out:
    reducing `f1 -> 14` head-first would ask f1 for its current value and file
    the clause under that number instead of under f1, losing the clause.
    """
    arg = arg.deref()
    # A clause built under a backquote is the clause: the quote is what kept
    # it from being worked out while it was being put together, and
    # std_expander hands each clause it generates over as
    # `` `(NewHead :- Code) ``.
    while True:
        _sym_bq = arg.type.keyword.symbol if (arg.type and arg.type.keyword) else ''
        if _sym_bq != '`' or '1' not in arg.attr_list or len(arg.attr_list) != 1:
            break
        arg = arg.attr_list['1'].deref()
    sym = arg.type.keyword.symbol if (arg.type and arg.type.keyword) else ''
    if sym in (':-', '->') and '1' in arg.attr_list and '2' in arg.attr_list:
        # A rule is filed as written.  The expression in its body is part of
        # the clause rather than a sum to work out, and freezing it keeps a
        # later strict call from working it out on the clause's behalf: after
        # `assert(f2 -> X)` the X of `X:(1+2)` reads as 1 + 2 everywhere.
        from wild_life.inference import _mark_arith_non_strict as _mans_asrt
        _mans_asrt(arg, None, eng)
        from wild_life.inference import (
            _thaw_non_strict_freeze as _thaw_asrt)
        _thaw_asrt(arg)
        return arg
    return _normalize_arith_in_term(arg, eng)


def bi_assert(goal: PsiTerm, eng) -> bool:
    """assert(Clause) / assertz(Clause)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    arg = _normalize_clause_for_assert(arg, eng)
    eng.assert_first = False
    eng.assert_clause(arg)
    return True


def bi_asserta(goal: PsiTerm, eng) -> bool:
    """asserta(Clause) — add at front."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    arg = _normalize_clause_for_assert(arg, eng)
    eng.assert_first = True
    eng.assert_clause(arg)
    eng.assert_first = False
    return True


# What each one-argument function undoes, where undoing it says one thing.
_UNARY_INVERSES = {
    'sin': math.asin, 'cos': math.acos, 'tan': math.atan,
    'asin': math.sin, 'acos': math.cos, 'atan': math.tan,
    'exp': math.log, 'log': math.exp,
    'sqrt': lambda v: v * v,
}


def _invert_unary_call(known: PsiTerm, call: PsiTerm, eng):
    """Read `Value = f(X)` backwards, as (X, the value X must have).

    Returns None where the call is not one function of one free argument, or
    where undoing it would say nothing definite.
    """
    sym = _get_sym(call)
    inverse = _UNARY_INVERSES.get(sym)
    if inverse is None:
        return None
    if len(call.attr_list) != 1:
        return None
    arg = call.attr_list.get('1')
    if arg is None:
        return None
    arg = arg.deref()
    if arg.value is not None or arg.attr_list:
        return None
    ok, value = _eval_arith(known, eng)
    if not ok:
        return None
    try:
        return arg, float(inverse(value))
    except (ValueError, OverflowError):
        return None


# Two-argument operators that stand for a function until both arguments are
# there: `and(B)` is waiting for its second, not a term with room for one.
_CURRIABLE_BINARY_OPS = (frozenset(('and', 'or', 'xor', '==='))
                         | _ARITH_OPS_SET)


def report_static_definition(defn) -> None:
    """Say that a closed definition was asked to change."""
    kw = getattr(defn, 'keyword', None)
    if kw is None:
        return
    module = getattr(kw, 'module', None)
    name = (f"{module.module_name}#{kw.symbol}"
            if module is not None and module.module_name else kw.symbol)
    sys.stderr.write(f"*** Error: the predicate '{name}' may not be changed.\n")


def bi_retract(goal: PsiTerm, eng) -> bool:
    """retract(Clause) — remove first matching clause (non-deterministic).

    Handles both :- and -> clause forms:
      retract((head :- body))   for predicate rules
      retract((head -> value))  for functional rules
      retract(head)             for facts / any rule
    """
    from wild_life.data_structures import GoalType as _GT
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    arg = arg.deref()
    wl = eng.wl
    sym = arg.type.keyword.symbol if arg.type and arg.type.keyword else ''
    if sym in (':-', '->'):
        # Clause or functional rule: (head :- body) or (head -> value)
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
    if getattr(defn, 'is_static', False):
        # A closed definition gives nothing up.
        report_static_definition(defn)
        return False
    # Build a body term if none given (unifies with 'true' / any body)
    if body is None:
        body = PsiTerm(type_def=wl.top)  # fresh var — will match any body
    # Use the engine's non-deterministic clause_aim machinery:
    # Push a DEL_CLAUSE goal with (master_list, start_idx=0) so clause_aim
    # always deletes from the master list at the correct position.
    rule_list = defn.rule  # live mutable list
    eng.push_goal(_GT.DEL_CLAUSE, head, body, (rule_list, 0))
    return True


def _mark_persistent_deep(t, seen: set) -> None:
    """Note every node of a term that a persistent write puts away."""
    if t is None:
        return
    t = t.deref()
    if id(t) in seen:
        return
    seen.add(id(t))
    t._wl_persistent_cell = True
    t._wl_persistent_written = True
    for _sub in t.attr_list.values():
        _mark_persistent_deep(_sub, seen)


def bi_store_arrow(goal: PsiTerm, eng) -> bool:
    """X <- V / X <<- V — assignment operators.

    '<-'  is BACKTRACKABLE assignment (trailed, reversible on backtracking).
    '<<-' is NON-BACKTRACKABLE (destructive, permanent) assignment.

    Two modes:
    1. Global/persistent variable (LHS has a function/predicate definition):
       Retract all existing rules and assert LHS -> RHS (like setq).
       Always destructive (global state is not backtracked).
    2. Local variable / already-bound term:
       For '<-': trail the coref pointer then rebind (backtrackable).
       For '<<-': destructively update in-place (non-backtrackable).
    """
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    if a1 is None or a2 is None:
        return False

    # Determine whether this is backtrackable (<-) or destructive (<<-)
    _op_sym = goal.type.keyword.symbol if (goal.type and goal.type.keyword) else '<<-'
    _backtrackable = (_op_sym == '<-')

    # Deref LHS
    lhs = a1.deref()
    # A name declared with `global` stands for a cell every reference reads,
    # so writing to the name writes into that cell: eratosthenes's
    # `limit <- 20` has to be visible to the `M < limit` that follows.
    _g_cell = _global_cell(lhs, eng)
    if _g_cell is not None:
        lhs = _g_cell.deref()
    elif (lhs.type is not None and lhs.type.keyword is not None
            and lhs.type.keyword.symbol == '.'):
        # `sieve.M <- multiple_of(P)` writes the feature, not the dot-term:
        # the sieve keeps what was written under M.
        _dot_cell = _resolve_dot_feat(lhs, eng)
        if _dot_cell is None:
            return False
        lhs = _dot_cell.deref()

    # `<-` writes something the query may take back, and what is in
    # persistent store is not the query's to take back.
    if _backtrackable and lhs.__dict__.get('_wl_persistent_written', False):
        from wild_life.print_term import term_to_string as _t2s_arrow
        from wild_life.unification import AbortException as _Abort_arrow
        _lhs_str = _t2s_arrow(lhs, quoted=True, wl=eng.wl)
        _rhs_str = _t2s_arrow(a2.deref(), quoted=True, wl=eng.wl)
        sys.stderr.write(
            f"*** Error: cannot use '<-' on persistent value in"
            f" {_lhs_str} <- {_rhs_str}\n\n*** Abort\n")
        raise _Abort_arrow(hook_called=True)

    defn = lhs.type

    # Mode 1: LHS is a named function/predicate symbol (global variable).
    # A name with rules that take arguments is not a global variable but a
    # function of the program's own: termsize marks the term it has counted
    # with `X <- Seen`, and when that term is the name `f` of `f(X) -> X*X`
    # the mark must not rewrite what f is.
    _lhs_is_global = (
        defn is not None and
        hasattr(defn, 'rule') and
        defn.rule is not None and
        lhs.value is None and
        not lhs.attr_list and
        (defn.type == DefType.GLOBAL
         or all(h is None or not h.deref().attr_list
                for h, _b in defn.rule)))
    if _lhs_is_global:
        # Use setq-like behavior: clear all rules, assert new value
        # Always destructive for global variables (global state is intentional)
        from wild_life.unification import copy_term as _copy_term
        rhs_d = a2.deref()
        # Try arithmetic eval for numeric RHS
        ok, val = _eval_arith(a2, eng)
        if ok:
            # Build a numeric psi-term as the new value
            new_val = PsiTerm()
            new_val.type = eng.wl.real
            new_val.value = val
            rhs_d = new_val
        elif rhs_d.attr_list:
            # What is written in is a value, so a comparison written on the
            # right is the answer it gives: structures.lf's
            # `res <<- (S :== true)` stores true or false, not the question.
            # A call is likewise asked for its answer, which is what makes
            # term_expansion.lf's `load_option <<- assert_rules or expand2file`
            # store true rather than the question it is written as.
            _rhs_sc = (_eval_sort_comparison(rhs_d, eng)
                       or _try_eval_any_func(rhs_d, eng))
            if _rhs_sc is not None and _rhs_sc.deref() is not rhs_d:
                rhs_d = _rhs_sc.deref()
        defn.rule = []          # clear existing rules
        defn.type = DefType.FUNCTION
        _vm: dict = {}
        head_copy = _copy_term(lhs, _vm)
        defn.rule.append((head_copy, rhs_d))
        return True

    # Mode 2: Local variable / bound term
    # Evaluate RHS (arithmetic or term)
    ok_arith, val = _eval_arith(a2, eng)
    if ok_arith:
        rhs_term = PsiTerm()
        rhs_term.type = eng.wl.real
        rhs_term.value = val
    else:
        rhs_term = a2.deref()
        # What is written in is a value, so a call is asked for the one it
        # answers: `V <<- term_explore(X, Seen)` stores the count, not the
        # call.  A call nothing can work out yet is stored as it stands.
        if rhs_term.attr_list:
            _rhs_ev = (_eval_sort_comparison(rhs_term, eng)
                       or _try_eval_any_func(rhs_term, eng))
            if _rhs_ev is not None and _rhs_ev.deref() is not rhs_term:
                rhs_term = _rhs_ev.deref()
        if not _backtrackable:
            # `X <<- s([1+6],X)` reads X before it writes it, so the X inside
            # the new term is the 3 that was there — `s([7],3)`, not a term
            # that points back at itself the way `X <- s(1+4,X)` does.
            rhs_term = _substitute_old_self(rhs_term, lhs, eng)

    # `X <- T` points X at T rather than taking a copy of what T holds, so a
    # later `T <- U` moves X along with it: boites keeps a list of pointers
    # into a configuration and deletes each in turn with `L <- Tl`, and a
    # pointer in front has to follow the one behind it out of the list.  A
    # copy would leave it holding the cell that was just removed.  A number
    # is written in as the number it is, since there is nothing to follow.
    if _backtrackable and not ok_arith and rhs_term.deref() is not lhs:
        eng.trail.trail_psi(lhs, 'coref')
        lhs.coref = rhs_term
        eng.unifier._wakeup_resid(lhs, lhs)
        return True

    # Both forms update the dereferenced endpoint in place, so that every
    # variable pointing into this chain sees the new value.  `<-` trails each
    # changed field, so a failed query leaves X the 3 it was; `<<-` writes for
    # good, which is what lets term_size count a term in one pass, fail back
    # out of the counting, and still read the number it arrived at.
    # `X <<- V` on a variable that stands for nothing yet makes a cell of its
    # own — termsize's `V<<-@` calls it "an anonymous persistent term" — and
    # what is written into such a cell stays written: term_size counts a term,
    # fails back out of the counting to undo its marks, and still reads the
    # number it arrived at.  Writing over a term that already holds something
    # is an ordinary write, undone with everything else the query did.
    _persistent = (not _backtrackable
                   and (lhs.__dict__.get('_wl_persistent_cell', False)
                        or (lhs.value is None and not lhs.attr_list
                            and (lhs.type is None or lhs.type is eng.wl.top))))
    if _persistent:
        eng.persistent_store_touched = True
        lhs._wl_persistent_cell = True
        # What is written here stays written, so an equation may read it
        # but not narrow it.  The whole of it: `A <<- p(a,b,c)` puts the p
        # and its three arguments away together, and display_persistent
        # writes a ` $` in front of each of them.
        lhs._wl_persistent_written = True
        # What goes into persistent store is a copy, not the working term:
        # term_expansion.lf files an expander with
        # `expansion_methods_table.combined_name(S) <<- (E,T)`, and E and T
        # are the caller's variables — left shared, the query that made the
        # entry takes its own bindings back out of the store on the way out
        # and leaves `(@,@)` behind.
        from wild_life.unification import copy_term as _ct_pers
        if rhs_term.attr_list or rhs_term.coref is not None:
            rhs_term = _ct_pers(rhs_term, {})
        _mark_persistent_deep(rhs_term, set())
    else:
        eng.trail.trail_psi(lhs, 'value')
        eng.trail.trail_psi(lhs, 'coref')
        eng.trail.trail_psi(lhs, 'type')
        eng.trail.trail_psi(lhs, 'attr_list')
        eng.trail.trail_psi(lhs, 'flags')
    if ok_arith:
        lhs.value = val
        lhs.coref = None
        lhs.attr_list = {}
        # A number written over a term is the number, sort and all: what
        # `X <<- a` left behind is not what `X <<- 1234` now stands for.
        lhs.type = _make_number(eng, float(val)).type
    else:
        rhs = rhs_term
        lhs.value = rhs.value
        lhs.type = rhs.type
        lhs.attr_list = dict(rhs.attr_list)
        lhs.coref = rhs.coref if _backtrackable else None
        lhs.flags = rhs.flags
    # An equation suspended on this variable was waiting for exactly this:
    # `A = B+5` answers A = 11 once `B <- 6` says what B is.
    eng.unifier._wakeup_resid(lhs, lhs)
    return True


def _substitute_old_self(rhs: PsiTerm, lhs: PsiTerm, eng) -> PsiTerm:
    """Replace references to `lhs` inside `rhs` with lhs's current value."""
    snapshot = PsiTerm()
    snapshot.type = lhs.type
    snapshot.value = lhs.value
    snapshot.attr_list = dict(lhs.attr_list)
    snapshot.coref = lhs.coref
    snapshot.flags = lhs.flags
    seen: set = set()
    stack = [rhs]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for key, ref in list(node.attr_list.items()):
            sub = ref.deref()
            if sub is lhs:
                eng.unifier.set_attr(node, key, snapshot)
            else:
                stack.append(sub)
    return rhs


def bi_setq(goal: PsiTerm, eng) -> bool:
    """setq(X, V) — set (global) functional fact X -> V.

    Retracts all existing X -> @ rules and asserts X -> V.
    Used for global variable assignment:  setq(counter, 5).

    Like <<-, setq evaluates V arithmetically before storing so that the
    stored value is concrete and survives backtracking.  (If V is not a
    pure arithmetic expression the raw term is stored instead.)
    """
    args = list(goal.attr_list.values()) if goal.attr_list else []
    if len(args) < 2:
        return False
    x_term = args[0].deref()
    wl = eng.wl

    defn = x_term.type
    if defn is None:
        return False

    # Make X dynamic if it is not already; set its type to FUNCTION
    from wild_life.data_structures import DefType
    if defn.rule is None or callable(defn.rule):
        defn.rule = []
    defn.type = DefType.FUNCTION  # ensure it's recognised as a function
    # A global the program writes to is dynamic whether or not it was declared
    # so: what it stands for is whatever the last setq put there.
    defn.is_dynamic = True

    # Remove ALL existing -> rules for X (retract all functional clauses)
    defn.rule = []  # wipe all rules

    # Evaluate V arithmetically if possible (so the stored value is a
    # concrete number that survives backtracking), otherwise store the
    # dereffed term directly (like <<- does for global variables).
    ok_arith, val = _eval_arith(goal.attr_list.get('2'), eng)
    if ok_arith:
        v_stored = PsiTerm()
        v_stored.type = eng.wl.real
        v_stored.value = val
    else:
        v_stored = goal.attr_list.get('2').deref()

    # Build head = x_term (fresh copy) with value = v_stored
    from wild_life.unification import copy_term
    _vm: dict = {}
    head_copy = copy_term(x_term, _vm)
    # What is filed is a copy, as `assert((X -> Value))` would file: a global
    # outlives the proof that set it.  Storing the live term let backtracking
    # take the value away again, which is exactly what
    # `read_all(L), setq(list_of_words, L), fail` relies on not happening.
    v_stored = copy_term(v_stored.deref(), {})
    # The rule body for -> is the return value
    defn.rule.append((head_copy, v_stored))
    return True


def bi_clause(goal: PsiTerm, eng) -> bool:
    """clause(Head) / clause(Head, Body) — non-deterministically match clauses.

    Succeeds once for each matching clause of the predicate or function.
    For a fact p, clause(p) succeeds if p has at least one clause.
    For clause(Head, Body), unifies Head and Body with each matching clause.
    """
    from wild_life.data_structures import GoalType as _GT
    args_raw = list(goal.attr_list.values()) if goal.attr_list else []
    if not args_raw:
        return False
    head = args_raw[0].deref()
    body = args_raw[1].deref() if len(args_raw) >= 2 else None
    wl = eng.wl

    # Handle the LIFE clause form: clause(X:(Pred->Body))?
    # When head.type is the clause arrow — '->' for a function, ':-' for a
    # predicate — the actual head is in attr '1' and the body in attr '2'.
    clause_container = None  # the head->body term to unify with full clause
    if (head.type and head.type.keyword
            and head.type.keyword.symbol in ('->', ':-')
            and not head.value):
        # head is a psi-term of sort '->': this is the X:(f1->Y) form
        pred_term = head.attr_list.get('1')
        body_from_head = head.attr_list.get('2')
        if pred_term is not None:
            pred_d = pred_term.deref()
            defn = pred_d.type
            if defn is not None and defn.rule is not None and not callable(defn.rule):
                if body_from_head is not None and body is None:
                    # Store the head container for unification
                    clause_container = head
                    # Use the body variable from inside the arrow form
                    body = body_from_head.deref() if body_from_head else wl.make_var()
                    head = pred_d  # The actual predicate head to match against rule heads
    else:
        defn = head.type
        if defn is None or defn.rule is None or callable(defn.rule):
            # Try treating it as a 0-arity predicate/fact
            return False

    defn = head.type
    if defn is None or defn.rule is None or callable(defn.rule):
        return False

    rule_list = defn.rule
    if not rule_list:
        return False

    # Use body = fresh var if not supplied (for clause/1 form)
    if body is None:
        body = wl.make_var()

    if clause_container is not None:
        # For the X:(Pred->Body) form, we need to unify the full clause term
        # clause_container is the term X was bound to (Pred->Body structure).
        # We push a special CLAUSE goal that handles this form.
        # When a rule (h, b) matches:
        #   - Unify head (pred term) with h
        #   - Unify body with b
        # The X variable is already bound to the container, and the head/body
        # sub-terms inside it will unify through the trail.
        eng.push_goal(_GT.CLAUSE, head, body, rule_list)
    else:
        eng.push_goal(_GT.CLAUSE, head, body, rule_list)
    return True


def bi_children(goal: PsiTerm, eng) -> bool:
    """children(Sort, List) — unify List with immediate subtypes of Sort."""
    wl = eng.wl
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    if a1 is None or a2 is None:
        return False
    sort_term = a1.deref()
    defn = sort_term.type
    if defn is None:
        return _unify(eng, a2, wl.make_atom('[]', wl.user_module))
    # Use the Definition's own children list to avoid duplicates from aliases.
    # Each Definition has a .children list populated by _make_type_link.
    children = []
    for child_defn in getattr(defn, 'children', []):
        if child_defn.keyword is None:
            continue
        child_atom = wl.make_atom(child_defn.keyword.symbol, wl.bi_module)
        children.append(child_atom)
    result_list = wl.make_list(children)
    return _unify(eng, a2, result_list)


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
        result = PsiTerm(type_def=defn)
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
        while cur.type is wl.alist:
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
        result = PsiTerm(type_def=defn)
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
        while cur.type is wl.alist:
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
    """Convert WL list to Python list.

    A sub-sort of cons is walked like a cons cell, so `int_cons <| cons.`
    makes an int_cons spine just as traversable as a plain list.
    """
    wl = eng.wl
    items = []
    cur = t.deref()
    while cur.type is not None and cur.type.is_subtype_of(wl.alist):
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
            var = PsiTerm(type_def=wl.top)
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
    # Set up choice points for each member.
    # Use UNIFY choice points (not PROVE) so backtracking unifies x with the
    # alternative item via unify_aim, not prove_aim which expects a rule list.
    x = a1
    for item in reversed(items[1:]):
        eng.push_choice_point(GoalType.UNIFY, x, item, None)
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
    import sys as _sys
    arg = _get_one_arg(goal)
    code = 0
    if arg and arg.value is not None:
        try:
            code = int(float(arg.value))
        except Exception:
            pass
    # Print final newline before halting (matches original C interpreter behaviour)
    _sys.stdout.write("\n")
    _sys.stdout.flush()
    raise HaltException(code)


def bi_abort(goal: PsiTerm, eng) -> bool:
    """abort — call aborthook (if set), then abort current query.

    The aborthook is a predicate name stored via setq(aborthook, foo).
    When set, we call it before raising AbortException.  The hook's output
    appears on the same line as the already-printed prompt ('> ').
    """
    hook_called = False
    wl = eng.wl

    # Look up the 'aborthook' symbol — the user sets it via setq(aborthook, foo).
    # update_symbol returns the existing Definition (creating one if new, but an
    # unset symbol will have rule=None or an empty list).
    try:
        hook_defn = wl.update_symbol(None, 'aborthook')
        if hook_defn is not None and isinstance(hook_defn.rule, list) and hook_defn.rule:
            # rule is a list of (head_copy, v_term) tuples stored by bi_setq.
            _, v_term = hook_defn.rule[0]
            hook_psi = v_term.deref() if v_term is not None else None
            if hook_psi is not None:
                try:
                    # Prove the hook predicate (e.g. foo, which writes "I'm outta here!")
                    eng.prove(hook_psi)
                except AbortException:
                    raise  # propagate nested abort
                except Exception:
                    pass  # ignore hook failures
                hook_called = True
    except AbortException:
        raise
    except Exception:
        pass

    raise AbortException(hook_called=hook_called)


def bi_nl_err(goal: PsiTerm, eng) -> bool:
    """nl_err — newline to stderr."""
    print(file=sys.stderr)
    return True


def bi_assert_ok(goal: PsiTerm, eng) -> bool:
    """succeed if last assert succeeded."""
    return True


# Where a loaded file is looked for when the name does not name one outright,
# which is what `bi_load_path` stands for in built_ins.lf.
def _life_load_dirs() -> list:
    import os as _os_lp
    _root = _os_lp.path.dirname(_os_lp.path.dirname(_os_lp.path.abspath(__file__)))
    return ['',
            _os_lp.path.join(_root, 'lib'),
            _os_lp.path.join(_root, 'Tools'),
            _os_lp.path.join(_root, 'examples'),
            _os_lp.path.join(_root, 'examples', 'SuperLint')]


def _resolve_life_file(filename: str) -> str:
    """The file a load names.

    The name is taken as written when a file of that name is there —
    `load("FILES/t3203.1")` names the file itself — and `.lf` is only added
    when it is not.  A bare name that names nothing here is looked for where
    the libraries and examples live, which is how `import("superlint")` finds
    a module that does not sit beside the program.
    """
    import os as _os_rf
    for _d in _life_load_dirs():
        for _name in ((filename,) if filename.endswith('.lf')
                      else (filename, filename + '.lf')):
            _p = _os_rf.path.join(_d, _name) if _d else _name
            if _os_rf.path.exists(_p):
                return _p
    return filename if filename.endswith('.lf') else filename + '.lf'


def _announce_load(filename: str) -> None:
    """Say which file is being read, the way built_ins.lf's load_2 does.

    The line goes out where the reader's prompt was last written, so the
    prompt and the announcement share a line exactly as they do in the C
    interpreter.
    """
    sys.stdout.write('*** Loading File "%s"\n' % filename)


def bi_load(goal: PsiTerm, eng) -> bool:
    """load(File) — load a LIFE source file."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    filename = str(arg.value) if arg.value else (
        arg.type.keyword.symbol if arg.type and arg.type.keyword else '')
    filename = _resolve_life_file(filename)
    _announce_load(filename)

    wl = eng.wl
    delay_count_before = len(wl.delay_rules)
    result = eng.load_file(filename)
    # The module is put back by load_file, to the one the reader was in when
    # the file was opened -- which is not always user: `module("parser")?`
    # followed by `load("tokenizer")?` carries on in parser, and the opens
    # that follow belong to parser rather than to whoever asked for it.

    # In C Wild Life, load(X) is implemented via user-defined predicates in
    # built_ins.lf: features(X) and load_2/2.  After loading and encode_types(),
    # load_2([], X) is proved; the nil term in load_2([]) gets eval_copy'd with
    # status=0 (nil inherits alist properties via type propagation), and
    # check_out fires the cons delay rule (:: C:cons | write(C.1), nl.) for it.
    # Inside that delay, write(C.1) accesses nil's "1" attribute (which is
    # unbound → prints as '@'), and _collect_literal_integers pre-fires the
    # integer delay for the literal 1 inside C.1 (→ "1 ").  nl adds "\n".
    # Net output: "1 @\n".
    #
    # Reproduce this behaviour: after loading a file that introduced new delay
    # rules, fire the cons delay for a fresh empty-alist term (representing the
    # nil from load_2([])).
    if result and len(wl.delay_rules) > delay_count_before and wl.alist is not None:
        nil_for_delay = PsiTerm()
        nil_for_delay.type = wl.alist
        nil_for_delay.attr_list = {}
        unifier = getattr(eng, 'unifier', None)
        if unifier is not None:
            unifier._fire_delay_rules(nil_for_delay, wl.alist)

    return result


def bi_op(goal: PsiTerm, eng) -> bool:
    """op(Prec, Type, Name) — declare, query, or enumerate operators.

    全引数が束縛されている場合: バリデーション + 演算子宣言モード。
    少なくとも1引数が自由変数の場合: 列挙/クエリモード (バックトラック対応)。
    """
    import sys
    from wild_life.data_structures import OperatorType, GoalType, Goal, ChoicePoint

    wl = eng.wl
    # An argument the caller left out is a free variable, not a missing one:
    # `op(X,3 => (+))` asks for the precedence and kind of `+` without naming
    # the kind, so it enumerates over the second position.
    for _k in ('1', '2', '3'):
        if goal.attr_list.get(_k) is None:
            eng.unifier.set_attr(goal, _k, PsiTerm(type_def=wl.top))
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
        return False
    # An operator answers to its positions by name as well: the three are
    # precedence, kind and functor, and `bagof(X.functor,X:op)` reads the
    # last of them out of every operator there is.
    def _sym(_d, _fallback):
        return (_d.symbol if (_d is not None and getattr(_d, 'keyword', None))
                else _fallback)
    for _feat, _pos in (('precedence', '1'), ('kind', '2'), ('functor', '3')):
        _key = _sym(getattr(wl, _feat, None), _feat)
        _have = goal.attr_list.get(_key)
        if _have is None:
            eng.unifier.set_attr(goal, _key, goal.attr_list[_pos])
        elif _have.deref() is not goal.attr_list[_pos].deref():
            if not _unify(eng, _have, goal.attr_list[_pos]):
                return False
    prec = a1.deref()
    typ  = a2.deref()
    name = a3.deref()

    prec_is_var = _is_var(prec, eng)
    typ_is_var  = _is_var(typ,  eng)
    name_is_var = _is_var(name, eng)

    op_type_names = {
        OperatorType.FX: 'fx', OperatorType.FY: 'fy',
        OperatorType.XF: 'xf', OperatorType.YF: 'yf',
        OperatorType.XFX: 'xfx', OperatorType.XFY: 'xfy',
        OperatorType.YFX: 'yfx',
    }
    valid_op_kinds = set(op_type_names.values())
    op_map = {v: k for k, v in op_type_names.items()}

    # ─── ヘルパー: エラーメッセージ用の項を文字列化 ──────────────────────────
    def _fmt(t):
        """PSI項をエラーメッセージ用に文字列化する。"""
        if (t.type is not None and
                hasattr(wl, 'quoted_string') and
                t.type is not None and
                getattr(t.type, 'is_subtype_of', None) is not None and
                t.type.is_subtype_of(wl.quoted_string) and
                t.value is not None):
            return f'"{t.value}"'
        if t.value is not None:
            v = t.value
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v)
        if t.type and t.type.keyword:
            return t.type.keyword.symbol
        return '?'

    # ─── 宣言/バリデーションモード (全引数が束縛されている) ─────────────────
    if not prec_is_var and not typ_is_var and not name_is_var:
        # 名前が文字列か数値でないか確認
        name_is_string = (
            name.type is not None and
            hasattr(wl, 'quoted_string') and
            getattr(name.type, 'is_subtype_of', None) is not None and
            name.type.is_subtype_of(wl.quoted_string) and
            name.value is not None
        )
        name_is_number = (name.value is not None and not name_is_string)

        if name_is_string or name_is_number:
            sys.stderr.write(
                f"*** Error: numbers or strings may not be operators"
                f" in c_op({_fmt(prec)},{_fmt(typ)},{_fmt(name)}).\n"
            )
            return False

        # 演算子種別が有効か確認
        typ_sym = typ.type.keyword.symbol if (typ.type and typ.type.keyword) else ''
        if typ_sym not in valid_op_kinds:
            sys.stderr.write(f"*** Error: bad operator kind '{typ_sym}'.\n")
            return False

        # 優先度が数値か確認
        if prec.value is None:
            sys.stderr.write(
                f"*** Error: precedence must be a positive integer"
                f" in c_op({_fmt(prec)},{typ_sym},{_fmt(name)}).\n"
            )
            return False

        p = int(float(prec.value))
        if p < 1 or p > 1200:
            sys.stderr.write(
                f"*** Error: precedence must range from 1 to 1200"
                f" in c_op({p},{typ_sym},{_fmt(name)}).\n"
            )
            return False

        # 有効: 演算子を宣言
        name_sym = name.type.keyword.symbol if (name.type and name.type.keyword) else ''
        wl.add_operator(p, op_map[typ_sym], name_sym)

        # Wild Life の REPL はこの宣言クエリが "depth" に入るよう
        # ダミーの choice point を積む (バックトラック時は即失敗)。
        # goal_stack.type が EVAL/EVAL_CUT 以外なら REPL が has_new_choices=True と判断する。
        from wild_life.data_structures import Goal, GoalType, ChoicePoint
        mark = eng.trail.mark()
        # UNIFY(nil, integer(0)) — 確実に失敗するダミーゴール
        dummy_lhs = wl.make_atom('@')
        dummy_rhs = wl.make_integer(0)
        sentinel = Goal(GoalType.UNIFY, dummy_lhs, dummy_rhs, None, next=None)
        cp = ChoicePoint(undo_point=mark, goal_stack=sentinel, next=eng.choice_stack)
        eng.choice_stack = cp
        return True

    # ─── 列挙/クエリモード (少なくとも1引数が自由変数) ──────────────────────
    # _enumerable_ops リストを使って順序通りに列挙する
    # (symbol_table の挿入順に依存しないため、期待する列挙順序が保証される)
    solutions = list(getattr(wl, '_enumerable_ops', []))

    # 束縛済み引数でフィルタリング
    def _matches(p, t, n):
        if not prec_is_var:
            if prec.value is None:
                return False
            if int(float(prec.value)) != p:
                return False
        if not typ_is_var:
            ts = typ.type.keyword.symbol if (typ.type and typ.type.keyword) else ''
            if ts != t:
                return False
        if not name_is_var:
            ns = name.type.keyword.symbol if (name.type and name.type.keyword) else ''
            if ns != n:
                return False
        return True

    filtered = [(p, t, n) for (p, t, n) in solutions if _matches(p, t, n)]
    if not filtered:
        return False

    # 最初の解以外を選択点としてスタックに積む (逆順で積む → 先頭が次の解)
    for (p, t, n) in reversed(filtered[1:]):
        p_term = wl.make_integer(p)
        t_term = wl.make_atom(t)
        n_term = wl.make_atom(n)
        # UNIFY ゴールを3つ連結: prec → typ → name → 現在のゴールスタック
        g_name = Goal(GoalType.UNIFY, name, n_term, None, next=eng.goal_stack)
        g_typ  = Goal(GoalType.UNIFY, typ,  t_term, None, next=g_name)
        g_prec = Goal(GoalType.UNIFY, prec, p_term, None, next=g_typ)
        mark = eng.trail.mark()
        cp = ChoicePoint(undo_point=mark, goal_stack=g_prec, next=eng.choice_stack)
        eng.choice_stack = cp

    # 最初の解を試みる
    p0, t0, n0 = filtered[0]
    p0_term = wl.make_integer(p0)
    t0_term = wl.make_atom(t0)
    n0_term = wl.make_atom(n0)
    mark0 = eng.trail.mark()
    ok = (_unify(eng, prec, p0_term) and
          _unify(eng, typ,  t0_term) and
          _unify(eng, name, n0_term))
    if not ok:
        eng.trail.undo_to(mark0)
        return False
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


def _rule_to_string(h, b, wl, inline: bool = False):
    """ルール (head, body) を共有 PrintState で文字列化する。

    body 中の conjunction ','( left, right ) を個々のゴールに分解し、
    head_str と goal_str のリストを返す。
    同一 PsiTerm を head/body で共有する変数は同じ名前 (_A, _B, ...) で表示。
    """
    import io
    from wild_life.print_term import (
        PrintState, _pretty_tag_or_psi_term, MAX_PRECEDENCE
    )

    def split_conj(t):
        """Recursively split conjunction into list of individual goals."""
        if t is None:
            return []
        t = t.deref()
        # Conjunction operator ','  (no functor-style extra attr)
        if (t.type and t.type.keyword and t.type.keyword.symbol == ','
                and '1' in t.attr_list and '2' in t.attr_list
                and 'functor' not in t.attr_list):
            return split_conj(t.attr_list['1']) + split_conj(t.attr_list['2'])
        return [t]

    # 共有 PrintState: head / body 全体をまとめてスキャン
    ps = PrintState(outfile=io.StringIO())
    ps.const_quote = True
    # A listing shows the clause as written: `a(1+2).` lists as `a(1 + 2)`,
    # not as the 3 it would evaluate to when the clause is used.
    ps.no_arith_eval = True
    # A listing writes one goal to a line, so `,` and `:-` carry the break.
    # A delay rule is written on one line, and keeps its goals on it.
    ps.listing_flag = not inline

    # The body is one term, not a row of goals: a clause that names itself —
    # `assert(X:(p :- retbool, inc, assert(X), fail))` — shares its whole
    # body with the copy inside, and it is that conjunction which wants a
    # name, not each of the four goals under it.
    ps.go_through(h)
    ps.go_through(b)
    ps.insert_variables({}, False)

    # head を出力
    # A head whose functor binds no tighter than `:-` itself cannot be written
    # in operator form there, so it is written as a call: `pred(a) :- succeed`,
    # not `pred a :- succeed`, which would read back as something else.
    from wild_life.print_term import _opcheck as _opchk_h, NOTOP as _NOTOP_h
    _h_kind, _h_prec, _h_type = _opchk_h(h.deref())
    _was_canon = ps.write_canon
    if _h_kind != _NOTOP_h and _h_prec >= 1200:
        ps.write_canon = True
    try:
        _pretty_tag_or_psi_term(ps, h, MAX_PRECEDENCE + 1, 0, wl)
    finally:
        ps.write_canon = _was_canon
    head_str = ps.take()

    body_str = None
    if b is not None:
        _pretty_tag_or_psi_term(ps, b, MAX_PRECEDENCE + 1, 0, wl)
        body_str = ps.take()

    return head_str, body_str


# What the C interpreter calls a built-in function rather than a built-in
# predicate: it answers with a value where a predicate answers yes or no.
_BUILTIN_FUNCTION_SYMS = frozenset((
    'and', 'or', 'not', 'xor',
    'var', 'nonvar', 'is_function', 'is_predicate', 'is_sort',
    'is_number', 'is_value', 'has_feature',
    'int2str', 'str2int', 'str2psi', 'psi2str', 'str2num', 'num2str',
    'strcon', 'strlen', 'substr', 'chr', 'asc', 'upper', 'lower',
    'root_sort', 'sort', 'combined_name',
    'features', 'feature_values', 'parents', 'children',
    'least_sorts', 'glb', 'lub', 'copy_term', 'eval',
    '+', '-', '*', '/', '//', 'mod', '^', 'min', 'max', 'abs',
    'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'floor', 'ceiling', 'round', 'truncate',
    '>', '<', '>=', '=<', '=:=', '=\\=',
    ':=<', ':>=', ':<', ':>', ':==', ':\\==',
))


def _bi_listing_one(defn, wl, imported: bool = False) -> None:
    """Helper: list clauses for a single Definition.

    imported=True  : 別モジュールからインポートされた述語。
                     dynamic ヘッダなし、常に ':-' ボディ付きで表示。
                     body ゴールは ',' で改行区切り。
    imported=False : 現在のモジュール所有の述語。
                     'dynamic(name)?' ヘッダ付き、succeed ボディは省略。
    """
    from wild_life.data_structures import DefType

    if defn is None or defn.keyword is None:
        return
    active_rules = [(h, b) for h, b in (defn.rule or []) if h is not None]
    if not active_rules:
        return
    func_name = defn.keyword.symbol
    is_function = (defn.type == DefType.FUNCTION)
    succeed_sym = wl.succeed.keyword.symbol if wl.succeed and wl.succeed.keyword else 'succeed'

    # 各定義の前に空行 (built_ins.lf の listing_2 が挟む nl に相当)。
    print()
    if getattr(defn, 'is_dynamic', False):
        # `dynamic(P)?` を宣言された述語だけがヘッダを持つ (assert2.lf 末尾の
        # dynamic(p)? / dynamic(f)? がその例)。宣言のない long.lf の q は
        # ヘッダなしで列挙される。
        print(f"dynamic({func_name})?")

    for h, b in active_rules:
        head_str, body_str = _rule_to_string(h, b, wl)

        if is_function:
            vs = body_str if body_str else 'true'
            print(f"{head_str} -> {vs}.")
        else:
            # 述語: ボディは常に ':-' 付きで表示 (各ゴール改行)。
            # ファクトも `HEAD :- succeed.` として列挙される。
            bs = body_str if body_str else 'succeed'
            print(f"{head_str} :-\n        {bs}.")


def _bi_listing_all(eng, wl) -> None:
    """List all user-defined predicates/functions."""
    from wild_life.data_structures import DefType
    if not hasattr(wl, 'user_module') or not wl.user_module:
        return
    # Gather all definitions that have rules
    seen = set()
    for sym, defn in list(wl.user_module.symbol_table.items()):
        if defn is None or id(defn) in seen:
            continue
        seen.add(id(defn))
        if defn.rule and defn.type in (DefType.PREDICATE, DefType.FUNCTION):
            _bi_listing_one(defn, wl)


def bi_listing(goal: PsiTerm, eng) -> bool:
    """listing(F, ...) — list clauses for one or more functors.

    引数なし: 全ユーザ定義述語/関数を列挙。
    引数あり: 指定したシンボルの節を列挙。複数引数可 (例: listing(aa,bb)?)。

    表示形式:
      - 自モジュール述語 (PREDICATE/FUNCTION):
          \\ndynamic(NAME)?
          HEAD :- BODY.   (succeed ボディは省略して HEAD. のみ)
      - インポート述語 (別モジュール由来):
          HEAD :- BODY.   (dynamic ヘッダなし、succeed でも表示)
          エントリ間は空行で区切る
      - 空定義 (自モジュール): % 'NAME' is a user-defined predicate...
      - UNDEF / 衝突ブロック済: 無出力で成功
    """
    from wild_life.data_structures import DefType

    wl = eng.wl

    # 引数なし: 全ユーザ述語を列挙
    if not goal.attr_list:
        _bi_listing_all(eng, wl)
        return True

    # 全引数を順に処理
    # imported_pending: 連続するインポート述語をまとめて空行区切りで出力
    imported_pending = []   # list of defn (imported, with rules)

    def flush_imported():
        """collected imported entries を出力してリセット"""
        for d in imported_pending:
            _bi_listing_one(d, wl, imported=True)
        imported_pending.clear()

    i = 1
    while True:
        a = goal.attr_list.get(str(i))
        if a is None:
            break
        a_deref = a.deref() if hasattr(a, 'deref') else a
        defn = a_deref.type if a_deref.type else None

        if defn is not None and defn._builtin_func is not None:
            # A built-in has no clauses to show, so listing says what it is.
            flush_imported()
            kind = ('function' if (defn.keyword is not None
                                   and defn.keyword.symbol in _BUILTIN_FUNCTION_SYMS)
                    else 'predicate')
            name = defn.keyword.symbol if defn.keyword else '?'
            print()
            print(f"% '{name}' is a built-in {kind}.")
        elif defn is not None and defn.type in (DefType.PREDICATE, DefType.FUNCTION):
            is_imported = (defn.keyword and defn.keyword.module is not None
                           and defn.keyword.module != wl.user_module)
            active_rules = [(h, b) for h, b in (defn.rule or []) if h is not None]

            if is_imported:
                if active_rules:
                    imported_pending.append(defn)
                # インポート述語で節なし: 無出力で成功
            else:
                # 自モジュール述語が来たらインポート分を先に出力
                flush_imported()
                func_name = defn.keyword.symbol if defn.keyword else '?'
                if not active_rules and getattr(defn, 'is_persistent', False):
                    # A global that has not been assigned yet holds a plain @.
                    _note_global_used(eng, defn)
                    print()
                    print(f"% '{func_name}' is a user-defined global variable "
                          f"worth @.")
                elif not active_rules:
                    # What it was defined as is what listing calls it: a
                    # function whose every clause has been retracted is still
                    # a function.
                    _kind_empty = ('function' if defn.type == DefType.FUNCTION
                                   else 'predicate')
                    print()
                    print(f"% '{func_name}' is a user-defined {_kind_empty} "
                          f"with an empty definition.")
                else:
                    _bi_listing_one(defn, wl, imported=False)
        elif defn is not None and defn.type == DefType.TYPE:
            # A sort lists as its membership condition, if it was defined with
            # one, followed by the sorts it sits under.
            flush_imported()
            name = defn.keyword.symbol if defn.keyword else '?'
            print()
            for _pat, _cond in (defn.rule or []):
                if _pat is None or _cond is None:
                    continue
                # `s := t` carries no membership condition; it says only that
                # an s is a t, which the `<|` lines below already report.
                if _cond.deref().type is wl.succeed:
                    continue
                # The pattern is shown as the sort being defined, not as the
                # sort it was written against: `positive := I:int | I > 0`
                # lists as `:: _A: positive | _A > 0`.  The swap is on the
                # pattern itself, so that it and the condition still share the
                # variable and print under one name.
                from wild_life.data_structures import SORT_VAR as _SV_LST
                _pat_d = _pat.deref()
                _was_type, _was_flags = _pat_d.type, _pat_d.flags
                _pat_d.type, _pat_d.flags = defn, _pat_d.flags | _SV_LST
                try:
                    _pat_str, _cond_str = _rule_to_string(_pat_d, _cond, wl,
                                                          inline=True)
                finally:
                    _pat_d.type, _pat_d.flags = _was_type, _was_flags
                print(f":: {_pat_str} | {_cond_str or 'succeed'}.")
            if defn.parents:
                for _parent in defn.parents:
                    _pname = _parent.keyword.symbol if _parent.keyword else '@'
                    print(f"{name} <| {_pname}.")
            else:
                print(f"{name} <| @.")
            # The sorts that sit under this one are part of what it is, so
            # listing a sort shows them too.
            for _child in getattr(defn, 'children', []):
                _cname = _child.keyword.symbol if _child.keyword else '@'
                print(f"{_cname} <| {name}.")
        elif defn is not None and defn.type == DefType.GLOBAL:
            # C Wild Life lists a global by name only — it does not report the
            # value the cell currently holds.
            flush_imported()
            _note_global_used(eng, defn)
            name = defn.keyword.symbol if defn.keyword else '?'
            # The leading blank ends the prompt line for the first entry and
            # separates the entries after that.
            print()
            print(f"% '{name}' is a user-defined global variable "
                  f"worth *null psi_term*.")
        elif defn is not None and defn.type == DefType.UNDEF:
            # UNDEF の場合:
            #   グローバル遅延規則 (:: X:bar | Goal.) の宛先ソート → その規則を列挙
            #   clash_blocked スタブ → 衝突検出で作成済みのブロック → 無音成功
            #   それ以外 (未定義/非公開) → "% 'name' is undefined." を表示
            _delay_for_sort = [
                _dr for _dr in (getattr(wl, 'delay_rules', None) or [])
                if (_dr.attr_list.get('1') is not None
                    and _dr.attr_list['1'].deref().type is defn)
            ]
            if _delay_for_sort:
                flush_imported()
                name = defn.keyword.symbol if defn.keyword else '?'
                print()
                for _dr in _delay_for_sort:
                    _dpat = _dr.attr_list.get('1')
                    _dgoal = _dr.attr_list.get('2')
                    if _dpat is None or _dgoal is None:
                        continue
                    _dpat_str, _dgoal_str = _rule_to_string(_dpat.deref(),
                                                            _dgoal, wl,
                                                            inline=True)
                    print(f":: {_dpat_str} | {_dgoal_str or 'succeed'}.")
                # A sort named only by a delay rule sits directly under @.
                _dparents = defn.parents or []
                if _dparents:
                    for _parent in _dparents:
                        _pname = _parent.keyword.symbol if _parent.keyword else '@'
                        print(f"{name} <| {_pname}.")
                else:
                    print(f"{name} <| @.")
            elif not getattr(defn, 'clash_blocked', False):
                func_name = defn.keyword.symbol if defn.keyword else '?'
                flush_imported()
                print()   # プロンプト行の末尾に改行を入れる
                print(f"% '{func_name}' is undefined.")
        # else: defn が None など → 無出力で成功

        i += 1

    # 残留インポート述語を出力
    flush_imported()

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


def _random_gen(eng):
    """The interpreter's random generator, seeded by initrandom/1."""
    import random as _random_mod
    wl = eng.wl
    gen = getattr(wl, '_random_gen', None)
    if gen is None:
        gen = _random_mod.Random()
        wl._random_gen = gen
    return gen


def bi_initrandom(goal: PsiTerm, eng) -> bool:
    """initrandom(Seed) — restart the random generator from Seed, so that the
    same seed replays the same sequence of draws."""
    import random as _random_mod
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    arg = arg.deref()
    ok, v = _eval_arith(arg, eng)
    if not ok:
        return False
    eng.wl._random_gen = _random_mod.Random(int(v))
    return True


def bi_rand(goal: PsiTerm, eng) -> bool:
    """random(X) — X is a random float [0,1).

    The functional form random(N), an integer in [0,N), is evaluated in
    _eval_arith; this is the predicate form, which draws into an unbound X.
    """
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    return _unify(eng, arg, eng.wl.make_number(_random_gen(eng).random()))


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


def bi_subsort(goal: PsiTerm, eng) -> bool:
    """subsort(A, B) — A is a subtype of B, with residuation on B.

    When B is unbound (type=top, no value, no attrs), suspends on B (registers
    a pending goal on B so it wakes when B is narrowed) and returns True.
    B will display as @~ until narrowed.

    When B is concrete:
      - If B has a value: A must have the same value.
      - If B has only a type (sort): A's type must be a subtype of B's type.
    On success, re-registers the pending goal on B so further narrowing triggers
    another check (B shows type~). Returns False if the check fails.
    """
    from wild_life.data_structures import Goal, GoalType, Residuation, SORT_VAR
    from wild_life.runtime import WL

    a1_raw = goal.attr_list.get('1')
    a2_raw = goal.attr_list.get('2')
    if a1_raw is None or a2_raw is None:
        return False

    d1 = a1_raw.deref()
    d2 = a2_raw.deref()

    # Check if B is unbound: type=top, no value, no attributes
    b_is_unbound = (d2.type is WL.top and d2.value is None and not d2.attr_list)

    def _register_on(var_pt):
        """Register a new pending subsort goal on var_pt."""
        pending_goal = Goal(GoalType.PROVE, goal, None, None, pending=True)
        if var_pt.resid is None:
            eng.trail.trail_psi(var_pt, 'resid')
            var_pt.resid = [Residuation(goal=pending_goal)]
        else:
            eng.trail.trail_copy(var_pt, 'resid')
            var_pt.resid.append(Residuation(goal=pending_goal))
        if not (var_pt.flags & SORT_VAR):
            eng.trail.trail_psi(var_pt, 'flags')
            var_pt.flags |= SORT_VAR

    if b_is_unbound:
        # B is free — suspend and return True
        if eng is not None:
            _register_on(d2)
        return True

    # B is concrete — check A :=< B
    if d2.value is not None:
        # B is a concrete value: A must equal B
        success = (d1.value is not None and d1.value == d2.value)
    else:
        # B is a sort (no value): A's type must be a subtype of B's type
        if d1.type is None or d2.type is None:
            success = False
        else:
            success = d1.type.is_subtype_of(d2.type)

    if success and eng is not None:
        # Re-register so further narrowing of B triggers another check
        _register_on(d2)

    return success


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
    result = PsiTerm(type_def=defn)
    return _unify(eng, a2, result)


def bi_type_of(goal: PsiTerm, eng) -> bool:
    """type_of(T, Type)."""
    return bi_functor_of(goal, eng)


# ─────────────────────────────────────────────────────────────────────────────
# substitute/3 — sort substitution in a psi-term
# ─────────────────────────────────────────────────────────────────────────────

def bi_substitute(goal: PsiTerm, eng) -> bool:
    """substitute(A, B, X) — In psi-term X, replace every occurrence of sort A with sort B.

    Traverses X recursively. For each node:
      - If the node is a *sort atom* (no .value) whose sort matches A's sort,
        change its sort to B's sort. Integer/float/string values (.value is not
        None) are left untouched even when their sort matches.
      - Feature labels (attribute-dict keys) equal to A's sort symbol are renamed
        to B's sort symbol. When the renamed label already exists, keep the
        existing feature value and discard the renamed one.

    All modifications are trailed so backtracking restores the original structure.
    Always succeeds (returns True) even when no changes are made.
    """
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if a1 is None or a2 is None or a3 is None:
        return False

    d_a = a1.deref()
    d_b = a2.deref()
    d_x = a3.deref()

    if d_a.type is None or d_b.type is None:
        return False

    sort_a_kw = d_a.type.keyword
    sort_b_def = d_b.type
    sort_b_kw = sort_b_def.keyword

    sort_a_sym = sort_a_kw.symbol if sort_a_kw else None
    sort_b_sym = sort_b_kw.symbol if sort_b_kw else None

    if sort_a_sym is None or sort_b_sym is None:
        return True  # unknown sorts — vacuous success

    # No-op when A and B are the same sort
    if sort_a_sym == sort_b_sym:
        return True

    visited: set = set()

    def _subst(t: PsiTerm) -> None:
        t = t.deref()
        t_id = id(t)
        if t_id in visited:
            return
        visited.add(t_id)

        # Change sort if this node is a sort atom (no value) matching A
        if (t.type is not None and t.type.keyword is not None
                and t.value is None
                and t.type.keyword.symbol == sort_a_sym):
            eng.trail.trail_psi(t, 'type')
            t.type = sort_b_def

        # Rename matching feature label A → B
        if t.attr_list and sort_a_sym in t.attr_list:
            val_a = t.attr_list[sort_a_sym]
            if sort_b_sym not in t.attr_list:
                # No conflict: rename the feature label
                eng.trail.trail_copy(t, 'attr_list')
                del t.attr_list[sort_a_sym]
                t.attr_list[sort_b_sym] = val_a
                # val_a is now accessible under sort_b_sym; will be processed below
            else:
                # Conflict: keep existing sort_b_sym value, discard renamed one.
                # Still recursively process val_a in case it's referenced elsewhere.
                eng.trail.trail_copy(t, 'attr_list')
                del t.attr_list[sort_a_sym]
                _subst(val_a)

        # Recursively process current attribute values
        if t.attr_list:
            for v in list(t.attr_list.values()):
                _subst(v)

    _subst(d_x)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# alias/2 — sort alias
# ─────────────────────────────────────────────────────────────────────────────

def bi_alias(goal: PsiTerm, eng) -> bool:
    """alias(X, Y) — Make sort X an alias for sort Y.

    After alias(X,Y), references to X resolve to Y.
    Prints a warning to stderr.
    """
    wl = eng.wl
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    if a1 is None or a2 is None:
        return False
    t1 = a1.deref()
    t2 = a2.deref()

    # Get the Definition objects for X and Y
    defn1 = t1.type
    defn2 = t2.type
    if defn1 is None or defn2 is None:
        return False

    # Get keyword info for the warning message
    kw1 = defn1.keyword
    kw2 = defn2.keyword
    sym1 = kw1.symbol if kw1 else str(defn1)
    sym2 = kw2.symbol if kw2 else str(defn2)
    mod1 = kw1.module if kw1 else None
    mod2 = kw2.module if kw2 else None
    mod1_name = mod1.module_name if mod1 else 'user'
    mod2_name = mod2.module_name if mod2 else 'user'

    # Print warning to stderr (matches original C Wild Life behaviour)
    sys.stderr.write(
        f"*** Warning: alias: '{mod1_name}#{sym1}' has now been overwritten by '{mod2_name}#{sym2}'\n"
    )

    # Perform the alias: update the symbol table entry for sym1 to refer to defn2.
    # Also update any other entries that already pointed to defn1 (transitive chain).
    # Search all modules for entries pointing to defn1 and redirect them to defn2.
    for mod in list(wl._all_modules()):
        for k, d in list(mod.symbol_table.items()):
            if d is defn1:
                mod.symbol_table[k] = defn2

    return True


# ─────────────────────────────────────────────────────────────────────────────
# trace / notrace / spy / nospy
# ─────────────────────────────────────────────────────────────────────────────

def _say_trace_state(on: bool) -> None:
    """Say which way tracing has just been turned, as new_trace says it."""
    print("*** Tracing is turned %s" % ("on." if on else "off."))


def bi_trace(goal: PsiTerm, eng) -> bool:
    """trace — turn execution tracing the other way.

    toggle_trace calls new_trace(trace?0:1), so asking twice turns it back
    off, and the answer goes to the ordinary output rather than to the
    error stream.
    """
    eng.trace = not eng.trace
    _say_trace_state(eng.trace)
    return True


def bi_notrace(goal: PsiTerm, eng) -> bool:
    """notrace — turn execution tracing off."""
    eng.trace = False
    _say_trace_state(False)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# open_in / open_out / close  (stream-based I/O redirection)
# ─────────────────────────────────────────────────────────────────────────────

def bi_open_in(goal: PsiTerm, eng) -> bool:
    """open_in(File) or open_in(File, Stream) — open file for reading,
    redirect stdin (or bind Stream to the file object).
    """
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    if a1 is None:
        return False
    a1d = a1.deref()
    # Get filename string
    if a1d.value is not None:
        filename = str(a1d.value)
    elif a1d.type and a1d.type.keyword:
        filename = a1d.type.keyword.symbol
    else:
        return False
    if filename == 'stdin':
        # `open_in(stdin, S)` names the standard input rather than a file.
        f = sys.__stdin__
    else:
        try:
            f = open(filename, 'r')
        except OSError:
            return False
    if a2 is not None:
        # 2-arg form: bind stream token to a2
        stream_term = PsiTerm()
        stream_term.value = f          # store file object as value
        stream_term.type = eng.wl.top  # generic type
        if not hasattr(eng, '_open_streams'):
            eng._open_streams = {}
        eng._open_streams[id(stream_term)] = f
        # Push the old stdin
        if not hasattr(eng, '_stdin_stack'):
            eng._stdin_stack = []
        eng._stdin_stack.append(sys.stdin)
        sys.stdin = f
        return _unify(eng, a2.deref(), stream_term)
    else:
        # 1-arg form: redirect global stdin
        if not hasattr(eng, '_stdin_stack'):
            eng._stdin_stack = []
        eng._stdin_stack.append(sys.stdin)
        sys.stdin = f
        return True


def bi_open_out(goal: PsiTerm, eng) -> bool:
    """open_out(File) or open_out(File, Stream) — open file for writing."""
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    if a1 is None:
        return False
    a1d = a1.deref()
    if a1d.value is not None:
        filename = str(a1d.value)
    elif a1d.type and a1d.type.keyword:
        filename = a1d.type.keyword.symbol
    else:
        return False
    if filename in ('stdout', 'stderr'):
        # `open_out(stdout, S3)` names the standard stream, which is what
        # copy_file holds on to so that closing the target file puts the
        # "done." back on the terminal.
        f = sys.__stdout__ if filename == 'stdout' else sys.__stderr__
    else:
        try:
            f = open(filename, 'w')
        except OSError:
            return False
    if a2 is not None:
        stream_term = PsiTerm()
        stream_term.value = f
        stream_term.type = eng.wl.top
        if not hasattr(eng, '_open_streams'):
            eng._open_streams = {}
        eng._open_streams[id(stream_term)] = f
        if not hasattr(eng, '_stdout_stack'):
            eng._stdout_stack = []
        eng._stdout_stack.append(sys.stdout)
        sys.stdout = f
        return _unify(eng, a2.deref(), stream_term)
    else:
        if not hasattr(eng, '_stdout_stack'):
            eng._stdout_stack = []
        eng._stdout_stack.append(sys.stdout)
        sys.stdout = f
        return True


def _stream_file(t: PsiTerm, eng):
    """The Python file object a stream psi-term stands for, or None."""
    t = t.deref()
    if t.value is not None and hasattr(t.value, 'write') or (
            t.value is not None and hasattr(t.value, 'read')):
        return t.value
    streams = getattr(eng, '_open_streams', None)
    if streams is not None and id(t) in streams:
        return streams[id(t)]
    if t.type is not None and t.type.keyword is not None:
        sym = t.type.keyword.symbol
        if sym == 'stdout':
            return sys.__stdout__
        if sym == 'stderr':
            return sys.__stderr__
        if sym == 'stdin':
            return sys.__stdin__
    return None


def bi_set_output(goal: PsiTerm, eng) -> bool:
    """set_output(Stream) — send what follows to Stream.

    The stream that was current is stacked, so closing Stream puts the output
    back where it was, which is how copy_file's "done." reaches the terminal.
    """
    a1 = goal.attr_list.get('1')
    if a1 is None:
        return False
    f = _stream_file(a1, eng)
    if f is None:
        return False
    if not hasattr(eng, '_stdout_stack'):
        eng._stdout_stack = []
    eng._stdout_stack.append(sys.stdout)
    sys.stdout = f
    return True


def bi_set_input(goal: PsiTerm, eng) -> bool:
    """set_input(Stream) — read what follows from Stream."""
    a1 = goal.attr_list.get('1')
    if a1 is None:
        return False
    f = _stream_file(a1, eng)
    if f is None:
        return False
    if not hasattr(eng, '_stdin_stack'):
        eng._stdin_stack = []
    eng._stdin_stack.append(sys.stdin)
    sys.stdin = f
    return True


def bi_close(goal: PsiTerm, eng) -> bool:
    """close(Stream) — close an open stream and restore stdin/stdout."""
    a1 = goal.attr_list.get('1')
    if a1 is None:
        return False
    a1d = a1.deref()
    f = None
    if a1d.value is not None and hasattr(a1d.value, 'close'):
        f = a1d.value
    elif hasattr(eng, '_open_streams') and id(a1d) in eng._open_streams:
        f = eng._open_streams.pop(id(a1d))
    if f is None:
        return True  # nothing to close
    try:
        f.close()
    except Exception:
        pass
    # Restore stdin if this was the current stdin
    if sys.stdin is f:
        if hasattr(eng, '_stdin_stack') and eng._stdin_stack:
            sys.stdin = eng._stdin_stack.pop()
        else:
            sys.stdin = sys.__stdin__
    # Restore stdout if this was the current stdout
    if sys.stdout is f:
        if hasattr(eng, '_stdout_stack') and eng._stdout_stack:
            sys.stdout = eng._stdout_stack.pop()
        else:
            sys.stdout = sys.__stdout__
    return True


# ─────────────────────────────────────────────────────────────────────────────
# read_token(T) — read one token from stdin, used by makestr.lf
# ─────────────────────────────────────────────────────────────────────────────

def bi_read_token(goal: PsiTerm, eng) -> bool:
    """read_token(T) — read one token from the current input stream.

    Used by makestr.lf to parse a string token written to a temp file:
      write('"'), write(X), write('"')  →  file contains "X"
      read_token(Y)                     →  Y = the quoted string "X"
    """
    arg = _get_one_arg(goal)
    # One token, and the rest of the line kept for the next call: what is
    # left on the input after it belongs to whoever reads next, and
    # eratosthenes has more queries waiting behind its `20`.
    stream = sys.stdin
    pending = getattr(eng, '_read_token_pending', None) or []
    if getattr(eng, '_read_token_stream', None) is not stream:
        # A different stream is a different queue: what was left over on the
        # one before belongs to it, not to this one.
        pending = []
        eng._read_token_stream = stream
    from wild_life.tokenizer import tokenizer_from_string
    while not pending:
        try:
            content = stream.readline()
        except (EOFError, KeyboardInterrupt, AttributeError):
            content = ''
        if not content:
            break
        ts = tokenizer_from_string(content)
        for _ in range(64):
            try:
                tok = ts.read_token_b()
            except Exception:
                break
            if tok is None:
                break
            _sym = (tok.type.keyword.symbol
                    if (tok.type is not None and tok.type.keyword) else '')
            if tok.value is None and _sym in ('eof', 'end_of_file'):
                break
            pending.append(tok)
            if ts.eof_flag:
                break
    eng._read_token_pending = pending
    if not pending:
        return False
    tok = pending.pop(0)
    if arg is None:
        return True
    return _unify(eng, arg.deref(), tok)


# ─────────────────────────────────────────────────────────────────────────────
# system(Command) — run a shell command
# ─────────────────────────────────────────────────────────────────────────────

def bi_system(goal: PsiTerm, eng) -> bool:
    """system(Command) — execute a shell command.

    Unifies the result (exit code as integer) with the second argument if present.
    Used by makestr.lf to remove the temp file: @=system("rm lifebuff").
    """
    import subprocess as _subp
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    if a1 is None:
        return False
    a1d = a1.deref()
    if a1d.value is not None:
        cmd = str(a1d.value)
    elif a1d.type and a1d.type.keyword:
        cmd = a1d.type.keyword.symbol
    else:
        return False
    try:
        ret = _subp.call(cmd, shell=True)
    except Exception:
        ret = -1
    if a2 is not None:
        wl = eng.wl
        result = wl.make_integer(ret)
        return _unify(eng, a2.deref(), result)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# map(F, List) → ResultList  (functional built-in)
# ─────────────────────────────────────────────────────────────────────────────

def _keep_call_value(call: PsiTerm, value: PsiTerm, eng) -> PsiTerm:
    """Let a call stand for what it answered, so every reader sees the one term.

    `entries(Square:grid)` reads the grid to run its rows together, and the
    squares it hands back are the ones Square holds: reading grid a second
    time would make a fresh square, and assigning a number to one of those
    would leave the first untouched.
    """
    if call is None or value is None or eng is None:
        return value
    _vd = value.deref()
    if _vd is call or call.coref is not None or not _is_user_function(call):
        return _vd
    eng.trail.trail_psi(call, 'coref')
    call.coref = _vd
    return _vd


def _eval_map_or_reduce(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """What a map or reduce written where a value belongs comes to."""
    if t is None or eng is None or t.type is None or t.type.keyword is None:
        return None
    _sym_mr = t.type.keyword.symbol
    if _sym_mr == 'map':
        if ('1' in t.attr_list and '2' in t.attr_list
                and '3' not in t.attr_list):
            return _eval_map_func(t, eng)
        return None
    if _sym_mr == 'reduce':
        if ('1' in t.attr_list and '2' in t.attr_list
                and '3' in t.attr_list and '4' not in t.attr_list):
            return _eval_reduce_func(t, eng)
    return None


def _eval_map_func(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """Evaluate map(F, List) functionally, returning the mapped list as a PsiTerm.

    Tries user-defined functions first (via _try_eval_any_func), then arithmetic,
    then string functions, and falls back to the unevaluated application term.
    Returns None if evaluation cannot proceed.
    """
    wl = eng.wl
    a1 = t.attr_list.get('1')  # F
    a2 = t.attr_list.get('2')  # List
    if a1 is None or a2 is None:
        return None
    f_term = a1.deref()
    list_term = a2.deref()

    # Evaluate the list argument if it's a function call (e.g. features(X))
    _list_ev = _try_eval_any_func(list_term, eng)
    if _list_ev is not None and _list_ev is not list_term:
        list_term = _keep_call_value(list_term, _list_ev, eng)

    results = []
    node = list_term
    while True:
        node = node.deref()
        sym = node.type.keyword.symbol if (node.type and node.type.keyword) else ''
        if sym in ('nil', '[]') or (node.value is None and not node.attr_list and node.type is wl.nil):
            break
        if sym in ('cons', '.', '|') or node.type is wl.alist:
            head_ref = node.attr_list.get('1')
            tail_ref = node.attr_list.get('2')
            if head_ref is None:
                break
            head = head_ref.deref()
            applied = _apply_func(f_term, head, eng)
            if applied is None:
                return None
            # Try user-defined function first (handles feature_value etc.)
            result = _try_eval_any_func(applied, eng)
            if result is not None:
                results.append(result)
            else:
                ok, val = _eval_arith(applied, eng)
                if ok:
                    results.append(_make_number(eng, val))
                else:
                    str_result = _try_eval_string_func(applied, eng)
                    if str_result is not None:
                        results.append(str_result)
                    else:
                        results.append(applied)
            node = tail_ref if tail_ref is not None else wl.make_atom('nil', wl.bi_module)
        elif (node.value is None and not node.attr_list
              and (node.type is None or node.type is wl.top)):
            # Nothing has said what the list is yet, so there is nothing to
            # walk: `[X|map(F,L)] | g(L)` reaches the map before the guard
            # has given L its list, and mapping over the variable itself
            # would put an F of it into the answer.  Leave the call as it
            # stands and it is worked out once L arrives.
            return None
        else:
            # Not a proper cons cell — apply to the element directly
            applied = _apply_func(f_term, node, eng)
            if applied is None:
                return None
            result = _try_eval_any_func(applied, eng)
            if result is not None:
                results.append(result)
            else:
                ok, val = _eval_arith(applied, eng)
                results.append(_make_number(eng, val) if ok else applied)
            break

    return wl.make_list(results)


def _apply_func(f_term: PsiTerm, arg: PsiTerm, eng) -> Optional[PsiTerm]:
    """Apply functor f_term to one argument, returning the result term.

    In Wild Life, F(X) is written as a psi-term whose type is F and whose
    '1' attribute is X.  For partial applications like *(2=>4), F already
    carries some attributes — we merge the new positional arg into position
    '1' (or the next free position).
    """
    # Build a fresh application node with arg placed into the first available
    # positional slot: if f_term has no '1', use '1'; otherwise use '2', etc.
    # The copy is shallow on purpose — a partial application captures the
    # psi-terms already bound to it, so `feature_value(2 => X)` must keep the
    # very node X, not a clone of it, or `X.A` would reach a fresh cell and
    # every coreference in X would be lost.
    f_d = f_term.deref()
    f_copy = PsiTerm(type_def=f_d.type, value=f_d.value,
                     attr_list=dict(f_d.attr_list))
    f_copy.flags = f_d.flags
    if '1' not in f_copy.attr_list:
        f_copy.attr_list['1'] = arg
    elif '2' not in f_copy.attr_list:
        f_copy.attr_list['2'] = arg
    else:
        f_copy.attr_list['1'] = arg
    return f_copy


def _eval_reduce_func(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """reduce(F, E, List) — fold the list from the right with F.

    `reduce(F,E,[H|T]) -> F(H, reduce(F,E,T))`, and `reduce(F,E,[]) -> E`,
    which is how `sum_up(L) -> reduce((+),0,L)` adds a list up and how
    `entries(S) -> reduce(append,[],S)` runs a list of lists together.
    """
    a1 = t.attr_list.get('1')   # F
    a2 = t.attr_list.get('2')   # E
    a3 = t.attr_list.get('3')   # List
    if a1 is None or a2 is None or a3 is None:
        return None
    f_term = a1.deref()
    lst = a3.deref()
    _lst_ev = _try_eval_any_func(lst, eng)
    if _lst_ev is not None and _lst_ev.deref() is not lst:
        lst = _keep_call_value(lst, _lst_ev, eng)
    elems = _proper_list_elems(lst, eng)
    if elems is None:
        return None
    acc = a2.deref()
    for _h in reversed(elems):
        applied = _apply_func(f_term, _h.deref(), eng)
        if applied is None:
            return None
        applied = _apply_func(applied, acc, eng)
        if applied is None:
            return None
        _v = _try_eval_any_func(applied, eng)
        if _v is None:
            _ok, _n = _eval_arith(applied, eng)
            _v = _make_number(eng, _n) if _ok else None
        acc = (_v.deref() if _v is not None else applied)
    return acc


def bi_reduce(goal: PsiTerm, eng) -> bool:
    """reduce(F, E, List[, Result]) — the predicate form of the fold."""
    if '4' in goal.attr_list:
        _call = PsiTerm(type_def=goal.type)
        _call.attr_list = {k: v for k, v in goal.attr_list.items() if k != '4'}
        _val = _eval_reduce_func(_call, eng)
        if _val is None:
            return False
        return _unify(eng, goal.attr_list['4'].deref(), _val)
    return _eval_reduce_func(goal, eng) is not None


def bi_map(goal: PsiTerm, eng) -> bool:
    """map(F, List) → MappedList  — apply function F to each element.

    Supports:
      map(F, List, Result)  — 3-arg predicate form
      X = map(F, List)      — 2-arg functional form (via bi_unify)
    """
    wl = eng.wl
    a1 = goal.attr_list.get('1')  # F
    a2 = goal.attr_list.get('2')  # List
    a3 = goal.attr_list.get('3')  # Result (optional)
    if a1 is None or a2 is None:
        return False
    f_term = a1.deref()
    list_term = a2.deref()

    # Walk the list
    results = []
    node = list_term
    while True:
        node = node.deref()
        sym = node.type.keyword.symbol if (node.type and node.type.keyword) else ''
        if sym in ('nil', '[]') or (node.value is None and not node.attr_list and node.type is wl.nil):
            break
        if sym in ('cons', '.', '|') or node.type is wl.alist:
            head_ref = node.attr_list.get('1')
            tail_ref = node.attr_list.get('2')
            if head_ref is None:
                break
            head = head_ref.deref()
            applied = _apply_func(f_term, head, eng)
            if applied is None:
                return False
            # Evaluate the applied function
            ok, val = _eval_arith(applied, eng)
            if ok:
                results.append(_make_number(eng, val))
            else:
                # Try string evaluation
                str_result = _try_eval_string_func(applied, eng)
                if str_result is not None:
                    results.append(str_result)
                else:
                    # A rule of the program's own says what the function
                    # answers: `map(term_explore(2 => Seen), FV)` asks
                    # term_explore of each feature value.
                    _uf = (_eval_user_func_sync(applied, eng, 0)
                           if _is_user_function(applied) else None)
                    if _uf is not None and _uf.deref() is not applied:
                        results.append(_uf)
                    else:
                        # Leave as unevaluated application term
                        results.append(applied)
            node = tail_ref if tail_ref is not None else wl.make_atom('nil', wl.bi_module)
        else:
            # Not a list — apply to the single element
            applied = _apply_func(f_term, node, eng)
            if applied is None:
                return False
            ok, val = _eval_arith(applied, eng)
            results.append(_make_number(eng, val) if ok else applied)
            break

    result_list = wl.make_list(results)
    if a3 is not None:
        return _unify(eng, a3.deref(), result_list)
    # 2-arg form used as function — the call site (bi_unify) handles unification
    # by calling bi_map and using the return value; since we can't return a term
    # from a bool function, we need the goal's result to be accessible.
    # Workaround: unify '0' attribute (return slot) if present, else fail.
    ret_slot = goal.attr_list.get('0')
    if ret_slot is not None:
        return _unify(eng, ret_slot.deref(), result_list)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# residuate(X) — force X to display as X~ (pending residuation)
# ─────────────────────────────────────────────────────────────────────────────

def bi_residuate(goal: PsiTerm, eng) -> bool:
    """residuate(X) — mark X as having a pending residuation (show X~)."""
    from wild_life.unification import Residuation
    a1 = goal.attr_list.get('1')
    if a1 is None:
        return True
    v = a1.deref()
    if v.resid is None:
        eng.trail.trail_psi(v, 'resid')
        v.resid = [Residuation(pending=True)]
    elif not any(getattr(r, 'pending', False) for r in v.resid):
        eng.trail.trail_psi(v, 'resid')
        v.resid = list(v.resid) + [Residuation(pending=True)]
    return True


# ─────────────────────────────────────────────────────────────────────────────
# global(X1, X2, ...) — declare mutable global variables
# ─────────────────────────────────────────────────────────────────────────────

def bi_global(goal: PsiTerm, eng) -> bool:
    """global(X1, X2, ...) — declare and optionally initialise global variables.

    Each argument is either a name to declare, or `X <- Value` to declare X
    with an initial value.  A name that already means something else cannot
    become a global, and neither can a literal, so every argument is checked
    before any of them is declared: one bad argument leaves the whole
    declaration undone, which is why `global(q1,...,5,...,q8)` declares none
    of the q's.
    """
    wl = eng.wl
    from wild_life.data_structures import DefType as _DT

    def _reject(what: str) -> None:
        line = getattr(wl, 'line_count', 0)
        sys.stderr.write(f"*** Error: {what} (near line {line}).\n")

    _KIND = {_DT.FUNCTION: 'function', _DT.TYPE: 'sort', _DT.PREDICATE: 'predicate'}

    def _target(a):
        """The name `a` would declare, or None once the reason is reported."""
        a = a.deref()
        sym = a.type.keyword.symbol if (a.type and a.type.keyword) else ''
        if sym == '<-':
            if a.attr_list.get('2') is None or a.attr_list.get('1') is None:
                _reject(f"{_term_to_str(a, eng)} is an incorrect global "
                        f"variable declaration")
                return None
            a = a.attr_list['1'].deref()
            sym = a.type.keyword.symbol if (a.type and a.type.keyword) else ''
        disp = _term_to_str(a, eng)
        if a.value is not None or not sym:
            # A literal stands for its own sort, so it cannot name a global.
            _reject(f"sort {disp} cannot be redeclared as a global variable")
            return None
        defn = wl.current_module.symbol_table.get(sym)
        if defn is None and a.type is not None:
            defn = a.type
        kind = _KIND.get(defn.type) if defn is not None else None
        # An arithmetic operator is a function whatever its symbol table entry
        # says: it is only registered there as an operator.
        if kind is None and sym in _ARITH_OPS_SET:
            kind = 'function'
        if kind is not None:
            _reject(f"{kind} {disp} cannot be redeclared as a global variable")
            return None
        return sym

    def _cell_for(sym: str) -> PsiTerm:
        """The cell named sym, declaring it on first mention."""
        defn = wl.update_symbol(wl.current_module, sym)
        if defn.type is not _DT.GLOBAL or defn.global_value is None:
            defn.type = _DT.GLOBAL
            defn.global_value = PsiTerm(type_def=wl.top)
        if defn not in wl.global_defs:
            wl.global_defs.append(defn)
        return defn.global_value

    def _declare(a) -> None:
        a = a.deref()
        sym = a.type.keyword.symbol if (a.type and a.type.keyword) else ''
        value = None
        if sym == '<-':
            lhs = a.attr_list['1'].deref()
            sym = lhs.type.keyword.symbol if (lhs.type and lhs.type.keyword) else ''
            rhs_d = a.attr_list['2'].deref()
            # `global(b <- a)` makes b share a's cell rather than hold a's
            # name, so whatever binds a later is what b reads.
            value = _global_cell(rhs_d, eng)
            if value is None:
                ok, val = _eval_arith(rhs_d, eng)
                value = _make_number(eng, val) if ok else rhs_d
        # A global is a cell, not a rule: every reference reads the same
        # psi-term, so binding it through one name is visible through all the
        # others (`global(a, b<-a)` then `a=23` shows 23 for b too).
        if value is not None:
            defn = wl.update_symbol(wl.current_module, sym)
            _note_global_used(eng, defn)
            defn.global_value = value

    args = []
    i = 1
    while True:
        arg_ref = goal.attr_list.get(str(i))
        if arg_ref is None:
            break
        args.append(arg_ref)
        i += 1

    names = [_target(arg) for arg in args]
    if any(name is None for name in names):
        return False
    # Give every name its cell before any initial value is worked out, so that
    # `global(e <- f, f)` lets e share the cell f is about to get.
    for name in names:
        _cell_for(name)
    for arg in args:
        _declare(arg)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# writeq_err / put_err — write to stderr
# ─────────────────────────────────────────────────────────────────────────────

def bi_writeq_err(goal: PsiTerm, eng) -> bool:
    """writeq_err(T) — write T in quoted form to stderr."""
    return _write_all_args(goal, eng, quoted=True, stream=sys.stderr)


def bi_put_err(goal: PsiTerm, eng) -> bool:
    """put_err(C) — write character C to stderr."""
    a1 = goal.attr_list.get('1')
    if a1 is None:
        return False
    a1d = a1.deref()
    c = None
    if a1d.value is not None:
        v = a1d.value
        if isinstance(v, (int, float)):
            c = chr(int(v))
        else:
            c = str(v)[0] if str(v) else ''
    elif a1d.type and a1d.type.keyword:
        s = a1d.type.keyword.symbol
        c = s[0] if s else ''
    if c is not None:
        sys.stderr.write(c)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel used inside inference.py
# ─────────────────────────────────────────────────────────────────────────────
from wild_life.inference import _DEFRULES, _INNER_RUN_BARRIER
_DEFRULES_SENTINEL = _DEFRULES


# ─────────────────────────────────────────────────────────────────────────────
# Registration helper
# ─────────────────────────────────────────────────────────────────────────────

def register_all(wl) -> None:
    """Register all built-in predicates on the runtime wl."""
    _reg = wl.new_built_in

    # I/O
    _reg('write', bi_write)
    _reg('pretty_write', bi_pretty_write)   # pretty_write uses pretty-printing
    _reg('writeq', bi_writeq)
    _reg('pretty_writeq', bi_pretty_writeq)  # pretty_writeq uses pretty-printing
    _reg('write_canonical', bi_write_canonical)
    _reg('print', bi_print)
    _reg('print_depth', bi_print_depth)
    _reg('page_width', bi_page_width)
    _reg('nl', bi_nl)
    _reg('write_err', bi_write_err)
    _reg('writeln', bi_writeln)
    _reg('put', bi_put_char)
    _reg('put_char', bi_put_char)
    _reg('get_char', bi_get_char)
    _reg('get', bi_get_code)
    _reg('read', bi_read)
    _reg('read_term', bi_read_term)
    _reg('parse', bi_parse)
    _reg('format', bi_format)
    _reg('nl_err', bi_nl_err)
    _reg('with_output_to', bi_with_output_to)
    _reg('writeq_err', bi_writeq_err)
    _reg('put_err', bi_put_err)
    # File stream I/O
    _reg('open_in', bi_open_in)

    def _bi_readf(goal, eng):
        """readf(File[, L]) — L is the file's characters, as their codes."""
        _call = PsiTerm(type_def=goal.type)
        _call.attr_list = {'1': goal.attr_list['1']} if '1' in goal.attr_list else {}
        _val = _try_eval_string_func(_call, eng)
        if _val is None:
            return False
        _out = goal.attr_list.get('2')
        if _out is None:
            return True
        return _unify(eng, _out.deref(), _val)
    _reg('readf', _bi_readf, def_type=DefType.FUNCTION)
    _reg('open_out', bi_open_out)
    _reg('close', bi_close)
    _reg('set_output', bi_set_output)
    _reg('set_input', bi_set_input)
    _reg('read_token', bi_read_token)
    _reg('system', bi_system)

    # Arithmetic
    _reg('is', bi_is)
    _reg('=:=', bi_arith_eq)
    _reg('=\\=', bi_arith_ne)
    _reg('<', bi_arith_lt)
    _reg('=<', bi_arith_le)
    _reg('>', bi_arith_gt)
    _reg('>=', bi_arith_ge)

    # String comparison  (A$>B  A$>=B  A$<B  A$=<B  A$==B  A$\==B)
    _reg('$>', bi_str_gt)
    _reg('$>=', bi_str_ge)
    _reg('$<', bi_str_lt)
    _reg('$=<', bi_str_le)
    _reg('$==', bi_str_eq)
    _reg('$\\==', bi_str_ne)

    # Unification
    _reg('=', bi_unify)
    _reg('\\=', bi_not_unify)
    _reg('==', bi_identical)
    _reg('\\==', bi_not_identical)
    _reg(':\\==', bi_not_identical)   # Flag:\==error colon-form alias
    _reg('compare', bi_compare)
    _reg('<-', bi_store_arrow)      # destructive assignment
    _reg('<<-', bi_store_arrow)     # strict destructive assignment (same semantics)

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
    _reg('repeat', bi_repeat)
    _reg('not', bi_not)
    _reg('\\+', bi_not)
    _reg('and', bi_and)
    _reg('or', bi_or)
    _reg('call', bi_call)
    _reg('implies', bi_implies)
    _reg('once', bi_once)
    _reg('call_once', bi_call_once)
    _reg('cond', bi_cond)
    _reg('findall', bi_findall)
    _reg('bagof', bi_findall)   # simplified
    _reg('setof', bi_findall)   # simplified
    _reg('aggregate_all', bi_aggregate_all)
    # Tracing / debugging
    _reg('trace', bi_trace)
    _reg('notrace', bi_notrace)
    _reg('spy', bi_trace)       # simplified: spy = trace
    _reg('nospy', bi_notrace)   # simplified: nospy = notrace
    # Higher-order
    _reg('map', bi_map)
    _reg('reduce', bi_reduce)

    def _bi_maprel(goal, eng):
        """maprel(P, List) — prove P of each element, left to right.

        built_ins.lf says it in LIFE: `maprel(P,[H|T]) :- !, root_sort(P) &
        @(H), maprel(P,T).`
        """
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None or a2 is None:
            return False
        elems = _proper_list_elems(a2.deref(), eng)
        if elems is None:
            return False
        _p = a1.deref()
        # Pushed last first, so the list is gone through in order.
        for _e in reversed(elems):
            _call = _apply_func(_p, _e.deref(), eng)
            if _call is None:
                return False
            eng.push_goal(GoalType.PROVE, _call, _DEFRULES_SENTINEL, None)
        return True
    _reg('maprel', _bi_maprel)

    def _bi_mresiduate(goal, eng):
        """mresiduate(List, Goal) — wait on every term in List at once.

        The goal is proven as soon as any one of them is given something,
        and only that once: `mresiduate([X,Y], write(qwe))` writes qwe when
        X is bound, and says nothing more when Y is bound after it.  A term
        that is not a list of its own — an unfinished list, a disjunction —
        is nothing to wait on, so the call fails.
        """
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None or a2 is None:
            return False
        from wild_life.data_structures import (
            Goal as _MRGoal, Residuation as _MRResid, SORT_VAR as _MR_SV)
        _pending = _MRGoal(GoalType.PROVE, a2.deref(), _DEFRULES_SENTINEL,
                           None, pending=True)
        # Walked a cell at a time, and each term met is given the goal to
        # wait for as it is met: a list that turns out not to end in `[]` is
        # reported with the tildes the walk so far has put on it.
        _node = a1.deref()
        _seen_mr: set = set()
        while True:
            if _node.type is eng.wl.nil or _get_sym(_node) in ('nil', '[]'):
                return True
            if (not _is_list_term(_node, eng) or id(_node) in _seen_mr
                    or '1' not in _node.attr_list
                    or '2' not in _node.attr_list):
                import sys as _sys_mr
                _sys_mr.stderr.write(
                    "*** Error: %s should be a nil-terminated list in "
                    "mresiduate.\n" % _term_to_str(a1.deref(), eng))
                return False
            _seen_mr.add(id(_node))
            _v = _node.attr_list['1'].deref()
            _node = _node.attr_list['2'].deref()
            if _v.resid is None:
                eng.trail.trail_psi(_v, 'resid')
                _v.resid = [_MRResid(goal=_pending)]
            elif not any(r.goal is _pending for r in _v.resid):
                eng.trail.trail_copy(_v, 'resid')
                _v.resid = list(_v.resid) + [_MRResid(goal=_pending)]
            # A plain variable is marked bindable so the unifier goes on
            # binding it although it now carries a residuation; a term that
            # already is something must not start looking like a variable.
            _v_plain = (not _v.attr_list and _v.value is None
                        and (_v.type is None or _v.type is eng.wl.top))
            if _v_plain and not (_v.flags & _MR_SV):
                eng.trail.trail_psi(_v, 'flags')
                _v.flags |= _MR_SV
    _reg('mresiduate', _bi_mresiduate)
    # Residuation
    _reg('residuate', bi_residuate)
    # Globals
    _reg('global', bi_global)

    # Assert / retract
    _reg('assert', bi_assert)
    _reg('assertz', bi_assert)
    _reg('asserta', bi_asserta)
    _reg('retract', bi_retract)
    _reg('abolish', bi_abolish)
    _reg('clause', bi_clause)
    _reg('setq', bi_setq)
    _reg('listing', bi_listing)

    # Type hierarchy
    _reg('children', bi_children)

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
    _reg('strlen', bi_string_length)   # alias: strlen(String, Len)
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
    _reg('initrandom', bi_initrandom)

    # Sort comparison — these compare the sorts of their two arguments.  They
    # are declared as operators in the syntax module, so that is where their
    # definitions belong.
    for _sc_name, _sc_fn in (
            (':==', bi_sort_eq), (':\\==', bi_sort_ne),
            (':=<', bi_sort_le), (':<', bi_sort_lt),
            (':>=', bi_sort_ge), (':>', bi_sort_gt),
            (':\\=<', bi_sort_not_le), (':\\<', bi_sort_not_lt),
            (':\\>=', bi_sort_not_ge), (':\\>', bi_sort_not_gt),
            (':><', bi_sort_comparable), (':\\><', bi_sort_incomparable)):
        _reg(_sc_name, _sc_fn, module=wl.syntax_module)

    # System
    _reg('halt', bi_halt)
    _reg('quit', bi_halt)   # alias for halt (not in original Wild Life)
    _reg('abort', bi_abort)
    # gc — garbage collection (memory management).  In Wild Life, heap
    # compaction discards choice points that contain stale retract (DEL_CLAUSE)
    # backtrack information, but leaves regular PROVE/CLAUSE choice points
    # (which point to live clause lists) intact.  We model this by removing
    # DEL_CLAUSE choice points from the stack while keeping all others.
    def _bi_gc(goal, eng):
        from wild_life.data_structures import GoalType as _GoalType
        # Walk the choice stack and filter out DEL_CLAUSE nodes.
        # The stack is a singly-linked list (newest at head, oldest at tail).
        # Build a new list of kept nodes, then relink them.
        kept = []
        cp = eng.choice_stack
        while cp is not None:
            gs = cp.goal_stack
            if gs is None or gs.type != _GoalType.DEL_CLAUSE:
                kept.append(cp)
            cp = cp.next
        # Relink the kept nodes
        if kept:
            for i in range(len(kept) - 1):
                kept[i].next = kept[i + 1]
            kept[-1].next = None
            eng.choice_stack = kept[0]
        else:
            eng.choice_stack = None
        return True
    _reg('gc', _bi_gc)
    _reg('garbage_collect', _bi_gc)
    _reg('load', bi_load)

    def _bi_chdir(goal, eng):
        """chdir(Dir) — make Dir the current directory."""
        arg = _get_one_arg(goal)
        if arg is None:
            return False
        _d = _get_str_val(arg.deref(), eng)
        if not _d:
            return False
        import os as _os_cd
        try:
            _os_cd.chdir(_d)
        except OSError:
            return False
        return True
    _reg('chdir', _bi_chdir)

    def _bi_getenv(goal, eng):
        """getenv(Name, Value) — what the environment says Name is worth."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        _n = _get_str_val(a1.deref(), eng)
        if _n is None:
            return False
        import os as _os_ge2
        _v = _os_ge2.environ.get(_n)
        if _v is None:
            return False
        return _unify(eng, a2.deref(), _make_string(eng, _v)) if a2 else True
    _reg('getenv', _bi_getenv)
    _reg('op', bi_op)
    _reg('statistics', bi_statistics)
    _reg('current_prolog_flag', bi_current_prolog_flag)
    _reg('set_prolog_flag', bi_set_prolog_flag)

    # Type hierarchy
    _reg('sub_type', bi_sub_type)
    _reg('subsort', bi_subsort)    # subsort(A,B): A:=<B with residuation on B
    _reg('get_attribute', bi_get_attribute)
    _reg('set_attribute', bi_set_attribute)
    _reg('type_of', bi_type_of)
    _reg('functor_of', bi_functor_of)

    # Alias / sort manipulation
    _reg('substitute', bi_substitute)   # substitute(A,B,X): replace sort A with B in X
    _reg('alias', bi_alias)

    # ── LIFE meta-predicates (no-ops or minimal stubs) ─────────────────────
    def _bi_non_strict(goal, eng):
        """non_strict(P): mark P as non-strict (lazy evaluation of arguments)."""
        arg = goal.attr_list.get('1')
        if arg is None:
            return False
        arg = arg.deref()
        if arg.type is not None and arg.type is not eng.wl.top:
            # Register this sort/function as non-strict
            if not hasattr(eng, 'non_strict_set'):
                eng.non_strict_set = set()
            eng.non_strict_set.add(arg.type)
        return True
    _reg('non_strict', _bi_non_strict)

    def _bi_delay_check(goal, eng):
        """delay_check(S, …): hold S's prototype and delay rules until a term
        of that sort is modified.

        Without it a term reaching S takes S's prototype and fires its rules
        straight away.  Under it merely being S is not yet the final word on
        what the term is, so `A = person` stays person, and the rules run once
        the term gains a feature.  Several sorts may be named in one call.
        """
        i = 1
        while True:
            arg = goal.attr_list.get(str(i))
            if arg is None:
                break
            arg_d = arg.deref()
            if arg_d.type is not None:
                arg_d.type.always_check = False
            i += 1
        return True
    _reg('delay_check', _bi_delay_check)

    def _bi_dynamic(goal, eng):
        """dynamic(P, …): declare each P dynamic, with an empty rule list."""
        from wild_life.data_structures import featcmp_key as _fck_dyn
        if not goal.attr_list:
            return True
        # One declaration may name several predicates: cb writes
        # `dynamic(varcount, elim)?` and means both of them.
        for _k in sorted(goal.attr_list, key=_fck_dyn):
            arg = goal.attr_list[_k].deref()
            # If the type has no rule, set it to an empty list so
            # assert/retract work
            if arg.type:
                if arg.type.rule is None:
                    arg.type.rule = []
                # listing prints a `dynamic(P)?` header for a predicate
                # declared this way, so that its listing can be read back in.
                arg.type.is_dynamic = True
                arg.type.is_static = False
        return True
    _reg('dynamic', _bi_dynamic)

    def _bi_static(goal, eng):
        """static(P, …): close P's definition.

        A static predicate takes no more clauses and gives none up: asserting
        one is quietly accepted and changes nothing, and retracting fails.
        """
        i = 1
        while True:
            arg = goal.attr_list.get(str(i))
            if arg is None:
                break
            i += 1
            arg_d = arg.deref()
            if arg_d.type is not None:
                arg_d.type.is_static = True
                arg_d.type.is_dynamic = False
        return True
    _reg('static', _bi_static)

    def _bi_persistent(goal, eng):
        """persistent(X1, X2, ...) — declare global variables that keep their
        value across garbage collection.

        Each name's definition is initialised as a FUNCTION with an empty rule
        list, so that a later `X <<- Value` takes the global-variable path in
        bi_store_arrow, and is recorded as a global so that a query reading one
        is worth keeping.
        """
        if not goal.attr_list:
            return False
        # A name that already has a definition of its own cannot become a
        # global, and one bad name refuses the whole declaration: after
        # `d -> 4`, `persistent(a,…,d,…,j)` declares none of them.
        names: list = []
        i = 1
        while True:
            arg = goal.attr_list.get(str(i))
            if arg is None:
                break
            i += 1
            arg_d = arg.deref()
            defn = arg_d.type
            if defn is None:
                continue
            if (defn.rule and defn.type in (DefType.FUNCTION, DefType.PREDICATE)
                    and not getattr(defn, 'is_persistent', False)):
                kind = ('function' if defn.type == DefType.FUNCTION
                        else 'predicate')
                name = defn.keyword.symbol if defn.keyword else '?'
                sys.stderr.write(
                    f"*** Error: {kind} {name} cannot be redeclared persistent"
                    f" (near line {getattr(wl, 'line_count', 0)}).\n")
                return False
            names.append(defn)
        for defn in names:
            if defn.rule is None:
                defn.rule = []
            if defn.type not in (DefType.FUNCTION, DefType.PREDICATE):
                defn.type = DefType.FUNCTION
            defn.is_persistent = True
            if defn not in wl.global_defs:
                wl.global_defs.append(defn)
        return True
    _reg('persistent', _bi_persistent)

    # ── quiet — whether the interpreter was asked to keep quiet ───────────
    def _bi_quiet(goal, eng):
        """quiet — succeeds when the interpreter is running quietly.

        The library files ask it before each warning they would print.
        """
        return bool(getattr(wl, 'quietflag', False))
    _reg('quiet', _bi_quiet)

    def _bi_print_variables(goal, eng):
        """print_variables — write out the variables the session holds."""
        from wild_life.print_term import print_variables as _pv_bi, \
            PRINT_DEPTH as _PD_bi
        merged: dict = {}
        for vt in (getattr(eng, '_frame_var_trees', None) or []):
            if vt:
                merged.update(vt)
        own = getattr(eng, '_last_var_tree', None)
        if own:
            merged.update(own)
        if not merged:
            return True
        _pv_bi(merged, outfile=sys.stdout, wl=wl,
               print_depth=getattr(wl, 'print_depth', _PD_bi))
        sys.stdout.write("\n")
        return True
    _reg('print_variables', _bi_print_variables)

    def _bi_delay_until(goal, eng):
        """delay_until(Cond,Goal): simplified — just try to prove Goal immediately."""
        from wild_life.data_structures import GoalType as _GT
        arg1 = goal.attr_list.get('1')
        arg2 = goal.attr_list.get('2')
        if arg2:
            eng.push_goal(_GT.PROVE, arg2.deref(), None, None)
        return True
    _reg('delay_until', _bi_delay_until)

    # ── LIFE string built-ins ─────────────────────────────────────────────────
    def _bi_psi2str(goal, eng):
        """psi2str(T, S): S = string representation of T."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        a1 = a1.deref()
        s = _term_to_display_string(a1, eng)
        result = _make_string(eng, s)
        if a2 is None:
            # Unary form psi2str(T): print and return
            return True
        return _unify(eng, a2.deref(), result)
    _reg('psi2str', _bi_psi2str)

    def _bi_makestr(goal, eng):
        """makestr(T, S): S = compact string representation of T (C Wild Life built-in).
        Unbound variables are represented as "@".
        Used as function: makestr(T) = S in queries.
        """
        import io as _io
        from wild_life.print_term import write_term as _write_term
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        a1 = a1.deref()
        if _term_is_unbound(a1, eng):
            s = '@'
        else:
            _buf = _io.StringIO()
            _write_term(a1, outfile=_buf, wl=eng.wl, quoted=False, max_col=1_000_000)
            s = _buf.getvalue()
        result = _make_string(eng, s)
        if a2 is None:
            return True
        return _unify(eng, a2.deref(), result)
    _reg('makestr', _bi_makestr)

    def _bi_str2psi(goal, eng):
        """str2psi(S, T): T = atom parsed from string S."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        a1 = a1.deref()
        if a1.type and a1.type is eng.wl.quoted_string and a1.value is not None:
            name = str(a1.value)
        elif a1.type and a1.type.keyword:
            name = a1.type.keyword.symbol
        else:
            name = _term_to_display_string(a1, eng)
        result = _make_atom(eng, name)
        if a2 is None:
            return True
        return _unify(eng, a2.deref(), result)
    _reg('str2psi', _bi_str2psi)

    def _bi_strcon(goal, eng):
        """strcon(A, B, C): C = A ++ B (string concatenation)."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        a3 = goal.attr_list.get('3')
        if a1 is None or a2 is None:
            return False
        a1, a2 = a1.deref(), a2.deref()
        # Only evaluate when at least one string is concrete
        def _is_str(x):
            return x.value is not None and x.type is not None and x.type.is_subtype_of(eng.wl.quoted_string)
        if not (_is_str(a1) or _is_str(a2)):
            return False
        # Evaluate nested string funcs
        a1e = _try_eval_string_func(a1, eng)
        if a1e is not None: a1 = a1e
        a2e = _try_eval_string_func(a2, eng)
        if a2e is not None: a2 = a2e
        s1 = str(a1.value) if a1.value is not None else ''
        s2 = str(a2.value) if a2.value is not None else ''
        result = _make_string(eng, s1 + s2)
        if a3 is None:
            return True
        return _unify(eng, a3.deref(), result)
    _reg('strcon', _bi_strcon)

    def _bi_substr(goal, eng):
        """substr(String, Start, Length, Result): Result = substring of String."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        a3 = goal.attr_list.get('3')
        a4 = goal.attr_list.get('4')
        if a1 is None or a2 is None or a3 is None:
            return False
        s_t = a1.deref()
        if s_t.value is None or s_t.type is None or not s_t.type.is_subtype_of(eng.wl.quoted_string):
            return False
        s = str(s_t.value)
        ok2, start_f = _eval_arith(a2.deref(), eng)
        ok3, length_f = _eval_arith(a3.deref(), eng)
        if not ok2 or not ok3:
            return False
        start = int(start_f) - 1  # 1-indexed to 0-indexed
        length = int(length_f)
        if start < 0:
            start = 0
        result_s = s[start:start + length] if start < len(s) else ''
        result = _make_string(eng, result_s)
        if a4 is None:
            return True
        return _unify(eng, a4.deref(), result)
    _reg('substr', _bi_substr)

    # ── LIFE type/sort built-ins ───────────────────────────────────────────────
    def _bi_root_sort(goal, eng):
        """root_sort(T, R): R = the root sort of T.
        For numeric/string atoms the root sort is the value itself."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        t = a1.deref()
        defn = t.type
        if defn is None or defn.keyword is None:
            return False
        # Unwrap backtick-quoted atoms: `foo → root sort is foo
        if defn.keyword.symbol == '`':
            inner = t.attr_list.get('1')
            if inner is not None:
                t = inner.deref()
                defn = t.type
                if defn is None or defn.keyword is None:
                    return False
        # For concrete numeric/string values the root sort is the value itself
        wl = eng.wl
        if t.value is not None:
            if defn.is_subtype_of(wl.integer):
                result = wl.make_integer(int(t.value))
            elif defn.is_subtype_of(wl.real):
                result = wl.make_number(float(t.value))
            elif defn.is_subtype_of(wl.quoted_string):
                result = _make_string(eng, str(t.value))
            else:
                result = wl.make_atom(defn.keyword.symbol, wl.user_module)
        else:
            result = wl.make_atom(defn.keyword.symbol, wl.user_module)
        if a2 is None:
            return True
        return _unify(eng, a2.deref(), result)
    _reg('root_sort', _bi_root_sort)

    # ── combined_name(T[, N]) — the sort's name with its module ───────────
    def _bi_combined_name(goal, eng):
        """combined_name(T[, N]) — N is T's sort named with its module."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        defn = a1.deref().type
        if defn is None or defn.keyword is None:
            return False
        result = wl.make_atom(defn.keyword.combined_name, wl.user_module)
        if a2 is None:
            return True
        return _unify(eng, a2.deref(), result)
    _reg('combined_name', _bi_combined_name, def_type=DefType.FUNCTION)

    def _bi_features(goal, eng):
        """features(T): return list of attribute labels of T."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        t = a1.deref()
        # Try to evaluate t first (e.g. local_time built-in)
        _t_ev = _try_eval_string_func(t, eng)
        if _t_ev is not None:
            t = _t_ev
        # Build list of attribute keys, in feature order (the same order the
        # printer uses), not in insertion order: features(@('' => A,0 => 22))
        # is ['',0].
        from wild_life.data_structures import featcmp_key as _featcmp_key
        keys = sorted(t.attr_list.keys(), key=_featcmp_key)
        # Build WL list from keys
        wl = eng.wl
        lst = wl.nil
        for key in reversed(keys):
            # Key may be numeric ("1","2") or named.
            # Negative integer keys (e.g. "-34") must be returned as atoms
            # (they display with quotes like '-34'), not as integer values.
            try:
                n = int(key)
                if n >= 0:
                    kterm = wl.make_integer(n)
                else:
                    kterm = wl.make_atom(key, wl.user_module)
            except (ValueError, TypeError):
                kterm = wl.make_atom(key, wl.user_module)
            pair = PsiTerm()
            pair.type = wl.alist
            pair.attr_list = {'1': kterm, '2': lst}
            lst = pair
        if a2 is None:
            return True
        return _unify(eng, a2.deref(), lst)
    _reg('features', _bi_features)

    def _bi_feature_values(goal, eng):
        """feature_values(T[, MOD], L) — L is the list of T's feature values."""
        _keys = sorted(goal.attr_list.keys(), key=lambda k: k)
        if '1' not in goal.attr_list:
            return False
        _out_key = '3' if '3' in goal.attr_list else (
            '2' if len(goal.attr_list) >= 2 else None)
        _call = PsiTerm(type_def=goal.type)
        _call.attr_list = {k: v for k, v in goal.attr_list.items()
                           if k != _out_key}
        _val = _try_eval_string_func(_call, eng)
        if _val is None:
            return False
        if _out_key is None:
            return True
        return _unify(eng, goal.attr_list[_out_key].deref(), _val)
    _reg('feature_values', _bi_feature_values, def_type=DefType.FUNCTION)

    def _make_strip_result(src, use_src_type, eng):
        """Core of strip / copy_pointer.

        For each *positional* attribute of *src* (keys '1', '2', ...):
          - If the current value is already an unbound variable, share it as-is.
          - Otherwise create a fresh unbound PsiTerm and set its coref to the
            current attribute value so that fresh.deref() == old_value.
        Both src's attr_list and the new result share the same fresh var objects
        for those positional keys; new attrs added to either term after the call
        will not affect the other (separate dicts).

        *src* is modified in place (trailed) to replace raw values with fresh
        variables so that the printer sees them as SHARED and emits "name: val".

        Returns the new result PsiTerm with:
          type = wl.top       (strip)
          type = src.type     (copy_pointer)
        """
        wl = eng.wl
        new_src_attrs: dict = {}
        new_res_attrs: dict = {}

        for k, v in src.attr_list.items():
            # Positional keys are numeric strings ('1', '2', ...)
            try:
                int(k)
                is_pos = True
            except (ValueError, TypeError):
                is_pos = False

            if is_pos:
                v_d = v.deref()
                if _term_is_unbound(v_d, eng):
                    # Already a free variable — share directly
                    fresh = v_d
                else:
                    # Create a fresh variable and bind it to the old value.
                    # We set coref = v (the original cell, preserving the chain)
                    # so that fresh.deref() ultimately reaches v_d.
                    # The printer deref()s before lookup, so both this fresh var
                    # in A's attr_list and the copy in B's attr_list dereference
                    # to the same underlying psiterm, making it SHARED and giving
                    # it a generated name ("_A: q" on first print, "_A" later).
                    fresh = PsiTerm()
                    fresh.coref = v  # binds fresh → v → v_d (atom / value)
                new_src_attrs[k] = fresh
                new_res_attrs[k] = fresh
            else:
                # Named (non-positional) attrs: keep in src, skip in result
                new_src_attrs[k] = v

        # Trail the entire attr_list of src so backtracking restores raw values
        eng.trail.trail_psi(src, 'attr_list')
        src.attr_list = new_src_attrs

        # Build result psiterm
        res = PsiTerm()
        res.type = src.type if use_src_type else wl.top
        res.attr_list = new_res_attrs
        return res

    def _bi_strip(goal, eng):
        """strip(S, R): R has type @, sharing S's positional args as variables.

        In LIFE, strip(S) creates a new term R with anonymous type (@) whose
        positional attributes are aliases for S's attributes.  New attributes
        added to either S or R after the call are independent.

        1-arg form: strip(S)   — just validates (always succeeds)
        2-arg form: strip(S,R) — R is the stripped copy
        """
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        src = a1.deref()
        res = _make_strip_result(src, False, eng)
        if a2 is None:
            return True
        return _unify(eng, a2.deref(), res)
    _reg('strip', _bi_strip)

    def _bi_copy_pointer(goal, eng):
        """copy_pointer(S, R): R has the same type as S, sharing positional args.

        Like strip but preserves S's sort: R.type == S.type.
        """
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        src = a1.deref()
        res = _make_strip_result(src, True, eng)
        if a2 is None:
            return True
        return _unify(eng, a2.deref(), res)
    _reg('copy_pointer', _bi_copy_pointer)

    def _bi_sort_of(goal, eng):
        """sort_of(T): synonym for root_sort."""
        return _bi_root_sort(goal, eng)
    _reg('sort', _bi_sort_of)

    def _bi_is_sort(goal, eng):
        """is_sort(T): succeed if T is a sort (type definition)."""
        from wild_life.data_structures import DefType as _DT
        a1 = _get_one_arg(goal)
        if a1 is None:
            return False
        t = a1.deref()
        defn = t.type
        return defn is not None and defn.type == _DT.TYPE
    _reg('is_sort', _bi_is_sort)

    def _bi_is_function(goal, eng):
        """is_function(T): succeed if T is a user-defined function."""
        from wild_life.data_structures import DefType as _DT
        a1 = _get_one_arg(goal)
        if a1 is None:
            return False
        t = a1.deref()
        defn = t.type
        if defn is None:
            return False
        if defn._builtin_func is not None:
            return (defn.keyword is not None
                    and defn.keyword.symbol in _BUILTIN_FUNCTION_SYMS)
        return defn.type == _DT.FUNCTION
    _reg('is_function', _bi_is_function)

    def _bi_is_predicate(goal, eng):
        """is_predicate(T): succeed if T is a user-defined predicate."""
        from wild_life.data_structures import DefType as _DT
        a1 = _get_one_arg(goal)
        if a1 is None:
            return False
        t = a1.deref()
        defn = t.type
        if defn is None:
            return False
        if defn._builtin_func is not None:
            return (defn.keyword is not None
                    and defn.keyword.symbol not in _BUILTIN_FUNCTION_SYMS)
        return defn.type == _DT.PREDICATE
    _reg('is_predicate', _bi_is_predicate)

    def _bi_glb(goal, eng):
        """glb(X, Y, Z) — Z is the GLB (unification) of X and Y.
        Also handles functional 2-arg form via bi_unify interception."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        a3 = goal.attr_list.get('3')
        if a1 is None or a2 is None:
            return False
        t = goal
        if a3 is None:
            # 2-arg predicate form called directly: just unify the two args
            return _unify(eng, a1.deref(), a2.deref())
        # 3-arg form: use _apply_glb_to_var so multiple GLBs create choice points
        return _apply_glb_to_var(goal, a3.deref(), eng)
    _reg('glb', _bi_glb)

    def _bi_lub(goal, eng):
        """lub(X, Y, Z) — Z is the LUB of types X and Y.
        Also handles functional 2-arg form via bi_unify interception."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        a3 = goal.attr_list.get('3')
        if a1 is None or a2 is None:
            return False
        t = goal
        if a3 is None:
            return True  # 2-arg with no result: trivially succeed
        return _apply_lub_to_var(goal, a3.deref(), eng)
    _reg('lub', _bi_lub)

    def _bi_children(goal, eng):
        """children(X) → list of direct subsorts of X (1-arg functional form).
        children(X, L) → L is the list of direct subsorts of X (2-arg predicate)."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        lst = _eval_children_func(goal, eng)
        if a2 is None:
            # 1-arg: used as a goal (e.g. listloop(children(@))) — succeeds
            return True
        return _unify(eng, a2.deref(), lst)
    _reg('children', _bi_children)

    # ── succeed ──────────────────────────────────────────────────────────────
    def _bi_succeed(goal, eng):
        """succeed — always succeeds (like true/0)."""
        return True
    _reg('succeed', _bi_succeed)

    # ── genint — global counter; 0-ary increments on each call ────────────
    def _bi_genint(goal, eng):
        """genint — 0-ary: increment and return the global integer counter.
        genint(N) — 1-arg: non-deterministically bind N to next counter value.
        """
        a1 = goal.attr_list.get('1')
        if a1 is None:
            # 0-ary form used as a predicate — just succeed (evaluation happens
            # via _eval_arith when genint appears in arithmetic context)
            return True
        a1d = a1.deref()
        # 1-arg form: bind N to the next counter value
        current = getattr(wl, '_genint_counter', 0) + 1
        wl._genint_counter = current
        return _unify(eng, a1d, wl.make_integer(current))
    _reg('genint', _bi_genint)

    # ── is_number(X) — true if X is a numeric value ──────────────────────
    def _bi_is_number(goal, eng):
        """is_number(X) — succeeds if X is a number (integer or real)."""
        a1 = goal.attr_list.get('1')
        if a1 is None:
            return False
        a1d = a1.deref()
        if a1d.value is None:
            return False
        return (a1d.type is not None and
                wl.real is not None and
                a1d.type.is_subtype_of(wl.real))
    _reg('is_number', _bi_is_number)

    # ── is_value(X) — true if X is a concrete (ground) value ─────────────
    def _bi_is_value(goal, eng):
        """is_value(X) — succeeds if X is a concrete value (number or string)."""
        a1 = goal.attr_list.get('1')
        if a1 is None:
            return False
        a1d = a1.deref()
        if a1d.value is None:
            return False
        return (a1d.type is not None and (
            (wl.real is not None and a1d.type.is_subtype_of(wl.real)) or
            (wl.quoted_string is not None and a1d.type.is_subtype_of(wl.quoted_string))
        ))
    _reg('is_value', _bi_is_value)

    # ── has_feature(F, T) — true if term T has feature named F ───────────
    def _bi_has_feature(goal, eng):
        """has_feature(F, T[, V]) — succeeds if T has a feature named F.

        Given a third argument it is what the feature holds, which is how
        acc_declarations.lf reads a table: `has_feature(Acc,accumulators,
        AccInfo)` asks for the entry filed under Acc and gets it.
        """
        a1 = goal.attr_list.get('1')  # feature name
        a2 = goal.attr_list.get('2')  # term
        a3 = goal.attr_list.get('3')  # what the feature holds
        if a1 is None or a2 is None:
            return False
        fname = _feature_name_of(_feature_arg_term(a1, eng), wl)
        if fname is None:
            return False
        _host = a2.deref()
        # A name declared `global` or `persistent` stands for its cell, and
        # the features belong to the cell: acc_declarations.lf asks
        # `has_feature(Acc,accumulators,AccInfo)` of the table it files
        # entries in.
        _host_cell = _global_cell(_host, eng) or _persistent_cell(_host, eng)
        if _host_cell is not None:
            _host = _host_cell.deref()
        if fname not in _host.attr_list:
            return False
        if a3 is not None:
            return _unify(eng, a3.deref(), _host.attr_list[fname].deref())
        return True
    _reg('has_feature', _bi_has_feature)

    # ── parents(X, L) — L is list of direct parent sorts of X ─────────────
    def _bi_parents(goal, eng):
        """parents(X, L) — L is the list of direct parent sorts of X.
        parents(X) is the functional 1-arg form (handled via bi_unify)."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        a1d = a1.deref()
        defn = a1d.type
        if defn is None:
            if a2 is None:
                return True
            return _unify(eng, a2.deref(), wl.make_list([]))
        parent_defs = getattr(defn, 'parents', [])
        parent_terms = []
        for pd in parent_defs:
            if pd is not None and pd.keyword is not None:
                parent_terms.append(wl.make_atom(pd.keyword.symbol, wl.bi_module))
        lst = wl.make_list(parent_terms)
        if a2 is None:
            # 1-arg functional form (via bi_unify interception)
            return True
        return _unify(eng, a2.deref(), lst)
    _reg('parents', _bi_parents)

    # ── least_sorts(X, L) — L is the list of most-specific sorts of X ─────
    def _bi_least_sorts(goal, eng):
        """least_sorts(X, L) — L is the list of minimal (most-specific) sorts of X."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None or a2 is None:
            return False
        a1d = a1.deref()
        defn = a1d.type
        if defn is None:
            return _unify(eng, a2.deref(), wl.make_list([]))
        # The most-specific sort of a term is the term's own type (leaf in hierarchy).
        # For atoms with children, the type itself is the sort; for values, use the type.
        least = [wl.make_atom(defn.keyword.symbol, wl.bi_module)] if defn.keyword else []
        return _unify(eng, a2.deref(), wl.make_list(least))
    _reg('least_sorts', _bi_least_sorts)

    # ── chr(N, C) / chr(N) — ASCII code N → character C ──────────────────
    def _bi_chr(goal, eng):
        """chr(N, C) — C is the character for ASCII code N (mod 256).
        chr(N) in functional position is handled via bi_unify."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        a1d = a1.deref()
        ok, v = _eval_arith(a1d, eng)
        if not ok:
            return False
        c = chr(int(v) % 256)
        char_term = _make_string(eng, c)
        if a2 is None:
            return True  # 1-arg predicate form just succeeds
        return _unify(eng, a2.deref(), char_term)
    _reg('chr', _bi_chr)

    # ── asc(C, N) / asc(C) — character C → ASCII code N ─────────────────
    def _bi_asc(goal, eng):
        """asc(C, N) — N is the ASCII code of character C.
        asc(C) in functional position is handled via bi_unify."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        a1d = a1.deref()
        ok, v = _eval_arith(a1d, eng)  # handles asc(chr(N)) too
        if ok:
            n = int(v) % 256
        else:
            # Direct character argument: string or atom
            if a1d.value is not None and a1d.type and a1d.type.is_subtype_of(wl.quoted_string):
                s = str(a1d.value)
                n = ord(s[0]) % 256 if s else 0
            elif a1d.type and a1d.type.keyword:
                s = a1d.type.keyword.symbol
                n = ord(s[0]) % 256 if len(s) == 1 else -1
                if n < 0:
                    return False
            else:
                return False
        if a2 is None:
            return True  # 1-arg just succeeds
        return _unify(eng, a2.deref(), wl.make_integer(n))
    _reg('asc', _bi_asc)

    # ── int2str(N, S) — integer N → string S ─────────────────────────────
    def _bi_int2str(goal, eng):
        """int2str(N, S) — S is the string representation of integer N."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None or a2 is None:
            return False
        a1d = a1.deref()
        ok, v = _eval_arith(a1d, eng)
        if not ok:
            return False
        iv = int(v)
        s = str(iv) if float(iv) == v else str(v)
        return _unify(eng, a2.deref(), _make_string(eng, s))
    _reg('int2str', _bi_int2str)

    # ── int(X, N) — coerce X to integer N ────────────────────────────────
    def _bi_int_coerce(goal, eng):
        """int(X, N) — N is the integer part of X (truncate towards zero)."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None or a2 is None:
            return False
        a1d = a1.deref()
        ok, v = _eval_arith(a1d, eng)
        if not ok:
            return False
        return _unify(eng, a2.deref(), wl.make_integer(int(v)))
    _reg('int', _bi_int_coerce)

    # ── real(X, R) — coerce X to real R ──────────────────────────────────
    def _bi_real_coerce(goal, eng):
        """real(X, R) — R is the floating-point value of X."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None or a2 is None:
            return False
        a1d = a1.deref()
        ok, v = _eval_arith(a1d, eng)
        if not ok:
            return False
        from wild_life.runtime import WildLifeRuntime
        real_t = PsiTerm()
        real_t.type = wl.real
        real_t.value = float(v)
        real_t.status = 4
        from wild_life.data_structures import QUOTED_TRUE
        real_t.flags = QUOTED_TRUE
        return _unify(eng, a2.deref(), real_t)
    _reg('real', _bi_real_coerce)


    # ── Module system predicates ──────────────────────────────────────────────

    def _get_string_or_atom(t: PsiTerm, eng) -> Optional[str]:
        """Extract a name from a string PsiTerm or atom PsiTerm, or None."""
        td = t.deref()
        if td.value is not None and td.type and td.type.is_subtype_of(eng.wl.quoted_string):
            return str(td.value)
        if td.type and td.type.keyword:
            return td.type.keyword.symbol
        return None

    def _bi_module(goal, eng):
        """module("Name") — switch the current module to Name, creating if needed."""
        a1 = goal.attr_list.get('1')
        if a1 is None:
            return False
        name = _get_string_or_atom(a1, eng)
        if name is None:
            return False
        mod = wl.create_module(name)
        # New user-created modules open bi and syntax automatically
        if wl.bi_module not in mod.open_modules:
            mod.open_modules.append(wl.bi_module)
        if wl.syntax_module not in mod.open_modules:
            mod.open_modules.append(wl.syntax_module)
        wl.set_current_module(mod)
        return True
    _reg('module', _bi_module)

    def _bi_public(goal, eng):
        """public(P, Q, ...) — declare symbols as public in the current module."""
        mod = wl.current_module
        if mod is None:
            return True
        i = 1
        while True:
            a = goal.attr_list.get(str(i))
            if a is None:
                break
            ad = a.deref()
            name = _get_string_or_atom(ad, eng)
            if name is None and ad.type and ad.type.keyword:
                name = ad.type.keyword.symbol
            if name:
                defn = wl.update_symbol(mod, name)
                if defn.keyword:
                    defn.keyword.public = True
                # A name this module took from one it opened is handed on
                # under this module's own name too: accumulators declares
                # acc_info public although acc_declarations defines it, and
                # tokenizer, which opens accumulators, reads it from there.
                # update_symbol files the name in the module's own table
                # before linking it to the definition it found, and this is
                # that filing.
                mod.symbol_table.setdefault(name, defn)
            i += 1
        return True
    _reg('public', _bi_public)

    def _bi_private(goal, eng):
        """private(P, …) — give the current module its own P.

        Without it a definition of `+` would add clauses to the syntax
        module's `+`; with it the module gets a `+` of its own, and the one it
        hides is still reachable as `'syntax#+'`.
        """
        from wild_life.data_structures import Keyword as _KW_pv, \
            Definition as _Def_pv
        mod = wl.current_module
        if mod is None:
            return True
        i = 1
        while True:
            a = goal.attr_list.get(str(i))
            if a is None:
                break
            i += 1
            ad = a.deref()
            name = _get_string_or_atom(ad, eng)
            if name is None and ad.type and ad.type.keyword:
                name = ad.type.keyword.symbol
            if not name or name in mod.symbol_table:
                continue
            hidden = wl.update_symbol(mod, name)
            kw = _KW_pv(name, mod)
            defn = _Def_pv(kw)
            kw.definition = defn
            mod.symbol_table[name] = defn
            _hidden_kw = getattr(hidden, 'keyword', None)
            _hidden_mod = getattr(_hidden_kw, 'module', None) if _hidden_kw else None
            if _hidden_mod is not None and _hidden_mod is not mod:
                sys.stderr.write(
                    f"*** Warning: local definition of '{name}' overrides "
                    f"'{_hidden_mod.module_name}#{name}'\n")
        return True
    _reg('private', _bi_private)

    def _bi_private_feature(goal, eng):
        """private_feature(F, ...) — mark features as private to current module.
        public 宣言済みの特性を private_feature にする場合は警告を出す。
        """
        import sys as _sys
        mod = wl.current_module
        if mod is None:
            return True
        i = 1
        while True:
            a = goal.attr_list.get(str(i))
            if a is None:
                break
            ad = a.deref()
            name = _get_string_or_atom(ad, eng)
            if name is None and ad.type and ad.type.keyword:
                name = ad.type.keyword.symbol
            if name:
                defn = wl.update_symbol(mod, name)
                if defn.keyword:
                    if defn.keyword.public:
                        # 既に public 宣言された特性を private にする → 警告
                        print(
                            f"*** Warning: feature '{defn.keyword.combined_name}'"
                            f" is now private, but was also declared public",
                            file=_sys.stderr,
                        )
                        defn.keyword.public = False
                    defn.keyword.private_feature = True
            i += 1
        return True
    _reg('private_feature', _bi_private_feature)

    def _bi_open(goal, eng):
        """open("Mod", ...) — add named module(s) to current module's open list."""
        import sys as _sys
        mod = wl.current_module
        if mod is None:
            return True
        i = 1
        while True:
            a = goal.attr_list.get(str(i))
            if a is None:
                break
            name = _get_string_or_atom(a, eng)
            if name:
                # Check whether the module was already defined.
                # If not, report the error (like the C interpreter) but continue.
                if name not in wl.module_table:
                    print(f'*** Error: module "{name}" not found', file=_sys.stderr)
                else:
                    target = wl.create_module(name)
                    # Ensure target itself opens bi/syntax
                    if wl.bi_module not in target.open_modules:
                        target.open_modules.append(wl.bi_module)
                    if wl.syntax_module not in target.open_modules:
                        target.open_modules.append(wl.syntax_module)
                    if target not in mod.open_modules:
                        mod.open_modules.append(target)
                        # 公開シンボルのモジュール名衝突を検出する。
                        # target より前に開かれているユーザモジュールの public シンボルと
                        # target の public シンボルが同名の場合はエラーを報告し、
                        # 衝突したシンボル名を現在のモジュールで UNDEF スタブとしてブロックする。
                        from wild_life.data_structures import (
                            Keyword as _Kw, Definition as _Def, DefType as _DT
                        )
                        prior_user_mods = [
                            m for m in mod.open_modules[:-1]
                            if m not in (wl.bi_module, wl.syntax_module) and m is not mod
                        ]
                        for existing_mod in prior_user_mods:
                            for sym_name, defn_t in list(target.symbol_table.items()):
                                if not (defn_t.keyword and defn_t.keyword.public):
                                    continue
                                if sym_name not in existing_mod.symbol_table:
                                    continue
                                defn_e = existing_mod.symbol_table[sym_name]
                                if not (defn_e.keyword and defn_e.keyword.public):
                                    continue
                                # 衝突検出: target の sym_name と existing_mod の sym_name が衝突
                                line_no = getattr(wl, 'line_count', 0) + 1
                                print(
                                    f'*** Error: serious module name clash: '
                                    f'"{target.module_name}#{sym_name}" and '
                                    f'"{existing_mod.module_name}#{sym_name}"',
                                    file=_sys.stderr
                                )
                                print(
                                    f'*** Syntax error: Module violation '
                                    f'(near line {line_no}).',
                                    file=_sys.stderr
                                )
                                # 現在のモジュールに UNDEF スタブを挿入して衝突シンボルをブロック
                                if sym_name not in mod.symbol_table:
                                    stub_kw = _Kw(sym_name, mod, public=False)
                                    stub_defn = _Def(stub_kw)
                                    stub_defn.type = _DT.UNDEF
                                    stub_defn.clash_blocked = True  # listing で無音成功
                                    stub_kw.definition = stub_defn
                                    mod.symbol_table[sym_name] = stub_defn
            i += 1
        return True
    _reg('open', _bi_open)

    def _bi_import(goal, eng):
        """import("A", "B", ...) — load each file and open the module it holds.

        built_ins.lf says the same thing in LIFE: `X:import :- load&strip(X),
        import_list(features(X), X)`, where each feature is loaded and then
        opened under the name the path ends in.  A file that holds no module
        of that name is loaded all the same.
        """
        import os as _os_im
        names = []
        i = 1
        while True:
            a = goal.attr_list.get(str(i))
            if a is None:
                break
            i += 1
            _n = _get_str_val(a.deref(), eng)
            if _n is None:
                return False
            names.append(_n)
        if not names:
            return False
        for _n in names:
            _path = _resolve_life_file(_n)
            _announce_load(_path)
            if not eng.load_file(_path):
                return False
        _open_defn = wl.update_symbol(wl.bi_module, 'open')
        for _n in names:
            _base = _os_im.path.basename(_n)
            if _base.endswith('.lf'):
                _base = _base[:-3]
            if _base in wl.module_table:
                _op = PsiTerm(type_def=_open_defn)
                _op.attr_list = {'1': _make_string(eng, _base)}
                _bi_open(_op, eng)
        return True
    _reg('import', _bi_import)

    def _bi_display_modules(goal, eng):
        """display_modules — enable module-qualified name display mode (like C Wild Life).

        In C Wild Life, calling display_modules enables module-qualified printing
        for all subsequent write/print operations.
        """
        # Enable module-qualified name display mode
        wl.display_modules_mode = True
        return True
    _reg('display_modules', _bi_display_modules)

    def _bi_display_persistent(goal, eng):
        """display_persistent — write a ` $` in front of every persistent term."""
        wl.display_persistent_mode = True
        return True
    _reg('display_persistent', _bi_display_persistent)

    def _bi_add_man(goal, eng):
        """add_man(Name, Text) — file a manual entry for Name.

        The library files each describe themselves this way as they load.
        Nothing reads the entries back here, so they are filed and left.
        """
        _a1 = goal.attr_list.get('1')
        _a2 = goal.attr_list.get('2')
        if _a1 is None or _a2 is None:
            return True
        _table = getattr(wl, 'manual_table', None)
        if _table is None:
            _table = {}
            wl.manual_table = _table
        _names = _proper_list_elems(_a1.deref(), eng)
        if _names is None:
            _names = [_a1.deref()]
        for _n in _names:
            _nd = _n.deref()
            _key = (_nd.type.keyword.symbol
                    if (_nd.type is not None and _nd.type.keyword) else None)
            if _key is None and _nd.value is not None:
                _key = str(_nd.value)
            if _key is not None:
                _table[_key] = _a2.deref()
        return True
    _reg('add_man', _bi_add_man)

    def _bi_import_clauses(goal, eng):
        """import_clauses(for => Module#Pred, replacing => [(Module#Old, New), ...])

        Copies all clauses (rules) of Module#Pred into the homonymous predicate /
        function in the current module.  The optional 'replacing' list maps old
        Definition references (Module#Name) inside the copied clause bodies to new
        local ones so that recursive calls target the local copy.

        Syntax note: import_clauses uses NAMED features, not positional args.
        The goal term itself carries 'for' and 'replacing' as attribute keys.
        """
        from wild_life.data_structures import DefType as _DT
        from wild_life.unification import copy_term

        # ── 1. Locate the 'for' feature directly on the goal term ─────────────
        # (import_clauses(for => X, replacing => Y) uses named features, not
        #  positional '1'/'2' args)
        for_part = goal.attr_list.get('for')
        repl_part = goal.attr_list.get('replacing')

        if for_part is None:
            # fall back: try positional arg wrapping a compound with 'for' feature
            a1 = goal.attr_list.get('1')
            if a1 is None:
                return True
            a1d = a1.deref()
            for_part = a1d.attr_list.get('for')
            repl_part = a1d.attr_list.get('replacing')
            if for_part is None:
                return True

        for_d = for_part.deref()

        # ── 2. Extract source module name and predicate/function name ──────────
        if for_d.type is None or for_d.type.keyword is None:
            return True
        sym = for_d.type.keyword.symbol
        src_mod_obj = for_d.type.keyword.module
        if src_mod_obj is None:
            # Unqualified name: try current module
            src_mod_obj = wl.current_module
        src_mod = wl.find_module(src_mod_obj.module_name) if src_mod_obj else None
        if src_mod is None:
            return True
        src_defn = src_mod.symbol_table.get(sym)
        if src_defn is None or not src_defn.rule:
            return True

        # ── 3. Parse the 'replacing' list: [(OldDef, NewDef), ...] ────────────
        # Build a mapping {old_Definition_id → new_Definition} for substitution.
        replacements: dict = {}   # id(old_defn) → new_defn
        if repl_part is not None:
            node = repl_part.deref()
            while node.type is not None and node.type is wl.alist:
                head_ref = node.attr_list.get('1')
                node = node.attr_list.get('2').deref() if node.attr_list.get('2') else wl.make_atom('nil', wl.bi_module).deref()
                if head_ref is None:
                    continue
                pair = head_ref.deref()
                # Pair: (OldQName, NewName) as a tuple-like term with '1' and '2'
                p1 = pair.attr_list.get('1')
                p2 = pair.attr_list.get('2')
                if p1 is None or p2 is None:
                    continue
                old_t = p1.deref()
                new_t = p2.deref()
                # old_t: module-qualified (e.g. lists#app) or bare atom
                if old_t.type and old_t.type.keyword:
                    old_defn = old_t.type
                    # new_t: bare atom → resolve in current module
                    if new_t.type and new_t.type.keyword:
                        new_name = new_t.type.keyword.symbol
                        new_defn = wl.update_symbol(wl.current_module, new_name)
                        replacements[id(old_defn)] = new_defn

        # ── 4. Ensure the local definition exists with correct type ─────────────
        local_defn = wl.update_symbol(wl.current_module, sym)
        if local_defn.rule is None:
            local_defn.rule = []
        if local_defn.type == _DT.UNDEF:
            local_defn.type = src_defn.type

        # ── 5. Copy each source clause and apply replacements ─────────────────
        def _replace_defns(t, visited=None):
            """Walk PsiTerm t and replace Definition references per 'replacements'."""
            if visited is None:
                visited = set()
            if id(t) in visited:
                return
            visited.add(id(t))
            if t.type is not None and id(t.type) in replacements:
                t.type = replacements[id(t.type)]
            for child in t.attr_list.values():
                cd = child.deref()
                _replace_defns(cd, visited)

        for (h0, b0) in src_defn.rule:
            _vm: dict = {}
            h_copy = copy_term(h0, _vm)
            b_copy = copy_term(b0, _vm)
            if replacements:
                _replace_defns(h_copy)
                _replace_defns(b_copy)
            local_defn.rule.append((h_copy, b_copy))

        return True
    _reg('import_clauses', _bi_import_clauses)

    def _bi_use_module(goal, eng):
        """use_module("Name") — alias for open (compatibility)."""
        return _bi_open(goal, eng)
    _reg('use_module', _bi_use_module)

    def _bi_module_info(goal, eng):
        """module_info(M, Info) — basic module info (stub)."""
        return True
    _reg('module_info', _bi_module_info)

    # ── Address equality operators (syntax_module operators) ─────────────────
    # '===' and '\===' are registered as operators in syntax_module by the
    # runtime's _init_built_ins (which was intended but never called), so we
    # register their implementations here, in the syntax_module, so that
    # prove_aim finds _builtin_func on the operator's Definition object.
    _reg('===', c_same_address,  def_type=DefType.FUNCTION, module=wl.syntax_module)
    _reg('\\===', c_diff_address, def_type=DefType.FUNCTION, module=wl.syntax_module)
