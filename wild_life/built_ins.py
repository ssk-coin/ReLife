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
    '+', '-', '*', '/', '//', 'mod', '**', '^',
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
    _unary_only = frozenset(('abs', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                              'exp', 'log', 'floor', 'ceiling', 'round', 'truncate',
                              'float', 'integer', 'sign', 'msb', '\\'))
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
    if is_free and (t.type is wl.top or t.type is None or bool(t.flags & SORT_VAR)):
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

    return None


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
        a1 = _try_eval_bool(a1, eng) or a1.deref()
        a2 = _try_eval_bool(a2, eng) or a2.deref()
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
        a1 = _try_eval_bool(a1, eng) or a1.deref()
        a2 = _try_eval_bool(a2, eng) or a2.deref()
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
        a1 = _try_eval_bool(a1.deref(), eng) or a1.deref()
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
        a1 = _try_eval_bool(a1, eng) or a1.deref()
        a2 = _try_eval_bool(a2, eng) or a2.deref()
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
                                                               'mod','**','^','max','min',
                                                               'abs','sqrt','sin','cos','tan',
                                                               'exp','log','floor','ceiling')):
        return None  # already a number, no evaluation needed
    result = _make_number(eng, v)
    # _eval_arith may have already fired delay rules for this value (e.g.
    # via the binary * path or literal evaluation). Mark _delay_fired=True so
    # that subsequent unification with a free variable does not re-fire.
    result._delay_fired = True
    # Memoize the result back into the compound arithmetic term (t) via coref.
    # This propagates the evaluated value through the variable chain:
    # after evaluation, any variable that pointed to this expression will deref to
    # the concrete number.  We trail the old coref so backtracking can undo this.
    if eng is not None and t.coref is None and t.value is None and t.attr_list:
        eng.trail.trail_psi(t, 'coref')
        t.coref = result
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

    # If the whole term is an arithmetic expression, evaluate it
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


def _try_eval_string_func(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """Try to evaluate string built-in functions (psi2str, str2psi, strcon).

    Returns evaluated PsiTerm or None if not applicable.
    """
    if t is None:
        return None
    t = t.deref()
    sym = _get_sym(t)

    if sym == 'psi2str':
        # psi2str(T) -> string representation of T
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = a1.deref()
        s = _term_to_display_string(a1, eng)
        return _make_string(eng, s)

    elif sym == 'str2psi':
        # str2psi(S) -> atom from string S
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = a1.deref()
        if a1.type and a1.type is eng.wl.quoted_string and a1.value is not None:
            name = str(a1.value)
        elif a1.type and a1.type.keyword:
            name = a1.type.keyword.symbol
        else:
            name = _term_to_display_string(a1, eng)
        return _make_atom(eng, name)

    elif sym == 'strcon':
        # strcon(A, B) -> concatenation of strings A and B
        a1 = t.attr_list.get('1')
        a2 = t.attr_list.get('2')
        if a1 is None or a2 is None:
            return None
        a1, a2 = a1.deref(), a2.deref()
        # Only evaluate when BOTH arguments are concrete strings
        if eng is not None:
            wl = eng.wl
            def _is_string(x):
                return (x.value is not None and x.type is not None
                        and x.type.is_subtype_of(wl.quoted_string))
            if not (_is_string(a1) and _is_string(a2)):
                return None
        # Recursively evaluate if needed
        a1e = _try_eval_string_func(a1, eng)
        if a1e is not None:
            a1 = a1e
        a2e = _try_eval_string_func(a2, eng)
        if a2e is not None:
            a2 = a2e
        s1 = str(a1.value) if (a1.value is not None) else ''
        s2 = str(a2.value) if (a2.value is not None) else ''
        return _make_string(eng, s1 + s2)

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

    elif sym == 'root_sort' or sym == 'sort':
        # root_sort(T) -> the root sort of T.
        # For numeric/string atoms, the root sort is the value itself.
        # For compound terms, it is the functor name as an atom.
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
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
        return _make_atom(eng, defn.keyword.symbol)

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

    elif sym == 'features':
        # features(T) -> list of attribute labels
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = a1.deref()
        # Try to evaluate a1 first (e.g. local_time built-in)
        _a1_ev = _try_eval_string_func(a1, eng)
        if _a1_ev is not None:
            a1 = _a1_ev
        keys = list(a1.attr_list.keys())
        wl = eng.wl
        lst = PsiTerm(type_def=wl.nil)
        lst.type = wl.nil
        for key in reversed(keys):
            try:
                n = int(key)
                # Negative integer feature names must be returned as atoms
                # (quoted when printed, e.g. '-34'), not as integer values,
                # because they are identifiers, not numbers.
                if n >= 0:
                    kterm = wl.make_integer(n)
                else:
                    kterm = _make_atom(eng, key)
            except (ValueError, TypeError):
                kterm = _make_atom(eng, key)
            pair = PsiTerm()
            pair.type = wl.alist
            pair.attr_list = {'1': kterm, '2': lst}
            lst = pair
        return lst

    elif sym == '.':
        # T.F — feature access: get feature F of term T
        a1 = t.attr_list.get('1')  # T
        a2 = t.attr_list.get('2')  # F (feature label)
        if a1 is None or a2 is None:
            return None
        term = a1.deref()
        feat = a2.deref()
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
            fkey = feat.type.keyword.symbol
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
        return s

    t1 = _eval_side(t1)
    if t1 is None:
        return None
    t2 = _eval_side(t2)
    if t2 is None:
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
            if _check_sort_member(e_d, filter_side):
                surviving.append(e_d)
        if not surviving:
            return None  # empty disjunction = fail (No)
        return _make_disjunction_psi(surviving, wl)

    # Unify t1 and t2 through a fresh variable to find their meet
    fresh = PsiTerm()
    fresh.type = wl.top
    mark = eng.trail.mark()
    ok1 = eng.unifier.unify(fresh, t1)
    if not ok1:
        eng.trail.undo_to(mark)
        return None
    fresh_d = fresh.deref()
    ok2 = eng.unifier.unify(fresh_d, t2)
    if not ok2:
        eng.trail.undo_to(mark)
        return None
    return fresh.deref()


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


def _write_term(t: PsiTerm, eng, stream=None, quoted=True) -> None:
    from wild_life.print_term import write_term
    var_tree = getattr(eng, '_last_var_tree', None)
    wl = eng.wl if eng else None

    # ── psi-term conjunction (&): evaluate before printing ──────────────────
    # writeq(`X & Y) should evaluate the conjunction and print the result.
    # If the conjunction fails, the predicate fails.
    if wl and t.type is not None and t.type is wl.and_sym:
        evaluated = _eval_and_conjunction(t, eng)
        if evaluated is None:
            raise _WriteFailure("conjunction failed")
        t = evaluated

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
    _arith_binary_ops = frozenset(('+', '-', '*', '/', '//', 'mod', '**', '^',
                                   'max', 'min'))
    _arith_unary_ops = frozenset(('abs', 'sqrt', 'sin', 'cos', 'tan', 'exp',
                                  'log', 'floor', 'ceiling', 'round',
                                  'truncate'))
    from wild_life.data_structures import NON_STRICT_TERM as _WT_NST
    _is_nst = bool(t.flags & _WT_NST)
    try:
        t_eval = _try_eval_arith_to_term(t, eng) if not _is_nst else None
        if t_eval is not None:
            t = t_eval
        else:
            # Evaluate '.' (feature access) in write context.
            # write(C.1) evaluates C.1 and prints the result (e.g. 'a' for a cons head).
            t_str = _try_eval_string_func(t, eng)
            if t_str is not None:
                t = t_str
            elif sym == '.':
                # T.F where feature F doesn't exist on T → unbound var → write '@'.
                # In C Wild Life, accessing a non-existent attribute yields a fresh
                # unbound variable, and write prints it as '@' (the top sort).
                _a1 = t.attr_list.get('1')
                _a2 = t.attr_list.get('2')
                if _a1 is not None and _a2 is not None:
                    _term_d = _a1.deref()
                    _feat_d = _a2.deref()
                    _fsym_raw = _feat_d.type.keyword.symbol if (
                        _feat_d.type and _feat_d.type.keyword) else None
                    if _fsym_raw in ('integer', 'real', 'int', 'float', 'number') and \
                            _feat_d.value is not None:
                        _fkey = str(int(_feat_d.value))
                    elif _fsym_raw is not None:
                        _fkey = _fsym_raw
                    else:
                        _fkey = None
                    if _fkey is not None and _fkey not in _term_d.attr_list:
                        _fresh_var = PsiTerm()
                        if wl:
                            _fresh_var.type = wl.top
                        t = _fresh_var
            elif _is_user_function(t) and eng is not None:
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
                        # For 0-arity globals: only substitute a concrete numeric
                        # result.  Compound bodies with unbound variables must not
                        # replace the atom name during display.
                        _evd = _evaled.deref()
                        if _evd.value is not None:
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

    from wild_life.print_term import PRINT_DEPTH as _PRINT_DEPTH
    _pd = getattr(eng.wl, 'print_depth', _PRINT_DEPTH) if eng and eng.wl else _PRINT_DEPTH
    write_term(t, outfile=stream or sys.stdout, quoted=quoted, wl=eng.wl,
               var_tree=var_tree, print_depth=_pd)


def _term_to_str(t: PsiTerm, eng, quoted=True) -> str:
    from wild_life.print_term import term_to_string
    return term_to_string(t, quoted=quoted, wl=eng.wl)


def _is_var(t: PsiTerm, eng) -> bool:
    wl = eng.wl
    return t.type == wl.top and t.value is None and not t.attr_list and t.coref is None


# ─────────────────────────────────────────────────────────────────────────────
# I/O predicates
# ─────────────────────────────────────────────────────────────────────────────

def _write_all_args(goal: PsiTerm, eng, quoted: bool, stream=None) -> bool:
    """Write all positional arguments of goal, concatenated (no separator).

    In LIFE, write(a,b,c) writes each argument in order without separator.
    If the goal has no positional args, write the goal's sort name.
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
            _write_term(arg, eng, stream=stream, quoted=quoted)
        except _WriteFailure:
            return False
        written_any = True
        i += 1
    if not written_any:
        # No positional args: treat as write of goal itself
        try:
            _write_term(goal, eng, stream=stream, quoted=quoted)
        except _WriteFailure:
            return False
    return True


def bi_write(goal: PsiTerm, eng) -> bool:
    """write(T) — write term T (or all positional args) without quoting."""
    return _write_all_args(goal, eng, quoted=False)


def bi_writeq(goal: PsiTerm, eng) -> bool:
    """writeq(T) — write term T (or all positional args) with quoting."""
    return _write_all_args(goal, eng, quoted=True)


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
        is_backtick = (arg.type is not None and arg.type.keyword is not None
                       and arg.type.keyword.symbol == '`')
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


def bi_print_depth(goal: PsiTerm, eng) -> bool:
    """print_depth(N) — set the global print depth limit.

    TRUE MODEL (from C Wild Life REFERR/REFOUT analysis):
      N >= 0  → wl.print_depth = N + 1  (pd(0)?→1, pd(1)?→2, pd(3)?→4, ...)
      N < 0   → error; if current pd<=1, show "..." in msg; else show actual arg value;
                reset pd to 4 (C Wild Life default) regardless.
    """
    arg = _get_one_arg(goal)
    if arg is None:
        # 0-arg case: print_depth? resets pd to initial (effectively unlimited)
        eng.wl.print_depth = 1000000000
        return True
    arg = arg.deref()
    wl = eng.wl
    if arg.value is not None and arg.type and arg.type.is_subtype_of(wl.real):
        n = int(float(arg.value))
        if n < 0:
            # Error: negative argument not allowed
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
            # TRUE MODEL: pd(N)? → wl.print_depth = N + 1
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

def _eval_arith(t: PsiTerm, eng, _depth: int = 0) -> Tuple[bool, float]:
    """Evaluate an arithmetic expression. Returns (ok, value)."""
    if t is None or _depth > 40:
        return False, 0.0
    t = t.deref()
    wl = eng.wl
    sym = t.type.keyword.symbol if t.type and t.type.keyword else ''

    if t.value is not None and t.type and t.type.is_subtype_of(wl.real):
        # Fire int/real delay rule for parsed literal integers (not computed by _make_number).
        # In C Wild Life, literal integers in expressions act like narrowed sort-vars
        # and fire the :: I:int | ... delay when they are "evaluated".
        from wild_life.runtime import WL as _WL_ea
        if (_WL_ea.delay_rules and eng is not None
                and not getattr(eng, '_in_fire_delay', False)
                and not getattr(t, '_delay_fired', False)):
            t._delay_fired = True
            eng.unifier._fire_delay_rules(t, t.type)
        return True, float(t.value)

    # User-defined function: try to evaluate it inline (no condition case)
    if t.type is not None and t.type.type == DefType.FUNCTION and t.type.rule:
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
                t_pre.attr_list[_k] = _evaled_arg if _evaled_arg is not None else _vd
        _cp_save = eng.choice_stack  # Save choice stack before user-func unification
        for _ri, (h0, b0) in enumerate(active):
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

    # Binary operators — early exit if sym is not a known arithmetic binary op.
    # This prevents infinite recursion on cyclic terms like cons(A,A) where
    # the 'cons' symbol is not arithmetic but the pre-check would recurse forever.
    _arith_binary_syms = frozenset(('+', '-', '*', '/', '//', 'mod', '**', '^',
                                     'max', 'min', '/\\', '\\/', 'xor', '>>', '<<'))
    if sym not in _arith_binary_syms:
        return False, 0.0
    arg1, arg2 = _get_two_args(t)
    ok1, v1 = _eval_arith(arg1, eng, _depth + 1) if arg1 else (False, 0.0)
    ok2, v2 = _eval_arith(arg2, eng, _depth + 1) if arg2 else (False, 0.0)

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
        # Bitwise operators
        '/\\': lambda a, b: float(int(a) & int(b)),
        '\\/': lambda a, b: float(int(a) | int(b)),
        'xor': lambda a, b: float(int(a) ^ int(b)),
        '>>': lambda a, b: float(int(a) >> int(b)),
        '<<': lambda a, b: float(int(a) << int(b)),
    }
    if sym in ops2 and ok1 and ok2:
        try:
            _result_val = float(ops2[sym](v1, v2))
            # Fire int/real delay for multiplication results.
            # In C Wild Life, each intermediate product of N*fact(N-1) triggers
            # the :: I:int global delay rule as the partial result is narrowed.
            # Only fire for '*' to avoid double-firing subtraction results that
            # are already handled by the pre-eval computed-term firing above.
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
        # Bitwise NOT
        '\\': lambda a: float(~int(a)),
    }
    if sym in ops1 and ok1 and arg2 is None:
        try:
            return True, float(ops1[sym](v1))
        except Exception:
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
                return _eval_arith(inner.deref(), eng, _depth + 1)
            return False, 0.0
        return _eval_arith(a1d, eng, _depth + 1)

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
        a1 = t.attr_list.get('1')
        if a1 is None:
            return False, 0.0
        a1d = a1.deref()
        # First try to evaluate as chr(N) or another string function
        char_term = _try_eval_string_func(a1d, eng)
        if char_term is not None:
            if char_term.value is not None:
                s = str(char_term.value)
                if s:
                    return True, float(ord(s[0]) % 256)
            return False, 0.0
        # Try as a direct string value
        if a1d.value is not None and eng is not None:
            wl = eng.wl
            if a1d.type and a1d.type.is_subtype_of(wl.quoted_string):
                s = str(a1d.value)
                return (True, float(ord(s[0]) % 256)) if s else (False, 0.0)
        # Try as atom (symbol name like 'a', 'b', etc.)
        if a1d.type is not None and a1d.type.keyword is not None:
            s = a1d.type.keyword.symbol
            if len(s) == 1:
                return True, float(ord(s[0]) % 256)
        return False, 0.0

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
    elif sym == '/':
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
    # x * x = 0 → x = 0
    if sym == '*':
        arg1_d = arg1.deref()
        arg2_d = arg2.deref()
        if id(arg1_d) == id(x_var) and id(arg2_d) == id(x_var):
            if abs(v_lhs) < 1e-12:
                return 0.0
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
            b_psi = _make_number(eng, r1[1] * v2) if ok_b else r1[1]
            return (r1[0] * v2, b_psi)
        return None
    elif sym == '/':
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


def _term_contains_disjunction(t: PsiTerm, eng, depth: int = 0) -> bool:
    """Return True if t (or any subterm up to depth 10) is a disjunction."""
    if depth > 10:
        return False
    t = t.deref()
    if t.type is None:
        return False
    if t.type is eng.wl.disjunction:
        return True
    for v in t.attr_list.values():
        if _term_contains_disjunction(v, eng, depth + 1):
            return True
    return False


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
    t = t.deref()
    # Backtick-quoted terms (QUOTED_TRUE) are sort references, not function calls
    from wild_life.data_structures import QUOTED_TRUE
    if t.flags & QUOTED_TRUE:
        return False
    defn = t.type
    if defn is None:
        return False
    if defn.type != DefType.FUNCTION:
        return False
    if defn._builtin_func is not None:
        return False
    if not defn.rule:
        return False
    return True


def _eval_user_func_sync(t: PsiTerm, eng, _depth: int = 0) -> Optional[PsiTerm]:
    """Synchronously evaluate a user-defined function call.

    This is used to eagerly evaluate function-call arguments before pattern
    matching (e.g., reverse([1,2,3,4]) in rev(reverse([1,2,3,4]),[])).
    Only evaluates simple (unconditional, deterministic first-rule) cases.

    Returns the result PsiTerm, or None if evaluation can't proceed.
    The engine trail is NOT rolled back — bindings persist on the trail.
    """
    if _depth > 40:
        return None
    if t is None:
        return None
    t = t.deref()
    if not _is_user_function(t):
        return None

    from wild_life.unification import copy_term
    from wild_life.data_structures import QUOTED_TRUE

    # Try each rule in order (no backtracking support here)
    rules = t.type.rule or []
    active = [(h, b) for (h, b) in rules if h is not None and b is not None]
    for h0, b0 in active:
        _vm: dict = {}
        head = copy_term(h0, _vm)
        body = copy_term(b0, _vm)
        body_d = body.deref()

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
            # Evaluate built-in / user-defined functional sub-terms inside
            # the guard goal (e.g. genChildren(children(X), A) → the
            # children(X) arg must be reduced before the predicate is called).
            _cond_d = cond_part.deref()
            _eval_embedded_user_funcs(_cond_d, eng, _depth + 1, set())
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
                val_d = val_part.deref()
                ok_a, val = _eval_arith(val_d, eng)
                if ok_a:
                    return _make_number(eng, val)
                return val_d
            else:
                eng.trail.undo_to(mark)
                continue

        # Pre-evaluate any user-defined or built-in functional sub-terms in
        # the input term's arguments before trying to unify with the head.
        # This mirrors the EVAL goal handler in inference.py (lines ~843-857)
        # and is necessary so that e.g. app([1], rev([2,3])) can match
        # app(L, [H|T]) after rev([2,3]) is reduced to [3,2].
        t_copy_attrs = dict(t.attr_list)
        for _key in list(t.attr_list.keys()):
            _attr = t.attr_list[_key].deref()
            _ev = _try_eval_any_func(_attr, eng)
            if _ev is not None and _ev is not _attr:
                t.attr_list[_key] = _ev

        mark = eng.trail.mark()
        ok = eng.unifier.unify(t, head)
        if not ok:
            # Restore original attrs in case we modified them
            t.attr_list = t_copy_attrs
            eng.trail.undo_to(mark)
            continue

        body_d2 = body_d.deref()
        result = _eval_body_sync(body_d2, eng, _depth + 1)
        if result is None and _is_user_function(body_d2):
            # Body is a user function that can't eval synchronously
            return None
        return result if result is not None else body_d2

    return None


def _is_cond_builtin_local(t: 'PsiTerm') -> bool:
    """Return True if t is the built-in cond(…) call."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    if t.type.keyword.symbol != 'cond':
        return False
    return getattr(t.type, '_builtin_func', None) is not None


def _is_copy_term_func(t: 'PsiTerm') -> bool:
    """Return True if t is copy_term(X) with exactly 1 argument (functional use)."""
    if t is None or t.type is None or t.type.keyword is None:
        return False
    if t.type.keyword.symbol != 'copy_term':
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
            new_src_attrs[k] = v  # keep non-positional attrs in src only

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
            for k, v in new_vt.items():
                if k not in lv:
                    lv[k] = v

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
    _arith_syms_p = frozenset(('+', '-', '*', '/', '//', 'mod', '**', '^',
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


def _eval_lub_func(t: 'PsiTerm', eng) -> Optional['PsiTerm']:
    """Evaluate lub(X, Y) → first LUB; non-determinism handled by _apply_lub_to_var."""
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
    return _compute_all_lubs(d1, d2, wl)


def _apply_lub_to_var(t: 'PsiTerm', target: 'PsiTerm', eng) -> bool:
    """Unify target with lub(X,Y), creating choice points for multiple LUBs."""
    from wild_life.data_structures import GoalType as _GT
    lubs = _compute_all_lubs_from_t(t, eng)
    if not lubs:
        return False
    # Push choice points for alternatives (reversed so first fires next)
    for alt_def in reversed(lubs[1:]):
        alt_psi = PsiTerm(type_def=alt_def)
        eng.push_choice_point(_GT.UNIFY, target, alt_psi, None)
    first_psi = PsiTerm(type_def=lubs[0])
    return _unify(eng, target, first_psi)


def _eval_body_sync(body_d: 'PsiTerm', eng, _depth: int) -> Optional['PsiTerm']:
    """Synchronously evaluate a function body expression.

    Handles: arithmetic, user-defined function calls, built-in cond(C,T,E),
    and compound terms with embedded user-function sub-terms.
    Returns the evaluated PsiTerm or None if evaluation cannot proceed.
    """
    if _depth > 40:
        return None

    # Arithmetic expression?
    ok_a, val = _eval_arith(body_d, eng)
    if ok_a:
        return _make_number(eng, val)

    # User-defined function call?
    if _is_user_function(body_d):
        return _eval_user_func_sync(body_d, eng, _depth)

    # Built-in copy_term(X) functional use — return a fresh copy
    if _is_copy_term_func(body_d):
        return _eval_copy_term_func(body_d)

    # Built-in cond(C, T, E) — evaluate functionally
    if _is_cond_builtin_local(body_d):
        args = list(body_d.attr_list.values()) if body_d.attr_list else []
        if len(args) < 2:
            return None
        cond_g = args[0].deref()
        then_g = args[1].deref()
        else_g = args[2].deref() if len(args) >= 3 else None

        from wild_life.inference import GoalType as _GT, _DEFRULES as _DR, _INNER_RUN_BARRIER as _IRB
        mark_c = eng.trail.mark()
        cp_save = eng.choice_stack
        gs_save = eng.goal_stack
        eng.goal_stack = None
        eng.push_goal(_GT.PROVE, cond_g, _DR, None)
        old_ok = eng.main_loop_ok
        barrier = cp_save if cp_save is not None else _IRB
        cond_ok = eng.run(cs_barrier=barrier)
        eng.main_loop_ok = old_ok
        eng.choice_stack = cp_save
        eng.goal_stack = gs_save

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

    # Compound term: evaluate embedded user-function and cond sub-terms in-place
    _eval_embedded_user_funcs(body_d, eng, _depth, set())
    return body_d


def _try_eval_any_func(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """Try to evaluate t as any functional form (user-defined or built-in).

    Returns the evaluated PsiTerm, or None if t is not a functional form
    (or evaluation fails).  Used to eagerly reduce function sub-terms that
    appear in predicate-argument position inside function bodies.
    """
    if t is None or eng is None:
        return None
    td = t.deref()
    if td.type is None:
        return None

    # User-defined function
    if _is_user_function(td):
        return _eval_user_func_sync(td, eng, 0)

    # Built-in copy_term
    if _is_copy_term_func(td):
        return _eval_copy_term_func(td)

    # Built-in cond(C,T,E)
    if _is_cond_builtin_local(td):
        return _eval_body_sync(td, eng, 0)

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

    # General string function (strcon, substr, strlen, int2str, …)
    r = _try_eval_string_func(td, eng)
    if r is not None:
        return r

    # Arithmetic expression (+, -, *, /, abs, sqrt, …)
    # _eval_arith returns (False, 0.0) quickly for non-arithmetic terms,
    # so calling it unconditionally is safe.
    ok, v = _eval_arith(td, eng)
    if ok:
        return _make_number(eng, v)

    return None



# Built-in predicates whose first argument ('1') should NOT be eagerly evaluated
# by _eval_embedded_user_funcs.  These predicates treat their first argument as a
# FUNCTION/PREDICATE NAME (a symbol to look up), not as a value to evaluate.
# For example, `setq(seed, 99)` should treat `seed` as a name, not call the
# function `seed` to get 1 and then set the integer-1 definition to 99.
_NON_STRICT_ARG1_BUILTINS: frozenset = frozenset({
    'setq', 'dynamic', 'static', 'assert', 'asserta', 'retract',
    'clause', 'abolish', 'listing',
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
    """
    if _depth > 40:
        return
    td = t.deref()
    if id(td) in visited:
        return
    visited.add(id(td))
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
        evaled = _try_eval_any_func(child, eng)
        if evaled is not None and evaled is not child:
            td.attr_list[key] = evaled
            _eval_embedded_user_funcs(evaled, eng, _depth + 1, visited)
        elif child.attr_list:
            _eval_embedded_user_funcs(child, eng, _depth + 1, visited)


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
    _ops2 = frozenset(('+', '-', '*', '/', '//', 'mod', '**', '^',
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
                '//': lambda a, b: float(int(a) // int(b)) if b != 0 else 0.0,
                'mod': lambda a, b: float(int(a) % int(b)) if b != 0 else 0.0,
                '**': lambda a, b: a ** b, '^': lambda a, b: a ** b,
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
    _ops = frozenset(('+', '-', '*', '/', '//', 'mod', '**', '^',
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


def _resolve_dot_feat(dot_term: 'PsiTerm', eng) -> 'Optional[PsiTerm]':
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
    # If the host is itself a dot-access expression (nested chain like A.a.b.c.d),
    # resolve it recursively to get the actual psi-term that holds the feature.
    if (host.type is not None and host.type.keyword is not None
            and host.type.keyword.symbol == '.'):
        host = _resolve_dot_feat(host, eng)
        if host is None:
            return None
        host = host.deref()
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
            fkey = feat.type.keyword.symbol
    else:
        return None
    existing = host.attr_list.get(fkey)
    if existing is not None:
        return existing  # caller will deref as needed
    # Attr absent — create a fresh variable (type=top = unbound), insert it (trailed)
    wl_rd = eng.wl
    fresh = PsiTerm()
    fresh.type = wl_rd.top  # must be WL.top so unification recognises it as a free var
    eng.trail.trail_psi(host, 'attr_list')
    new_attrs = dict(host.attr_list)
    new_attrs[fkey] = fresh
    host.attr_list = new_attrs
    # Fix D: fire pending daemon resids on host after a new attribute is added.
    # e.g. X.set = true? fires the daemon write(X) that was set by such_that.
    if host.resid and eng is not None and getattr(eng, 'unifier', None) is not None:
        eng.unifier._wakeup_resid(host, fresh)
    return fresh


def bi_unify(goal: PsiTerm, eng) -> bool:
    """X = Y — LIFE sort unification (with functional evaluation)."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return a is b

    a_d = a.deref()
    b_d = b.deref()

    # Handle T.F = V and V = T.F (dot feature access / creation).
    # When T.F does not yet exist as an attribute, a fresh variable is
    # inserted into T's attr_list (trailed) and unified with V.
    _dot_sym_check = (lambda td: td.type is not None and td.type.keyword is not None
                      and td.type.keyword.symbol == '.')
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
        return _unify(eng, _attr_cell.deref(), b_d)
    if _dot_sym_check(b_d):
        _attr_cell = _resolve_dot_feat(b_d, eng)
        if _attr_cell is None:
            return False
        return _unify(eng, a_d, _attr_cell.deref())

    # Unwrap backtick-quoted terms: `Expr = X → bind X to inner Expr (marked NON_STRICT_TERM).
    # In Wild Life, `Expr (backtick-quoted) "freezes" the expression to prevent evaluation.
    # We unwrap here so that subsequent feature unification (e.g. A=@(2=>val)) operates
    # on the actual arithmetic term rather than the backtick wrapper.
    from wild_life.data_structures import NON_STRICT_TERM as _BI_BQ_NST  # noqa: F811
    from wild_life.inference import _mark_arith_non_strict as _BI_MANS  # noqa: F811
    _bq_sym_check = (lambda td: td.type is not None and td.type.keyword is not None
                     and td.type.keyword.symbol == '`')
    _b_was_backtick = False
    if _bq_sym_check(b_d):
        _bq_inner = b_d.attr_list.get('1')
        if _bq_inner is not None:
            _bq_inner_d = _bq_inner.deref()
            _BI_MANS(_bq_inner_d)  # recursively mark arithmetic sub-terms as NON_STRICT
            b_d = _bq_inner_d
            _b_was_backtick = True
    _a_was_backtick = False
    if _bq_sym_check(a_d):
        _bq_inner = a_d.attr_list.get('1')
        if _bq_inner is not None:
            _bq_inner_d = _bq_inner.deref()
            _BI_MANS(_bq_inner_d)
            a_d = _bq_inner_d
            _a_was_backtick = True

    # Detect non-frozen arithmetic operator being applied via @(1,2)-style term.
    # Example: A=(+), A=@(1,2) — without backtick-freeze, `+` is an eager operator,
    # not a curriable function value; attempting to add args via apply merging is an error.
    from wild_life.data_structures import NON_STRICT_TERM as _BI_UNI_NST
    _wl_uni = eng.wl
    def _is_bare_arith_op(td):
        sym = td.type.keyword.symbol if (td.type and td.type.keyword) else ''
        return (sym in _ARITH_OPS_SET
                and not td.attr_list        # no existing args
                and not (td.flags & _BI_UNI_NST))  # not frozen
    # Use symbol-based check for apply type — the parsed @(1,2) may use the '@' symbol
    # definition rather than wl.apply which is set up later during boot.
    _b_sym_apply = b_d.type.keyword.symbol if (b_d.type and b_d.type.keyword) else ''
    _a_sym_apply = a_d.type.keyword.symbol if (a_d.type and a_d.type.keyword) else ''
    _b_is_apply_type = (b_d.type is not None and
                        (_b_sym_apply == '@' or b_d.type is _wl_uni.apply))
    _a_is_apply_type = (a_d.type is not None and
                        (_a_sym_apply == '@' or a_d.type is _wl_uni.apply))
    if _is_bare_arith_op(a_d) and _b_is_apply_type and b_d.attr_list:
        import sys as _sys_uni
        _sym_uni = a_d.type.keyword.symbol if (a_d.type and a_d.type.keyword) else '?'
        _sys_uni.stderr.write(f'*** Error: attempt to unify with curried function {_sym_uni}\n')
        return False
    if _is_bare_arith_op(b_d) and _a_is_apply_type and a_d.attr_list:
        import sys as _sys_uni2
        _sym_uni2 = b_d.type.keyword.symbol if (b_d.type and b_d.type.keyword) else '?'
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

    # Try to evaluate b as a user-defined function call (f -> result style).
    # EXCEPTION: 0-arity user functions (global variables like `result` declared
    # with `persistent` or `setq`) are handled later by direct synchronous
    # evaluation (line ~4061), NOT via an EVAL goal.  Using EVAL goals for them
    # would create arithmetic constraints when the stored value is an arithmetic
    # expression with unbound variables, causing spurious `real~` display.
    if _is_user_function(b_d) and b_d.attr_list:
        result = PsiTerm(type_def=eng.wl.top)
        # LIFO: push UNIFY first, then EVAL on top (EVAL executes first)
        eng.push_goal(GoalType.UNIFY, a_d, result, None)
        eng.push_goal(GoalType.EVAL, b_d, result, b_d.type.rule)
        return True

    # Try to evaluate a as a user-defined function call
    if _is_user_function(a_d) and a_d.attr_list:
        result = PsiTerm(type_def=eng.wl.top)
        eng.push_goal(GoalType.UNIFY, result, b_d, None)
        eng.push_goal(GoalType.EVAL, a_d, result, a_d.type.rule)
        return True

    # Handle cond(C, T, E) in functional position:
    #   X = cond(3 < 2, {}, f(N))  →  evaluate cond, unify result with X
    if _is_cond_builtin_local(b_d):
        evaled = _eval_body_sync(b_d, eng, 0)
        if evaled is None:
            return False
        evaled = _evaluate_result_for_display(evaled.deref(), eng, 1)
        return _unify(eng, a_d, evaled)
    if _is_cond_builtin_local(a_d):
        evaled = _eval_body_sync(a_d, eng, 0)
        if evaled is None:
            return False
        evaled = _evaluate_result_for_display(evaled.deref(), eng, 1)
        return _unify(eng, b_d, evaled)

    # Handle copy_term(X) functional use: Y = copy_term(X) → Y = fresh copy of X
    if _is_copy_term_func(b_d):
        c = _eval_copy_term_func(b_d)
        return _unify(eng, a_d, c)
    if _is_copy_term_func(a_d):
        c = _eval_copy_term_func(a_d)
        return _unify(eng, b_d, c)

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
        # If functor is a non-frozen arithmetic operator (no NON_STRICT_TERM), refuse
        # partial application — it's an eager operator, not a function value.
        # Frozen operators (`+`, `*`, etc.) carry NON_STRICT_TERM and are allowed.
        from wild_life.data_structures import NON_STRICT_TERM as _BI_APPLY_NST_CHK  # noqa: F811
        _fval_sym = _ftype.keyword.symbol if _ftype.keyword else ''
        if (_fval_sym in _ARITH_OPS_SET
                and not (_functor_val.flags & _BI_APPLY_NST_CHK)
                and not _functor_val.attr_list):  # bare arithmetic operator (no args yet)
            import sys as _sys_apply
            _sys_apply.stderr.write(f'*** Error: attempt to unify with curried function {_fval_sym}\n')
            return False
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

    # Handle asc(C) functional use: N = asc(C) → ASCII code of character C
    if _is_asc_func(b_d):
        ok, v = _eval_arith(b_d, eng)
        return _unify(eng, a_d, _make_number(eng, v)) if ok else False
    if _is_asc_func(a_d):
        ok, v = _eval_arith(a_d, eng)
        return _unify(eng, b_d, _make_number(eng, v)) if ok else False

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

    # Handle embedded disjunctions in RHS (e.g. [{1;2;3}|T] → [1|T], [2|T], [3|T])
    # Only do this when LHS is an unbound variable (binding case)
    a_is_var = (a_d.type is None or (a_d.type is eng.wl.top and not a_d.attr_list))
    if a_is_var and b_d.type is not None and _term_contains_disjunction(b_d, eng):
        alts = _expand_term_disjunctions(b_d, eng)
        if len(alts) > 1:
            for alt in reversed(alts[1:]):
                eng.push_choice_point(GoalType.UNIFY, a_d, alt, None)
            return _unify(eng, a_d, alts[0])

    # Pre-check: detect concrete non-boolean arguments in and/or expressions.
    # Wild Life emits "Non-boolean argument or result in '...'." when any direct
    # argument of an and/or operator is a concrete atom that is neither true nor
    # false (free variables and nested bool expressions are OK).
    # This check must run BEFORE _try_eval_bool so that 'true and c' shows
    # 'true and c' in the error message (not just 'c' after simplification).
    def _check_nonbool_bool_arg(expr_t):
        """Return True and print error if expr_t is and/or with a concrete non-boolean arg."""
        _sym_nb = _get_sym(expr_t)
        if _sym_nb not in ('and', 'or'):
            return False
        _a1_nb = expr_t.attr_list.get('1')
        _a2_nb = expr_t.attr_list.get('2')
        if _a1_nb is None or _a2_nb is None:
            return False
        _a1_nb = _a1_nb.deref()
        _a2_nb = _a2_nb.deref()
        _wl_nb = eng.wl
        from wild_life.data_structures import SORT_VAR as _SV_NB
        def _bool_arg_ok(t_ok):
            t_ok = t_ok.deref()
            s_ok = _get_sym(t_ok)
            if s_ok in ('true', 'false'):
                return True
            if _is_proper_bool_expr(t_ok):
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
    if _b_is_user_fn and not _b_is_non_strict and not b_d.attr_list and not _b_was_backtick:
        _0a_mark = eng.trail.mark()
        _b_evaled = _eval_user_func_sync(b_d, eng, 0)
        eng.trail.undo_to(_0a_mark)  # undo coref-linking of atom with rule-head copy
        if _b_evaled is not None:
            _b_evaled_d = _b_evaled.deref()
            # Only accept the evaluation if it produced a CONCRETE numeric value.
            # If it returned a compound expression (unevaluated or self-referential),
            # try one more arithmetic evaluation pass.  If that also fails, fall
            # through to normal unification (treats `result` as a variable).
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
                # else: keep original b_d (the 0-arity function atom) so that
                # normal unification treats `result` as a sort variable.
    b_arith = _try_eval_arith_to_term(b_d, eng) if (not _b_is_user_fn and not _b_is_non_strict) else None
    if b_arith is not None:
        # Expression fully evaluated — proceed to unify LHS with result.
        b_d = b_arith
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
                    return _unify(eng, _a_arith_lhs, b_d)
        # Arithmetic expression that couldn't be fully evaluated (has variables).
        wl = eng.wl
        b_sym = b_d.type.keyword.symbol if b_d.type and b_d.type.keyword else ''
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
            if b_simplified is not None:
                # Simplification succeeded — recurse to handle the simplified form.
                b_d = b_simplified
                b_arith2 = _try_eval_arith_to_term(b_d, eng)
                if b_arith2 is not None:
                    b_d = b_arith2
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
                    if ok_lhs and len(vars_in_expr) == 1:
                        x_var = vars_in_expr[0].deref()
                        coeffs = _get_linear_coeff(b_d, x_var, eng)
                        if coeffs is not None:
                            a_coeff, b_const = coeffs
                            # v_lhs = a_coeff * x + b_const  →  x = (v_lhs - b_const) / a_coeff
                            if a_coeff != 0.0:
                                x_val = (v_lhs - b_const) / a_coeff
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
                    # Case 1c: v = A/B with both A,B free vars
                    if ok_lhs and len(vars_in_expr) == 2:
                        b_d_sym2 = b_d.type.keyword.symbol if b_d.type and b_d.type.keyword else ''
                        if b_d_sym2 == '/':
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
                        # No free vars in expression — evaluate it and unify.
                        ok_eval, v_eval = _eval_arith(b_d, eng)
                        if ok_eval:
                            b_d = _make_number(eng, v_eval)
                        else:
                            # Concrete but unevaluable (e.g. division by zero, non-numeric
                            # atom argument).  Check if any immediate arg is a concrete
                            # non-numeric atom — if so, emit the standard Wild Life warning.
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
                        from wild_life.data_structures import Goal, Residuation
                        eq_defn = getattr(wl, 'eqsym', None) or wl.syntax_module.symbol_table.get('=')
                        eq_term = PsiTerm(type_def=eq_defn)
                        eq_term.attr_list['1'] = a_d
                        eq_term.attr_list['2'] = b_d
                        eq_term._resid_marker = True
                        pending_goal = Goal(GoalType.PROVE, eq_term, None, None, pending=True)
                        for v in vars_in_expr:
                            _attach_arith_resid(v, wl, pending_goal, eng)
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

    # Non-delaying string functions (psi2str, root_sort, children, chr evaluated already above)
    b_str = _try_eval_string_func(b_d, eng)
    if b_str is not None:
        b_d = b_str
    else:
        a_str = _try_eval_string_func(a_d, eng)
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
    eng.push_goal(GoalType.PROVE, arg1d, _DEFRULES_SENTINEL, None)
    _barrier = cp_save if cp_save is not None else _INNER_RUN_BARRIER
    result1 = eng.run(cs_barrier=_barrier)
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

    # ── Disjunction (;) ──
    if defn is wl.life_or or defn is wl.disjunction:
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

    # ── Built-in FUNCTION: arithmetic comparisons ──
    if defn._builtin_func is not None and defn.type == DefType.FUNCTION:
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
    args = list(goal.attr_list.values()) if goal.attr_list else []
    if len(args) < 2:
        return True   # degenerate: succeed
    cond_g = args[0].deref()
    then_g = args[1].deref()
    else_g = args[2].deref() if len(args) >= 3 else None

    if else_g is None:
        # ── 2-arg form: fully functional evaluation ──
        cond_result = _eval_as_bool_func(cond_g, eng)
        if cond_result is not True:
            # Cond is false or unresolvable → succeed silently (no Then)
            return True
        # Cond is true → evaluate Then as a boolean function
        then_result = _eval_as_bool_func(then_g, eng)
        return then_result is True

    # ── 3-arg form: predicate if-then-else ──
    mark = eng.trail.mark()
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack
    # IMPORTANT: clear goal_stack before the inner run so only cond_g is
    # proved — the continuation must NOT run inside the inner run.
    eng.goal_stack = None
    eng.push_goal(GoalType.PROVE, cond_g, _DEFRULES_SENTINEL, None)
    old_main_loop_ok = eng.main_loop_ok
    _barrier = cp_save if cp_save is not None else _INNER_RUN_BARRIER
    cond_ok = eng.run(cs_barrier=_barrier)
    eng.main_loop_ok = old_main_loop_ok

    eng.choice_stack = cp_save   # discard Cond's choice points either way
    eng.goal_stack = gs_save     # restore the outer continuation

    if cond_ok:
        # Cond succeeded → push Then
        eng.push_goal(GoalType.PROVE, then_g, _DEFRULES_SENTINEL, None)
        return True
    else:
        # Cond failed → undo its bindings, push Else
        eng.trail.undo_to(mark)
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
            # Copy template_copy with current bindings resolved
            collected.append(copy_term(template_copy))
            if not eng.choice_stack or eng.choice_stack is cp_save:
                break
            eng.backtrack()
        else:
            break

    eng.trail.undo_to(mark)
    eng.choice_stack = cp_save
    eng.goal_stack = gs_save
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


def bi_assert(goal: PsiTerm, eng) -> bool:
    """assert(Clause) / assertz(Clause)."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    # Evaluate arithmetic sub-expressions before storing so that
    # assert(mynum(N+1)) with N=31 stores mynum(32) not mynum(31+1).
    arg = _normalize_arith_in_term(arg, eng)
    eng.assert_first = False
    eng.assert_clause(arg)
    return True


def bi_asserta(goal: PsiTerm, eng) -> bool:
    """asserta(Clause) — add at front."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    # Evaluate arithmetic sub-expressions before storing.
    arg = _normalize_arith_in_term(arg, eng)
    eng.assert_first = True
    eng.assert_clause(arg)
    eng.assert_first = False
    return True


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
    # Build a body term if none given (unifies with 'true' / any body)
    if body is None:
        body = PsiTerm(type_def=wl.top)  # fresh var — will match any body
    # Use the engine's non-deterministic clause_aim machinery:
    # Push a DEL_CLAUSE goal with (master_list, start_idx=0) so clause_aim
    # always deletes from the master list at the correct position.
    rule_list = defn.rule  # live mutable list
    eng.push_goal(_GT.DEL_CLAUSE, head, body, (rule_list, 0))
    return True


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
    defn = lhs.type

    # Mode 1: LHS is a named function/predicate symbol (global variable)
    if (defn is not None and
            hasattr(defn, 'rule') and
            defn.rule is not None and
            lhs.value is None and
            not lhs.attr_list):
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

    if _backtrackable:
        # '<-': backtrackable in-place update of the dereferenced endpoint.
        # We must mutate `lhs` (the end of the deref chain) so that ALL
        # variables pointing into this chain see the new value, while trailing
        # each changed field so backtracking restores the original state.
        eng.trail.trail_psi(lhs, 'value')
        eng.trail.trail_psi(lhs, 'coref')
        eng.trail.trail_psi(lhs, 'type')
        eng.trail.trail_psi(lhs, 'attr_list')
        eng.trail.trail_psi(lhs, 'flags')
        if ok_arith:
            lhs.value = val
            lhs.coref = None
            lhs.attr_list = {}
            if lhs.type is None or lhs.type is eng.wl.top:
                lhs.type = eng.wl.real
        else:
            rhs = rhs_term
            lhs.value = rhs.value
            lhs.type = rhs.type
            lhs.attr_list = dict(rhs.attr_list)
            lhs.coref = rhs.coref
            lhs.flags = rhs.flags
        return True

    # '<<-': non-backtrackable (destructive in-place update)
    if ok_arith:
        lhs.value = val
        lhs.coref = None
        lhs.attr_list = {}
        if lhs.type is None or lhs.type is eng.wl.top:
            lhs.type = eng.wl.real
        return True

    # Non-arithmetic RHS: destructively copy rhs structure into lhs
    rhs = rhs_term
    lhs.value = rhs.value
    lhs.type = rhs.type
    lhs.attr_list = dict(rhs.attr_list)
    lhs.coref = None   # clear any forwarding pointer
    lhs.flags = rhs.flags
    lhs.resid = rhs.resid
    return True


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
    # When head.type is '->' (the clause arrow), the actual predicate is in attr '1'
    # and the body variable is in attr '2'.
    clause_container = None  # the head->body term to unify with full clause
    if (head.type and head.type.keyword and head.type.keyword.symbol == '->'
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


def bi_load(goal: PsiTerm, eng) -> bool:
    """load(File) — load a LIFE source file."""
    arg = _get_one_arg(goal)
    if arg is None:
        return False
    filename = str(arg.value) if arg.value else (
        arg.type.keyword.symbol if arg.type and arg.type.keyword else '')
    if not filename.endswith('.lf'):
        filename += '.lf'

    wl = eng.wl
    delay_count_before = len(wl.delay_rules)
    result = eng.load_file(filename)

    # C版 Wild Life は .lf ファイルのロード後にカレントモジュールを
    # user モジュールへ戻す。Python 版でも同じ動作を再現する。
    wl.current_module = wl.user_module

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
    a1 = goal.attr_list.get('1')
    a2 = goal.attr_list.get('2')
    a3 = goal.attr_list.get('3')
    if not (a1 and a2 and a3):
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


def _rule_to_string(h, b, wl):
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

    body_goals = split_conj(b) if b is not None else []

    # 共有 PrintState: head / body ゴール全体をまとめてスキャン
    ps = PrintState(outfile=io.StringIO())
    ps.const_quote = True
    ps.indent = False

    ps.go_through(h)
    for g in body_goals:
        ps.go_through(g)
    ps.insert_variables({}, False)

    # head を出力
    _pretty_tag_or_psi_term(ps, h, MAX_PRECEDENCE + 1, 0, wl)
    head_str = ps.outfile.getvalue()

    # body ゴールを個別に出力 (outfile を切り替えて再利用)
    goal_strs = []
    for g in body_goals:
        ps.outfile = io.StringIO()
        _pretty_tag_or_psi_term(ps, g, MAX_PRECEDENCE + 1, 0, wl)
        goal_strs.append(ps.outfile.getvalue())

    return head_str, goal_strs


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

    if not imported:
        # 自モジュール述語: dynamic 宣言ヘッダを表示
        print(f"\ndynamic({func_name})?")

    for h, b in active_rules:
        head_str, goal_strs = _rule_to_string(h, b, wl)

        if is_function:
            vs = goal_strs[0] if goal_strs else 'true'
            print(f"{head_str} -> {vs}.")
        elif imported:
            # インポート述語: 常に ':-' ボディ付きで表示 (各ゴール改行)
            if goal_strs:
                bs = ',\n        '.join(goal_strs)
            else:
                bs = 'succeed'
            print(f"{head_str} :-\n        {bs}.")
        else:
            # 自モジュール述語: succeed ボディは省略
            has_body = (b is not None and b.type is not None
                        and b.type.keyword is not None
                        and b.type.keyword.symbol != succeed_sym)
            if has_body:
                bs = ',\n        '.join(goal_strs) if goal_strs else 'succeed'
                print(f"{head_str} :-\n        {bs}.")
            else:
                print(f"{head_str}.")


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
        """collected imported entries を空行区切りで出力してリセット"""
        for k, d in enumerate(imported_pending):
            # k==0: プロンプト直後なので改行1つでプロンプト行を終わらせる
            # k>0 : 前エントリの末尾 \n に続く空行区切り
            print()
            _bi_listing_one(d, wl, imported=True)
        imported_pending.clear()

    i = 1
    while True:
        a = goal.attr_list.get(str(i))
        if a is None:
            break
        a_deref = a.deref() if hasattr(a, 'deref') else a
        defn = a_deref.type if a_deref.type else None

        if defn is not None and defn.type in (DefType.PREDICATE, DefType.FUNCTION):
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
                if not active_rules:
                    print(f"% '{func_name}' is a user-defined predicate with an empty definition.\n")
                else:
                    _bi_listing_one(defn, wl, imported=False)
        elif defn is not None and defn.type == DefType.UNDEF:
            # UNDEF の場合:
            #   clash_blocked スタブ → 衝突検出で作成済みのブロック → 無音成功
            #   それ以外 (未定義/非公開) → "% 'name' is undefined." を表示
            if not getattr(defn, 'clash_blocked', False):
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
    result = PsiTerm(type=defn)
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

def bi_trace(goal: PsiTerm, eng) -> bool:
    """trace — enable execution tracing."""
    if not eng.trace:
        eng.trace = True
        print("*** Tracing is turned on.", file=sys.stderr)
    return True


def bi_notrace(goal: PsiTerm, eng) -> bool:
    """notrace — disable execution tracing."""
    eng.trace = False
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
# map(F, List) → ResultList  (functional built-in)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_func(f_term: PsiTerm, arg: PsiTerm, eng) -> Optional[PsiTerm]:
    """Apply functor f_term to one argument, returning the result term.

    In Wild Life, F(X) is written as a psi-term whose type is F and whose
    '1' attribute is X.  For partial applications like *(2=>4), F already
    carries some attributes — we merge the new positional arg into position
    '1' (or the next free position).
    """
    from wild_life.unification import copy_term as _copy
    # Build a copy of f_term with arg placed into the first available
    # positional slot: if f_term has no '1', use '1'; otherwise use '2', etc.
    f_copy = _copy(f_term)
    f_copy = f_copy.deref()
    if '1' not in f_copy.attr_list:
        f_copy.attr_list['1'] = arg
    elif '2' not in f_copy.attr_list:
        f_copy.attr_list['2'] = arg
    else:
        # Fallback: create a new application term
        app = PsiTerm(type=f_copy.type)
        app.attr_list = dict(f_copy.attr_list)
        app.attr_list['1'] = arg
        f_copy = app
    return f_copy


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

    Each argument is either:
      - An atom name: declare it as a global (0-ary predicate returning its value)
      - A term  X <- Value: declare X and set its initial value to Value
      - A term  <-(X):     declare X as a global reference
    """
    wl = eng.wl

    def _do_global_arg(a):
        a = a.deref()
        sym = a.type.keyword.symbol if (a.type and a.type.keyword) else ''

        # Form: X <- Value
        if sym == '<-':
            lhs_ref = a.attr_list.get('1')
            rhs_ref = a.attr_list.get('2')
            if lhs_ref is None:
                return True
            lhs = lhs_ref.deref()
            lhs_sym = lhs.type.keyword.symbol if (lhs.type and lhs.type.keyword) else ''
            if not lhs_sym:
                return True
            # Evaluate rhs as arithmetic if possible
            rhs = None
            if rhs_ref is not None:
                rhs_d = rhs_ref.deref()
                ok, val = _eval_arith(rhs_d, eng)
                if ok:
                    rhs = _make_number(eng, val)
                else:
                    rhs = rhs_d
            # Register as a 0-ary function in current module
            defn = wl.update_symbol(wl.current_module, lhs_sym)
            from wild_life.data_structures import DefType as _DT
            defn.type = _DT.FUNCTION
            result_term = rhs if rhs is not None else PsiTerm(type=wl.top)
            defn.rule = [(PsiTerm(type=defn), result_term)]
            return True

        # Form: <-(X) — declare X as global reference
        if sym == '<-' and not a.attr_list.get('2'):
            inner = a.attr_list.get('1')
            if inner is not None:
                return _do_global_arg(inner)
            return True

        # Bare atom: declare as global (no initial value — evaluates to itself)
        if sym:
            defn = wl.update_symbol(wl.current_module, sym)
            from wild_life.data_structures import DefType as _DT
            if defn.type == _DT.UNDEF:
                defn.type = _DT.FUNCTION
                head = PsiTerm(type=defn)
                defn.rule = [(head, head)]  # f -> f (returns itself)
            return True

        return True

    # Iterate over positional arguments 1, 2, 3, ...
    i = 1
    while True:
        arg_ref = goal.attr_list.get(str(i))
        if arg_ref is None:
            break
        _do_global_arg(arg_ref)
        i += 1
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
    _reg('pretty_write', bi_write)     # alias: pretty_write = write
    _reg('writeq', bi_writeq)
    _reg('pretty_writeq', bi_writeq)   # alias: pretty_writeq = writeq
    _reg('write_canonical', bi_write_canonical)
    _reg('print', bi_print)
    _reg('print_depth', bi_print_depth)
    _reg('nl', bi_nl)
    _reg('write_err', bi_write_err)
    _reg('writeln', bi_writeln)
    _reg('put', bi_put_char)
    _reg('put_char', bi_put_char)
    _reg('get_char', bi_get_char)
    _reg('get', bi_get_char)
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
    _reg('open_out', bi_open_out)
    _reg('close', bi_close)

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
    _reg('implies', bi_call)   # implies(Goal) is an alias for call(Goal)
    _reg('once', bi_once)
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
        """delay_check(P): register delay checking for P. No-op here."""
        return True
    _reg('delay_check', _bi_delay_check)

    def _bi_dynamic(goal, eng):
        """dynamic(P): declare P as dynamic. Ensure the predicate has an empty rule list."""
        arg = _get_one_arg(goal)
        if arg is None:
            return True
        arg = arg.deref()
        # If the type has no rule, set it to an empty list so assert/retract work
        if arg.type and arg.type.rule is None:
            arg.type.rule = []
        return True
    _reg('dynamic', _bi_dynamic)

    def _bi_persistent(goal, eng):
        """persistent(P): declare P as a persistent (global) function variable.

        This initializes P's definition as a FUNCTION with an empty rule list
        so that subsequent `P <<- Value` calls use the global-variable (Mode 1)
        assignment path in bi_store_arrow — updating the shared Definition's
        rule list rather than destructively modifying a single PsiTerm instance.
        """
        arg = goal.attr_list.get('1')
        if arg is None:
            return True
        arg_d = arg.deref()
        defn = arg_d.type
        if defn is not None:
            # Ensure the definition is typed as FUNCTION with an initialized rule list
            if defn.rule is None:
                defn.rule = []
            if defn.type not in (DefType.FUNCTION, DefType.PREDICATE):
                defn.type = DefType.FUNCTION
        return True
    _reg('persistent', _bi_persistent)

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
        # Build list of attribute keys
        keys = list(t.attr_list.keys())
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

    def _make_strip_result(src, use_src_type):
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
        res = _make_strip_result(src, False)
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
        res = _make_strip_result(src, True)
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
        return defn is not None and defn.type == _DT.FUNCTION
    _reg('is_function', _bi_is_function)

    def _bi_is_predicate(goal, eng):
        """is_predicate(T): succeed if T is a user-defined predicate."""
        from wild_life.data_structures import DefType as _DT
        a1 = _get_one_arg(goal)
        if a1 is None:
            return False
        t = a1.deref()
        defn = t.type
        return defn is not None and defn.type == _DT.PREDICATE
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
        """has_feature(F, T) — succeeds if T has a feature named F."""
        a1 = goal.attr_list.get('1')  # feature name
        a2 = goal.attr_list.get('2')  # term
        if a1 is None or a2 is None:
            return False
        feat = a1.deref()
        term = a2.deref()
        # Determine feature name
        if feat.type is not None and feat.type.keyword is not None:
            fname = feat.type.keyword.symbol
        elif feat.value is not None:
            fname = str(feat.value)
        else:
            return False
        return fname in term.attr_list
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
            i += 1
        return True
    _reg('public', _bi_public)

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

    def _bi_display_modules(goal, eng):
        """display_modules — print info about all known modules."""
        for name, mod in sorted(wl.module_table.items()):
            opens = [m.module_name for m in mod.open_modules
                     if m.module_name not in ('bi', 'syntax')]
            sym_count = len(mod.symbol_table)
            if opens:
                print(f"Module '{name}': {sym_count} symbols, opens {opens}")
            else:
                print(f"Module '{name}': {sym_count} symbols")
        return True
    _reg('display_modules', _bi_display_modules)

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
