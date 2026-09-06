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


def _get_sym(t: PsiTerm) -> str:
    """Get the functor symbol of a term."""
    if t is None:
        return ''
    t = t.deref()
    return t.type.keyword.symbol if (t.type and t.type.keyword) else ''


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
        return None

    return None


def _try_eval_arith_to_term(t: PsiTerm, eng) -> Optional[PsiTerm]:
    """Try arithmetic evaluation, returning a PsiTerm or None."""
    ok, v = _eval_arith(t, eng)
    if not ok:
        return None
    # Only return if the original term was NOT already a number
    # (to avoid infinite recursion)
    if t is None:
        return None
    t = t.deref()
    if t.value is not None and not (t.type and t.type.keyword and
                                    t.type.keyword.symbol in ('+','-','*','/','//',
                                                               'mod','**','^','max','min',
                                                               'abs','sqrt','sin','cos','tan',
                                                               'exp','log','floor','ceiling')):
        return None  # already a number, no evaluation needed
    return _make_number(eng, v)


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
        # Only evaluate when both arguments are concrete strings
        if eng is not None:
            wl = eng.wl
            def _is_string(x):
                return (x.value is not None and x.type is not None
                        and x.type.is_subtype_of(wl.quoted_string))
            if not (_is_string(a1) or _is_string(a2)):
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
        # children(Sort) -> list of immediate child sorts
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = a1.deref()
        defn = a1.type
        if defn is None or eng is None:
            return None
        wl = eng.wl
        child_atoms = []
        for mod in wl._all_modules():
            for sym_name, child_defn in mod.symbol_table.items():
                if defn in getattr(child_defn, 'parents', []):
                    child_atoms.append(wl.make_atom(sym_name, mod))
        return wl.make_list(child_atoms)

    elif sym == 'features':
        # features(T) -> list of attribute labels
        a1 = t.attr_list.get('1')
        if a1 is None:
            return None
        a1 = a1.deref()
        keys = list(a1.attr_list.keys())
        wl = eng.wl
        lst = PsiTerm(type_def=wl.nil)
        lst.type = wl.nil
        for key in reversed(keys):
            try:
                n = int(key)
                kterm = wl.make_integer(n)
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
        if feat.value is not None and feat.type and feat.type.keyword:
            fsym = feat.type.keyword.symbol
            if fsym in ('integer', 'real'):
                fkey = str(int(feat.value))
            else:
                fkey = fsym
        elif feat.type and feat.type.keyword:
            fkey = feat.type.keyword.symbol
        else:
            return None
        val = term.attr_list.get(fkey)
        if val is None:
            return None
        return val.deref()

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

    # Recursively evaluate nested conjunctions
    if t1.type is not None and t1.type is wl.and_sym:
        t1 = _eval_and_conjunction(t1, eng)
        if t1 is None:
            return None
    else:
        t1 = _strip_bq(t1)

    if t2.type is not None and t2.type is wl.and_sym:
        t2 = _eval_and_conjunction(t2, eng)
        if t2 is None:
            return None
    else:
        t2 = _strip_bq(t2)

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

    # ── bottom type ({} / disj_nil): cannot be written ──────────────────────
    sym = t.type.keyword.symbol if (t.type and t.type.keyword) else ''
    if sym == '{}':
        raise _WriteFailure("bottom type '{}'")
    if wl and wl.disj_nil is not None and t.type is wl.disj_nil:
        raise _WriteFailure("bottom type disj_nil")

    # ── arithmetic evaluation (e.g. 23+23 → 46) ─────────────────────────────
    # Guard against cyclic terms (e.g. X = s(X)) causing infinite recursion.
    _arith_binary_ops = frozenset(('+', '-', '*', '/', '//', 'mod', '**', '^',
                                   'max', 'min'))
    _arith_unary_ops = frozenset(('abs', 'sqrt', 'sin', 'cos', 'tan', 'exp',
                                  'log', 'floor', 'ceiling', 'round',
                                  'truncate'))
    try:
        t_eval = _try_eval_arith_to_term(t, eng)
        if t_eval is not None:
            t = t_eval
        else:
            t_str = _try_eval_string_func(t, eng)
            if t_str is not None:
                t = t_str
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

    write_term(t, outfile=stream or sys.stdout, quoted=quoted, wl=eng.wl,
               var_tree=var_tree)


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
        # Evaluate arithmetic first
        try:
            t_eval = _try_eval_arith_to_term(arg, eng)
            if t_eval is not None:
                arg = t_eval
        except (RecursionError, Exception):
            pass
        # Unwrap backtick-quoted terms: `(X) → write X canonically
        if (arg.type is not None and arg.type.keyword is not None
                and arg.type.keyword.symbol == '`'):
            inner = arg.attr_list.get('1')
            if inner is not None:
                arg = inner.deref()
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

_ARITH_DEBUG = False  # Set True to debug arithmetic evaluation

def _eval_arith(t: PsiTerm, eng, _depth: int = 0) -> Tuple[bool, float]:
    """Evaluate an arithmetic expression. Returns (ok, value)."""
    if t is None or _depth > 40:
        return False, 0.0
    t = t.deref()
    wl = eng.wl
    sym = t.type.keyword.symbol if t.type and t.type.keyword else ''

    if t.value is not None and t.type and t.type.is_subtype_of(wl.real):
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
            _evaled_arg = _try_eval_string_func(_vd, eng)
            t_pre.attr_list[_k] = _evaled_arg if _evaled_arg is not None else _vd
        for _ri, (h0, b0) in enumerate(active):
            _vm: dict = {}
            head = copy_term(h0, _vm)
            body = copy_term(b0, _vm)
            body_d = body.deref()
            # Skip sort-constrained rules: head is a bare variable (no attrs).
            # These rules are X:sort -> body, designed for eval_aim not direct
            # arithmetic. Evaluating them causes infinite recursion because the
            # body typically calls features(X) which can't be resolved here.
            head_d = head.deref()
            if not head_d.attr_list:
                continue
            # Handle conditional: body = (value | condition) — skip if conditioned
            if body_d.type is not None and body_d.type is wl.such_that:
                continue  # Can't evaluate conditionals without engine; skip
            # Unify head with pre-evaluated copy of t to bind arguments
            mark = eng.trail.mark()
            ok = eng.unifier.unify(t_pre, head)
            if ok:
                result = _eval_arith(body_d, eng, _depth + 1)
                eng.trail.undo_to(mark)
                if result[0]:
                    return result
                # Evaluation failed; try next rule
            eng.trail.undo_to(mark)

    # Feature access: T.F → evaluate as arithmetic if possible
    if sym == '.':
        arg1, arg2 = _get_two_args(t)
        feat_val = _try_eval_string_func(t, eng)
        if feat_val is not None:
            return _eval_arith(feat_val, eng, _depth + 1)
        return False, 0.0

    # Binary operators
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
        # Bitwise NOT
        '\\': lambda a: float(~int(a)),
    }
    if sym in ops1 and ok1:
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
            # Run the guard in an inner proof loop.
            # IMPORTANT: clear goal_stack so only the guard is proved;
            # the outer continuation must not run inside this inner loop.
            from wild_life.inference import GoalType as _GoalType, _DEFRULES as _DR, _INNER_RUN_BARRIER as _IRB
            cp_save = eng.choice_stack
            gs_save = eng.goal_stack
            eng.goal_stack = None
            eng.push_goal(_GoalType.PROVE, cond_part, _DR, None)
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

        mark = eng.trail.mark()
        ok = eng.unifier.unify(t, head)
        if not ok:
            eng.trail.undo_to(mark)
            continue

        body_d2 = body_d.deref()
        # Try arithmetic evaluation
        ok_a, val = _eval_arith(body_d2, eng)
        if ok_a:
            return _make_number(eng, val)

        # Try recursive user-function evaluation
        if _is_user_function(body_d2):
            inner = _eval_user_func_sync(body_d2, eng, _depth + 1)
            if inner is not None:
                return inner
            # Body is a user function but can't eval synchronously — signal failure
            # (don't return body_d2 as that would be a wrong replacement for t)
            return None

        # Body is a compound with possible embedded user-function sub-terms
        # (e.g. [X|app2(L1,L2)] where the tail is a recursive function call).
        # Evaluate those sub-terms so the result is a fully-reduced term.
        _eval_embedded_user_funcs(body_d2, eng, _depth + 1, set())
        return body_d2

    return None


def _eval_embedded_user_funcs(
        t: PsiTerm, eng, _depth: int, visited: set) -> None:
    """Walk t's attribute tree and evaluate any user-function sub-terms.

    Modifies t's attr_list in-place (replacing function calls with their
    evaluated results).  t must be a fresh copy (not a stored rule term).
    """
    if _depth > 40:
        return
    td = t.deref()
    if id(td) in visited:
        return
    visited.add(id(td))
    for key in list(td.attr_list.keys()):
        child = td.attr_list[key].deref()
        if _is_user_function(child):
            evaled = _eval_user_func_sync(child, eng, _depth + 1)
            if evaled is not None and evaled is not child:
                td.attr_list[key] = evaled
                _eval_embedded_user_funcs(evaled, eng, _depth + 1, visited)
            else:
                _eval_embedded_user_funcs(child, eng, _depth + 1, visited)
        elif child.attr_list:
            _eval_embedded_user_funcs(child, eng, _depth + 1, visited)


def bi_unify(goal: PsiTerm, eng) -> bool:
    """X = Y — LIFE sort unification (with functional evaluation)."""
    a, b = _get_two_args(goal)
    if a is None or b is None:
        return a is b

    a_d = a.deref()
    b_d = b.deref()

    # Try to evaluate b as a user-defined function call (f -> result style)
    if _is_user_function(b_d):
        result = PsiTerm(type_def=eng.wl.top)
        # LIFO: push UNIFY first, then EVAL on top (EVAL executes first)
        eng.push_goal(GoalType.UNIFY, a_d, result, None)
        eng.push_goal(GoalType.EVAL, b_d, result, b_d.type.rule)
        return True

    # Try to evaluate a as a user-defined function call
    if _is_user_function(a_d):
        result = PsiTerm(type_def=eng.wl.top)
        eng.push_goal(GoalType.UNIFY, result, b_d, None)
        eng.push_goal(GoalType.EVAL, a_d, result, a_d.type.rule)
        return True

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

    # Handle conjunction on RHS: A = t1 & t2  →  A = t1, A = t2 (psi-term merge)
    # '&' is psi-term conjunction: the result must satisfy BOTH constraints.
    if b_d.type is not None and b_d.type is eng.wl.and_sym:
        t1 = b_d.attr_list.get('1')
        t2 = b_d.attr_list.get('2')
        if t1 is not None and t2 is not None:
            # Push the second unification as the next goal; do the first inline.
            eng.push_goal(GoalType.UNIFY, a_d, t2.deref(), None)
            return _unify(eng, a_d, t1.deref())

    # Handle conjunction on LHS: t1 & t2 = B  →  t1 = B, t2 = B (symmetric)
    if a_d.type is not None and a_d.type is eng.wl.and_sym:
        t1 = a_d.attr_list.get('1')
        t2 = a_d.attr_list.get('2')
        if t1 is not None and t2 is not None:
            eng.push_goal(GoalType.UNIFY, t2.deref(), b_d, None)
            return _unify(eng, t1.deref(), b_d)

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

    # Try to evaluate functional terms before unifying (boolean ops)
    b_evaled = _try_eval_bool(b_d, eng)
    if b_evaled is not None:
        b_d = b_evaled
    else:
        a_evaled = _try_eval_bool(a_d, eng)
        if a_evaled is not None:
            a_d = a_evaled
    # Try arithmetic evaluation on the RHS (for A = 1+2 style)
    b_arith = _try_eval_arith_to_term(b_d, eng)
    if b_arith is not None:
        b_d = b_arith
    # Try string function evaluation on RHS (psi2str, str2psi, strcon)
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
            collected.insert(0, copy_term(template_copy))
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
    """
    args = list(goal.attr_list.values()) if goal.attr_list else []
    if len(args) < 2:
        return False
    x_term = args[0].deref()
    v_term = args[1].deref()
    wl = eng.wl

    defn = x_term.type
    if defn is None:
        return False

    # Make X dynamic if it is not already
    from wild_life.data_structures import DefType
    if defn.rule is None or callable(defn.rule):
        defn.rule = []

    # Remove ALL existing -> rules for X (retract all functional clauses)
    # A functional rule is stored as (head, body) where head matches x_term
    rule_list = defn.rule
    new_rules = [(h, b) for (h, b) in rule_list if h is None]  # keep tombstones? No — clear all
    defn.rule = []  # wipe all rules

    # Assert X -> V  (a single-arg functional rule)
    # Build head = x_term (fresh copy) with value = v_term
    from wild_life.unification import copy_term
    _vm: dict = {}
    head_copy = copy_term(x_term, _vm)
    # The rule body for -> is the return value directly
    defn.rule.append((head_copy, v_term))
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
    # Collect subtypes
    children = []
    for mod in wl._all_modules():
        for sym_name, child_defn in mod.symbol_table.items():
            if defn in getattr(child_defn, 'parents', []):
                child_atom = wl.make_atom(sym_name, mod)
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


def _bi_listing_one(defn, wl) -> None:
    """Helper: list clauses for a single Definition."""
    from wild_life.data_structures import DefType
    from wild_life.print_term import term_to_string

    if defn is None or defn.keyword is None:
        return
    active_rules = [(h, b) for h, b in (defn.rule or []) if h is not None]
    if not active_rules:
        return
    func_name = defn.keyword.symbol
    print(f"\ndynamic({func_name})?")
    is_function = (defn.type == DefType.FUNCTION)
    succeed_sym = wl.succeed.keyword.symbol if wl.succeed and wl.succeed.keyword else 'succeed'
    for h, b in active_rules:
        hs = term_to_string(h, wl=wl)
        if is_function:
            vs = term_to_string(b, wl=wl) if b is not None else 'true'
            print(f"{hs} -> {vs}.")
        else:
            has_body = (b is not None and b.type is not None
                        and b.type.keyword is not None
                        and b.type.keyword.symbol != succeed_sym)
            if has_body:
                bs = term_to_string(b, wl=wl)
                print(f"{hs} :-\n        {bs}.")
            else:
                print(f"{hs}.")


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
    """listing(F) — list clauses for functor F.

    Expected output format (matching original Wild Life):
      - Empty predicate: % 'NAME' is a user-defined predicate with an empty definition.\\n
      - Non-empty predicate:
          \\ndynamic(NAME)?
          HEAD :-
                  BODY.
      - Functional rules:
          \\ndynamic(NAME)?
          HEAD -> VALUE.
    """
    from wild_life.data_structures import DefType
    from wild_life.print_term import term_to_string

    wl = eng.wl
    arg = _get_one_arg(goal)
    if arg is None:
        # listing with no args: list all user-defined predicates/functions
        _bi_listing_all(eng, wl)
        return True
    defn = arg.type if arg.type else None
    if defn is None:
        return False

    func_name = defn.keyword.symbol if defn.keyword else '?'

    # Collect non-deleted rules
    active_rules = [(h, b) for h, b in (defn.rule or []) if h is not None]

    if not active_rules:
        # Empty definition
        print(f"% '{func_name}' is a user-defined predicate with an empty definition.\n")
        return True

    # Print dynamic declaration header (with leading blank line)
    print(f"\ndynamic({func_name})?")

    is_function = (defn.type == DefType.FUNCTION)
    succeed_sym = wl.succeed.keyword.symbol if wl.succeed and wl.succeed.keyword else 'succeed'

    for h, b in active_rules:
        hs = term_to_string(h, wl=wl)
        if is_function:
            # Functional rule: HEAD -> VALUE.
            vs = term_to_string(b, wl=wl) if b is not None else 'true'
            print(f"{hs} -> {vs}.")
        else:
            # Regular predicate clause
            has_body = (b is not None and b.type is not None
                        and b.type.keyword is not None
                        and b.type.keyword.symbol != succeed_sym)
            if has_body:
                bs = term_to_string(b, wl=wl)
                print(f"{hs} :-\n        {bs}.")
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
    _reg('get_attribute', bi_get_attribute)
    _reg('set_attribute', bi_set_attribute)
    _reg('type_of', bi_type_of)
    _reg('functor_of', bi_functor_of)

    # Alias / sort manipulation
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
        """persistent(P): declare P as persistent. No-op here."""
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
        # Build list of attribute keys
        keys = list(t.attr_list.keys())
        # Build WL list from keys
        wl = eng.wl
        lst = wl.nil
        for key in reversed(keys):
            # Key may be numeric ("1","2") or named
            try:
                n = int(key)
                kterm = wl.make_integer(n)
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

    def _bi_strip(goal, eng):
        """strip(T): return T without sort constraints (just top-level copy)."""
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is None:
            return False
        t = a1.deref()
        # Return the term as-is (stripping sort annotation not implemented)
        if a2 is None:
            return True
        return _unify(eng, a2.deref(), t)
    _reg('strip', _bi_strip)

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
