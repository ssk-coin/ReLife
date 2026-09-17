"""
inference.py — The Wild Life proof / inference engine.
Corresponds to login.c + lefun.c in the original C source.

The engine is goal-stack based (no Python recursion for the main loop),
using explicit choice points for backtracking.
"""

from __future__ import annotations
import sys
import time
from typing import Optional, List, Tuple, Any, Dict

from wild_life.data_structures import (
    PsiTerm, Definition, GoalType, Goal, ChoicePoint,
    DefType, FACT, QUERY, ERROR, featcmp_key,
)
from wild_life.unification import (
    UnificationFailure, CutException, HaltException, AbortException,
    SortCycleException, Trail, Unifier, copy_term, compute_lub, types_compatible
)


# ─────────────────────────────────────────────────────────────────────────────
# Non-strict predicate helpers
# ─────────────────────────────────────────────────────────────────────────────

_ARITH_OPS_NON_STRICT = frozenset((
    '+', '-', '*', '/', '//', 'mod', '**', '^',
    'max', 'min', 'abs', 'sqrt', 'floor', 'ceiling',
    'round', 'truncate', 'exp', 'log', 'sin', 'cos', 'tan',
))

# Built-in functions whose reduction a such-that rule may have to wait for:
# the guard is what binds their arguments.
_DEFERRABLE_BUILTIN_FUNCS = frozenset((
    'psi2str', 'str2psi', 'strcon', 'makestr', 'strlen', 'substr',
    'root_sort', 'children', 'length', 'append', 'features', 'chr', 'int2str',
))


def _leftmost_goal(t: 'PsiTerm', wl) -> 'PsiTerm':
    """The first goal a conjunction runs, or t itself when it is not one."""
    seen = 0
    t = t.deref()
    while seen < 64:
        if t.type is not wl.commasym or '1' not in t.attr_list:
            return t
        t = t.attr_list['1'].deref()
        seen += 1
    return t


def _occurs_by_identity(target: PsiTerm, t: PsiTerm, visited: set = None) -> bool:
    """Whether *t* holds the very psi-term *target*, anywhere under it.

    A rule head that reads its own call — the `X` of `X:sum -> …` — is the
    same psi-term as the X in the body, because head and body are copied
    together.  A rule head that merely names the function — `quadruple` in
    `quadruple -> *(2 => 4)` — is not.
    """
    if target is None or t is None:
        return False
    if visited is None:
        visited = set()
    td = t.deref()
    if td is target or td is target.deref():
        return True
    if id(td) in visited:
        return False
    visited.add(id(td))
    return any(_occurs_by_identity(target, v, visited)
               for v in td.attr_list.values())


def _mark_non_strict_args(t: PsiTerm, eng, visited: set = None) -> None:
    """Freeze the arithmetic that a non-strict call's arguments stand for.

    A predicate declared non_strict does not evaluate what it is given, and in
    C Wild Life that reaches the whole clause the call sits in: once `foo(N)`
    is non-strict, the `N:(2*4)` elsewhere in the same clause reads as `2 * 4`
    for every other call too, since both are the one variable.
    """
    non_strict = getattr(eng, 'non_strict_set', None)
    if not non_strict or t is None:
        return
    if visited is None:
        visited = set()
    t = t.deref()
    if id(t) in visited:
        return
    visited.add(id(t))
    if t.type in non_strict:
        for arg in t.attr_list.values():
            _mark_arith_non_strict(arg)
    for sub in t.attr_list.values():
        _mark_non_strict_args(sub, eng, visited)


_STRICT_ARITH_SYMS = frozenset((
    '+', '-', '*', '/', '//', 'mod', '**', '^', 'max', 'min',
    '/\\', '\\/', 'xor', '>>', '<<'))


def _mark_arith_non_strict(t: PsiTerm, visited: set = None, eng=None) -> None:
    """Recursively mark arithmetic operator psiterms with NON_STRICT_TERM.

    Called after head unification for a non-strict predicate so that
    arithmetic sub-expressions in the bound result are not eagerly
    evaluated during printing.

    Pass *eng* when the marking belongs to one solution rather than to the
    program text: the flag is then trailed, so backtracking hands the term
    back unfrozen.  `assert(jolly(3+X) :- …)` freezes the sum it stores for
    that X, and the next X finds `3+X` ready to be worked out again.
    """
    from wild_life.data_structures import NON_STRICT_TERM
    if visited is None:
        visited = set()
    if t is None:
        return
    # Follow coref chain to the actual bound psiterm, then deduplicate
    td = t.deref()
    if td is None:
        return
    tdid = id(td)
    if tdid in visited:
        return
    visited.add(tdid)
    sym = td.type.keyword.symbol if (td.type and td.type.keyword) else ''
    if sym in _ARITH_OPS_NON_STRICT and td.value is None:
        if eng is not None and not (td.flags & NON_STRICT_TERM):
            eng.trail.trail_psi(td, 'flags')
        td.flags |= NON_STRICT_TERM
    for v in td.attr_list.values():
        _mark_arith_non_strict(v, visited, eng)


# ─────────────────────────────────────────────────────────────────────────────
# Disjunction expansion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_disj_elems(t: PsiTerm, wl) -> list:
    """Collect all leaf elements from a disjunction linked-list {a;b;c}.
    {a;b;c} is stored as disj(a, disj(b, disj(c, disj_nil))).
    Returns [a, b, c].
    """
    elems = []
    node = t
    while node is not None:
        node = node.deref()
        if node.type is None or node.type is wl.disj_nil:
            break
        if node.type is wl.disjunction:
            head = node.attr_list.get('1')
            tail = node.attr_list.get('2')
            if head is not None:
                elems.append(head.deref())
            node = tail.deref() if tail else None
        else:
            elems.append(node)
            break
    return elems


def _expand_head_disj(head: PsiTerm, wl, depth: int = 0) -> list:
    """Expand disjunctions in a head term into a list of alternative terms.

    E.g., f({a;b}, {c;d}) → [f(a,c), f(a,d), f(b,c), f(b,d)]
         pick_arg({5;3;7}) → [pick_arg(5), pick_arg(3), pick_arg(7)]

    Returns [head] if no disjunctions are found.
    """
    if depth > 8:
        return [head]
    head_d = head.deref()
    if head_d.type is None:
        return [head_d]
    if head_d.type is wl.disjunction:
        return _collect_disj_elems(head_d, wl)

    attr_keys = list(head_d.attr_list.keys())
    if not attr_keys:
        return [head_d]

    # Build Cartesian product of attribute alternatives
    combos = [{}]
    has_disj = False
    for key in attr_keys:
        attr_val = head_d.attr_list[key]
        val_d = attr_val.deref()
        # Only expand if the attribute IS the disjunction directly (attr_val is val_d),
        # not when deref'd THROUGH a variable wrapper (attr_val is not val_d).
        # A variable wrapper (type=Def('variable')) with coref pointing to a disjunction
        # represents a SORT-CONSTRAINED FORMAL PARAMETER, e.g. A:{1;2;3} in a clause head.
        # Expanding it at load time would lose the shared reference between head and body;
        # the disjunction must instead be expanded at RUNTIME during head unification.
        if attr_val is not val_d:
            # Dereffed through a variable: keep the attribute as-is (no expansion).
            alts = [attr_val]
        else:
            alts = _expand_head_disj(val_d, wl, depth + 1)
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
        return [head_d]

    result = []
    for attrs in combos:
        new_term = PsiTerm(type_def=head_d.type, value=head_d.value)
        new_term.attr_list = attrs
        result.append(new_term)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cut barrier helper
# ─────────────────────────────────────────────────────────────────────────────

def _patch_cut_barriers(term: PsiTerm, wl, cut_point, seen=None) -> None:
    """Recursively set cut atoms' .value to cut_point in a copied body.

    In WAM semantics, '!' inside a predicate's body cuts to the choice point
    that was current when the predicate was CALLED (B0 register).  After
    copy_term the cut atoms in the copy still have value=None, so we patch
    them here before pushing the body onto the goal stack.
    """
    if term is None:
        return
    if seen is None:
        seen = set()
    oid = id(term)
    if oid in seen:
        return
    seen.add(oid)
    # Dereference (may be a bound variable — PsiTerm uses .coref)
    while term.coref is not None:
        term = term.coref
    if term.type is wl.cut:
        term.value = cut_point
        return   # cut atom has no meaningful subterms
    for v in term.attr_list.values():
        _patch_cut_barriers(v, wl, cut_point, seen)


# ─────────────────────────────────────────────────────────────────────────────
# Functional cond(C, T, E) evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _is_cond_builtin(t: 'PsiTerm') -> bool:
    """Return True if t is a built-in cond(…) call (not a user-defined one)."""
    if t is None:
        return False
    t = t.deref()
    if t.type is None or t.type.keyword is None:
        return False
    if t.type.keyword.symbol != 'cond':
        return False
    return getattr(t.type, '_builtin_func', None) is not None


def _eval_body_to_result(branch: 'PsiTerm', result: 'PsiTerm', eng) -> bool:
    """Evaluate an expression branch into result.

    Handles: arithmetic, user-defined functions, nested cond, and compound
    terms with embedded user-function sub-terms.
    Called from _eval_cond_functional and eval_aim.
    """
    from wild_life.built_ins import _eval_arith, _make_number, _is_user_function

    branch_d = branch.deref()

    # Arithmetic?
    arith_ok, arith_val = _eval_arith(branch_d, eng)
    if arith_ok:
        num = _make_number(eng, arith_val)
        return eng.unifier.unify(result, num)

    # User-defined function call?
    if _is_user_function(branch_d):
        eng.push_goal(GoalType.EVAL, branch_d, result, branch_d.type.rule)
        return True

    # Nested built-in cond?
    if _is_cond_builtin(branch_d):
        return _eval_cond_functional(branch_d, result, eng)

    # Disjunction branch: evaluate each element (including arithmetic like
    # 1 + posint_stream_to(N)) via _eval_body_sync, then unify result with
    # the produced disjunction.  The generic _collect_embedded_func_goals path
    # below cannot evaluate 1 + {disjunction} because the arithmetic wrapping
    # the embedded function call is not reduced after the EVAL goal fires.
    wl = eng.wl
    if branch_d.type is not None and branch_d.type is wl.disjunction:
        from wild_life.built_ins import _eval_body_sync
        evaled = _eval_body_sync(branch_d, eng, 0)
        if evaled is not None:
            return eng.unifier.unify(result, evaled)
        # fall through on sync failure

    # Sort-conjunction `A & where(B)` pattern — evaluate where(B) synchronously
    # first so that its arguments are evaluated for side effects (binding
    # variables in A), then evaluate A with those variables bound.
    # Without this, LIFO goal-stack ordering would run bodify_list(B) before
    # copy_body binds B, causing an infinite loop.
    _br_kw = branch_d.type.keyword if branch_d.type else None
    if (_br_kw is not None and _br_kw.symbol == '&'
            and '1' in branch_d.attr_list and '2' in branch_d.attr_list):
        from wild_life.built_ins import _is_user_function as _iuf_and, _eval_body_sync as _ebs_and
        _b_lhs = branch_d.attr_list['1'].deref()
        _b_rhs = branch_d.attr_list['2'].deref()
        # If RHS is a user function (e.g. where(side_effects)), evaluate it
        # synchronously first so that its argument's side effects fire before
        # any EVAL goals from LHS run.
        if _iuf_and(_b_rhs):
            _ebs_and(_b_rhs, eng, 0)   # side effects: binds vars in _b_lhs
            return _eval_body_to_result(_b_lhs, result, eng)

    # Compound with embedded user-function sub-terms
    from wild_life.built_ins import _is_user_function as _iuf_btr
    eval_goals = _collect_embedded_func_goals(branch_d, eng, set())
    eng.push_goal(GoalType.UNIFY, branch_d, result, None)
    # Push top-level eval goals first, then push all deferred sub-goals last.
    # Deferring sub-goals ensures they run BEFORE top-level goals in LIFO order,
    # so e.g. copy_body (sub of where's arg) runs before bodify_list (top-level).
    deferred_subs = []
    for ft, rv, rl in eval_goals:
        eng.push_goal(GoalType.EVAL, ft, rv, rl)
        for _ak in list(ft.attr_list.keys()):
            _av = ft.attr_list[_ak].deref()
            if not _iuf_btr(_av) and _av.attr_list:
                _sub_goals = _collect_embedded_func_goals(_av, eng, set())
                for _sft, _srv, _srl in _sub_goals:
                    deferred_subs.append((_sft, _srv, _srl))
    for _sft, _srv, _srl in deferred_subs:
        eng.push_goal(GoalType.EVAL, _sft, _srv, _srl)
    return True


def _boolean_value(t: 'PsiTerm'):
    """True/False for a `true`/`false` psi-term, None for anything else."""
    if t is None:
        return None
    d = t.deref()
    if d.type is None or d.type.keyword is None:
        return None
    sym = d.type.keyword.symbol
    return True if sym == 'true' else (False if sym == 'false' else None)


def prove_cond(cond_g: 'PsiTerm', eng) -> bool:
    """Prove the condition of a cond(C, T[, E]) in an inner run.

    `&` in a condition is LIFE's type intersection rather than a conjunction:
    in `cond(deja_vu(X,Table) & bool(Copy), ...)` the left side reduces to
    `true(V)` or `false`, intersecting it with `bool(Copy)` picks V up as
    Copy, and the condition holds exactly when the value is `true`.  Proving
    the two sides as separate goals would instead fail on `bool`, which is a
    sort and not a predicate — so an `&` condition is evaluated first, and
    only a value that is neither `true` nor `false` falls back to a proof.
    """
    if cond_g.type is not None and cond_g.type is eng.wl.and_sym:
        from wild_life.built_ins import _eval_body_sync
        mark = eng.trail.mark()
        truth = _boolean_value(_eval_body_sync(cond_g, eng, 0))
        if truth is not None:
            return truth
        eng.trail.undo_to(mark)
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack
    # Clear the goal stack so only cond_g is proved — the outer continuation
    # must not run inside the inner loop.
    eng.goal_stack = None
    eng.push_goal(GoalType.PROVE, cond_g, _DEFRULES, None)
    old_ok = eng.main_loop_ok
    barrier = cp_save if cp_save is not None else _INNER_RUN_BARRIER
    ok = eng.run(cs_barrier=barrier)
    eng.main_loop_ok = old_ok
    eng.choice_stack = cp_save
    eng.goal_stack = gs_save
    return ok


def _is_open_head_var(h: 'PsiTerm', wl) -> bool:
    """A head position that is a variable: matching binds it to the call's term."""
    from wild_life.data_structures import SORT_VAR as _SV
    return (not h.attr_list and h.value is None
            and (h.type is None or h.type is wl.top or bool(h.flags & _SV)))


class DeclarationError(Exception):
    """A declaration the reader could parse but that says too little."""


def _is_open_call_term(c: 'PsiTerm', wl) -> bool:
    """Whether the call's own term can still become something narrower.

    A variable can: `i(X:t1(l => t3))` waits for X and answers once it is an
    a.  A term the call states outright cannot — `g(t2)` is a t2, so a rule
    whose head asks for a t1 will never apply to it and simply fails.
    """
    from wild_life.data_structures import SORT_VAR as _SV_c
    if c.type is None or c.type is wl.top:
        return True
    if c.flags & _SV_c:
        return True
    if c.value is not None:
        # A number carries features like anything else: `X = 23` waiting on
        # `h(23(1),…)` is answered by `X = @(1)`.
        return True
    return bool(c.resid)


def _add_blocker(out: list, t: 'PsiTerm') -> None:
    if not any(b is t for b in out):
        out.append(t)


def _match_one(c: 'PsiTerm', h: 'PsiTerm', out: list, eng, seen: set,
               bindings: dict, stuck: list, depth: int = 0):
    """How far the call's term c already matches the head's term h.

    Matching is one-way: a head position that is a variable takes the call's
    term, and a second position naming that same variable asks for that very
    term again — which is why `f(X,s(X))` applied to f(X',s(Z)) waits for X'
    and Z to become one rather than making them one.  `bindings` carries what
    each head variable has taken so far.

    Returns 'never' when no narrowing of the call could ever match, and
    otherwise whether the head's demands left anything inside c unsettled,
    having noted in `out` the terms whose narrowing would settle it.
    """
    if depth > 20:
        return False
    c = c.deref()
    h = h.deref()
    if c is h:
        return False
    key = (id(c), id(h))
    if key in seen:
        return False
    seen.add(key)
    wl = eng.wl
    if _is_open_head_var(h, wl):
        _sort_pending = False
        if h.type is not None and h.type is not wl.top:
            # `X:c` still asks the call's term to be under that sort.
            if c.type is None or not c.type.is_subtype_of(h.type):
                if not types_compatible(c.type, h.type):
                    return 'never'
                _add_blocker(out, c)
                _sort_pending = True
        taken = bindings.get(id(h))
        if _sort_pending:
            # The sort is not met yet, but the variable still stands for this
            # term as far as the positions after it are concerned: `h(X:c,X)`
            # applied to h(A:a,B) waits on B as well as on A.
            if taken is None:
                bindings[id(h)] = c
                return True
        if taken is None:
            bindings[id(h)] = c
            return False
        if taken is c:
            return False
        _paired = _pair_blockers(taken, c, out, set(), eng)
        if _paired == 'never':
            return 'never'
        if not _paired:
            # The two differ nowhere a binding could reach, yet they are still
            # two terms and the head asks for one.
            stuck[0] = True
        # What the two terms wait on is noted between themselves; the term
        # that holds them does not wait with them.
        return False
    taken = bindings.get(id(h))
    if taken is not None:
        if taken is c:
            return False
        # The head asks for this very term again — `f(X:s(X))` asks the call's
        # term to be its own first feature.
        _paired = _pair_blockers(taken, c, out, set(), eng)
        if _paired == 'never':
            return 'never'
        if not _paired:
            stuck[0] = True
        return False
    bindings[id(h)] = c

    noted = False
    if h.type is not None and h.type is not wl.top:
        if c.type is None or not c.type.is_subtype_of(h.type):
            if not types_compatible(c.type, h.type):
                return 'never'
            # c can still be narrowed under h's sort, and what it already
            # carries is waiting on the head's demands just the same.
            _add_blocker(out, c)
            noted = True
    if h.value is not None:
        if c.value is None:
            _add_blocker(out, c)
            noted = True
        elif c.value != h.value:
            return 'never'
    for k, hv in h.attr_list.items():
        cv = c.attr_list.get(k)
        if cv is None:
            if not _is_open_call_term(c, wl):
                return 'never'
            # The call can still gain the feature.
            _add_blocker(out, c)
            return True
        below = _match_one(cv, hv, out, eng, seen, bindings, stuck, depth + 1)
        if below == 'never':
            return 'never'
        if below:
            noted = True
    if noted:
        # Something the head asks for is unsettled inside c, so c is waiting
        # on it too — disequality5 marks X as well as the Y within it.
        _add_blocker(out, c)
    return noted


def _rule_match_status(head: 'PsiTerm', call: 'PsiTerm', eng):
    """Whether this rule applies to the call, cannot, or is not settled yet.

    Matching is one-way: the head's variables take the call's terms, while the
    call's own terms are never narrowed to make a rule fit.  A call that is
    not specific enough yet therefore waits — the answer is the list of terms
    whose narrowing would settle it — and one that no narrowing could ever
    make fit reports 'never', so the next rule is tried.  'stuck' is the third
    case: the call is as settled as it will get and still does not match, so
    it neither applies the rule nor waits.
    """
    if head is None or call is None or not head.attr_list:
        return 'ready'
    wl = eng.wl

    keys = [k for k in sorted(head.attr_list, key=featcmp_key)
            if k in call.attr_list]
    if not keys:
        return 'ready'

    # Positions belong together where either side shares a term: a call that
    # passes one term to several positions has to meet what all of them ask at
    # once, and a head naming one variable in several positions asks those
    # positions of the call to agree.
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    first_call: dict = {}
    first_head: dict = {}
    for k in keys:
        cd = call.attr_list[k].deref()
        hd = head.attr_list[k].deref()
        if id(cd) in first_call:
            union(k, first_call[id(cd)])
        else:
            first_call[id(cd)] = k
        if _is_open_head_var(hd, wl):
            if id(hd) in first_head:
                union(k, first_head[id(hd)])
            else:
                first_head[id(hd)] = k

    groups: dict = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)

    # Can a group's demands be met at all?  `h(X:c,X)` asks both of h(a,b)'s
    # arguments to be one term under c, and a and b have no common sub-sort
    # under c, so no narrowing could ever make that rule fit.  Asked on copies,
    # so that nothing the call carries is narrowed by the question.
    _never_mark = eng.trail.mark()
    _was_firing = getattr(eng, '_in_fire_delay', False)
    # The question is asked by unifying copies, and a delay rule firing on one
    # of them would be an answer written out to the user: manual8's
    # `:: I:int | write(I," ")` would report the 0 in `fact(0)`'s head before
    # the call had anything to do with it.
    eng._in_fire_delay = True
    _was_skipping = eng.unifier._skip_prototypes
    eng.unifier._skip_prototypes = True
    try:
        _hm: dict = {}
        _cm: dict = {}
        for ks in groups.values():
            merged = PsiTerm(type_def=wl.top)
            for k in ks:
                try:
                    if not eng.unifier.unify(merged,
                                             copy_term(head.attr_list[k], _hm)):
                        return 'never'
                    if not eng.unifier.unify(merged,
                                             copy_term(call.attr_list[k], _cm)):
                        return 'never'
                except UnificationFailure:
                    return 'never'
    finally:
        eng._in_fire_delay = _was_firing
        eng.unifier._skip_prototypes = _was_skipping
        eng.trail.undo_to(_never_mark)

    blockers: list = []
    bindings: dict = {}
    stuck = [False]
    seen: set = set()
    mark = eng.trail.mark()
    try:
        head_map: dict = {}
        asked: dict = {}
        for ks in groups.values():
            merged = copy_term(head.attr_list[ks[0]], head_map)
            for k in ks[1:]:
                try:
                    if not eng.unifier.unify(
                            merged, copy_term(head.attr_list[k], head_map)):
                        return 'never'
                except UnificationFailure:
                    return 'never'
            for k in ks:
                asked[k] = merged
        for k in keys:
            if _match_one(call.attr_list[k], asked[k], blockers, eng, seen,
                          bindings, stuck) == 'never':
                return 'never'
    finally:
        eng.trail.undo_to(mark)
    if blockers:
        return blockers
    return 'stuck' if stuck[0] else 'ready'


def _is_cyclic(t: 'PsiTerm', _seen: frozenset = frozenset(),
               depth: int = 0) -> bool:
    """Whether the term comes round to itself, and so stands for an infinite one."""
    if depth > 30:
        return True
    t = t.deref()
    if id(t) in _seen:
        return True
    if not t.attr_list:
        return False
    below = _seen | {id(t)}
    return any(_is_cyclic(v, below, depth + 1) for v in t.attr_list.values())


def _pair_blockers(a: 'PsiTerm', b: 'PsiTerm', out: list, seen: set, eng,
                   depth: int = 0) -> bool:
    """Note the terms whose narrowing could still make a and b one term.

    Whether they can differ at all is what decides it: two terms that already
    agree down to the very same variables are noted nowhere, because no
    binding could bring the two of them together.  Where they do differ, every
    term on the way down to the difference is waiting on it — which is why
    disequality3 marks each cell of `s(a(a(a(B))))` and not only B.

    Returns 'never' when no narrowing could ever bring them together, and
    otherwise whether anything was noted.
    """
    if depth > 30:
        return False
    a = a.deref()
    b = b.deref()
    if a is b:
        return False
    key = (id(a), id(b))
    if key in seen:
        return False
    seen.add(key)
    if _is_cyclic(a) or _is_cyclic(b):
        # An infinite term: walking it can never confirm that the two are
        # alike, so the comparison settles nothing and both keep waiting.
        _add_blocker(out, a)
        _add_blocker(out, b)
        for sub_key, sub in a.attr_list.items():
            other = b.attr_list.get(sub_key)
            if other is not None:
                _pair_blockers(sub, other, out, seen, eng, depth + 1)
        return True
    if not types_compatible(a.type, b.type):
        return 'never'
    if a.value is not None and b.value is not None and a.value != b.value:
        return 'never'
    settled = lambda t: bool(t.attr_list) or t.value is not None
    if not settled(a) or not settled(b):
        # One of them is still a bare variable: it is what the call waits on,
        # and there is nothing below it to compare.
        _add_blocker(out, a)
        _add_blocker(out, b)
        return True
    differs = (a.type is not b.type or a.value != b.value
               or a.attr_list.keys() != b.attr_list.keys())
    for sub_key, sub in a.attr_list.items():
        other = b.attr_list.get(sub_key)
        if other is None:
            continue
        below = _pair_blockers(sub, other, out, seen, eng, depth + 1)
        if below == 'never':
            return 'never'
        if below:
            differs = True
    if differs:
        _add_blocker(out, a)
        _add_blocker(out, b)
    return differs


def _eval_cond_functional(cond_term: 'PsiTerm', result: 'PsiTerm', eng) -> bool:
    """Evaluate cond(C, T [, E]) as a functional expression, binding result.

    This is called from eval_aim when cond appears as the body of a function
    rule (or as a sub-expression being evaluated functionally), so that
    cond acts as a value-producing conditional rather than a predicate.

    Semantics:
      - Prove C via inner run (preserving any bindings it makes).
      - If C succeeds → evaluate T into result.
      - If C fails   → undo C's bindings, evaluate E into result.
        If there is no E (2-arg form), fail.
    """
    wl = eng.wl
    args = list(cond_term.attr_list.values()) if cond_term.attr_list else []
    if len(args) < 2:
        return False
    cond_g = args[0].deref()
    then_g = args[1].deref()
    else_g = args[2].deref() if len(args) >= 3 else None

    mark = eng.trail.mark()
    eng._arith_error = False
    cond_ok = prove_cond(cond_g, eng)

    if cond_ok:
        return _eval_body_to_result(then_g, result, eng)
    else:
        eng.trail.undo_to(mark)
        # A condition that could not be computed at all — `(N/M) =:= floor(N/M)`
        # with M zero — is not a condition that came out false, so the
        # alternative is not taken and the call has no value.
        if getattr(eng, '_arith_error', False):
            eng._arith_error = False
            return False
        if else_g is None:
            return False
        return _eval_body_to_result(else_g, result, eng)


# ─────────────────────────────────────────────────────────────────────────────
# Goal-stack based embedded-function-call lifter
# ─────────────────────────────────────────────────────────────────────────────

def _collect_embedded_func_goals(t: 'PsiTerm', eng, visited: set) -> list:
    """Walk t (non-recursively via an explicit work-list) and replace any
    user-function sub-terms with fresh unbound variables.

    Returns a list of (func_term, result_var, rule) tuples for EVAL goals.
    Modifies t's attr_list in-place (safe because t is already a copy_term
    copy).  Does NOT push goals itself — the caller pushes them in the
    correct order:

        eval_goals = _collect_embedded_func_goals(body, eng, set())
        eng.push_goal(UNIFY, body, result, None)   # pushed first → runs last
        for ft, rv, rl in eval_goals:              # pushed after → run first
            eng.push_goal(EVAL, ft, rv, rl)

    Unlike _eval_embedded_user_funcs, this helper does NOT recurse in Python
    for each level of a deeply-recursive function body — instead it hands off
    to the engine's own iterative goal-dispatch loop.
    """
    from wild_life.built_ins import _is_user_function

    t = t.deref()
    if not t.attr_list:
        return []

    # Work-list: (parent_term, key) pairs to examine.
    work_queue = []
    for key in list(t.attr_list.keys()):
        work_queue.append((t, key))

    # Nodes we've already examined (avoid revisiting shared sub-terms)
    examined = set(visited)
    examined.add(id(t))

    eval_goals = []   # collected (func_term, result_var, rule)

    i = 0
    while i < len(work_queue):
        parent, key = work_queue[i]
        i += 1
        child = parent.attr_list[key].deref()
        child_id = id(child)
        if child_id in examined:
            continue
        examined.add(child_id)

        if _is_user_function(child):
            # Replace with fresh variable; record EVAL goal.
            v = PsiTerm(type_def=eng.wl.top)
            parent.attr_list[key] = v
            eval_goals.append((child, v, child.type.rule))
            # Do NOT enqueue children of child — they belong to the EVAL goal.
        else:
            # Handle sort-conjunction `A & B` — when either side contains an
            # evaluable sub-term (user function, built-in, or nested `&`),
            # evaluate the whole conjunction via _eval_body_sync to get the
            # sort intersection (e.g. `Copy & root_sort(X) & bodify_list(B)`
            # → `ww(a=>1,b=>2)` after copy_body has bound B).
            _child_kw = child.type.keyword if child.type else None
            if _child_kw is not None and _child_kw.symbol == '&':
                _c1 = child.attr_list.get('1')
                _c2 = child.attr_list.get('2')
                _c1d = _c1.deref() if _c1 is not None else None
                _c2d = _c2.deref() if _c2 is not None else None
                _c1_kw = _c1d.type.keyword if (_c1d is not None and _c1d.type) else None
                _c2_kw = _c2d.type.keyword if (_c2d is not None and _c2d.type) else None
                from wild_life.built_ins import (
                    _try_eval_string_func as _tesf_cj,
                    _eval_body_sync as _ebs_cj,
                )
                _has_evaluable = (
                    (_c1d is not None and (_is_user_function(_c1d) or _tesf_cj(_c1d, eng) is not None)) or
                    (_c2d is not None and (_is_user_function(_c2d) or _tesf_cj(_c2d, eng) is not None)) or
                    (_c1_kw is not None and _c1_kw.symbol == '&') or
                    (_c2_kw is not None and _c2_kw.symbol == '&')
                )
                if _has_evaluable:
                    _ev_conj = _ebs_cj(child, eng, 0)
                    if _ev_conj is not None and _ev_conj is not child:
                        parent.attr_list[key] = _ev_conj
                        # Examined the conjunction; don't re-enqueue its children.
                        continue
            # Try evaluating as a pure built-in (root_sort, features, etc.)
            from wild_life.built_ins import _try_eval_string_func as _tesf_cefg
            _bi_ev = _tesf_cefg(child, eng)
            if _bi_ev is not None and _bi_ev is not child:
                parent.attr_list[key] = _bi_ev
            else:
                # Not an evaluable built-in; walk its children.
                for sub_key in list(child.attr_list.keys()):
                    work_queue.append((child, sub_key))

    return eval_goals


# Keep old name as an alias so any other callers don't break.
def _push_embedded_func_goals(t: 'PsiTerm', eng, visited: set) -> 'PsiTerm':
    """Deprecated alias: collects AND immediately pushes EVAL goals.
    New code should use _collect_embedded_func_goals instead so the
    UNIFY goal can be pushed in between (correct LIFO ordering).
    """
    eval_goals = _collect_embedded_func_goals(t, eng, visited)
    for ft, rv, rl in eval_goals:
        eng.push_goal(GoalType.EVAL, ft, rv, rl)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class Engine:
    """
    The Wild Life inference engine.

    State mirrors the C globals in login.c / lefun.c:
      goal_stack, choice_stack, undo_stack (trail), aim
    """

    def __init__(self, wl):
        self.wl = wl           # WildLifeRuntime singleton
        self.trail: Trail = Trail()
        self.unifier: Unifier = Unifier(self.trail, engine=self)
        self.goal_stack: Optional[Goal] = None
        self.choice_stack: Optional[ChoicePoint] = None
        self.aim: Optional[Goal] = None
        self.goal_count: int = 0
        self.interrupted: bool = False
        self.main_loop_ok: bool = True
        self.verbose: bool = False
        self.trace: bool = False
        self.assert_first: bool = False
        self.var_occurred: bool = False
        self.noisy: bool = True
        self._start_time: float = 0.0

    # ─── goal stack helpers ──────────────────────────────────────────────────

    def push_goal(self, gtype: GoalType, a=None, b=None, c=None) -> Goal:
        g = Goal(gtype, a, b, c)
        g.next = self.goal_stack
        self.goal_stack = g
        return g

    def push_choice_point(self, gtype: GoalType, a=None, b=None, c=None) -> ChoicePoint:
        """Create a choice point with an alternative goal."""
        alt = Goal(gtype, a, b, c)
        alt.next = self.goal_stack
        mark = self.trail.mark()
        cp = ChoicePoint(
            undo_point=mark,
            goal_stack=alt,
            next=self.choice_stack
        )
        self.choice_stack = cp
        return cp

    def drop_choice_point(self, cp) -> None:
        """Unlink one choice point, keeping the ones created after it.

        Unlike cut_to this is a 'soft cut': it only discards the alternative
        held by `cp` itself.  A guarded function rule uses it to commit to its
        clause while the choice points its guard created stay re-satisfiable.
        """
        if cp is None:
            return
        if self.choice_stack is cp:
            self.choice_stack = cp.next
            return
        prev = self.choice_stack
        while prev is not None and prev.next is not cp:
            prev = prev.next
        if prev is not None:
            prev.next = cp.next

    def backtrack(self) -> bool:
        """Undo to the previous choice point and set goal_stack to its alt."""
        if not self.choice_stack:
            return False
        cp = self.choice_stack
        self.trail.undo_to(cp.undo_point)
        self.goal_stack = cp.goal_stack
        self.choice_stack = cp.next
        return True

    def cut_to(self, cut_point) -> None:
        """Remove choice points up to (not including) cut_point."""
        while self.choice_stack and self.choice_stack is not cut_point:
            self.choice_stack = self.choice_stack.next

    # ─── assertion helpers ───────────────────────────────────────────────────

    def add_rule(self, head: PsiTerm, body: Optional[PsiTerm],
                 typ: DefType) -> bool:
        """Add a clause to the database (implements assert_clause logic)."""
        wl = self.wl
        _mark_non_strict_args(body, self)
        head = head.deref()
        # A head written through a functor variable — `X(Args)`, which parses as
        # apply(Args, functor => X) — names the predicate X stands for, so the
        # clause is filed under that rather than under apply.
        if getattr(wl, 'apply', None) is not None and head.type is wl.apply:
            from wild_life.built_ins import _apply_to_call
            _head_call = _apply_to_call(head, self)
            if _head_call is not None:
                head = _head_call
        defn = head.type
        if defn is None:
            return False
        if getattr(defn, 'is_static', False):
            # `static(p)?` closes p: a further clause changes nothing, so a
            # later listing shows what p was.
            from wild_life.built_ins import report_static_definition
            report_static_definition(defn)
            return True

        if defn.type == DefType.UNDEF:
            defn.type = typ
        elif defn.type != typ:
            if defn._builtin_func is not None:
                # Built-in with different type — user definition takes over.
                # Allow type change (e.g. built-in PREDICATE → user FUNCTION).
                defn.type = typ
            elif defn.type == DefType.TYPE:
                # TYPE sorts can't be redefined as PREDICATE/FUNCTION
                return False
            else:
                print(f"*** Error: cannot redefine {defn.keyword.symbol} as {typ}.",
                      file=sys.stderr)
                return False

        if defn._builtin_func is not None:
            # Allow user rules to shadow builtins — clear the builtin function
            # so user-defined rules take over (Wild Life original behavior).
            defn._builtin_func = None
            if defn.rule is None:
                defn.rule = []

        # Expand disjunctions in the head before copying.
        # e.g. pick_arg({5;3;7}). → three facts: pick_arg(5). pick_arg(3). pick_arg(7).
        alt_heads = _expand_head_disj(head, wl)

        rules_to_add = []
        for alt_head in alt_heads:
            # Copy head & body to heap-permanent storage.
            # Shared var_map ensures the same original variable maps to the
            # same fresh copy in both head and body.
            shared_map: dict = {}
            head_copy = copy_term(alt_head, shared_map)
            if body is not None:
                body_copy = copy_term(body, shared_map)
            else:
                # Facts: body = succeed
                body_copy = wl.make_atom('succeed', wl.bi_module)
                if body_copy is None:
                    body_copy = PsiTerm(type_def=wl.succeed)
            rules_to_add.append((head_copy, body_copy))

        if self.assert_first:
            defn.rule = list(reversed(rules_to_add)) + (defn.rule or [])
        else:
            if defn.rule is None:
                defn.rule = []
            defn.rule.extend(rules_to_add)
        return True

    def assert_clause(self, t: PsiTerm) -> None:
        """Top-level assertion. Dispatch on head functor."""
        wl = self.wl
        t = t.deref()
        sym = t.type.keyword.symbol if t.type and t.type.keyword else ''

        def get_two(attrs):
            return attrs.get('1'), attrs.get('2')

        # A sort declaration says which sort stands under which, and needs
        # both of them: `<|(2 => s)` names neither, so there is nothing to
        # declare and the interpreter says so.
        if sym in ('<|', ':=') and not ('1' in t.attr_list and '2' in t.attr_list):
            raise DeclarationError('argument missing in sort declaration')
        if sym == '::' and '1' not in t.attr_list:
            raise DeclarationError('argument missing in sort declaration')

        if sym == ':-':
            h, b = get_two(t.attr_list)
            if h and b:
                self.add_rule(h, b, DefType.PREDICATE)
        elif sym == '->':
            h, b = get_two(t.attr_list)
            if h and b:
                self.add_rule(h, b, DefType.FUNCTION)
        elif sym in ('<|', ':='):
            self._assert_type(t)
        elif sym == '::':
            # :: Inner — either delay rule (:: Pattern | Goal) or sort prototype (:: Sort(attrs))
            inner = t.attr_list.get('1')
            if inner is not None:
                inner_d = inner.deref()
                inner_sym = (inner_d.type.keyword.symbol
                             if inner_d.type and inner_d.type.keyword else '')
                if inner_sym == '|':
                    # :: Pattern | Goal — global delay rule
                    wl.delay_rules.append(inner_d)
                else:
                    # :: Sort(attrs) — sort-level prototype attributes
                    self._assert_colon_colon_proto(inner_d)
        else:
            # Bare fact
            self.add_rule(t, None, DefType.PREDICATE)

    def _assert_colon_colon_proto(self, proto: PsiTerm) -> None:
        """Handle :: Sort(attrs) — stores sort-level prototype attributes.

        :: cleopatra(nose => pretty, occupation => queen).
        means: the sort 'cleopatra' has prototype attrs nose=pretty, occupation=queen.
        Any variable narrowed to sort 'cleopatra' automatically gets these attrs.
        """
        proto = proto.deref()
        if proto.type is None or not proto.attr_list:
            return
        sort_def = proto.type
        # Ensure the sort has a prototype_attrs dict
        if sort_def.prototype_attrs is None:
            sort_def.prototype_attrs = {}
        # Store copies of the prototype attrs (deep copy to avoid shared state)
        for key, val in proto.attr_list.items():
            val_d = val.deref()
            sort_def.prototype_attrs[key] = val_d
        # Register in the global proto_sorts list so _try_sort_narrowing can find it
        # even when the sort is not reachable via WL.top.children (e.g. 'person' is
        # not explicitly declared as 'person <| @').
        if sort_def not in self.wl.proto_sorts:
            self.wl.proto_sorts.append(sort_def)

    def _assert_type(self, t: PsiTerm) -> None:
        """Handle type declarations (<| or :=).

        <|  (sub-sort): ``A <| B`` means A is a sub-sort of B.
            → child=A, parent=B

        := (sort definition): ``A := {B;C;D}`` means B, C, D are sub-sorts
            of A.  If RHS is a plain atom ``A := B``, treat it the same way
            (B is the only direct sub-sort of A).
            → child=element, parent=A (for each element in the RHS disjunction)

        Raises SortCycleException if the new edge would create a cycle in the
        sort hierarchy.
        """
        from wild_life.data_structures import DefType
        from wild_life.unification import SortCycleException
        arg1 = t.attr_list.get('1')
        arg2 = t.attr_list.get('2')
        if not arg1 or not arg2:
            return
        arg1 = arg1.deref()
        arg2 = arg2.deref()
        sym = t.type.keyword.symbol if t.type and t.type.keyword else ''

        # Determine the (child, parent) pairs to add.
        pairs = []   # list of (child_def, parent_def)
        if sym == '<|':
            # A <| B → child=A, parent=B
            if arg1.type and arg2.type:
                pairs.append((arg1.type, arg2.type))
        else:
            # := → for each element e in the RHS disjunction: child=e, parent=LHS
            if not arg1.type:
                return
            super_def = arg1.type   # LHS is the super-sort

            # Conditional sort definition:  S := P:T | Condition
            # The RHS is a such_that(pattern, condition) node.
            # Store the (pattern, condition) pair as a sort-membership rule on
            # super_def; also add the pattern's sort as a parent of super_def so
            # that type-compatibility checks work.
            # `S := T` with a single sort or term on the right is the same
            # shape without a condition: S is a T, and takes what T states.
            _plain_rhs = (arg2.type is not None
                          and arg2.type is not self.wl.such_that
                          and arg2.type is not self.wl.disjunction)
            if (arg2.type is not None and arg2.type is self.wl.such_that) or _plain_rhs:
                if _plain_rhs:
                    pat = t.attr_list.get('2')
                    cond = self.wl.make_atom('succeed', self.wl.bi_module)
                    if cond is None:
                        cond = PsiTerm(type_def=self.wl.succeed)
                else:
                    pat  = arg2.attr_list.get('1')  # e.g. P:posint
                    cond = arg2.attr_list.get('2')  # e.g. number_of_factors(P) = one
                if pat is not None:
                    _ct2 = copy_term  # copy_term imported at module level
                    pat_d = pat.deref()
                    # Add the pattern's sort as a parent of the conditional sort
                    if pat_d.type is not None and pat_d.type is not super_def:
                        parent_def = pat_d.type
                        if parent_def.type == DefType.UNDEF:
                            parent_def.type = DefType.TYPE
                        if super_def.type == DefType.UNDEF:
                            super_def.type = DefType.TYPE
                        if parent_def not in super_def.parents:
                            super_def.parents.append(parent_def)
                        if super_def not in parent_def.children:
                            parent_def.children.append(super_def)
                    # Store sort-membership rule: (head_pattern, condition)
                    rule_entry = (pat, cond)
                    if super_def.rule is None:
                        super_def.rule = [rule_entry]
                    else:
                        super_def.rule.append(rule_entry)
                return  # Don't fall through to the generic pairs logic

            # Collect all leaf elements from the RHS (may be a disjunction or atom)
            rhs_elems = _collect_disj_elems(arg2, self.wl) if (
                arg2.type is not None and arg2.type is self.wl.disjunction
            ) else [arg2]
            for elem in rhs_elems:
                elem_d = elem.deref()
                if elem_d.type:
                    pairs.append((elem_d.type, super_def))

        for child, parent in pairs:
            # Mark both as TYPE sorts (they may have been UNDEF if newly created)
            if child.type == DefType.UNDEF:
                child.type = DefType.TYPE
            if parent.type == DefType.UNDEF:
                parent.type = DefType.TYPE
            if parent not in child.parents:
                child.parents.append(parent)
            if child not in parent.children:
                parent.children.append(child)

                # ---- Cycle detection ----------------------------------------
                # The new edge (child <| parent) creates a cycle if there is
                # already a path from `parent` UP to `child` via existing
                # parent links.
                #
                # The C Wild Life interpreter reports cycles using a specific
                # traversal order (most-recently-added parents/children first,
                # equivalent to prepend-order in C linked lists).  In Python
                # we append to lists, so "most recent first" = reversed().
                #
                # Algorithm (mirrors the C interpreter's output):
                #  1. DFS from `parent` going UP via reversed().parents to find
                #     `child`.  This detects the cycle and records the path.
                #  2. Descend at most 2 levels from `parent` via
                #     reversed().children to find a deeper "terminal" node.
                #  3. DFS from terminal going UP via reversed().parents to
                #     find `child`.  This builds the displayed path.
                #  4. Emit [child <| terminal <| ... <| child].

                def _dfs_up(start, target, visited):
                    """Return path [start, …, target] via .parents (reversed),
                    or None if target is not reachable."""
                    if start is target:
                        return [start]
                    if start in visited:
                        return None
                    visited.add(start)
                    for p in reversed(start.parents):
                        result = _dfs_up(p, target, visited)
                        if result is not None:
                            return [start] + result
                    return None

                # Step 1: does a cycle exist?
                if _dfs_up(parent, child, set()) is None:
                    continue  # no cycle — proceed to next pair

                # Cycle confirmed.  Remove the just-added edge so the
                # hierarchy remains consistent.
                child.parents.remove(parent)
                parent.children.remove(child)

                # Step 2: descend ≤2 levels from parent via children
                # (reversed order), recording the last node at each level.
                terminal = parent
                lvl1_nodes = list(reversed(parent.children))
                if lvl1_nodes:
                    for c1 in lvl1_nodes:
                        lvl2_nodes = list(reversed(c1.children))
                        if lvl2_nodes:
                            for c2 in lvl2_nodes:
                                terminal = c2
                        else:
                            terminal = c1  # c1 has no children; it IS level 1

                # Step 3: DFS from terminal UP to child (reversed parents).
                path = _dfs_up(terminal, child, set())
                if path is None:
                    # Fallback: use parent itself as start.
                    path = _dfs_up(parent, child, set()) or [parent, child]

                # Step 4: emit error + cycle string.
                child_name = child.keyword.symbol if child.keyword else "?"
                elems = [child_name] + \
                        [d.keyword.symbol if d.keyword else "?" for d in path]
                cycle_str = "[" + " <| ".join(elems) + "]"
                sys.stderr.write(
                    "*** Error: there is a cycle in the sort hierarchy\n"
                )
                sys.stderr.write(f"*** Cycle: {cycle_str}\n")
                raise SortCycleException(path)

    # ─── prove helpers ───────────────────────────────────────────────────────

    def _deref_term(self, t: PsiTerm) -> PsiTerm:
        return t.deref() if t else t

    def prove_aim(self) -> bool:
        """Handle a 'prove' goal. Returns success flag."""
        wl = self.wl
        aim = self.aim
        thegoal = aim.a
        rule_or_sentinel = aim.b  # DEFRULES sentinel or specific rule list

        if not thegoal:
            return False

        thegoal = thegoal.deref()
        defn = thegoal.type

        # ── AND (conjunction) ──
        # commasym (',') is the standard Prolog-style conjunction;
        # and_sym ('&') is the functional-pair form — both split into two goals.
        if defn == wl.and_sym or defn == wl.commasym:
            self.goal_stack = aim.next
            self.goal_count += 1
            arg1 = thegoal.attr_list.get('1')
            arg2 = thegoal.attr_list.get('2')
            if arg2:
                self.push_goal(GoalType.PROVE, arg2, _DEFRULES, None)
            if arg1:
                self.push_goal(GoalType.PROVE, arg1, _DEFRULES, None)
            return True

        # ── CUT ──
        if defn == wl.cut:
            self.goal_stack = aim.next
            self.goal_count += 1
            cut_point = thegoal.value  # stored choice point
            self.cut_to(cut_point)
            return True

        # ── OR / disjunction ──
        # Both wl.disjunction ({a;b} curly form) and wl.life_or (a;b infix form)
        if defn == wl.disjunction or defn == wl.life_or:
            self.goal_stack = aim.next
            self.goal_count += 1
            arg1 = thegoal.attr_list.get('1')
            arg2 = thegoal.attr_list.get('2')
            if arg2:
                self.push_choice_point(GoalType.PROVE, arg2, _DEFRULES, None)
            if arg1:
                self.push_goal(GoalType.PROVE, arg1, _DEFRULES, None)
            return True

        # ── TRUE / FALSE atoms ──
        if defn == wl.true:
            self.goal_stack = aim.next
            self.goal_count += 1
            return True
        if defn == wl.false:
            self.goal_stack = aim.next
            self.goal_count += 1
            return False

        # ── BUILT-IN ──
        if defn is not None and defn._builtin_func is not None:
            self.goal_stack = aim.next
            self.goal_count += 1
            if self.trace:
                print(f"[trace] prove built-in {defn.keyword.symbol}", file=sys.stderr)
            try:
                result = defn._builtin_func(thegoal, self)
                return bool(result)
            except UnificationFailure:
                return False
            except CutException as e:
                self.cut_to(e.cut_point)
                return True
            except AbortException:
                raise  # propagate to main.py (AbortException carries hook_called flag)
            except HaltException as e:
                raise

        # ── UNDEFINED or LOOKUP from DEFRULES ──
        rules = rule_or_sentinel
        if rules is _DEFRULES:
            # Check if the goal term is an unbound free variable.
            # Free vars have type=wl.top (DefType.TYPE) with no attr_list/value/coref,
            # OR type=None. In Wild Life, calling a free variable as a goal succeeds
            # immediately and suspends the prove as a pending residuated goal on the
            # variable. When the variable is later bound, the woken goal proves the
            # bound value. The variable displays as @~ (top sort with pending constraint).
            _goal_is_free_var = (
                (defn is None or (defn is wl.top)) and
                not thegoal.attr_list and
                thegoal.value is None and
                thegoal.coref is None
            )
            if _goal_is_free_var:
                from wild_life.data_structures import Residuation as _RuVar, SORT_VAR as _SORT_VAR_FV
                _g_var_pending = Goal(GoalType.PROVE, thegoal, aim.b, aim.c,
                                      next=None, pending=True)
                _r_var = _RuVar(goal=_g_var_pending, bestsort=None, value=None,
                                next=None, pending=True)
                if thegoal.resid is None:
                    self.trail.trail_psi(thegoal, 'resid')
                    thegoal.resid = [_r_var]
                else:
                    self.trail.trail_psi(thegoal, 'resid')
                    thegoal.resid = thegoal.resid + [_r_var]
                # Set SORT_VAR flag so the Unifier treats this variable as
                # bindable (not as a ground term) even though resid is non-empty.
                # Without this flag, Unifier.unify sees `not u.resid` as False
                # and skips _wakeup_resid when the variable is later bound.
                if not (thegoal.flags & _SORT_VAR_FV):
                    self.trail.trail_psi(thegoal, 'flags')
                    thegoal.flags |= _SORT_VAR_FV
                self.goal_stack = aim.next
                self.goal_count += 1
                return True
            if defn is None:
                return False
            if defn.type == DefType.PREDICATE:
                rules = defn.rule or []
            elif defn.type == DefType.FUNCTION:
                rules = defn.rule or []
            elif defn.type == DefType.UNDEF:
                if (defn.keyword is not None and defn.keyword.symbol == '.'
                        and thegoal.attr_list):
                    # `F.ground` standing where a goal is expected is the
                    # feature's value proved in its place, so `cond(F.ground,
                    # …)` goes by what the feature holds.
                    from wild_life.built_ins import _resolve_dot_feat as _rdf_pg
                    _cell_pg = _rdf_pg(thegoal, self)
                    if _cell_pg is not None:
                        self.goal_stack = aim.next
                        self.goal_count += 1
                        self.push_goal(GoalType.PROVE, _cell_pg.deref(),
                                       _DEFRULES, None)
                        return True
                if defn.rule is None:
                    # Never declared (not via dynamic/assert) → error + abort
                    # フルターム表現 (例: 'b(@)') を表示する
                    from wild_life.print_term import term_to_string as _t2s
                    term_str = _t2s(thegoal, quoted=True, wl=wl)
                    sys.stderr.write(
                        f"*** Error: '{term_str}' is not a predicate or a function.\n"
                        f"\n*** Abort\n"
                    )
                    raise AbortException(hook_called=True)
                # rule == [] → declared via dynamic but no clauses → fail silently
                self.goal_stack = aim.next
                self.goal_count += 1
                return False
            else:
                self.goal_stack = aim.next
                self.goal_count += 1
                return False
        elif rules is None:
            self.goal_stack = aim.next
            self.goal_count += 1
            return False

        # For user-defined FUNCTION calls with free arguments:
        # Residuate instead of eagerly matching clauses, so that f(X)? with
        # free X suspends until X is bound, rather than proceeding with an
        # unbound sort-typed variable (which would print int~ or similar).
        # This mirrors the residuation check in eval_aim (lines ~1102-1150).
        if (defn is not None and defn.type == DefType.FUNCTION and
                thegoal.attr_list and rules):
            _h0, _b0 = rules[0]
            _h0d = _h0.deref() if _h0 is not None else None
            if _h0d is not None and _h0d.attr_list:
                _fn_free_args = []
                for _fk_fn, _fv_psi_fn in thegoal.attr_list.items():
                    _fv_fn = _fv_psi_fn.deref()
                    _fv_fn_free = (
                        (_fv_fn.type is None or _fv_fn.type is wl.top) and
                        not _fv_fn.attr_list and
                        _fv_fn.value is None and
                        _fv_fn.coref is None
                    )
                    if _fv_fn_free:
                        _h_arg_fn = _h0d.attr_list.get(_fk_fn)
                        if _h_arg_fn is not None:
                            _h_arg_d_fn = _h_arg_fn.deref()
                            if (_h_arg_d_fn.type is not None and
                                    _h_arg_d_fn.type is not wl.top):
                                _fn_free_args.append(_fv_fn)
                if _fn_free_args:
                    from wild_life.data_structures import Goal as _FnGoal, Residuation as _FnResid, SORT_VAR as _SV_FN
                    _pending_prove_fn = _FnGoal(GoalType.PROVE, thegoal, _DEFRULES,
                                                None, next=None, pending=True)
                    for _fv_free_fn in _fn_free_args:
                        if _fv_free_fn.resid is None:
                            self.trail.trail_psi(_fv_free_fn, 'resid')
                            _fv_free_fn.resid = [_FnResid(goal=_pending_prove_fn)]
                        else:
                            if not any(rv.goal is _pending_prove_fn for rv in _fv_free_fn.resid):
                                self.trail.trail_copy(_fv_free_fn, 'resid')
                                _fv_free_fn.resid.append(_FnResid(goal=_pending_prove_fn))
                        # Set SORT_VAR flag so Unifier treats variable as bindable
                        # even when resid is non-empty.
                        if not (_fv_free_fn.flags & _SV_FN):
                            self.trail.trail_psi(_fv_free_fn, 'flags')
                            _fv_free_fn.flags |= _SV_FN
                    self.goal_stack = aim.next
                    self.goal_count += 1
                    return True

        # A function standing where a goal is expected is evaluated, and the
        # value it comes to is then proven in its place: `f(44)` with
        # `f(X:int) -> write(boo,X)` writes once, and `call(p(X))` with
        # `call(X) -> (S | (X,S=true ; S=false))` holds for the X that make
        # p(X) hold and not for the others.
        if defn is not None and defn.type == DefType.FUNCTION and rules:
            _fn_res = PsiTerm(type_def=wl.top)
            self.goal_stack = aim.next
            self.goal_count += 1
            # Pushed in LIFO order: evaluate, then prove what it came to.
            self.push_goal(GoalType.PROVE, _fn_res, _DEFRULES, None)
            self.push_goal(GoalType.EVAL, thegoal, _fn_res, rules)
            return True

        # Filter out retracted clauses
        active = [(h, b) for (h, b) in (rules if rules else [])
                  if h is not None and b is not None]
        if not active:
            self.goal_stack = aim.next
            self.goal_count += 1
            return False

        # A strict predicate is given values, not calls: `pick_op(X:ran)` asks
        # ran for its number once and every clause of pick_op then reads that
        # one number, where reducing the call per clause would draw a fresh one
        # each time round.  One call is reduced and the goal put back, so the
        # next is found on the way round.
        if (defn is not None and defn.type == DefType.PREDICATE
                and defn._builtin_func is None and thegoal.attr_list
                and not (hasattr(self, 'non_strict_set')
                         and defn in self.non_strict_set)):
            from wild_life.built_ins import _is_user_function as _iuf_pa
            _call_arg = None
            for _av_pa in thegoal.attr_list.values():
                _ad_pa = _av_pa.deref()
                if (_iuf_pa(_ad_pa) and not _ad_pa.attr_list
                        and not getattr(_ad_pa.type, 'is_dynamic', False)):
                    _call_arg = _ad_pa
                    break
            if _call_arg is not None:
                self.goal_stack = aim.next
                self.goal_count += 1
                # Asked once: a call the evaluation leaves standing must not
                # send the goal round again for the same argument.
                from wild_life.data_structures import REDUCED as _RED_pa
                self.trail.trail_psi(_call_arg, 'flags')
                _call_arg.flags |= _RED_pa
                _R_pa = wl.make_var()
                self.push_goal(GoalType.PROVE, thegoal, aim.b, aim.c)
                self.push_goal(GoalType.UNIFY, _call_arg, _R_pa, None)
                self.push_goal(GoalType.EVAL, _call_arg, _R_pa, _call_arg.type.rule)
                return True

        self.goal_stack = aim.next
        self.goal_count += 1

        if self.trace:
            sym = defn.keyword.symbol if defn and defn.keyword else '?'
            print(f"[trace] prove {sym}", file=sys.stderr)

        # ── DISJUNCTION EXPANSION IN ACTUAL ARGUMENTS ────────────────────────
        # When any ACTUAL argument of thegoal is a disjunction (e.g. p({1;2;3})?),
        # expand into multiple PROVE alternatives BEFORE setting the cut barrier.
        # This ensures that '!' inside the clause body only cuts the clause's own
        # alternatives, NOT the disjunction alternatives from the call site.
        #
        # Example: p({1;2;3})? with  p(A) :- !, write(A).
        #   → PROVE(p(3)) and PROVE(p(2)) are pushed here (before cut_barrier),
        #     then thegoal = p(1).  '!' inside p cuts its own choices, NOT p(2)/p(3).
        #
        # Contrast: q(X)? with q(A:{1;2;3}) :- !, write(A).
        #   → X is unbound (not a disjunction at the call site), so NO expansion here.
        #     The disjunction comes from the clause head; those choice points are
        #     created during head unification (after cut_barrier) → '!' DOES cut them.
        _goal_alts = _expand_head_disj(thegoal, wl)
        if len(_goal_alts) > 1:
            for _alt in reversed(_goal_alts[1:]):
                self.push_choice_point(GoalType.PROVE, _alt, _DEFRULES, None)
            thegoal = _goal_alts[0]
        elif len(_goal_alts) == 0:
            return False  # empty disjunction in argument → fail

        # Multiple clauses → set up choice point for first, then proceed.
        # Record cut_barrier BEFORE pushing the multi-clause choice point so
        # that '!' inside the clause body only cuts choices that belong to
        # THIS predicate call, not choices from the calling context.
        cut_barrier = self.choice_stack   # WAM B0 register

        head_orig, body_orig = active[0]
        if len(active) > 1:
            self.push_choice_point(GoalType.PROVE, thegoal, active[1:], None)

        _vm: dict = {}
        head = copy_term(head_orig, _vm)
        body = copy_term(body_orig, _vm)
        # A call written into a clause head's argument is there for its value:
        # `p_a(pair(foo_a(Y:titi_a), …))` matches against pair(t(Y), …), which
        # is what foo_a answers, not against the call itself.  Only a call a
        # rule already fits is reduced: `hanoi(…, Ms:ensuite(Ms1,…))` has to
        # wait on Ms1, and reducing it here would pick the empty-list rule and
        # settle a question the clause has not asked.
        if head.attr_list:
            self._reduce_settled_head_calls(head)

        # Fix A: such_that daemon setup for FUNCTION rules.
        # When the body is `val | cond` (such_that), and the call has free
        # arguments that correspond to sort-constrained head parameters
        # (X:@(set=>true)), set up `cond` as a daemon residuation on those
        # free variables instead of eagerly running the clause.
        # The daemon fires later when the sort constraint is satisfied.
        _body_d_fx = body.deref() if body is not None else None
        if (defn is not None and defn.type == DefType.FUNCTION
                and _body_d_fx is not None and _body_d_fx.type is wl.such_that
                and thegoal.attr_list):
            _head_d_fx = head.deref() if head is not None else None
            _daemon_pairs_fx = []  # list of (actual_free_var, head_param_deref)
            if _head_d_fx is not None and _head_d_fx.attr_list:
                for _k_fx, _actual_psi_fx in thegoal.attr_list.items():
                    _actual_fx = _actual_psi_fx.deref()
                    _is_free_fx = (
                        (_actual_fx.type is None or _actual_fx.type is wl.top)
                        and not _actual_fx.attr_list
                        and _actual_fx.value is None
                        and _actual_fx.coref is None
                    )
                    if _is_free_fx:
                        _hp_psi_fx = _head_d_fx.attr_list.get(_k_fx)
                        if _hp_psi_fx is not None:
                            _hp_d_fx = _hp_psi_fx.deref()
                            # Sort constraint if head param has attrs or non-top type
                            _has_constraint_fx = (
                                _hp_d_fx.attr_list or
                                (_hp_d_fx.type is not None and _hp_d_fx.type is not wl.top)
                            )
                            if _has_constraint_fx:
                                _daemon_pairs_fx.append((_actual_fx, _hp_psi_fx, _k_fx))
            if _daemon_pairs_fx:
                from wild_life.data_structures import (
                    Goal as _DGfx, Residuation as _DRfx, SORT_VAR as _SVfx
                )
                _cond_psi_fx = _body_d_fx.attr_list.get('2')
                _val_psi_fx = _body_d_fx.attr_list.get('1')
                for (_act_fx, _hp_psi_fx_ref, _k_fx_ref) in _daemon_pairs_fx:
                    # Build a var_map copy that maps the head-param's deref
                    # (the sort-constraint psi-term) back to the actual free var.
                    # This gives us cond(X_actual) so write(X') → write(X_actual).
                    _vm_dae: dict = {}
                    _hp_d_ref = _hp_psi_fx_ref.deref()
                    _vm_dae[id(_hp_d_ref)] = _act_fx
                    _cond_copy_fx = copy_term(
                        _cond_psi_fx.deref() if _cond_psi_fx else None,
                        _vm_dae
                    ) if _cond_psi_fx is not None else None
                    if _cond_copy_fx is not None:
                        _pending_g = _DGfx(GoalType.PROVE, _cond_copy_fx,
                                           _DEFRULES, None, pending=True)
                        _dr_fx = _DRfx(goal=_pending_g, daemon=True)
                        if _act_fx.resid is None:
                            self.trail.trail_psi(_act_fx, 'resid')
                            _act_fx.resid = [_dr_fx]
                        else:
                            self.trail.trail_copy(_act_fx, 'resid')
                            _act_fx.resid = list(_act_fx.resid) + [_dr_fx]
                    # Mark as SORT_VAR so unification treats it as bindable variable
                    if not (_act_fx.flags & _SVfx):
                        self.trail.trail_psi(_act_fx, 'flags')
                        _act_fx.flags |= _SVfx
                # Push PROVE(val) – typically 'true'/succeed, skip if succeed
                if _val_psi_fx is not None:
                    _val_d_fx = _val_psi_fx.deref()
                    if _val_d_fx.type is not wl.succeed:
                        self.push_goal(GoalType.PROVE, _val_d_fx, _DEFRULES, None)
                self.goal_stack = aim.next
                self.goal_count += 1
                return True

        # Unify head with goal
        if body.type != wl.succeed:
            # Patch cut atoms in the body copy so they respect the cut barrier.
            _patch_cut_barriers(body, wl, cut_barrier)
            self.push_goal(GoalType.PROVE, body, _DEFRULES, None)

        # Bind head's coref to thegoal (= head ← thegoal)
        # For non-strict functions, suppress eager arithmetic evaluation of arguments.
        _non_strict = (defn is not None and
                       hasattr(self, 'non_strict_set') and
                       defn in self.non_strict_set)
        _prev_no_arith = getattr(self, 'no_arith_eval', False)
        if _non_strict:
            self.no_arith_eval = True
        else:
            # A strict predicate is given values, not calls: reduce a built-in
            # function in an argument before matching, so that a clause body
            # asserting its argument asserts `a1` and not `str2psi("a1")`.
            from wild_life.built_ins import (_try_eval_string_func as _tesf_pa,
                                              _try_eval_arith_to_term as _teat_pa)
            for _k_pa, _a_pa in list(thegoal.attr_list.items()):
                _a_pa_d = _a_pa.deref()
                _ev_pa = _tesf_pa(_a_pa_d, self)
                if _ev_pa is not None and _ev_pa is not _a_pa_d:
                    thegoal.attr_list[_k_pa] = _ev_pa
                    continue
                # An expression handed to a strict call is asked for its
                # value, whatever it was written as: `p(X:(1+2))` leaves X
                # worth 3, and everything reading X from then on reads 3.
                from wild_life.data_structures import (
                    NON_STRICT_TERM as _NST_pa)
                if (_a_pa_d.attr_list and _a_pa_d.value is None
                        and not (_a_pa_d.flags & _NST_pa)
                        and _a_pa_d.type is not None
                        and _a_pa_d.type.keyword is not None
                        and _a_pa_d.type.keyword.symbol in _STRICT_ARITH_SYMS):
                    _teat_pa(_a_pa_d, self)
        mark = self.trail.mark()
        ok = self.unifier.unify(thegoal, head)
        if _non_strict:
            self.no_arith_eval = _prev_no_arith
            if ok:
                _mark_arith_non_strict(head)
        if not ok:
            self.trail.undo_to(mark)
            # Try next clause if any
            if self.choice_stack and \
               self.choice_stack.goal_stack.type == GoalType.PROVE and \
               self.choice_stack.goal_stack.a is thegoal:
                return self.backtrack_and_succeed()
            return False
        return True

    def _reduce_settled_head_calls(self, head: 'PsiTerm') -> None:
        """Reduce the calls under a clause head that a rule already fits.

        A call whose arguments are not specific enough for any rule is left
        as written: it is part of the pattern, and settling it here would
        answer a question the clause has not asked.
        """
        from wild_life.built_ins import (_is_user_function as _iuf_h,
                                         _try_eval_any_func as _teaf_h)
        seen: set = set()

        def walk(t: 'PsiTerm', depth: int) -> None:
            if depth > 40:
                return
            td = t.deref()
            if id(td) in seen:
                return
            seen.add(id(td))
            for key in list(td.attr_list.keys()):
                child = td.attr_list[key].deref()
                if _iuf_h(child) and self._head_call_is_settled(child):
                    evaled = _teaf_h(child, self)
                    if evaled is not None and evaled.deref() is not child:
                        td.attr_list[key] = evaled
                        walk(evaled, depth + 1)
                        continue
                walk(child, depth + 1)

        walk(head, 0)

    def _head_call_is_settled(self, call: 'PsiTerm') -> bool:
        """Whether some rule of call's function fits it as it stands."""
        rules = call.type.rule if call.type is not None else None
        if not rules:
            return False
        for _h, _b in rules:
            if _h is None or _b is None:
                continue
            _hd = _h.deref()
            if set(_hd.attr_list.keys()) - set(call.attr_list.keys()):
                continue
            if _rule_match_status(_hd, call, self) == 'ready':
                return True
        return False

    def backtrack_and_succeed(self) -> bool:
        if not self.choice_stack:
            return False
        self.backtrack()
        return True  # will be re-evaluated in main_prove

    def unify_aim(self) -> bool:
        """Handle a 'unify' goal."""
        aim = self.aim
        u = aim.a
        v = aim.b
        if u is None or v is None:
            return False
        # Resolve dot-access terms (T.F) before structural unification.
        # After EVAL goals fire for user-function hosts, the host is already
        # bound and _try_eval_string_func can access the feature correctly.
        u_d = u.deref()
        if (u_d.type is not None and u_d.type.keyword is not None
                and u_d.type.keyword.symbol == '.'):
            from wild_life.built_ins import _try_eval_string_func as _tef_ua
            _dot_val = _tef_ua(u_d, self)
            if _dot_val is not None:
                mark = self.trail.mark()
                ok = self.unifier.unify(_dot_val, v)
                if not ok:
                    self.trail.undo_to(mark)
                return ok
        mark = self.trail.mark()
        ok = self.unifier.unify(u, v)
        if not ok:
            self.trail.undo_to(mark)
        return ok

    def suchthat_val_aim(self) -> bool:
        """Handle a 'suchthat_val' goal: reduce a such-that rule's value part
        once its guard has been proved, then unify it with the rule's result.

        aim.c holds the value's function call; eval_aim has pointed the rule's
        value variable at a fresh node, so any feature the guard added —
        `X = Y.A` in
        `bodify_list([(A,X)|T]) -> Y : bodify_list(T) | X = Y.A.` — sits on
        aim.a.  Unifying the two merges the guard's constraints with the value
        the call reduces to.
        """
        from wild_life.built_ins import _eval_user_func_sync
        aim = self.aim
        val_part = aim.a
        result = aim.b
        call = aim.c
        if val_part is None or result is None or call is None:
            return False

        mark = self.trail.mark()
        # A call that cannot be reduced here stands as its own value, as it did
        # when the reduction was attempted before the guard.
        from wild_life.built_ins import _try_eval_string_func as _tesf_stv
        evaled = _eval_user_func_sync(call, self)
        if evaled is None:
            evaled = _tesf_stv(call, self)
        if evaled is None:
            evaled = call
        ok = (self.unifier.unify(val_part, evaled)
              and self.unifier.unify(val_part, result))
        if not ok:
            self.trail.undo_to(mark)
        return ok

    def _push_embedded_func_goals_method(self, t: 'PsiTerm', visited: set) -> 'PsiTerm':
        """Walk t and replace user-function sub-terms with fresh vars, pushing
        EVAL goals for each.  Returns (possibly modified) term safe to UNIFY.
        Uses goal-stack instead of Python recursion so that deeply-recursive
        functions like largeterm(1000) don't blow the Python call stack.
        """
        return _push_embedded_func_goals(t, self, visited)

    def _preeval_funct_args(self, funct: 'PsiTerm') -> None:
        """Reduce the call's arguments before its head is matched.

        This enables patterns like f(g(x)) where g(x) has to be evaluated
        before pattern matching against f's head (e.g. rev(reverse(L),[])).

        The reduced argument is written back through the trail.  A rule whose
        head does not match undoes what the reduction bound on its way out, so
        an untrailed write would leave the next rule looking at the reduced
        term with its bindings gone — `split(2,[],ll(C,[1|l(C)]))` would see a
        couple whose left feature had become @ again.
        """
        from wild_life.built_ins import (
            _eval_user_func_sync, _is_user_function,
            _try_eval_string_func, _try_eval_arith_to_term,
            _eval_embedded_user_funcs,
        )
        for _key in list(funct.attr_list.keys()):
            _attr = funct.attr_list[_key].deref()
            if _is_user_function(_attr):
                _evaled = _eval_user_func_sync(_attr, self)
                if _evaled is not None and _evaled is not _attr:
                    self.unifier.set_attr(funct, _key, _evaled)
            else:
                # Try built-in function evaluation (features, root_sort, etc.)
                _evaled = _try_eval_string_func(_attr, self)
                if _evaled is not None:
                    self.unifier.set_attr(funct, _key, _evaled)
                else:
                    _evaled = _try_eval_arith_to_term(_attr, self)
                    if _evaled is not None:
                        self.unifier.set_attr(funct, _key, _evaled)
                    elif _attr.attr_list:
                        # Compound arg: synchronously evaluate any embedded
                        # user-function calls so that e.g.
                        #   where((B,Table) & copy_body(...))
                        # gets copy_body evaluated BEFORE where's body (@)
                        # discards the argument.  Without this, bodify_list(B)
                        # would run on the goal stack with B still unbound.
                        _eval_embedded_user_funcs(_attr, self, 0, set())

    def eval_aim(self) -> bool:
        """Handle an 'eval' goal (function evaluation)."""
        wl = self.wl
        aim = self.aim
        funct = aim.a
        result = aim.b
        rules = aim.c  # rule list

        if funct is None:
            return False
        funct = funct.deref()

        # Boolean built-in used as a function — `===` and `\===` carry a
        # _builtin_func instead of rules, so evaluating one means proving it
        # and taking the truth value as its result.
        if not rules and funct.type is not None:
            bi_fn = getattr(funct.type, '_builtin_func', None)
            if bi_fn is not None:
                truth = 'true' if bi_fn(funct, self) else 'false'
                return self.unifier.unify(result, wl.make_atom(truth))

        if rules is None:
            return False

        # Built-in function
        if isinstance(rules, int):
            # Built-in index — look up in wl.c_rules
            bi = getattr(wl, '_c_rules', {}).get(rules)
            if bi:
                try:
                    return bool(bi(funct, result, self))
                except UnificationFailure:
                    return False
            return False

        # Whether this EVAL was woken from a residuation (see curry3 residuation setup).
        # When all rules fail for a residuated call, we fall back to binding result
        # to the original (unevaluated) compound funct — the function "returns itself"
        # when no matching rule is found (LIFE's lazy/constructor semantics).
        # Also check funct._resid_refire so the flag survives across choice-point firings
        # (choice points store funct by reference, not aim).
        _is_resid_refire = getattr(aim, '_resid_marker', False) or getattr(funct, '_resid_refire', False)

        # User-defined function: find first active rule
        active = [(h, b) for (h, b) in (rules if rules else [])
                  if h is not None and b is not None]
        if not active:
            if _is_resid_refire:
                # No rules at all — return the compound as-is
                return self.unifier.unify(result, funct)
            return False

        head_orig, body_orig = active[0]
        # Choice point level before the remaining-clause alternatives are
        # pushed.  A guarded rule (`f(X) -> Val | Guard`) commits to its clause
        # once head matching and the guard both succeed, so the guard is
        # followed by a cut back to this level.
        _rule_cp = None
        if len(active) > 1:
            _rule_cp = self.push_choice_point(GoalType.EVAL, funct, result, active[1:])

        _vm: dict = {}
        head = copy_term(head_orig, _vm)
        body = copy_term(body_orig, _vm)

        # Handle conditional functional rule: body = (value | condition)
        # where '|' is the such-that / function-guard operator.
        # We must prove 'condition' as a goal and unify result with 'value'.
        body_d = body.deref()
        if body_d.type is not None and body_d.type is wl.such_that:
            val_part  = body_d.attr_list.get('1')  # return value
            cond_part = body_d.attr_list.get('2')  # condition to prove
            if val_part is not None and cond_part is not None:
                # For functions with input arguments (non-nullary), unify funct
                # with head FIRST to bind the argument variables.  This must
                # happen before we evaluate functional sub-terms in cond_part
                # (e.g. children(X) can only be reduced once X is bound to s1).
                # For nullary function sorts (head is a bare variable with no
                # attributes — e.g. `ran -> A | cond`), skip this step: linking
                # the head variable back to funct (which has a function sort)
                # would cause bi_unify to misidentify it as a function call when
                # the body assigns `A = computed_value`, triggering spurious
                # recursive evaluation.
                head_d = head.deref()
                if head_d.attr_list:
                    # If funct contains disjunction arguments, expand them at
                    # the EVAL level BEFORE calling unify(funct, head).
                    # Without this, BIND_DIRECT CPs pushed inside unify()
                    # have incomplete goal_stacks (missing the PROVE(cond)
                    # and UNIFY(val, result) goals pushed below), so cross-
                    # product backtracking fails for e.g. b({3;4}) with
                    # clause b(X) -> (X,Y) | Y={1;2}.
                    from wild_life.built_ins import (
                        _term_contains_disjunction as _st_tcd,
                        _expand_term_disjunctions  as _st_etd,
                    )
                    if _st_tcd(funct, self):
                        _st_alts = _st_etd(funct, self)
                        if len(_st_alts) > 1:
                            for _st_alt in reversed(_st_alts[1:]):
                                self.push_choice_point(GoalType.EVAL, _st_alt, result, active)
                            funct = _st_alts[0]
                            _vm_st: dict = {}
                            head = copy_term(head_orig, _vm_st)
                            body = copy_term(body_orig, _vm_st)
                            body_d = body.deref()
                            val_part  = body_d.attr_list.get('1')
                            cond_part = body_d.attr_list.get('2')
                            if val_part is None or cond_part is None:
                                return False
                            head_d = head.deref()
                    # Reduce the call's arguments before matching, the same as
                    # an unguarded rule does further down: `q_sort(l(LM))` has
                    # to become q_sort([1]) before the head `q_sort([H|T])`
                    # can be matched against it.
                    self._preeval_funct_args(funct)
                    mark = self.trail.mark()
                    ok = self.unifier.unify(funct, head)
                    if not ok:
                        self.trail.undo_to(mark)
                        return False
                # Now that argument variables are bound, eagerly evaluate any
                # built-in or user-defined functional sub-terms in cond_part
                # (e.g. genChildren(children(X), A) → children(X) → [a,b,c,d]).
                from wild_life.built_ins import (
                    _eval_embedded_user_funcs,
                    _eval_user_func_sync,
                    _is_user_function,
                    _try_eval_string_func,
                )
                # A value that is a function call is reduced only AFTER the
                # guard has been proven, because the guard may constrain the
                # call's result — as in
                #   bodify_list([(A,X)|T]) -> Y : bodify_list(T) | X = Y.A.
                # where `Y.A` inserts feature A into Y.  Y is pointed at a
                # fresh node first so that `Y.1` addresses that result rather
                # than colliding with the call's own first argument, and that
                # has to happen before the guard is touched at all, since
                # _eval_embedded_user_funcs already resolves `Y.A` in place.
                _st_call = None
                _vp_d = val_part.deref()
                # A built-in call standing as the value waits too: the guard is
                # what binds its arguments, so `q_sort([H|T]) -> append(L1,
                # [H|L2]) | …, L1 = q_sort(…), L2 = q_sort(…)` can only reduce
                # the append once the guard has run.
                _vp_sym = (_vp_d.type.keyword.symbol
                           if (_vp_d.type and _vp_d.type.keyword) else '')
                _vp_is_bi_call = (bool(_vp_d.attr_list)
                                  and _vp_sym in _DEFERRABLE_BUILTIN_FUNCS)
                if _is_user_function(_vp_d) or _vp_is_bi_call:
                    _st_call = PsiTerm(type_def=_vp_d.type)
                    _st_call.attr_list = dict(_vp_d.attr_list)
                    _st_call.flags = _vp_d.flags
                    self.trail.trail_psi(_vp_d, 'coref')
                    _vp_d.coref = PsiTerm(type_def=wl.top)
                _cond_d = cond_part.deref()
                # Reduce calls embedded in the guard — `genChildren(children(X),
                # A)` needs its children(X) argument reduced before the
                # predicate runs.  A conjunction is proven left to right, so
                # only its leftmost goal is ready: a later one is still waiting
                # on what the goals before it will bind, and reducing
                # `L1 = q_sort(l(LM))` before LM exists is how qsort2 lost its
                # first solution.
                _eval_embedded_user_funcs(_leftmost_goal(_cond_d, wl), self, 0, set())
                if _st_call is None:
                    # Any other value is reduced up front: its sub-terms are
                    # rewritten in place, which a later backtrack into the
                    # guard would not undo.
                    _eval_embedded_user_funcs(_vp_d, self, 0, set())
                    _sv = _try_eval_string_func(_vp_d, self)
                    if _sv is not None and _sv is not _vp_d:
                        val_part = _sv
                    self.push_goal(GoalType.UNIFY, val_part, result, None)
                else:
                    self.push_goal(GoalType.SUCHTHAT_VAL, val_part, result, _st_call)
                if _rule_cp is not None:
                    self.push_goal(GoalType.EVAL_COMMIT, _rule_cp, None, None)
                self.push_goal(GoalType.PROVE, _cond_d, _DEFRULES, None)
                return True

        # Expand disjunctions embedded in function arguments BEFORE pre-eval.
        # This must happen first so that nested function calls like f2(f2({1;2}))
        # push EVAL-level CPs (with the full continuation) rather than letting
        # _eval_user_func_sync push inner CPs that miss the outer evaluation.
        # e.g. f2(f2({1;2})) → push EVAL CP for f2(f2(2)), try f2(f2(1)) first.
        # Then pre-eval evaluates inner calls on each alternative separately.
        from wild_life.built_ins import _term_contains_disjunction, _expand_term_disjunctions
        if _term_contains_disjunction(funct, self):
            _alts = _expand_term_disjunctions(funct, self)
            if len(_alts) > 1:
                # Push choice points for alternatives 2..N (in reverse so first
                # alternative is tried next, then 2nd, etc.)
                for _alt in reversed(_alts[1:]):
                    self.push_choice_point(GoalType.EVAL, _alt, result, active)
                funct = _alts[0]
                # Recompute fresh head/body copies for the first alternative
                _vm = {}
                head = copy_term(head_orig, _vm)
                body = copy_term(body_orig, _vm)

        # Pre-evaluate any function call arguments in funct.
        # This enables patterns like f(g(x)) where g(x) needs to be evaluated
        # before pattern matching against f's head (e.g. rev(reverse(L),[]) ).
        from wild_life.built_ins import (
            _eval_user_func_sync, _is_user_function,
            _try_eval_string_func, _try_eval_arith_to_term,
        )
        self._preeval_funct_args(funct)

        # Arity check: if head has feature keys not present in funct, this rule
        # requires arguments that the call doesn't provide.  Skip the rule —
        # adding extra features to a function call is wrong semantics (unlike
        # sort unification where adding features is fine).
        _head_d_arity = head.deref()
        _funct_keys_set = set(funct.attr_list.keys())
        _head_only_keys = set(_head_d_arity.attr_list.keys()) - _funct_keys_set
        if _head_only_keys:
            # A rule asks for features the call does not carry, so the call is
            # a partial application: it may yet gain them, and which rule
            # applies is not settled.  It stands for itself rather than
            # reducing through a later rule — `X = f(b => 0)` with
            # `f(a => int) -> 1.` and `f(b => int) -> 2.` answers f(b => 0),
            # and only `X(a => string)` picks a rule.
            if _rule_cp is not None:
                self.drop_choice_point(_rule_cp)
            return self.unifier.unify(result, funct)

        # The other way round: the call carries features the rule's head does
        # not ask for.  A rule with a bare name for a head says what the name
        # is worth — `quadruple -> *(2 => 4)` — and the features the call
        # carries belong to that value, not to the name: `quadruple(5)` is
        # `*(2 => 4)` applied to 5, which is 20.  A head that is a sort
        # variable (`X:sum -> …`) is a different thing: it stands for the call
        # itself, features and all, and is left alone here.
        _extra_keys = _funct_keys_set - set(_head_d_arity.attr_list.keys())
        if (_extra_keys and not _head_d_arity.attr_list
                and not _occurs_by_identity(head, body)
                and getattr(wl, 'apply', None) is not None):
            _bare = PsiTerm(type_def=funct.type)
            _value = PsiTerm(type_def=wl.top)
            _applied = PsiTerm(type_def=wl.apply)
            _applied.attr_list = {k: funct.attr_list[k] for k in _extra_keys}
            _applied.attr_list['functor'] = _value
            if _rule_cp is not None:
                self.drop_choice_point(_rule_cp)
            # LIFO: the value is worked out first, then applied.
            self.push_goal(GoalType.UNIFY, result, _applied, None)
            self.push_goal(GoalType.EVAL, _bare, _value, rules)
            return True

        # A rule head that names the same variable twice asks for the very same
        # psi-term in both places.  `f(X,X)` therefore does not apply to
        # `f(a(a(X1)), a(a(X2)))` however alike the two arguments look, and
        # unifying them would answer a question the call has not settled — so
        # the call residuates on the variables that keep them apart, and is
        # retried when one of them is bound.
        # Matching is one-way: the head's variables take the call's terms, and
        # the call's own terms are never narrowed to make a rule fit.  A rule
        # no narrowing could ever fit is passed over; one the call is not yet
        # specific enough for makes the call wait on the terms that would
        # settle it.
        _match = _rule_match_status(_head_d_arity, funct, self)
        if _match == 'never':
            return False
        if _match == 'stuck':
            # The call's arguments are alike down to their variables yet are
            # still two terms, and the head asks for one: nothing left to bind
            # can settle it, so the call neither applies the rule nor waits —
            # it simply has no value.
            return True
        _free_args_for_resid = _match if isinstance(_match, list) else None
        if _free_args_for_resid:
            # The call as a whole is waiting, not this one rule: the pending
            # goal carries the entire rule list and starts again from the
            # first rule once the variable is bound.  Leaving the alternatives
            # for the later rules standing would let backtracking suspend the
            # same call once per rule, so `A = f(B)` with three rules for `f`
            # would answer three times over.
            if _rule_cp is not None:
                self.drop_choice_point(_rule_cp)
                _rule_cp = None
            # Set up residuation: attach a pending EVAL goal to each free variable.
            # When the variable gets bound, _wakeup_resid will push the EVAL goal
            # back onto the goal stack and f(bound_val) will be re-evaluated.
            from wild_life.data_structures import Goal as _ResidGoal, Residuation as _ResidR, SORT_VAR as _SV_R
            # Reuse the call's own pending goal across re-fires.  A fresh one
            # each time would never compare equal to the goals already on the
            # variables, so every re-evaluation would pile another copy on and
            # the term would show a tilde per round.
            _pending_eval = getattr(funct, '_resid_eval_goal', None)
            if _pending_eval is None:
                _pending_eval = _ResidGoal(GoalType.EVAL, funct, result, rules, pending=True)
                _pending_eval._resid_marker = True  # mark as residuation so re-fire knows
                funct._resid_eval_goal = _pending_eval
            # Firing the goal clears its pending flag; the call is suspending
            # again, so it is pending again — and shows a tilde again.
            _pending_eval.pending = True
            # Mark funct so eval_aim can detect resid re-fire even from a freshly pushed Goal.
            # _wakeup_resid calls push_goal(g.type, g.a, g.b, g.c) which creates a new Goal
            # without _resid_marker, so we propagate via funct (which is g.a and is preserved).
            funct._resid_refire = True
            # A term the call waited on last time round may not be one now:
            # `f(X,s(X))` waits on Y until Y is s(Z), and from then on it waits
            # on Z instead.  Drop the old marks before laying down the new
            # ones, so that only what the call is actually waiting on shows a
            # tilde.
            for _old_r in getattr(funct, '_resid_marked', ()) or ():
                if _old_r.resid and any(rv.goal is _pending_eval
                                        for rv in _old_r.resid):
                    self.trail.trail_copy(_old_r, 'resid')
                    _old_r.resid = [rv for rv in _old_r.resid
                                    if rv.goal is not _pending_eval]
            funct._resid_marked = list(_free_args_for_resid)
            for _fv_r in _free_args_for_resid:
                if _fv_r.resid is None:
                    self.trail.trail_psi(_fv_r, 'resid')
                    _fv_r.resid = [_ResidR(goal=_pending_eval)]
                else:
                    if not any(rv.goal is _pending_eval for rv in _fv_r.resid):
                        self.trail.trail_copy(_fv_r, 'resid')
                        _fv_r.resid.append(_ResidR(goal=_pending_eval))
                # A plain variable is marked bindable so the unifier keeps
                # binding it although it now carries a residuation.  A term
                # that already has a sort, a value or features is not a
                # variable and must not start looking like one — `b` would
                # then let itself be narrowed to anything.
                _fv_r_is_plain = (not _fv_r.attr_list and _fv_r.value is None
                                  and (_fv_r.type is None or _fv_r.type is wl.top))
                if _fv_r_is_plain and not (_fv_r.flags & _SV_R):
                    self.trail.trail_psi(_fv_r, 'flags')
                    _fv_r.flags |= _SV_R
            # result (and hence A) stays unbound — return True so the UNIFY(A,result)
            # goal fires and merges A with the free result variable.
            return True

        # Unify head with funct first (to bind head arguments)
        mark = self.trail.mark()
        ok = self.unifier.unify(funct, head)
        if not ok:
            self.trail.undo_to(mark)
            # If this is the last rule in a resid re-fire, fall back to returning
            # the compound as-is (function can't reduce, acts as constructor).
            if _is_resid_refire and len(active) == 1:
                return self.unifier.unify(result, funct)
            return False

        # Sort-constrained computation rule fix:
        # Rule form: X:sort -> body_expr(X, ...)
        # The parser stores head_orig as one SORT_VAR and body's X occurrences
        # as INDEPENDENT SORT_VAR tokens (different Python objects, different ids).
        # copy_term with shared _vm therefore produces a DIFFERENT copy X'_body
        # for the body than X'_head for the head — they don't share the binding.
        # After unify(funct, head) binds X'_head → funct, X'_body remains free.
        # Fix: walk body and bind every free SORT_VAR of the same sort to funct.
        from wild_life.data_structures import SORT_VAR as _SORT_VAR_FLAG
        _head_orig_d = head_orig  # head_orig is the stored (un-copied) head
        _will_bfsv = ((_head_orig_d.flags & _SORT_VAR_FLAG) and _head_orig_d.type is not None and not _head_orig_d.attr_list)
        # A rule whose head is a bare sort (`X:sum -> ...`, `foo -> ...`) binds
        # its head variable to the call itself, so the body reads funct as a
        # term rather than as a call.  Reducing it again in there would restart
        # this very rule instead of reading the term's features.
        if not _head_orig_d.attr_list:
            from wild_life.data_structures import REDUCED as _REDUCED_FLAG
            if not (funct.flags & _REDUCED_FLAG):
                self.trail.trail_psi(funct, 'flags')
                funct.flags |= _REDUCED_FLAG

        if _will_bfsv:
            _sort_type = _head_orig_d.type
            _sv_visited: set = set()

            def _bind_free_sort_vars(t: 'PsiTerm') -> None:
                """Bind free SORT_VARs of _sort_type to funct, in-place."""
                if id(t) in _sv_visited:
                    return
                _sv_visited.add(id(t))
                if ((t.flags & _SORT_VAR_FLAG) and
                        t.type is _sort_type and
                        t.coref is None):
                    self.trail.trail_psi(t, 'coref')
                    t.coref = funct
                    return
                td = t.deref()
                if id(td) not in _sv_visited:
                    _sv_visited.add(id(td))
                    for _child in list(td.attr_list.values()):
                        _bind_free_sort_vars(_child)

            _bind_free_sort_vars(body)

        # Now that head args are bound, try arithmetic evaluation of body.
        body_d2 = body.deref()
        from wild_life.built_ins import _eval_arith, _make_number

        # Body is a user-defined function call — push EVAL so it gets evaluated
        # (rather than UNIFY which would just structurally bind result to the term).
        #
        # IMPORTANT: Check this BEFORE _eval_arith.  _eval_arith can inline-evaluate
        # user-defined functions (e.g. last([2,3]) → 3.0), but doing so loses the
        # original psi-term object identity: it returns (True, 3.0) and we then call
        # _make_number to create a FRESH psi-term.  That fresh term has a different
        # Python id than the original node in the data structure (e.g. the integer 3
        # inside list A=[1,2,3]).  The shared-term detection in print_variables uses
        # Python object identity to detect sharing, so the freshly created term is
        # NOT seen as the same object as the element of A — breaking "A = [1,2,B]".
        # Pushing an EVAL goal instead lets the machinery recurse properly and at the
        # base case (body is a concrete literal, not a user function) preserves the
        # original term identity.
        if _is_user_function(body_d2):
            self.push_goal(GoalType.EVAL, body_d2, result, body_d2.type.rule)
            return True

        arith_ok, arith_val = _eval_arith(body_d2, self)
        if arith_ok:
            # Body evaluated to a number — unify result with it immediately.
            # Mark _delay_fired=True because _eval_arith already fired delay for
            # the result (via the binary * path or pre-eval computed-term firing).
            # This prevents a second delay fire during unification with result.
            #
            # If the body is already a concrete literal (value is not None), bind
            # result directly to preserve the original psi-term's Python identity.
            if body_d2.value is not None:
                # Concrete literal — bind directly (delay already fired by _eval_arith)
                ok2 = self.unifier.unify(result, body_d2)
            else:
                # Compound arithmetic expression — create a new numeric term
                num_term = _make_number(self, arith_val)
                num_term._delay_fired = True
                ok2 = self.unifier.unify(result, num_term)
            if not ok2:
                self.trail.undo_to(mark)
                return False
            return True

        # Body is a built-in cond(C, T, E) — evaluate it as a functional conditional
        # (not as a predicate). This makes cond usable in function rule bodies.
        if _is_cond_builtin(body_d2):
            return _eval_cond_functional(body_d2, result, self)

        # Body is built-in map(F, List) in functional position — evaluate it now.
        from wild_life.built_ins import _eval_map_func as _emf
        _body_sym_map = body_d2.type.keyword.symbol if (body_d2.type and body_d2.type.keyword) else ''
        if (_body_sym_map == 'map'
                and '1' in body_d2.attr_list and '2' in body_d2.attr_list
                and '3' not in body_d2.attr_list):
            mapped = _emf(body_d2, self)
            if mapped is None:
                return False
            ok2 = self.unifier.unify(result, mapped)
            if not ok2:
                self.trail.undo_to(mark)
                return False
            return True

        # Body is a built-in function in functional position (features,
        # root_sort, …).  Embedded ones are reduced below as sub-terms, but a
        # body that IS such a call — `F:f -> features(F)` — has to be evaluated
        # here, or the call's value would be the unevaluated term.
        from wild_life.built_ins import _try_eval_string_func as _tesf_body
        _bi_body = None
        if not any(_is_user_function(_bv.deref())
                   for _bv in body_d2.attr_list.values()):
            # A call still waiting on a user function of its own — copy1.lf's
            # `copy(X) -> memo_copy(X,[]).1` — is left to the sub-term handling
            # below, which evaluates the inner call first.
            _bi_body = _tesf_body(body_d2, self)
        if _bi_body is not None and _bi_body is not body_d2:
            if not self.unifier.unify(result, _bi_body):
                self.trail.undo_to(mark)
                return False
            return True

        # Body is a call through a functor variable — twice's body F(F(X)).
        # Put it through '=', which rebuilds the call once the functor is known
        # and otherwise suspends on it, so that binding F later still reduces.
        if (getattr(wl, 'apply', None) is not None and body_d2.type is wl.apply
                and '1' in body_d2.attr_list):
            _eq_defn_ap2 = (getattr(wl, 'eqsym', None)
                            or wl.syntax_module.symbol_table.get('='))
            if _eq_defn_ap2 is not None:
                _eq_ap2 = PsiTerm(type_def=_eq_defn_ap2)
                _eq_ap2.attr_list = {'1': result, '2': body_d2}
                self.push_goal(GoalType.PROVE, _eq_ap2, None, None)
                return True

        # Body is a compound with possible embedded user-function sub-terms
        # (e.g. [X|app2(L1,L2)] where app2 is a recursive function).
        # Push EVAL goals for each embedded user-function call onto the goal
        # stack so they are evaluated *iteratively* (not via Python recursion).
        # This avoids hitting Python's stack depth limit for deeply-recursive
        # functions like largeterm(1000).
        #
        # Correct LIFO ordering:
        #   1. Push UNIFY first  → it sits below EVAL goals on the stack
        #   2. Push EVAL goals after → they sit on top, so they run FIRST
        # This ensures the fresh variables are bound before UNIFY fires.
        eval_goals = _collect_embedded_func_goals(body_d2, self, set())

        # Choose how to bind result to body_d2:
        # - Arithmetic expressions: go through bi_unify (PROVE via '=') so that the
        #   arithmetic constraint machinery (real~ marking, residuation on free vars)
        #   fires correctly.  A plain UNIFY goal bypasses bi_unify entirely.
        # - Everything else: use a plain UNIFY goal (faster, avoids bi_unify overhead).
        from wild_life.built_ins import _is_complete_arith_expr as _is_cae
        _body_sym = body_d2.type.keyword.symbol if (body_d2.type and body_d2.type.keyword) else ''
        from wild_life.built_ins import _ARITH_OPS_SET as _AOS
        _body_is_arith = (_body_sym in _AOS and _is_cae(body_d2))

        if _body_is_arith and not eval_goals:
            # Arithmetic body with no embedded user-function calls:
            # push via bi_unify (PROVE) so arithmetic constraints fire properly.
            _eq_defn_ei = getattr(wl, 'eqsym', None)
            if _eq_defn_ei is None and hasattr(wl, 'syntax_module'):
                _eq_defn_ei = wl.syntax_module.symbol_table.get('=')
            if _eq_defn_ei is not None:
                _eq_term_ei = PsiTerm(type_def=_eq_defn_ei)
                _eq_term_ei.attr_list['1'] = result
                _eq_term_ei.attr_list['2'] = body_d2
                self.push_goal(GoalType.PROVE, _eq_term_ei, None, None)
            else:
                self.push_goal(GoalType.UNIFY, body_d2, result, None)
        else:
            # Push UNIFY first (runs LAST — body_d2 has fresh vars for embedded calls)
            self.push_goal(GoalType.UNIFY, body_d2, result, None)

            # Push each EVAL goal (runs FIRST — binds the fresh vars before UNIFY).
            # Also lift any embedded user-function calls from each EVAL goal's compound
            # arguments (e.g. where((B,T) & copy_body(...)) needs copy_body evaluated).
            from wild_life.built_ins import _is_user_function as _iuf_ea
            for ft, rv, rl in eval_goals:
                # Push ft FIRST (runs LATER in LIFO order — after sub-goals).
                self.push_goal(GoalType.EVAL, ft, rv, rl)
                # Then push sub-goals (run EARLIER — before ft).
                # This handles e.g. where((B,T) & copy_body(...)) where
                # copy_body must be evaluated BEFORE where's body (@) runs.
                for _ak in list(ft.attr_list.keys()):
                    _av = ft.attr_list[_ak].deref()
                    if not _iuf_ea(_av) and _av.attr_list:
                        _sub_goals = _collect_embedded_func_goals(_av, self, set())
                        for _sft, _srv, _srl in _sub_goals:
                            self.push_goal(GoalType.EVAL, _sft, _srv, _srl)

        return True

    def match_aim(self) -> bool:
        """
        'match' goal: one-way unification — pattern (b) is unified with
        call (a), but a may not be changed.
        """
        aim = self.aim
        u = aim.a  # calling term (read-only)
        v = aim.b  # pattern (from definition)
        if u is None or v is None:
            return False
        u = u.deref()
        v = v.deref()
        if u is v:
            return True

        # Types must be compatible
        if not types_compatible(u.type, v.type):
            return False

        # Values must match if both have values
        if v.value is not None:
            if u.value is None:
                return False
            if u.value != v.value:
                return False

        # Bind v's coref → u (one-way: v points to u)
        mark = self.trail.mark()
        self.trail.trail_psi(v, 'coref')
        v.coref = u

        # Match attributes
        for key, vpsi in v.attr_list.items():
            upsi = u.attr_list.get(key)
            if upsi is None:
                self.trail.undo_to(mark)
                return False
            self.push_goal(GoalType.MATCH, upsi, vpsi, None)
        return True

    def clause_aim(self, retract: bool) -> bool:
        """Handle clause / retract goals."""
        aim = self.aim
        head = aim.a
        body = aim.b
        # For CLAUSE: rule_list_ref is a list of rules (possibly a slice).
        # For DEL_CLAUSE (retract): rule_list_ref is (master_list, start_idx) tuple
        # so we always delete from the master list.
        rule_list_ref = aim.c

        if retract:
            # Unpack (master_list, start_from) for retract
            if isinstance(rule_list_ref, tuple):
                master_list, start_from = rule_list_ref
            else:
                master_list, start_from = rule_list_ref, 0
            # Find the first non-deleted rule starting from start_from
            idx = start_from
            while idx < len(master_list) and (
                    master_list[idx][0] is None or master_list[idx][1] is None):
                idx += 1
            if idx >= len(master_list):
                return False
            # Push choice point to retry from idx+1 (using master_list with new start)
            has_more = any(
                master_list[i][0] is not None for i in range(idx + 1, len(master_list))
            )
            if has_more:
                self.push_choice_point(GoalType.DEL_CLAUSE, head, body, (master_list, idx + 1))
            h0, b0 = master_list[idx]
            # Store (master_list, idx) so RETRACT modifies the correct master slot.
            self.push_goal(GoalType.RETRACT, (master_list, idx), None, None)
        else:
            if not rule_list_ref or not isinstance(rule_list_ref, list):
                return False
            # Find the first non-deleted rule
            idx = 0
            while idx < len(rule_list_ref) and (
                    rule_list_ref[idx][0] is None or rule_list_ref[idx][1] is None):
                idx += 1
            if idx >= len(rule_list_ref):
                return False
            # Push choice point with the remaining slice
            next_rules = rule_list_ref[idx + 1:]
            if next_rules:
                self.push_choice_point(GoalType.CLAUSE, head, body, next_rules)
            h0, b0 = rule_list_ref[idx]

        _vm: dict = {}
        rule_head = copy_term(h0, _vm)
        rule_body = copy_term(b0, _vm)
        self.push_goal(GoalType.UNIFY, body, rule_body, None)
        self.push_goal(GoalType.UNIFY, head, rule_head, None)
        return True

    def load_file(self, filename: str) -> bool:
        """Load a LIFE source file."""
        from wild_life.tokenizer import tokenizer_from_file
        from wild_life.parser_ import Parser
        try:
            ts = tokenizer_from_file(filename)
        except FileNotFoundError:
            print(f"*** Error: cannot open file '{filename}'.", file=sys.stderr)
            return False

        p = Parser(ts)
        while True:
            try:
                term, sort = p.parse()
            except Exception as e:
                print(f"*** Syntax error in '{filename}': {e}", file=sys.stderr)
                break

            if term is None:
                break
            t = term.deref()
            wl = self.wl
            if t.type == wl.eof:
                break
            if sort == FACT:
                self.assert_first = False
                try:
                    self.assert_clause(t)
                except SortCycleException:
                    # Cycle in .lf file: write a newline so refout matches
                    # (the C interpreter outputs \n before halting), then exit.
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    raise HaltException(1)
            elif sort == QUERY:
                # Execute query; push as goal.  A query in a file runs for its
                # first solution only: the alternatives it leaves behind are
                # dropped, so they do not turn the prompt that follows the load
                # into a '--1>' continuation of the file's last query.
                _cs_before = self.choice_stack
                self.push_goal(GoalType.PROVE, t, _DEFRULES, None)
                self.run(cs_barrier=_cs_before)
                self.choice_stack = _cs_before
                self.goal_stack = None
        return True

    # ─── main loop ──────────────────────────────────────────────────────────

    def run(self, cs_barrier=None) -> bool:
        """
        Run the main prove loop (main_prove in login.c).
        Returns True if the goal_stack was satisfied (at least once).

        cs_barrier: if set, do not backtrack past this choice point.
            Used when proving fresh queries at depth > 0 to prevent the new
            query from consuming choice points belonging to an outer query.
        """
        success = True
        self.main_loop_ok = True
        self.goal_count = 0

        while self.main_loop_ok and self.goal_stack:
            self.aim = self.goal_stack

            try:
                gtype = self.aim.type

                if gtype == GoalType.PROVE:
                    success = self.prove_aim()

                elif gtype == GoalType.UNIFY:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    success = self.unify_aim()

                elif gtype == GoalType.BIND_DIRECT:
                    # Disjunction expansion choice point: directly bind the
                    # disjunction psi-term (aim.a) to the alternative (aim.b)
                    # without going through full unification (which would trigger
                    # spurious cross-product for nested disjunctions).
                    # IMPORTANT: use aim.a directly (NOT deref'd) because
                    # backtracking restored aim.a.coref to None — aim.a is the
                    # disjunction node we always bind element-by-element.
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    _bd_u = self.aim.a   # the disjunction psi-term (not deref'd)
                    _bd_v = self.aim.b   # the next alternative element
                    if _bd_u is not None:
                        _bd_v_d = _bd_v.deref() if _bd_v is not None else _bd_v
                        # Try arithmetic evaluation on the alternative: if alt is
                        # a ground arithmetic expression (e.g. 1+2, A+5 with A=5),
                        # evaluate it and memoize the result so that deref through
                        # the disjunction node gives the concrete number directly.
                        _bd_val = _bd_v_d
                        if (_bd_v_d is not None and not getattr(self, 'no_arith_eval', False)
                                and _bd_v_d.type is not None and _bd_v_d.type.keyword is not None):
                            _bd_sym = _bd_v_d.type.keyword.symbol
                            _bd_arith_ops = frozenset(('+', '-', '*', '/', '//', 'mod',
                                                        '**', '^', 'max', 'min',
                                                        '/\\', '\\/', 'xor', '>>', '<<'))
                            if _bd_sym in _bd_arith_ops:
                                try:
                                    from wild_life.built_ins import (
                                        _eval_arith as _bd_ea, _make_number as _bd_mn)
                                    _bd_ok, _bd_num_v = _bd_ea(_bd_v_d, self)
                                    if _bd_ok:
                                        _bd_num = _bd_mn(self, _bd_num_v)
                                        _bd_val = _bd_num
                                        # Memoize into the arithmetic term
                                        if _bd_v_d.coref is None and _bd_v_d.value is None and _bd_v_d.attr_list:
                                            self.trail.trail_psi(_bd_v_d, 'coref')
                                            _bd_v_d.coref = _bd_num
                                except Exception:
                                    pass
                        self.unifier.trail.trail_psi(_bd_u, 'coref')
                        _bd_u.coref = _bd_val
                        self.unifier._wakeup_resid(_bd_u, _bd_u)
                        # Fire sort-level delay rules (:: SortName | goal) when the
                        # disjunction node gets bound to a sort-typed term.
                        _bd_val_canon = _bd_val.deref() if _bd_val is not None else None
                        from wild_life.runtime import WL as _bd_WL
                        if (_bd_WL.delay_rules and _bd_val_canon is not None
                                and _bd_val_canon.type is not None
                                and _bd_val_canon.type is not _bd_WL.top
                                and not getattr(_bd_val_canon, '_delay_fired', False)):
                            _bd_val_canon._delay_fired = True
                            self.unifier._fire_delay_rules(_bd_val_canon, _bd_val_canon.type)
                    success = True

                elif gtype == GoalType.UNIFY_NOEVAL:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    success = self.unify_aim()

                elif gtype == GoalType.EVAL:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    success = self.eval_aim()

                elif gtype == GoalType.SUCHTHAT_VAL:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    success = self.suchthat_val_aim()

                elif gtype == GoalType.MATCH:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    success = self.match_aim()

                elif gtype == GoalType.FAIL:
                    self.goal_stack = self.aim.next
                    success = False

                elif gtype == GoalType.CLAUSE:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    success = self.clause_aim(False)

                elif gtype == GoalType.DEL_CLAUSE:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    success = self.clause_aim(True)

                elif gtype == GoalType.RETRACT:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    retract_info = self.aim.a
                    if isinstance(retract_info, tuple):
                        # New-style: (master_list, index)
                        master_list, del_idx = retract_info
                        master_list[del_idx] = (None, None)
                    elif retract_info and isinstance(retract_info, list) and retract_info:
                        # Old-style fallback: list, mark first slot
                        retract_info[0] = (None, None)

                elif gtype == GoalType.WHAT_NEXT:
                    self.goal_stack = self.aim.next
                    success = self._what_next_aim()

                elif gtype == GoalType.GENERAL_CUT:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    self.cut_to(self.aim.a)

                elif gtype == GoalType.EVAL_COMMIT:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    self.drop_choice_point(self.aim.a)

                else:
                    print(f"*** Error: unknown goal type {gtype}", file=sys.stderr)
                    self.goal_stack = self.aim.next
                    success = False

            except HaltException:
                raise
            except AbortException:
                raise  # propagate to main.py (AbortException carries hook_called flag)
            except CutException as e:
                self.cut_to(e.cut_point)
                success = True

            if self.main_loop_ok:
                if not success:
                    # Backtrack to the most recent choice point, but not past
                    # cs_barrier (which marks the boundary of this fresh query).
                    can_backtrack = (
                        self.choice_stack is not None
                        and (cs_barrier is None or self.choice_stack is not cs_barrier)
                    )
                    if can_backtrack:
                        self.backtrack()
                        success = True
                    else:
                        if cs_barrier is None:
                            # No barrier: full cleanup (top-level query)
                            self.trail.undo_to(0)
                        # With barrier: don't undo trail — caller (main.py) handles it
                        if self.noisy:
                            print("\n*** No", end='', flush=True)
                        self.main_loop_ok = False

        return success

    def prove(self, goal: PsiTerm, cs_barrier=None) -> bool:
        """Prove a single goal. Returns True on success.

        cs_barrier: if set, do not backtrack past this choice point.
            Pass engine.choice_stack to prevent this fresh query from
            consuming choice points that belong to an enclosing query.
        """
        _mark_non_strict_args(goal, self)
        self.push_goal(GoalType.PROVE, goal, _DEFRULES, None)
        return self.run(cs_barrier=cs_barrier)

    def _what_next_aim(self) -> bool:
        """Handle user interaction at a query result."""
        aim = self.aim
        level = aim.c if isinstance(aim.c, int) else 0
        has_answer = bool(aim.a)
        wl = self.wl

        from wild_life.print_term import print_variables
        vt = getattr(wl, '_var_tree', {})

        if has_answer:
            print("\n*** Yes", end='', flush=True)
        else:
            print("\n*** No", end='', flush=True)

        if has_answer or level > 0:
            from wild_life.print_term import PRINT_DEPTH as _INF_PD
            _inf_pd = getattr(wl, 'print_depth', _INF_PD) if wl else _INF_PD
            print_variables(vt, sys.stdout, wl=wl, print_depth=_inf_pd)

        prompt = '--' * min(level, 4) + '?- '
        print(prompt, end='', flush=True)

        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            self.trail.undo_to(0)
            self.goal_stack = None
            self.choice_stack = None
            return True

        line = line.rstrip('\n')
        if line == '' or line == '\n':
            # Accept (cut remaining choices)
            while self.choice_stack:
                self.choice_stack = self.choice_stack.next
            return True

        if line.startswith(';'):
            # Request more solutions
            if self.choice_stack:
                self.backtrack()
                return True
            else:
                print("*** No more solutions.", flush=True)
                return True

        if line.startswith('.'):
            self.trail.undo_to(0)
            self.goal_stack = None
            self.choice_stack = None
            return True

        # Otherwise treat as a new query
        from wild_life.parser_ import parse_string
        from wild_life.tokenizer import tokenizer_from_string
        from wild_life.parser_ import Parser
        ts = tokenizer_from_string(line)
        p = Parser(ts)
        try:
            t, sort = p.parse()
        except Exception:
            return True

        if t and sort == QUERY:
            if level > 0:
                self.push_choice_point(GoalType.WHAT_NEXT, False, None, level)
            self.push_goal(GoalType.WHAT_NEXT, True, self.var_occurred, level + 1)
            self.push_goal(GoalType.PROVE, t, _DEFRULES, None)
            return True

        return True

    @property
    def var_occurred(self) -> bool:
        return self._var_occurred

    @var_occurred.setter
    def var_occurred(self, v: bool) -> None:
        self._var_occurred = v

    _var_occurred: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel: "use the type's own rule list"
# ─────────────────────────────────────────────────────────────────────────────
_DEFRULES = object()  # sentinel — same role as DEFRULES macro in C

# Sentinel used as cs_barrier when a built-in (bi_not, bi_once, bi_cond, …)
# calls eng.run() for a sub-proof.  When cs_barrier is this sentinel (non-None),
# run() will NOT undo trail entries to position 0 on failure — it only sets
# main_loop_ok=False and returns False, leaving outer bindings intact.
_INNER_RUN_BARRIER = object()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: collect all clauses for a predicate (for clause/2)
# ─────────────────────────────────────────────────────────────────────────────

def get_rules_for(defn: Definition):
    """Return the list of (head, body) pairs for defn, or []."""
    if defn is None or defn.rule is None:
        return []
    if callable(defn.rule):
        return []  # built-in
    return list(defn.rule)
