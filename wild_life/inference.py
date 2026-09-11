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
    DefType, FACT, QUERY, ERROR
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

def _mark_arith_non_strict(t: PsiTerm, visited: set = None) -> None:
    """Recursively mark arithmetic operator psiterms with NON_STRICT_TERM.

    Called after head unification for a non-strict predicate so that
    arithmetic sub-expressions in the bound result are not eagerly
    evaluated during printing.
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
        td.flags |= NON_STRICT_TERM
    for v in td.attr_list.values():
        _mark_arith_non_strict(v, visited)


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

    # Compound with embedded user-function sub-terms
    eval_goals = _collect_embedded_func_goals(branch_d, eng, set())
    eng.push_goal(GoalType.UNIFY, branch_d, result, None)
    for ft, rv, rl in eval_goals:
        eng.push_goal(GoalType.EVAL, ft, rv, rl)
    return True


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

    # Prove the condition via inner run (same pattern as bi_cond)
    mark = eng.trail.mark()
    cp_save = eng.choice_stack
    gs_save = eng.goal_stack
    eng.goal_stack = None
    eng.push_goal(GoalType.PROVE, cond_g, _DEFRULES, None)
    old_ok = eng.main_loop_ok
    barrier = cp_save if cp_save is not None else _INNER_RUN_BARRIER
    cond_ok = eng.run(cs_barrier=barrier)
    eng.main_loop_ok = old_ok
    eng.choice_stack = cp_save
    eng.goal_stack = gs_save

    if cond_ok:
        return _eval_body_to_result(then_g, result, eng)
    else:
        eng.trail.undo_to(mark)
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
            # Not a function call; walk its children.
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

    def push_choice_point(self, gtype: GoalType, a=None, b=None, c=None) -> None:
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
        head = head.deref()
        defn = head.type
        if defn is None:
            return False

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
                    body_copy = PsiTerm(type=wl.succeed)
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
            if arg2.type is not None and arg2.type is self.wl.such_that:
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

        # Filter out retracted clauses
        active = [(h, b) for (h, b) in (rules if rules else [])
                  if h is not None and b is not None]
        if not active:
            self.goal_stack = aim.next
            self.goal_count += 1
            return False

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
        mark = self.trail.mark()
        ok = self.unifier.unify(u, v)
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
        if len(active) > 1:
            self.push_choice_point(GoalType.EVAL, funct, result, active[1:])

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
                    mark = self.trail.mark()
                    ok = self.unifier.unify(funct, head)
                    if not ok:
                        self.trail.undo_to(mark)
                        return False
                # Now that argument variables are bound, eagerly evaluate any
                # built-in or user-defined functional sub-terms in cond_part
                # (e.g. genChildren(children(X), A) → children(X) → [a,b,c,d]).
                from wild_life.built_ins import _eval_embedded_user_funcs
                _cond_d = cond_part.deref()
                _eval_embedded_user_funcs(_cond_d, self, 0, set())
                # Push: unify result with val_part AFTER cond_part is proven
                self.push_goal(GoalType.UNIFY, val_part, result, None)
                self.push_goal(GoalType.PROVE, _cond_d, _DEFRULES, None)
                return True

        # Pre-evaluate any function call arguments in funct.
        # This enables patterns like f(g(x)) where g(x) needs to be evaluated
        # before pattern matching against f's head (e.g. rev(reverse(L),[]) ).
        from wild_life.built_ins import (
            _eval_user_func_sync, _is_user_function,
            _try_eval_string_func, _try_eval_arith_to_term,
        )
        for _key in list(funct.attr_list.keys()):
            _attr = funct.attr_list[_key].deref()
            if _is_user_function(_attr):
                _evaled = _eval_user_func_sync(_attr, self)
                if _evaled is not None and _evaled is not _attr:
                    funct.attr_list[_key] = _evaled
            else:
                # Try built-in function evaluation (features, root_sort, etc.)
                _evaled = _try_eval_string_func(_attr, self)
                if _evaled is not None:
                    funct.attr_list[_key] = _evaled
                else:
                    _evaled = _try_eval_arith_to_term(_attr, self)
                    if _evaled is not None:
                        funct.attr_list[_key] = _evaled

        # Expand disjunctions embedded in function arguments.
        # e.g. f(s({1;2;3})) → try f(s(1)), then f(s(2)), then f(s(3)).
        # Push choice points for alternatives 2..N before trying alt 1.
        from wild_life.built_ins import _term_contains_disjunction, _expand_term_disjunctions
        if _term_contains_disjunction(funct, self):
            _alts = _expand_term_disjunctions(funct, self)
            if len(_alts) > 1:
                # Push choice points for alternatives 2..N (in reverse so first
                # alternative is tried next, then 2nd, etc.)
                for _alt in reversed(_alts[1:]):
                    _vm2: dict = {}
                    _h2 = copy_term(head_orig, _vm2)
                    _b2 = copy_term(body_orig, _vm2)
                    self.push_choice_point(GoalType.EVAL, _alt, result, active)
                funct = _alts[0]
                # Recompute fresh head/body copies for the first alternative
                _vm = {}
                head = copy_term(head_orig, _vm)
                body = copy_term(body_orig, _vm)

        # Arity check: if head has feature keys not present in funct, this rule
        # requires arguments that the call doesn't provide.  Skip the rule —
        # adding extra features to a function call is wrong semantics (unlike
        # sort unification where adding features is fine).
        _head_d_arity = head.deref()
        _funct_keys_set = set(funct.attr_list.keys())
        _head_only_keys = set(_head_d_arity.attr_list.keys()) - _funct_keys_set
        if _head_only_keys:
            # Funct has fewer args than this rule requires — partial application.
            # In Wild Life, calling a function with fewer args than its head needs
            # is always a partial application: return funct as a constructor term.
            if len(active) == 1:
                # Last rule: return funct as partial application (constructor semantics).
                return self.unifier.unify(result, funct)
            # More rules exist; skip this one (try next via choice point).
            return False

        # Residuation check: if funct has completely free (unbound) arguments,
        # don't eagerly bind them to sorts just to match a head pattern.
        # Instead, suspend (residuate) on those free variables so that when they
        # get bound (by a later goal), the function is re-evaluated.
        # "Completely free" = type is top, no attrs, no value, no sort constraint.
        _free_args_for_resid = []
        for _fk_r, _fv_r_psi in funct.attr_list.items():
            _fv_r = _fv_r_psi.deref()
            _fv_r_is_free = (
                (_fv_r.type is None or _fv_r.type is wl.top) and
                not _fv_r.attr_list and
                _fv_r.value is None
            )
            if _fv_r_is_free:
                # Check if the corresponding head arg is a non-top constraint.
                _head_r_arg = _head_d_arity.attr_list.get(_fk_r)
                if _head_r_arg is not None:
                    _head_r_d = _head_r_arg.deref()
                    _head_r_is_constrained = (
                        _head_r_d.type is not None and _head_r_d.type is not wl.top
                    )
                    if _head_r_is_constrained:
                        _free_args_for_resid.append(_fv_r)
        if _free_args_for_resid:
            # Set up residuation: attach a pending EVAL goal to each free variable.
            # When the variable gets bound, _wakeup_resid will push the EVAL goal
            # back onto the goal stack and f(bound_val) will be re-evaluated.
            from wild_life.data_structures import Goal as _ResidGoal, Residuation as _ResidR, SORT_VAR as _SV_R
            _pending_eval = _ResidGoal(GoalType.EVAL, funct, result, rules, pending=True)
            _pending_eval._resid_marker = True  # mark as residuation so re-fire knows
            # Mark funct so eval_aim can detect resid re-fire even from a freshly pushed Goal.
            # _wakeup_resid calls push_goal(g.type, g.a, g.b, g.c) which creates a new Goal
            # without _resid_marker, so we propagate via funct (which is g.a and is preserved).
            funct._resid_refire = True
            for _fv_r in _free_args_for_resid:
                if _fv_r.resid is None:
                    self.trail.trail_psi(_fv_r, 'resid')
                    _fv_r.resid = [_ResidR(goal=_pending_eval)]
                else:
                    if not any(rv.goal is _pending_eval for rv in _fv_r.resid):
                        self.trail.trail_copy(_fv_r, 'resid')
                        _fv_r.resid.append(_ResidR(goal=_pending_eval))
                # Mark with SORT_VAR-like flag so display shows @~
                if not (_fv_r.flags & _SV_R):
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

            # Push each EVAL goal (runs FIRST — binds the fresh vars before UNIFY)
            for ft, rv, rl in eval_goals:
                self.push_goal(GoalType.EVAL, ft, rv, rl)

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
                # Execute query; push as goal
                self.push_goal(GoalType.PROVE, t, _DEFRULES, None)
                self.run()
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

                elif gtype == GoalType.UNIFY_NOEVAL:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    success = self.unify_aim()

                elif gtype == GoalType.EVAL:
                    self.goal_stack = self.aim.next
                    self.goal_count += 1
                    success = self.eval_aim()

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
            print_variables(vt, sys.stdout, wl=wl)

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
