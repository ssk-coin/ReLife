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
        val_d = head_d.attr_list[key].deref()
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
        else:
            # Bare fact
            self.add_rule(t, None, DefType.PREDICATE)

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
            if defn is None:
                return False
            if defn.type == DefType.PREDICATE:
                rules = defn.rule or []
            elif defn.type == DefType.FUNCTION:
                rules = defn.rule or []
            elif defn.type == DefType.UNDEF:
                if defn.rule is None:
                    # Never declared (not via dynamic/assert) → error + abort
                    name = defn.keyword.symbol if defn.keyword else '?'
                    sys.stderr.write(
                        f"*** Error: '{name}' is not a predicate or a function.\n"
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

        # User-defined function: find first active rule
        active = [(h, b) for (h, b) in (rules if rules else [])
                  if h is not None and b is not None]
        if not active:
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
                # Push: unify result with val_part AFTER cond_part is proven
                self.push_goal(GoalType.UNIFY, val_part, result, None)
                self.push_goal(GoalType.PROVE, cond_part, _DEFRULES, None)
                # For functions with input arguments (non-nullary), unify funct
                # with head to bind the argument variables before the body runs.
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

        # Unify head with funct first (to bind head arguments)
        mark = self.trail.mark()
        ok = self.unifier.unify(funct, head)
        if not ok:
            self.trail.undo_to(mark)
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
        if ((_head_orig_d.flags & _SORT_VAR_FLAG) and
                _head_orig_d.type is not None and
                not _head_orig_d.attr_list):
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

        # Now that head args are bound, try arithmetic evaluation of body
        body_d2 = body.deref()
        from wild_life.built_ins import _eval_arith, _make_number
        arith_ok, arith_val = _eval_arith(body_d2, self)
        if arith_ok:
            # Body evaluated to a number — unify result with it immediately
            num_term = _make_number(self, arith_val)
            ok2 = self.unifier.unify(result, num_term)
            if not ok2:
                self.trail.undo_to(mark)
                return False
            return True

        # Body is a user-defined function call — push EVAL so it gets evaluated
        # (rather than UNIFY which would just structurally bind result to the term)
        if _is_user_function(body_d2):
            self.push_goal(GoalType.EVAL, body_d2, result, body_d2.type.rule)
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
