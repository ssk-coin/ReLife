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
    Trail, Unifier, copy_term, compute_lub, types_compatible
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
        """Handle type declarations (<| or :=)."""
        from wild_life.data_structures import DefType
        # Simplified: mark the LHS type as a subtype of the RHS
        arg1 = t.attr_list.get('1')
        arg2 = t.attr_list.get('2')
        if arg1 and arg2:
            arg1 = arg1.deref()
            arg2 = arg2.deref()
            if arg1.type and arg2.type:
                # Add arg2.type as parent of arg1.type
                child = arg1.type
                parent = arg2.type
                # Mark both as TYPE sorts (they may have been UNDEF if newly created)
                if child.type == DefType.UNDEF:
                    child.type = DefType.TYPE
                if parent.type == DefType.UNDEF:
                    parent.type = DefType.TYPE
                if parent not in child.parents:
                    child.parents.append(parent)
                if child not in parent.children:
                    parent.children.append(child)

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
                self.main_loop_ok = False
                return False
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
                # Dynamic predicate with no clauses → fail silently
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
        mark = self.trail.mark()
        ok = self.unifier.unify(thegoal, head)
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

        # Unify head with funct first (to bind head arguments)
        mark = self.trail.mark()
        ok = self.unifier.unify(funct, head)
        if not ok:
            self.trail.undo_to(mark)
            return False

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

        # Body is not pure arithmetic — push as a UNIFY goal for later resolution
        self.push_goal(GoalType.UNIFY, body_d2, result, None)
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
                self.assert_clause(t)
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
                self.trail.undo_to(0)
                self.goal_stack = None
                self.choice_stack = None
                return False
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
