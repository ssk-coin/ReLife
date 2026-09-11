"""
unification.py - Wild Life 単一化エンジン (Python版)

C版の対応ファイル: login.c (前半のunification部分), lefun.c

LIFE言語の単一化 (Unification):
  通常の Prolog 単一化を拡張して psi-term (型付き特性項) に対応:

  1. 型の単一化: 型階層における最小上限 (LUB: Least Upper Bound)
     型 A と型 B の LUB = A と B を両方満たす最も特殊な型

  2. 特性の単一化: 対応する特性を再帰的に単一化

  3. 変数束縛: union-find アルゴリズムを使用
     バックトラック可能な束縛のために trail (アンドゥスタック) を使用

例:
  foo(x => 1) と foo(y => 2) を単一化すると:
  -> foo(x => 1, y => 2)

  integer と real を単一化すると失敗 (共通のサブタイプがない)
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import sys

from wild_life.data_structures import (
    PsiTerm, Definition, DefType, Rule, UndoEntry, ChoicePoint, Goal, GoalType,
    Residuation
)
from wild_life.runtime import WL


# ==================== 例外 ====================

class UnificationFailure(Exception):
    """単一化失敗"""
    pass


class CutException(Exception):
    """カット演算子 (!)"""
    def __init__(self, cut_point=None):
        self.cut_point = cut_point


class AbortException(Exception):
    """abort 例外

    hook_called: True ならば aborthook が既に呼ばれ、改行なしで出力が終わっている。
    その場合 main.py の例外ハンドラは余分な '\\n' を書かない。
    """
    def __init__(self, hook_called: bool = False):
        self.hook_called = hook_called


class HaltException(Exception):
    """halt 例外"""
    def __init__(self, code: int = 0):
        self.code = code


class SortCycleException(Exception):
    """ソート階層にサイクルが検出されたときに送出される例外。

    cycle_path: サイクルを形成する Definition オブジェクトのリスト
                (parent から child まで、child を末尾に含む)
    """
    def __init__(self, cycle_path: list):
        self.cycle_path = cycle_path


# ==================== トレイル (アンドゥスタック) ====================

class Trail:
    """バックトラック用トレイル
    C版の undo_stack に対応

    変数束縛をトレイルに記録し、バックトラック時に元に戻す。
    """

    def __init__(self):
        self._trail: List[Tuple[PsiTerm, str, Any]] = []
        # タプル: (psi_term, field_name, old_value)

    def mark(self) -> int:
        """現在のトレイル位置を記録 (バックトラック点)"""
        return len(self._trail)

    def trail_psi(self, t: PsiTerm, field: str):
        """PsiTerm のフィールドをトレイルに記録"""
        old_val = getattr(t, field)
        self._trail.append((t, field, old_val))

    def trail_copy(self, obj, field: str):
        """フィールドのシャローコピーをトレイルに記録 (リストなど可変オブジェクト用)"""
        import copy
        old_val = getattr(obj, field)
        saved = copy.copy(old_val)
        self._trail.append((obj, field, saved))

    def undo_to(self, mark: int):
        """mark 位置までトレイルを巻き戻す"""
        while len(self._trail) > mark:
            t, field, old_val = self._trail.pop()
            setattr(t, field, old_val)

    def __len__(self):
        return len(self._trail)


# ==================== 型の GLB (最大下限) 計算 ====================

def compute_lub(d1: Definition, d2: Definition) -> Optional[Definition]:
    """後方互換のため残す — compute_glb() を使うこと。"""
    return compute_glb(d1, d2)


def compute_glb(d1: Definition, d2: Definition) -> Optional[Definition]:
    """2つの型の最大下限 (GLB: Greatest Lower Bound) を計算する。

    LIFE の型単一化では「最も特殊な共通サブタイプ」が必要。
    型階層を *下向き* に BFS して d1 と d2 の両方のサブタイプを探す。

    d1 が d2 のサブタイプなら d1 を返す (d1 の方が特殊)。
    d2 が d1 のサブタイプなら d2 を返す。
    共通サブタイプがなければ None を返す (型が非互換)。

    注: 元の C 版の compute_lub() は実際には GLB を計算していた
    (型単一化は GLB = infimum が必要)。
    """
    if d1 is d2:
        return d1

    # top (@) との組み合わせ: top は全ての型のスーパータイプ
    if d1 is WL.top:
        return d2   # d2 の方が特殊 (または同じ)
    if d2 is WL.top:
        return d1

    # 一方が他方のサブタイプならそちらを返す (より特殊)
    if d1.is_subtype_of(d2):
        return d1
    if d2.is_subtype_of(d1):
        return d2

    # d1 の全サブタイプを収集 (children 方向に BFS)
    d1_subs: set = set()
    queue = list(d1.children)
    while queue:
        d = queue.pop(0)
        if d not in d1_subs:
            d1_subs.add(d)
            queue.extend(d.children)

    # d2 の全サブタイプの中で d1_subs に入っているものを探す
    common = []
    queue = list(d2.children)
    visited: set = set()
    while queue:
        d = queue.pop(0)
        if d not in visited:
            visited.add(d)
            if d in d1_subs:
                common.append(d)
            queue.extend(d.children)

    if not common:
        return None  # 共通サブタイプなし → 非互換

    # 最も特殊な共通サブタイプを選ぶ
    # (他のどの common メンバーのサブタイプでもないもの)
    most_specific = common[0]
    for d in common[1:]:
        if d.is_subtype_of(most_specific):
            most_specific = d
    return most_specific


def compute_all_glbs(d1: Definition, d2: Definition) -> List[Definition]:
    """2つの型の全ての最大下限 (GLB) を計算する。

    単一の GLB しかない場合は1要素リストを返す。
    複数の非比較可能なミニマルサブタイプがある場合は全てを返す。
    共通サブタイプがなければ空リストを返す。

    例: four_wheels と vehicle の GLB は [truck, car] の両方になりうる。
    """
    if d1 is d2:
        return [d1]
    if d1 is WL.top:
        return [d2]
    if d2 is WL.top:
        return [d1]
    if d1.is_subtype_of(d2):
        return [d1]
    if d2.is_subtype_of(d1):
        return [d2]

    # d1 の全サブタイプを収集 (children 方向に BFS)
    d1_subs: set = set()
    queue = list(d1.children)
    while queue:
        d = queue.pop(0)
        if d not in d1_subs:
            d1_subs.add(d)
            queue.extend(d.children)

    # d2 の全サブタイプの中で d1_subs に入っているものを探す
    common = []
    queue = list(d2.children)
    visited: set = set()
    while queue:
        d = queue.pop(0)
        if d not in visited:
            visited.add(d)
            if d in d1_subs:
                common.append(d)
            queue.extend(d.children)

    if not common:
        return []

    # 最大下限 (GLB) の元を選ぶ: common の中で他の要素のサブタイプでないもの
    # (より特殊な共通サブタイプが存在しない、つまり d1・d2 の直接の共通サブタイプ)
    # 例: four_wheels & vehicle → [truck, car] (rolls_royce は car のサブタイプなので除外)
    maximal: List[Definition] = []
    for d in common:
        if not any(other is not d and d.is_subtype_of(other) for other in common):
            maximal.append(d)
    return maximal


def types_compatible(d1: Definition, d2: Definition) -> bool:
    """2つの型が単一化可能かどうか判定。

    共通サブタイプ (GLB) が存在するとき True。
    integer と real のような直交した基本型は False になる。
    """
    if d1 is d2:
        return True
    if d1 is WL.top or d2 is WL.top:
        return True
    # 一方が他方のサブタイプなら互換
    if d1.is_subtype_of(d2) or d2.is_subtype_of(d1):
        return True
    # ユーザー定義の共通サブタイプがあれば互換
    return compute_glb(d1, d2) is not None


# ==================== 単一化エンジン ====================

class Unifier:
    """LIFE言語の単一化エンジン
    C版の global_unify(), global_unify_attr() などに対応 (login.c)
    """

    def __init__(self, trail: Trail, engine=None):
        self.trail = trail
        self.engine = engine  # back-reference to Engine (may be None)
        # Deferred wakeup support: goals woken during nested _unify_attrs calls
        # are queued here and pushed (in order) after the top-level unify returns.
        # This ensures that when multiple variables are bound during compound-term
        # unification, their pending goals fire in ATTRIBUTE ORDER (1 before 2 before
        # 3) rather than in reversed LIFO order.
        self._unify_nesting = 0      # depth counter for nested unify calls
        self._deferred_wakeups: list = []  # list of (gtype, ga, gb, gc) tuples

    def bind(self, var: PsiTerm, val: PsiTerm):
        """変数 var を val に束縛する (バックトラック可能)
        C版の push_ptr_value() / push_psi_ptr_value() に対応
        """
        # coref フィールドをトレイルに記録してから変更
        self.trail.trail_psi(var, 'coref')
        var.coref = val

    def bind_type(self, t: PsiTerm, new_type: Definition):
        """PsiTerm の型を変更する (バックトラック可能)"""
        self.trail.trail_psi(t, 'type')
        t.type = new_type

    def bind_value(self, t: PsiTerm, new_value: Any):
        """PsiTerm の値を変更する (バックトラック可能)"""
        self.trail.trail_psi(t, 'value')
        t.value = new_value

    def set_attr(self, t: PsiTerm, key: str, val: PsiTerm):
        """属性を設定する (バックトラック可能)"""
        # dict の変更をトレイルに記録
        old_attrs = dict(t.attr_list)
        self.trail.trail_psi(t, 'attr_list')
        t.attr_list = old_attrs
        t.attr_list[key] = val

    def unify(self, u: PsiTerm, v: PsiTerm) -> bool:
        """2つの psi-term を単一化する
        C版の global_unify() に対応 (login.c)

        Args:
            u, v: 単一化する2つの psi-term

        Returns:
            True if successful, False on failure
        """
        is_top_level = (self._unify_nesting == 0)
        self._unify_nesting += 1
        try:
            result = self._unify_impl(u, v)
        finally:
            self._unify_nesting -= 1
            # At the top-level unify call, flush deferred wakeup goals in the
            # order they were collected (attribute order 1,2,3,…).  Push them
            # onto the goal stack in REVERSE order so the first goal ends up on
            # top (LIFO → executes first → fires in the correct attribute order).
            if is_top_level and self._deferred_wakeups and self.engine is not None:
                for _gt, _ga, _gb, _gc in reversed(self._deferred_wakeups):
                    self.engine.push_goal(_gt, _ga, _gb, _gc)
                self._deferred_wakeups.clear()
        return result

    def _unify_impl(self, u: PsiTerm, v: PsiTerm) -> bool:
        """Internal unify implementation (called from unify with nesting tracking)."""
        u = u.deref()
        v = v.deref()

        if u is v:
            return True  # 同一オブジェクト

        # 変数の処理
        u_is_var = (u.type is WL.top and not u.attr_list and not u.resid)
        v_is_var = (v.type is WL.top and not v.attr_list and not v.resid)

        # Sort-constrained variables (X:sort — marked SORT_VAR by the parser, or
        # X:ran where ran is a FUNCTION sort) are treated as bindable variables.
        if not u_is_var and not v_is_var:
            from wild_life.data_structures import DefType, QUOTED_TRUE, SORT_VAR
            # SORT_VAR flag: set by parser for any X:sort syntax
            if u.flags & SORT_VAR:
                u_is_var = True
            elif (u.value is None and not u.attr_list and not u.resid and
                    not (u.flags & QUOTED_TRUE) and
                    u.type is not None and u.type.type == DefType.FUNCTION and
                    u.type._builtin_func is None):
                u_is_var = True
            if v.flags & SORT_VAR:
                v_is_var = True
            elif (v.value is None and not v.attr_list and not v.resid and
                    not (v.flags & QUOTED_TRUE) and
                    v.type is not None and v.type.type == DefType.FUNCTION and
                    v.type._builtin_func is None):
                v_is_var = True

        if u_is_var:
            # If u is a sort-constrained variable (type != WL.top) and v is a plain
            # top variable, bind v→u so that dereferencing v returns u (which
            # retains its sort constraint).  For FUNCTION sorts this preserves sort
            # information for _is_user_function checks; for regular SORT sorts it
            # ensures the sort constraint is visible after binding.
            from wild_life.data_structures import SORT_VAR as _SORT_VAR_FLAG
            u_is_fn_sort = (u.type is not WL.top)
            u_is_sort_var = bool(u.flags & _SORT_VAR_FLAG)  # user X:sort annotation
            if u_is_fn_sort and v_is_var:
                # u has a sort/function-sort constraint; v is a variable.
                # If v also has a sort constraint (SORT_VAR), we must verify
                # type compatibility — both sorts must have a common sub-sort.
                if u_is_sort_var and (v.flags & _SORT_VAR_FLAG) and v.type is not WL.top:
                    if not self._unify_types(u, v):
                        return False
                self.bind(v, u)   # v.coref = u; v.deref() = u (sort kept)
                self._wakeup_resid(u, v)
            elif u_is_sort_var and u_is_fn_sort and not v_is_var:
                # Sort-constrained variable (X:sort) vs ground/non-variable term.
                # Enforce the sort constraint: v's type must be a sub-sort of u's sort.
                if not self._unify_types(u, v):
                    return False
                self.bind(u, v)
                self._wakeup_resid(u, v)
                # Fire global delay rules: sort-constrained variable just got bound.
                # e.g. :: I:int | write(I," "). fires when I:int is unified with an integer.
                if WL.delay_rules and self.engine is not None:
                    _v_d = v.deref()
                    if _v_d.type is not None and _v_d.type is not WL.top:
                        self._fire_delay_rules(u, u.type)
            else:
                # If v is a disjunction psi-term, expand into UNIFY choice points.
                # These choice points are created AFTER the cut_barrier (during head
                # unification), so '!' inside the clause body will correctly cut them.
                # Example: q(A:{1;2;3}) :- !, write(A). with q(X)?
                #   → unify(X, {1;2;3}_copy) here; expand to X=1 (default) + CPs for 2,3
                #
                # IMPORTANT: choice points and initial binding are placed on v (the
                # disjunction object itself), NOT on u.  The clause body references the
                # SAME psi-term as the head attribute, so binding v → elem[0] makes all
                # body references deref to elem[0].  u is then bound to v so that the
                # top-level query variable also dereferences correctly.
                if not v_is_var and v.type is WL.disjunction and self.engine is not None:
                    from wild_life.built_ins import _collect_disjunction as _cdisj
                    _elems = _cdisj(v, self.engine)
                    if not _elems:
                        return False
                    for _alt in reversed(_elems[1:]):
                        # Push UNIFY choice point on v (disjunction) so backtracking
                        # re-binds v → next alternative (body refs also update).
                        self.engine.push_choice_point(GoalType.UNIFY, v, _alt, None)
                    # Bind v (the disjunction) to the first element
                    self.bind(v, _elems[0])
                    # Bind u (X) to v so u dereferences through v to elem[0]
                    self.bind(u, v)
                    self._wakeup_resid(u, v)
                    return True
                self.bind(u, v)
                self._wakeup_resid(u, v)
                # Sort narrowing: when a plain variable is bound to a term with
                # attributes, check if the term's sort can be narrowed based on
                # :: Sort(attrs) prototype declarations (e.g. @(nose=>pretty) → cleopatra).
                _v_canon = v.deref()
                # Fire global delay rules for the sort of the term being bound to.
                # e.g. :: C:cons | write(C.1), nl. fires when a plain var is bound to a cons.
                if WL.delay_rules and self.engine is not None and _v_canon.type is not None and _v_canon.type is not WL.top:
                    # Fire sub-terms first (bottom-up / post-order, matching C Wild Life behaviour).
                    self._fire_delay_rules_for_subterms(_v_canon)
                    if not getattr(_v_canon, '_delay_fired', False):
                        _v_canon._delay_fired = True
                        self._fire_delay_rules(_v_canon, _v_canon.type)
                if _v_canon.attr_list and _v_canon.type is not None and self.engine is not None:
                    self._try_sort_narrowing(_v_canon)
            return True

        if v_is_var:
            # Eagerly evaluate pure arithmetic expressions to prevent deeply-nested
            # expression chains in recursive predicates like loop(N-1).
            # Only apply when u is a compound arithmetic op (not a function sort or var).
            # Skip if engine is in non-strict call context (engine.no_arith_eval=True).
            from wild_life.data_structures import SORT_VAR as _SORT_VAR_FLAG
            v_is_sort_var = bool(v.flags & _SORT_VAR_FLAG) and v.type is not WL.top
            _skip_arith = getattr(self.engine, 'no_arith_eval', False) if self.engine else False
            if self.engine is not None and not u_is_var and not _skip_arith:
                _arith_ops = frozenset(('+', '-', '*', '/', '//', 'mod', '**', '^',
                                        'max', 'min', '/\\', '\\/', 'xor', '>>', '<<'))
                _sym = u.type.keyword.symbol if u.type and u.type.keyword else ''
                if _sym in _arith_ops:
                    try:
                        from wild_life.built_ins import _eval_arith as _ea, _make_number as _mn
                        _ok, _val = _ea(u, self.engine)
                        if _ok:
                            _u_num = _mn(self.engine, _val)
                            self.bind(v, _u_num)
                            self._wakeup_resid(v, _u_num)
                            # Memoize the evaluated result back into the compound term so
                            # that any query variable pointing to this compound (e.g. X → 1+2)
                            # also dereferences to the evaluated number (X → 1+2 → 3).
                            # This makes the binding display show X=3 instead of X=1+2 after
                            # a strict predicate evaluates the argument at call time.
                            # The coref update is trailed so backtracking correctly undoes it.
                            if u.coref is None and u.value is None and u.attr_list:
                                self.trail.trail_psi(u, 'coref')
                                u.coref = _u_num
                            return True
                    except Exception:
                        pass
            # Non-strict context: mark the arithmetic term so display doesn't evaluate it
            if _skip_arith and u.type and u.type.keyword:
                _sym2 = u.type.keyword.symbol
                _arith_ops2 = frozenset(('+', '-', '*', '/', '//', 'mod', '**', '^',
                                         'max', 'min', '/\\', '\\/', 'xor', '>>', '<<'))
                if _sym2 in _arith_ops2:
                    from wild_life.data_structures import NON_STRICT_TERM as _NST
                    self.trail.trail_psi(u, 'flags')
                    u.flags |= _NST
            # Sort-constrained variable (v) vs ground/non-variable (u):
            # enforce the sort constraint — u's type must be a sub-sort of v's sort.
            if v_is_sort_var:
                if not self._unify_types(v, u):
                    return False
            # If both u and v are disjunctions, compute cross-product intersection:
            # {u1;u2;...} vs {v1;v2;...} — try each pair (ui, vj) and collect
            # the successfully unified ui elements, then push choice points.
            if (u.type is WL.disjunction and v.type is WL.disjunction and
                    self.engine is not None):
                from wild_life.built_ins import _collect_disjunction as _cdisj2
                _u_elems = _cdisj2(u, self.engine)
                _v_elems = _cdisj2(v, self.engine)
                _successful = []
                for _ue in _u_elems:
                    _ue_d = _ue.deref()
                    for _ve in _v_elems:
                        _ve_d = _ve.deref()
                        _mark = self.trail.mark()
                        if self._unify_impl(_ue_d, _ve_d):
                            # Record the v-element (concrete, unchanged by undo)
                            _successful.append(_ve_d)
                        self.trail.undo_to(_mark)
                if not _successful:
                    return False
                # Push choice points for alternatives (reverse so first ends on top)
                for _alt in reversed(_successful[1:]):
                    self.engine.push_choice_point(GoalType.UNIFY, u, _alt, None)
                # Bind u (the disjunction cell) to the first successful element,
                # and bind v to u so that deref(v) → u → element
                self.trail.trail_psi(u, 'coref')
                u.coref = _successful[0]
                self.trail.trail_psi(v, 'coref')
                v.coref = u
                self._wakeup_resid(u, u)
                return True

            # If u is a disjunction psi-term, expand into UNIFY choice points.
            # Same logic as in the u_is_var+else branch above: bind u (the disjunction)
            # to elem[0] and bind v to u, so body refs deref correctly.
            if u.type is WL.disjunction and self.engine is not None:
                from wild_life.built_ins import _collect_disjunction as _cdisj
                _elems = _cdisj(u, self.engine)
                if not _elems:
                    return False
                for _alt in reversed(_elems[1:]):
                    self.engine.push_choice_point(GoalType.UNIFY, u, _alt, None)
                self.bind(u, _elems[0])
                self.bind(v, u)
                self._wakeup_resid(v, u)
                return True
            self.bind(v, u)
            self._wakeup_resid(v, u)
            # Sort narrowing for terms with attributes
            _u_canon = u.deref()
            # Fire global delay rules for the sort of the term being bound to.
            if WL.delay_rules and self.engine is not None and _u_canon.type is not None and _u_canon.type is not WL.top:
                # Fire sub-terms first (bottom-up / post-order, matching C Wild Life behaviour).
                self._fire_delay_rules_for_subterms(_u_canon)
                if not getattr(_u_canon, '_delay_fired', False):
                    _u_canon._delay_fired = True
                    self._fire_delay_rules(_u_canon, _u_canon.type)
            if _u_canon.attr_list and _u_canon.type is not None and self.engine is not None:
                self._try_sort_narrowing(_u_canon)
            return True

        # Disjunction × Disjunction: compute cross-product semantic intersection.
        # e.g. {1;2;3} vs {real;int} inside a({1;2;3})=a({real;int}) or reversed.
        # Try each pair (concrete_i, abstract_j); collect successful concrete elements.
        # Push choice points on the "concrete" disjunction so that variable display
        # (via deref of the attr_list entry) updates correctly on each backtrack.
        if (u.type is WL.disjunction and v.type is WL.disjunction and
                self.engine is not None):
            from wild_life.built_ins import _collect_disjunction as _cdisj_cross
            _u_elems = _cdisj_cross(u, self.engine)
            _v_elems = _cdisj_cross(v, self.engine)
            # Identify the "concrete" side (elements with numeric/string values)
            # vs the "abstract" side (sort atoms like real, int).  We make the
            # concrete side the outer loop so the results appear in the natural
            # order: 1,1,2,2,3,3 for {1;2;3}×{real;int} regardless of u/v order.
            _u0 = _u_elems[0].deref() if _u_elems else None
            _v0 = _v_elems[0].deref() if _v_elems else None
            _v_is_concrete = (
                _v0 is not None and _v0.value is not None and
                (_u0 is None or _u0.value is None)
            )
            # Swap so that the concrete side is always "outer" (= _c_elems),
            # abstract side is "inner" (= _a_elems).  Track which psi-term is which.
            if _v_is_concrete:
                _c_elems, _a_elems = _v_elems, _u_elems  # v=concrete, u=abstract
                _c_disj, _a_disj = v, u
            else:
                _c_elems, _a_elems = _u_elems, _v_elems  # u=concrete (or fallback)
                _c_disj, _a_disj = u, v
            _cross_ok = []
            for _ce in _c_elems:
                _ce_d = _ce.deref()
                for _ae in _a_elems:
                    _ae_d = _ae.deref()
                    _mark_cross = self.trail.mark()
                    # Trial: unify concrete element with abstract element
                    # (order matters for _unify_types GLB — keep concrete as u)
                    if self._unify_impl(_ce_d, _ae_d):
                        # _ce_d is concrete (value ≠ None) and is unchanged by undo
                        _cross_ok.append(_ce_d)
                    self.trail.undo_to(_mark_cross)
            if not _cross_ok:
                return False
            # Push choice points on _c_disj (the concrete disjunction).
            # When they fire, unify(_c_disj, alt) binds _c_disj.coref directly,
            # so any enclosing sort-var whose attr_list contains _c_disj updates.
            for _alt_cross in reversed(_cross_ok[1:]):
                self.engine.push_choice_point(GoalType.UNIFY, _c_disj, _alt_cross, None)
            self.trail.trail_psi(_c_disj, 'coref')
            _c_disj.coref = _cross_ok[0]
            self.trail.trail_psi(_a_disj, 'coref')
            _a_disj.coref = _c_disj
            self._wakeup_resid(_c_disj, _c_disj)
            return True

        # Disjunction × concrete: a choice point from disjunction×disjunction
        # expansion fires with unify(u_disj, element).  The element was already
        # vetted during cross-product computation, so just bind u to it directly.
        if u.type is WL.disjunction and self.engine is not None:
            self.trail.trail_psi(u, 'coref')
            u.coref = v
            self._wakeup_resid(u, u)
            return True
        if v.type is WL.disjunction and self.engine is not None:
            self.trail.trail_psi(v, 'coref')
            v.coref = u
            self._wakeup_resid(v, v)
            return True

        # Arithmetic evaluation: if one term is a concrete number and the other
        # is an arithmetic expression (compound with arithmetic op), evaluate the
        # expression and retry unification.  This is needed for LIFE's automatic
        # evaluation of numeric sub-terms, e.g. loop(N-1) where N=3.
        u_is_num = (u.type is WL.integer or u.type is WL.real) and u.value is not None and not u.attr_list
        v_is_num = (v.type is WL.integer or v.type is WL.real) and v.value is not None and not v.attr_list
        if (u_is_num or v_is_num) and self.engine is not None:
            try:
                from wild_life.built_ins import _eval_arith as _ea, _make_number as _mn
                eng = self.engine
                if not u_is_num:
                    ok_u, val_u = _ea(u, eng)
                    if ok_u:
                        u2 = _mn(eng, val_u)
                        return self.unify(u2, v)
                if not v_is_num:
                    ok_v, val_v = _ea(v, eng)
                    if ok_v:
                        v2 = _mn(eng, val_v)
                        return self.unify(u, v2)
            except Exception:
                pass  # evaluation failed, proceed with structural unification

        # 型の単一化
        if not self._unify_types(u, v):
            return False

        # 値の単一化 (数値・文字列)
        if not self._unify_values(u, v):
            return False

        # 特性の単一化
        if not self._unify_attrs(u, v):
            return False

        # ソート絞り込み: :: Sort(attrs) プロトタイプに基づいて、より特定のソートに絞り込む
        # cleopatra(nose=>pretty, occupation=>queen) の prototype があり、
        # u が person(nose=>pretty) になったとき、 u を cleopatra に絞り込む
        u_canon = u.deref()
        if u_canon.type is not None and u_canon.attr_list and self.engine is not None:
            self._try_sort_narrowing(u_canon)

        # After successful structural unification, merge the two psi-terms by
        # binding v → u (via coref).  This preserves the sharing relationship
        # so that print_variables can detect when two variables refer to the
        # same canonical term and show e.g. "Y = X" instead of "Y = !".
        # Only do this for non-numeric atoms (numbers are primitive values that
        # should remain separate; ChoicePoint values in '!' terms are OK to merge).
        from wild_life.data_structures import ChoicePoint as _CP_merge, NON_STRICT_TERM as _NST_merge
        _u_prim = isinstance(u.value, (int, float, str)) if u.value is not None else False
        _v_prim = isinstance(v.value, (int, float, str)) if v.value is not None else False
        if not _u_prim and not _v_prim and v.coref is None:
            # Propagate NON_STRICT_TERM from v to u before binding: if v is a frozen
            # arithmetic term (e.g. `+(23) with NST) and u is the new canonical
            # representative, the freeze must survive on u too.
            if (v.flags & _NST_merge) and not (u.flags & _NST_merge):
                self.trail.trail_psi(u, 'flags')
                u.flags |= _NST_merge
            # Bind v → u so deref(v) returns u (the canonical psi-term).
            self.bind(v, u)

        return True

    def _unify_types(self, u: PsiTerm, v: PsiTerm) -> bool:
        """型を単一化する (GLB = infimum を採用)。
        C版の global_unify() の型処理部分に対応。

        LIFE の型単一化は GLB (Greatest Lower Bound / 最大下限) を使う。
        2つの型 du, dv を単一化した結果は「より特殊な型 (サブタイプ)」。
        共通サブタイプがなければ単一化失敗。
        """
        du = u.type
        dv = v.type

        if du is dv:
            return True  # 同じ型

        # 一方が top (@) → もう一方の型に制約
        if du is WL.top:
            self.bind_type(u, dv)
            return True
        if dv is WL.top:
            self.bind_type(v, du)
            return True

        # サブタイプ関係: より特殊な型 (GLB) を採用
        if du.is_subtype_of(dv):
            self.bind_type(v, du)   # v の型を du (より特殊) に引き上げ
            # Numeric value compatibility: if v has a concrete numeric value,
            # verify that it is compatible with the narrowed type (du).
            # e.g. narrowing real(3.3) to integer must fail because 3.3 is not
            # an integer.  Without this check, f(3.3) incorrectly matches f(int).
            if v.value is not None and isinstance(v.value, float) and WL.integer is not None and du.is_subtype_of(WL.integer):
                import math as _math_ut
                if not _math_ut.isfinite(v.value) or v.value != int(v.value):
                    return False
            return True
        if dv.is_subtype_of(du):
            self.bind_type(u, dv)   # u の型を dv (より特殊) に引き上げ
            # Numeric value compatibility: if u has a concrete numeric value,
            # verify that it is compatible with the narrowed type (dv).
            if u.value is not None and isinstance(u.value, float) and WL.integer is not None and dv.is_subtype_of(WL.integer):
                import math as _math_ut
                if not _math_ut.isfinite(u.value) or u.value != int(u.value):
                    return False
            return True

        # 直交した型 (どちらもサブタイプでない) → 互換性チェック
        # ユーザー定義の共通サブタイプがあれば GLB が存在する
        if self.engine is not None:
            glbs = compute_all_glbs(du, dv)
        else:
            _g = compute_glb(du, dv)
            glbs = [_g] if _g is not None else []

        if not glbs:
            return False            # 共通サブタイプなし → 型が非互換

        # 複数の GLB がある場合: バックトラック用チョイスポイントを積む
        # (最初の GLB で進め、残りをチョイスポイントとして積む)
        if len(glbs) > 1 and self.engine is not None:
            for alt_glb in reversed(glbs[1:]):
                alt_psi = PsiTerm(type_def=alt_glb)
                self.engine.push_choice_point(GoalType.UNIFY, u, alt_psi, None)

        # 最初の GLB で進める
        glb = glbs[0]
        self.bind_type(u, glb)
        self.bind_type(v, glb)
        return True

    def _unify_values(self, u: PsiTerm, v: PsiTerm) -> bool:
        """値 (数値・文字列) を単一化する"""
        # ChoicePoint values are cut-barrier references stored in '!' (cut)
        # psi-terms for execution semantics only.  Two cut atoms are always
        # equal regardless of their stored cut points; skip value comparison.
        from wild_life.data_structures import ChoicePoint as _CP
        u_cp = isinstance(u.value, _CP)
        v_cp = isinstance(v.value, _CP)

        # 両方が値を持つ場合は等値チェック
        if u.value is not None and v.value is not None:
            # Both are ChoicePoint references → cut atoms are structurally equal
            if u_cp and v_cp:
                return True
            if isinstance(u.value, (int, float)) and isinstance(v.value, (int, float)):
                return float(u.value) == float(v.value)
            return u.value == v.value

        # 片方だけが値を持つ場合
        if u.value is not None and v.value is None:
            # Don't propagate a ChoicePoint cut-point to v — cut atoms share
            # the same sort and that is enough for structural equality.
            if not u_cp:
                self.bind_value(v, u.value)
            return True
        if v.value is not None and u.value is None:
            if not v_cp:
                self.bind_value(u, v.value)
            return True

        return True  # 両方 None

    def _unify_attrs(self, u: PsiTerm, v: PsiTerm) -> bool:
        """特性を単一化する
        C版の global_unify_attr() に対応

        u と v の全特性について:
        - u にあって v にない特性 -> v に追加
        - v にあって u にない特性 -> u に追加
        - 両方にある特性 -> 再帰的に単一化
        """
        u_attrs = dict(u.attr_list)
        v_attrs = dict(v.attr_list)

        # Sort attribute keys so that positional (numeric) keys are processed in
        # ascending order (1, 2, 3 …) and named keys follow alphabetically.
        # Together with the deferred-wakeup mechanism in _wakeup_resid / unify,
        # this ensures that pending goals woken during compound-term unification
        # fire in left-to-right (attribute 1 before 2 before 3) execution order.
        def _attr_sort_key(k):
            try:
                return (0, int(k))    # numeric keys: ascending numeric order
            except (ValueError, TypeError):
                return (1, k)          # named keys: alphabetic, after numerics
        all_keys = sorted(set(u_attrs.keys()) | set(v_attrs.keys()), key=_attr_sort_key)

        for key in all_keys:
            u_val = u_attrs.get(key)
            v_val = v_attrs.get(key)

            if u_val is not None and v_val is not None:
                # 両方に特性がある -> 再帰的に単一化
                if not self.unify(u_val, v_val):
                    return False
                # u と v の特性を統一
                unified = u_val.deref()
                if key not in u.attr_list or u.attr_list[key] is not unified:
                    self.set_attr(u, key, unified)
                if key not in v.attr_list or v.attr_list[key] is not unified:
                    self.set_attr(v, key, unified)

            elif u_val is not None:
                # u だけに特性がある -> v に追加
                self.set_attr(v, key, u_val)

            else:
                # v だけに特性がある -> u に追加
                self.set_attr(u, key, v_val)

        return True

    def _try_sort_narrowing(self, u: PsiTerm) -> bool:
        """ソートプロトタイプに基づいて u のソートを絞り込む試み。

        WL.proto_sorts に登録されているソートの中から、u の現在のソートのサブタイプで
        かつ u の属性がプロトタイプと互換なソートを探す。
        例: :: cleopatra(nose => pretty, occupation => queen). の後、
            u が @(nose => pretty) になると u を cleopatra に絞り込む。

        Note: WL.top.children には必ずしも全ユーザー定義ソートが含まれないため、
        BFS ではなくグローバルレジストリ WL.proto_sorts を使って検索する。
        """
        sort_def = u.type
        if sort_def is None:
            return False

        # Use the global proto_sorts registry (populated by _assert_colon_colon_proto).
        # WL.top.children may not contain all user sorts (e.g. 'person' is not
        # explicitly declared as 'person <| @'), so we must search the registry.
        candidates = []
        for child in WL.proto_sorts:
            # The candidate sort must be a subsort of (or equal to) the current sort.
            # When sort_def is WL.top (@), ALL sorts are implicitly subtypes of @.
            is_sub = (sort_def is WL.top) or child.is_subtype_of(sort_def)
            if not is_sub:
                continue
            # Skip if the candidate sort is the same as (or a supertype of) the current sort
            # (we only narrow DOWN, not stay the same or go up)
            if child is sort_def:
                continue

            proto = child.prototype_attrs
            if not proto:
                continue

            # Check compatibility: for each attr in child's prototype,
            # if u has that attr, their values must match
            compatible = True
            has_evidence = False
            for key, proto_val in proto.items():
                if key in u.attr_list:
                    has_evidence = True
                    u_val = u.attr_list[key]
                    u_val_d = u_val.deref()
                    proto_val_d = proto_val.deref()
                    # Compare: types and values must agree
                    if u_val_d.type is not proto_val_d.type:
                        compatible = False
                        break
                    if (u_val_d.value is not None and proto_val_d.value is not None
                            and u_val_d.value != proto_val_d.value):
                        compatible = False
                        break
            if compatible and has_evidence:
                candidates.append(child)

        if len(candidates) != 1:
            # 0 candidates: no narrowing; >1 candidates: ambiguous, skip
            return False

        child = candidates[0]
        # Narrow u's sort to child
        self.bind_type(u, child)

        # Merge prototype attrs into u (add missing attrs from prototype)
        proto = child.prototype_attrs
        for key, proto_val in proto.items():
            if key not in u.attr_list:
                self.set_attr(u, key, proto_val.deref())

        # Fire global delay rules for the new sort
        if self.engine is not None:
            self._fire_delay_rules(u, child)

        return True

    def _fire_delay_rules(self, u: PsiTerm, new_sort) -> None:  # noqa: E501
        """グローバル遅延ルール (:: Pattern | Goal) を起動する。

        u のソートが new_sort に絞り込まれたとき、パターンのソートが
        new_sort のスーパーソートである遅延ルールを起動する。
        """
        if self.engine is None:
            return
        # Re-entrancy guard: _fire_delay_rules calls self.unify() which could
        # in turn trigger _fire_delay_rules again, causing infinite recursion.
        # Skip the recursive call — the goal is already being set up.
        if getattr(self.engine, '_in_fire_delay', False):
            return
        self.engine._in_fire_delay = True
        # Collect concrete typed literals from goal copies for deferred firing.
        # (In C Wild Life, integer/real literals in rule bodies act as sort-constrained
        # variables that get "bound" when the rule fires, triggering the int delay.)
        deferred_literal_fires: list = []
        try:
            self._fire_delay_rules_inner(u, new_sort, deferred_literal_fires)
        finally:
            self.engine._in_fire_delay = False
        # Fire deferred delays for concrete integer/real literals found in goal copies.
        for _lit_term in deferred_literal_fires:
            self._fire_delay_rules(_lit_term, _lit_term.type)

    def _collect_literal_integers(self, t: PsiTerm, result: list, visited: set) -> None:
        """Walk t recursively and collect concrete integer/real PsiTerms.

        These correspond to numeric literals in rule bodies that, in the original
        C Wild Life implementation, act as sort-constrained variables narrowed to
        their value — causing the int/real delay rule to fire.
        """
        t = t.deref()
        tid = id(t)
        if tid in visited:
            return
        visited.add(tid)
        wl = WL
        if (t.value is not None and t.type is not None
                and t.type is not wl.top
                and t.type.keyword is not None
                and t.type.keyword.symbol in ('int', 'integer', 'real', 'float', 'number')):
            result.append(t)
        # In C Wild Life, the integer label in T.F (e.g. '1' in C.1) DOES fire the
        # int delay rule — the REFOUT for manual8 shows '1 d\n1 c\n1 b\n1 a\n' where
        # '1' comes from the feature key. So collect all sub-terms including .2.
        for k, val_ref in t.attr_list.items():
            self._collect_literal_integers(val_ref, result, visited)

    def _fire_delay_rules_inner(self, u: PsiTerm, new_sort,
                                deferred_literal_fires: list = None) -> None:
        """_fire_delay_rules の実処理 (再入禁止ガード外側から呼ぶ)。"""
        wl = WL
        for rule_inner in wl.delay_rules:
            # rule_inner is the | (Pattern | Goal) psiterm
            pattern_side = rule_inner.attr_list.get('1')
            goal_side = rule_inner.attr_list.get('2')
            if pattern_side is None or goal_side is None:
                continue
            pattern_d = pattern_side.deref()
            # Pattern sort must be a supersort of (or equal to) new_sort
            pat_sort = pattern_d.type
            if pat_sort is None or pat_sort is wl.top:
                pat_sort_ok = True
            else:
                pat_sort_ok = new_sort.is_subtype_of(pat_sort)
            if not pat_sort_ok:
                continue

            # Build a copy of pattern AND goal using the SAME shared_map
            # so that variables shared between pattern and goal stay shared.
            # IMPORTANT: copy pattern_side (the variable wrapper), not pattern_d,
            # so that the id of the wrapper is in shared_map for the goal copy to
            # find when it encounters the same wrapper object.
            shared_map = {}
            pattern_copy = copy_term(pattern_side, shared_map)
            goal_copy = copy_term(goal_side, shared_map)

            # Deref to get the actual sort-constrained term (past any variable wrapper)
            pattern_d_copy = pattern_copy.deref()

            # Collect concrete integer/real literals from the goal copy BEFORE unification.
            # In C Wild Life, integer literals in rule bodies act as sort-constrained
            # variables that get "narrowed" to their value during rule instantiation,
            # triggering the int/real delay rule.  We collect them here (before unify
            # changes any bindings) to avoid collecting already-bound sort-vars.
            if deferred_literal_fires is not None:
                pre_unify_literals: list = []
                self._collect_literal_integers(goal_copy, pre_unify_literals, set())

            # Unify pattern_d_copy with u (e.g. person(best_friend=>Q) with cleopatra_pt)
            # This binds u's attrs from the pattern (adds best_friend=Q_fresh)
            unify_ok = self.unify(pattern_d_copy, u)
            if not unify_ok:
                continue

            # Prove the goal (e.g. get_along(P, Q)) by pushing it onto the goal stack.
            # The engine will process it in the next iteration, after the current
            # unification step completes.
            from wild_life.data_structures import GoalType as _GT
            from wild_life.inference import _DEFRULES as _defrules_sentinel
            goal_d_copy = goal_copy.deref()
            # Create a trail-independent concrete copy of the goal by materialising all
            # current trail bindings.  This ensures that if the caller later undoes its
            # trail (as _eval_arith does after testing a function-rule head), the pushed
            # goal still contains the concrete integer values rather than unbound sort-vars.
            goal_materialized = copy_term(goal_d_copy, {})

            # Fire integer literal delays BEFORE the goal so they appear first in output.
            # In C Wild Life the integer feature key '1' in write(C.1) fires BEFORE
            # the element value is printed (e.g. '1 d' not 'd 1').
            # We temporarily release _in_fire_delay to allow _fire_delay_rules to run.
            if deferred_literal_fires is not None and pre_unify_literals:
                self.engine._in_fire_delay = False
                try:
                    for _lit in pre_unify_literals:
                        _lit_d = _lit.deref()
                        if not getattr(_lit_d, '_delay_fired', False):
                            _lit_d._delay_fired = True
                            self._fire_delay_rules(_lit_d, _lit_d.type)
                finally:
                    self.engine._in_fire_delay = True

            # Execute the goal synchronously so delay outputs appear in
            # triggering order (FIFO) rather than LIFO stack order.
            _exec_delay_goal_sync(goal_materialized, self.engine)
            # (Do NOT extend deferred_literal_fires — literals are fired above.)

    def _fire_delay_rules_for_subterms(self, t: PsiTerm, visited: set = None) -> None:
        """Fire delay rules recursively for all typed sub-terms of t.

        When a variable is bound to a compound structure (e.g. a list [a,b,c,d]),
        delay rules should fire not just for the top-level term but also for each
        typed sub-term (e.g. each cons cell in the list).  This mirrors C Wild Life
        behaviour where binding a variable to a structure propagates delay firing
        to all matching sub-terms.
        """
        if self.engine is None or not WL.delay_rules:
            return
        if visited is None:
            visited = set()
        t = t.deref()
        tid = id(t)
        if tid in visited:
            return
        visited.add(tid)
        for val_ref in t.attr_list.values():
            sub = val_ref.deref()
            sub_type = sub.type
            if sub_type is not None and sub_type is not WL.top:
                # Recurse into sub-term first (depth-first post-order = bottom-up firing).
                # In C Wild Life, delay fires for inner terms before outer ones.
                self._fire_delay_rules_for_subterms(sub, visited)
                self._fire_delay_rules(sub, sub_type)

    def _wakeup_resid(self, var: PsiTerm, val: PsiTerm):
        """残留ゴールを覚醒させる
        変数が束縛されたときに呼ばれる
        C版の wakeup() に対応

        pending=True のゴールをエンジンのゴールスタックに再投入する。
        同一ゴールオブジェクトを複数回投入しないよう管理する。
        """
        if var.resid is None:
            return
        if self.engine is None:
            return

        # Collect unique pending goals (by object identity)
        seen_goals: set = set()
        goals_to_wake = []
        for r in var.resid:
            g = getattr(r, 'goal', None)
            if g is not None and getattr(g, 'pending', False):
                gid = id(g)
                if gid not in seen_goals:
                    seen_goals.add(gid)
                    goals_to_wake.append(g)

        # Mark pending goals as no longer pending (they will be re-evaluated)
        # and push them back onto the goal stack.
        # IMPORTANT: trail the pending flag change so that on backtrack the
        # goal becomes pending again and can be re-awakened next time.
        #
        # When called from inside nested _unify_impl (nesting depth > 0, i.e.
        # from _unify_attrs), defer the push: collect into _deferred_wakeups.
        # The top-level unify() will flush them in reverse order after the full
        # unification completes, ensuring pending prove goals fire in the same
        # attribute order (1, 2, 3) in which the compound term's slots are bound.
        for g in goals_to_wake:
            self.trail.trail_psi(g, 'pending')  # restore pending=True on backtrack
            g.pending = False
            if self._unify_nesting > 0 and self.engine is not None:
                self._deferred_wakeups.append((g.type, g.a, g.b, g.c))
            else:
                self.engine.push_goal(g.type, g.a, g.b, g.c)

    def unify_noeval(self, u: PsiTerm, v: PsiTerm) -> bool:
        """評価なしの単一化
        C版の global_unify() の noeval バージョンに対応
        """
        return self.unify(u, v)

    def occurs_check(self, var: PsiTerm, term: PsiTerm) -> bool:
        """発生チェック (occur check)
        var が term の中に現れるかどうか判定

        Prolog では通常省略されるが、無限項を防ぐために使える。
        """
        term = term.deref()
        if var is term:
            return True
        for v in term.attr_list.values():
            if self.occurs_check(var, v):
                return True
        return False


# ==================== ユーティリティ ====================

def unify_terms(u: PsiTerm, v: PsiTerm,
                trail: Optional[Trail] = None) -> Tuple[bool, Trail]:
    """2つの psi-term を単一化する (スタンドアロン版)

    Args:
        u, v: 単一化する2つの psi-term
        trail: バックトラック用トレイル (None の場合は新規作成)

    Returns:
        (success, trail)
    """
    if trail is None:
        trail = Trail()
    unifier = Unifier(trail)
    success = unifier.unify(u, v)
    return success, trail


def _exec_delay_goal_sync(goal: PsiTerm, eng) -> None:
    """Execute a delay rule goal synchronously (for FIFO ordering of delay outputs).

    Handles conjunctions, write, nl, print and similar built-in predicates
    directly so that delay outputs appear in the order they were triggered
    (i.e. each delay fires and completes before the next one starts).
    Falls back to push_goal for complex or unknown predicates.
    """
    goal = goal.deref()
    sym = goal.type.keyword.symbol if (goal.type and goal.type.keyword) else ''

    if sym == ',':  # conjunction: execute each conjunct in order
        a1 = goal.attr_list.get('1')
        a2 = goal.attr_list.get('2')
        if a1 is not None:
            _exec_delay_goal_sync(a1.deref(), eng)
        if a2 is not None:
            _exec_delay_goal_sync(a2.deref(), eng)
        return

    # Direct dispatch for common built-in predicates
    try:
        from wild_life.built_ins import (bi_write, bi_nl, bi_writeln, bi_print,
                                          bi_writeq, bi_write_canonical, bi_write_err)
        _SYNC_BUILTINS = {
            'write': bi_write, 'nl': bi_nl, 'writeln': bi_writeln,
            'print': bi_print, 'writeq': bi_writeq,
            'write_canonical': bi_write_canonical,
            'write_err': bi_write_err,
        }
        if sym in _SYNC_BUILTINS:
            _SYNC_BUILTINS[sym](goal, eng)
            return
    except ImportError:
        pass
    # Also handle format/2 and put_char/1
    try:
        from wild_life.built_ins import bi_format
        if sym == 'format':
            bi_format(goal, eng)
            return
    except ImportError:
        pass
    try:
        from wild_life.built_ins import bi_put_char
        if sym == 'put_char':
            bi_put_char(goal, eng)
            return
    except ImportError:
        pass

    # Fallback: push as a deferred goal for the engine to process
    from wild_life.data_structures import GoalType as _GT
    from wild_life.inference import _DEFRULES as _defrules_sentinel
    eng.push_goal(_GT.PROVE, goal, _defrules_sentinel, None)


def copy_term(t: PsiTerm, var_map: Optional[Dict[int, PsiTerm]] = None) -> PsiTerm:
    """psi-term をコピーする (変数を新しい変数に置き換える)
    C版の copy.c の copy_term() に対応

    Args:
        t: コピーする psi-term
        var_map: 変数マッピング (旧変数ID -> 新変数)

    Returns:
        コピーされた psi-term
    """
    if var_map is None:
        var_map = {}

    # Sort-constrained variable (X:sort — marked with SORT_VAR flag by the parser).
    # The tokenizer creates a fresh proxy token for each occurrence of X (tok.coref = stored_X),
    # so the SORT_VAR flag ends up on stored_X (the deref target), not on the proxy token.
    # We check SORT_VAR both BEFORE and AFTER deref so all occurrences of X share the
    # same copy regardless of whether they come via a proxy token or a direct reference.
    # In both cases, key the var_map by id(stored_X) so all occurrences converge.
    #
    # IMPORTANT: If the SORT_VAR has already been bound (coref is not None), we must deref
    # and copy the concrete bound value rather than creating a fresh unbound sort-var.
    # This handles goal materialization where e.g. C:cons is already bound to a cons cell.
    from wild_life.data_structures import SORT_VAR
    if t.flags & SORT_VAR:
        if t.coref is not None:
            # Already bound — deref and fall through to copy the concrete value
            t = t.deref()
        else:
            tid = id(t)
            if tid not in var_map:
                new_var = PsiTerm()
                new_var.type = t.type  # same sort constraint
                new_var.flags = t.flags
                var_map[tid] = new_var
            return var_map[tid]

    t = t.deref()

    # Post-deref SORT_VAR check: handles proxy tokens (tok.coref = stored_X)
    # where the SORT_VAR flag is on stored_X, not on tok.
    if t.flags & SORT_VAR:
        if t.coref is not None:
            # Already bound — deref and fall through to copy the concrete value
            t = t.deref()
        else:
            tid = id(t)
            if tid not in var_map:
                new_var = PsiTerm()
                new_var.type = t.type
                new_var.flags = t.flags
                var_map[tid] = new_var
            return var_map[tid]

    # 変数 (未束縛 top)
    if t.type is WL.top and not t.attr_list and not t.resid:
        tid = id(t)
        if tid not in var_map:
            new_var = PsiTerm()
            new_var.type = WL.top
            var_map[tid] = new_var
        return var_map[tid]

    # 定数・アトム
    if not t.attr_list and t.value is not None:
        result = PsiTerm()
        result.type = t.type
        result.value = t.value
        result.flags = t.flags
        result.status = t.status
        # Copy delay-tracking flags so that goal copies don't re-fire delay rules.
        # Without this, _write_term → _eval_arith on a goal copy would fire delay again
        # for each fresh copy, causing infinite recursion.
        if getattr(t, '_delay_fired', False):
            result._delay_fired = True
        if getattr(t, '_is_computed', False):
            result._is_computed = True
        return result

    # 複合項
    # Preserve structural sharing: if the same Python object appears at
    # multiple positions in a rule (e.g. an empty sort-typed term X:sort
    # acting as a shared variable, or any shared sub-structure), all
    # occurrences must map to the SAME fresh copy.  Register the result in
    # var_map *before* recursing so that circular structures are also safe.
    tid = id(t)
    if tid in var_map:
        return var_map[tid]
    result = PsiTerm()
    var_map[tid] = result  # register before recursing
    result.type = t.type
    result.value = t.value
    result.flags = t.flags
    result.status = t.status

    for key, val in t.attr_list.items():
        result.attr_list[key] = copy_term(val, var_map)

    return result


def term_to_string(t: PsiTerm, quoted: bool = False,
                   depth: int = 0, max_depth: int = 100) -> str:
    """psi-term を文字列に変換 (デバッグ用)
    C版の print.c の print_psi_term() に対応
    """
    if depth > max_depth:
        return "..."

    t = t.deref()

    # 変数
    if t.type is WL.top and not t.attr_list:
        return f"_G{id(t)}"

    # 数値
    if t.type is WL.integer and t.value is not None:
        v = t.value
        if float(v) == int(float(v)):
            return str(int(float(v)))
        return str(v)

    if t.type is WL.real and t.value is not None:
        return str(t.value)

    # 文字列
    if t.type is WL.quoted_string and t.value is not None:
        if quoted:
            return f'"{t.value}"'
        return str(t.value)

    # nil (空リスト)
    if t.type is WL.nil:
        return "[]"

    # alist (非空リスト)
    if t.type is WL.alist:
        return _list_to_str(t, quoted, depth, max_depth)

    # アトム/定数
    sym = t.type.symbol if t.type else "?"
    if not t.attr_list:
        return sym

    # 複合項
    # 引数が "1", "2", ... の場合は f(arg1, arg2) 形式で表示
    keys = sorted(t.attr_list.keys(), key=lambda k: featcmp_key(k))
    positional = all(
        k == str(i+1) for i, k in enumerate(keys)
    )

    if positional and keys:
        args = ", ".join(
            term_to_string(t.attr_list[k], quoted, depth+1, max_depth)
            for k in keys
        )
        return f"{sym}({args})"
    else:
        attrs = ", ".join(
            f"{k}=>{term_to_string(v, quoted, depth+1, max_depth)}"
            for k, v in sorted(t.attr_list.items(),
                               key=lambda x: featcmp_key(x[0]))
        )
        return f"{sym}({attrs})"


def _list_to_str(t: PsiTerm, quoted: bool, depth: int,
                 max_depth: int) -> str:
    """リストを '[a,b,c]' 形式の文字列に変換"""
    items = []
    current = t
    tail = None

    while True:
        current = current.deref()
        if current.type is WL.nil:
            break
        if current.type is not WL.alist:
            tail = current
            break
        if depth > max_depth:
            items.append("...")
            break
        head = current.attr_list.get("1")
        if head:
            items.append(term_to_string(head, quoted, depth+1, max_depth))
        rest = current.attr_list.get("2")
        if rest is None:
            break
        current = rest

    result = "[" + ", ".join(items)
    if tail is not None:
        result += "|" + term_to_string(tail, quoted, depth+1, max_depth)
    result += "]"
    return result


# featcmp_key のインポート (term_to_string で使用)
from wild_life.data_structures import featcmp_key
