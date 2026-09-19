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
    Residuation, SORT_VAR as _SORT_VAR, QUOTED_TRUE as _QUOTED_TRUE
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

class _DictSnapshot:
    """Puts a dict's contents back when the trail rewinds past it."""

    __slots__ = ('target',)

    def __init__(self, target: dict):
        self.target = target

    @property
    def restore(self):
        return dict(self.target)

    @restore.setter
    def restore(self, saved: dict):
        self.target.clear()
        self.target.update(saved)


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

    def trail_dict(self, d: dict):
        """Record a dict's contents so backtracking restores them.

        The variables `parse` adds to the query's table belong to the solution
        that made them: retrying `(p(5,"B") ; p(5,"C"))` reports C's variables
        in place of B's, not both.
        """
        self._trail.append((_DictSnapshot(d), 'restore', dict(d)))

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


def _fired_rules_of(t: PsiTerm) -> set:
    """Delay rules this psi-term has already run, by rule identity.

    A term keeps the set as it is narrowed, so `X = b1` running `:: b1 |` does
    not run it a second time when X narrows to a1 — only `:: a1 |` is owed.
    """
    fs = getattr(t, '_delay_rules_fired', None)
    if fs is None:
        fs = set()
        t._delay_rules_fired = fs
    return fs


def _share_fired_rules(u: PsiTerm, v: PsiTerm) -> set:
    """Join two psi-terms' fired-rule histories into one shared set."""
    fs = _fired_rules_of(u)
    fs |= _fired_rules_of(v)
    u._delay_rules_fired = fs
    v._delay_rules_fired = fs
    return fs


# How far a rule body's literals may go on announcing literals of their own.
# `:: I:int | I <- 3` hands each firing a fresh 3 to announce, which would
# never end; two levels is what manual8's `write(C.1)` needs.
_LITERAL_FIRE_LIMIT = 2


def defers_check(defn, _depth: int = 0) -> bool:
    """Whether a term of this sort holds its prototype and delay rules back.

    `delay_check(S)?` marks S, and the mark reaches everything under it: being
    S — or anything below S — is not yet the final word on what a term is, so
    the rules wait until the term is modified.
    """
    if defn is None or _depth > 32:
        return False
    if not getattr(defn, 'always_check', True):
        return True
    for parent in getattr(defn, 'parents', ()) or ():
        if defers_check(parent, _depth + 1):
            return True
    return False


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

    # C版 Wild Life の実装に合わせ、GLB を型の生成順 (creation_id) でソートする。
    # 型は最初に参照された宣言の順に生成されるため、宣言順に基づく安定した順序が得られる。
    maximal.sort(key=lambda d: getattr(d, 'creation_id', 0))
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

def _delay_rules_by_specificity(wl) -> list:
    """The delay rules, most specific sort first.

    A term that becomes a d owes `:: X:d`, then what the sorts above d say
    about it, and lazyinherit shows the order: D, C, B, E, A.  That is the
    hierarchy read from the top — a, e, b, c, d — taken backwards, so a sort
    comes before every sort it sits under.
    """
    _ver = getattr(wl, 'hierarchy_version', 0)
    _cached = getattr(wl, '_delay_rule_order', None)
    if (_cached is not None and _cached[0] == _ver
            and _cached[1] == len(wl.delay_rules)):
        return _cached[2]

    top = getattr(wl, 'top', None)
    # The sorts the rules speak of, and every sort above them.
    involved: dict = {}
    stack = []
    for rule in wl.delay_rules:
        _ps = rule.attr_list.get('1')
        _pt = _ps.deref().type if _ps is not None else None
        if _pt is not None and _pt is not top:
            stack.append(_pt)
    while stack:
        node = stack.pop()
        if id(node) in involved:
            continue
        involved[id(node)] = node
        for parent in (getattr(node, 'parents', None) or ()):
            if id(parent) not in involved:
                stack.append(parent)
    # Read them from the top down: the sorts nothing sits above come first,
    # in the order they were declared, and each sort follows the ones it
    # sits under.
    roots = [d for d in involved.values()
             if not [p for p in (getattr(d, 'parents', None) or ())
                     if id(p) in involved]]
    roots.sort(key=lambda d: getattr(d, 'creation_id', 0))
    order: dict = {}
    queue = list(roots)
    seen = {id(d) for d in roots}
    while queue:
        node = queue.pop(0)
        order[id(node)] = len(order)
        for child in (getattr(node, 'children', None) or ()):
            if id(child) in involved and id(child) not in seen:
                seen.add(id(child))
                queue.append(child)

    def key(item):
        _i, rule = item
        pattern_side = rule.attr_list.get('1')
        pat = pattern_side.deref().type if pattern_side is not None else None
        if pat is None or pat is top:
            return (-1, -_i)
        return (order.get(id(pat), -1), -_i)

    ranked = sorted(enumerate(wl.delay_rules), key=key, reverse=True)
    result = [r for _i, r in ranked]
    wl._delay_rule_order = (_ver, len(wl.delay_rules), result)
    return result


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
        # Cycle detection for rational-tree (cyclic) unification.
        # Stores frozensets of {id(u), id(v)} pairs currently being unified.
        # If we encounter the same pair again (via circular attrs), we return True
        # immediately (the rational-tree assumption: cyclic terms can be unified).
        self._unifying_pairs: set = set()
        # How deep the announcing of a rule body's own literals has gone.
        self._literal_fire_depth: int = 0
        # Set while asking whether a narrowing could exist at all.
        self._skip_prototypes: bool = False
        # ids of psi-terms whose conditional sort is being checked.
        self._proving_sort: set = set()
        # ids of psi-terms whose :: Sort(attrs). prototype is being applied.
        self._applying_proto: set = set()
        # Re-entrancy guard for the deferred delay_check pass (see
        # _unify_impl_inner): a cyclic term would otherwise keep handing
        # itself a fresh copy of the prototype for ever.
        self._in_deferred_check: bool = False

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

    def _settle_disjunction(self, d: PsiTerm) -> bool:
        """Bind a disjunction node to one alternative, keeping the rest.

        A prototype may promise a feature worth `{@;@}`.  The term then has
        one of them, with the others there to come back to, the same as a
        disjunction written into the program.
        """
        if self.engine is None or d is None:
            return True
        d = d.deref()
        if d.type is not WL.disjunction or not d.attr_list:
            return True
        from wild_life.built_ins import _collect_disjunction as _cdisj_p
        _elems = _cdisj_p(d, self.engine)
        if not _elems:
            return False
        for _alt in reversed(_elems[1:]):
            self.engine.push_choice_point(GoalType.BIND_DIRECT, d, _alt, None)
        self.bind(d, _elems[0])
        return True

    def apply_prototypes_deep(self, t: PsiTerm) -> None:
        """Give every sort named inside a bound term its prototype features.

        `:: titi(arg => 1).` says what a titi is, so the titi that `X =
        f(titi)` hands X carries an arg just as the one `X = titi` hands it
        does.  A rule head is left out of this: it is a pattern, and a
        prototype is a consequence of narrowing rather than a bar to it.

        The nodes are collected before any of them is given its features, so
        a prototype that names its own sort — `:: P:married_person(spouse =>
        person(spouse => P))` — does not walk into what it just added.
        """
        if t is None or self._skip_prototypes or not WL.proto_sorts:
            return
        nodes: list = []
        seen: set = set()
        queue = [t]
        while queue:
            node = queue.pop()
            if node is None:
                continue
            node = node.deref()
            if id(node) in seen:
                continue
            seen.add(id(node))
            nodes.append(node)
            queue.extend(node.attr_list.values())
        for node in nodes:
            if node.type is None or node.type is WL.top:
                continue
            proto = getattr(node.type, 'prototype_attrs', None)
            if not proto:
                continue
            # A sort under delay_check holds its prototype back while the
            # term carries no features, the same as it does when bound.
            if not node.attr_list and defers_check(node.type):
                continue
            missing = [_pk for _pk in proto if _pk not in node.attr_list]
            if not missing:
                # What the sort promises is already there.  Unifying it with
                # the prototype again would walk a prototype that names its
                # own sort round for ever.
                continue
            var_map: dict = {}
            copies = {_pk: copy_term(_pv, var_map) for _pk, _pv in proto.items()}
            for _pk in missing:
                self.set_attr(node, _pk, copies[_pk])
                self._settle_disjunction(copies[_pk])

    def _apply_prototype_attrs(self, t: PsiTerm) -> bool:
        """Constrain t's features by the `:: Sort(attrs).` prototype of its sort.

        Declaring `:: int_cons(int, int_list).` gives every int_cons an int
        head and an int_list tail, so narrowing one list cell to int_cons
        narrows the rest of the spine along with it.  Features the prototype
        names but t lacks are added; the ones it already has are unified with
        the prototype, which is what fails an incompatible narrowing.

        Only a term that already carries features takes a prototype this way.
        A bare one is just a sort being named — `Q:person` meeting `julius`
        leaves Q as julius, not as julius(last_name => caesar).
        """
        if t.type is None or not t.attr_list:
            return True
        if self._skip_prototypes:
            # Asked only whether some narrowing could make a rule fit, and a
            # prototype is a consequence of narrowing, not a bar to it: a call
            # of `i(X:t1(l => t3))` against `i(t2)` waits on X rather than
            # ruling the rule out over `:: t2(l => t4)`.
            return True
        # A prototype is inherited: `:: a(x=>c).` with `b <| a` gives every b
        # an x as well, so the sorts above t's own are collected too.
        protos = []
        seen_sorts: set = set()
        queue = [t.type]
        while queue:
            sort_def = queue.pop(0)
            if sort_def is None or id(sort_def) in seen_sorts:
                continue
            seen_sorts.add(id(sort_def))
            sort_proto = getattr(sort_def, 'prototype_attrs', None)
            if sort_proto:
                protos.append(sort_proto)
            queue.extend(getattr(sort_def, 'parents', ()) or ())
        if not protos:
            return True
        # A cyclic term would otherwise re-enter through the recursive unify
        # below; the guard is per-unification, so a later retry still applies.
        if id(t) in self._applying_proto:
            return True
        self._applying_proto.add(id(t))
        try:
            for proto in protos:
                # One shared var_map per declaration, so variables the
                # prototype shares across features stay shared in the copies.
                var_map: dict = {}
                _fresh_arith = []
                for key, proto_val in proto.items():
                    copy = copy_term(proto_val, var_map)
                    existing = t.attr_list.get(key)
                    if existing is not None:
                        # A sum is read as the equation it states, not matched
                        # against the feature shape for shape: `today => A + Y`
                        # meeting 1992 says what A and Y come to between them.
                        if (self.engine is not None
                                and self._is_open_arith_proto(copy)):
                            if not self._proto_arith_eq(existing, copy):
                                return False
                            continue
                        if not self.unify(existing, copy):
                            return False
                    # Point the feature at the prototype's own node, so features
                    # the prototype shares stay shared on the term: every feature
                    # of `:: square(side => S, length => S, width => S)` is one
                    # node even where the term already carried equal values.
                    self.set_attr(t, key, copy)
                    if existing is None:
                        _fresh_arith.append(key)
                # A prototype feature written as a sum — `:: person(age => A,
                # yob => Y, today => A + Y)` — is what a person's today comes
                # to, not an expression the term carries around.  The term is
                # given a number waiting to be worked out, and the equation
                # waits on the features that would settle it.
                for key in _fresh_arith:
                    if not self._constrain_proto_arith(t, key):
                        return False
            t._proto_applied = True
            return True
        finally:
            self._applying_proto.discard(id(t))

    def _is_open_arith_proto(self, cell: PsiTerm) -> bool:
        """Whether a prototype feature is an arithmetic expression."""
        from wild_life.built_ins import _ARITH_OPS_SET as _AOS_pa
        from wild_life.data_structures import NON_STRICT_TERM as _NST_pa
        cell = cell.deref()
        sym = cell.type.keyword.symbol if (cell.type and cell.type.keyword) else ''
        return (sym in _AOS_pa and cell.value is None and bool(cell.attr_list)
                and not (cell.flags & _NST_pa))

    def _proto_arith_eq(self, target: PsiTerm, expr: PsiTerm) -> bool:
        """State a prototype's arithmetic feature as an equation.

        `=` is what knows how to read a sum: it works the sum out when the
        features it names are known and waits on them when they are not,
        which is what makes a person's today a number rather than a term.
        """
        from wild_life.built_ins import bi_unify as _bu_pa
        eq_defn = (getattr(WL, 'eqsym', None)
                   or WL.syntax_module.symbol_table.get('='))
        if eq_defn is None:
            return True
        eq = PsiTerm(type_def=eq_defn)
        eq.attr_list = {'1': target, '2': expr}
        return bool(_bu_pa(eq, self.engine))

    def _constrain_proto_arith(self, t: PsiTerm, key: str) -> bool:
        """Turn an arithmetic prototype feature into the equation it states."""
        if self.engine is None:
            return True
        cell = t.attr_list.get(key)
        if cell is None or not self._is_open_arith_proto(cell):
            return True
        cell = cell.deref()
        var = PsiTerm(type_def=WL.top)
        self.set_attr(t, key, var)
        return self._proto_arith_eq(var, cell)

    def _prove_sort_condition(self, t: PsiTerm) -> bool:
        """Prove the membership condition a conditional sort carries.

        `zero := I | I = 0.` makes every zero satisfy `I = 0`, so narrowing a
        term to zero has to prove that of the term — and keep what the proof
        binds, which is how `A = zero` answers `A = 0: zero`, and how
        `C = -5` refuses to be positive.
        """
        from wild_life.data_structures import DefType as _DT_sc
        defn = t.type
        if (self.engine is None or defn is None or defn.type is not _DT_sc.TYPE
                or not defn.rule):
            return True
        # The proof narrows sorts of its own, including the pattern it matches
        # against; without a guard it would ask the same question again on the
        # way, without end.
        if self._proving_sort:
            return True
        self._proving_sort.add(id(t))
        try:
            from wild_life.data_structures import GoalType as _GT_sc
            from wild_life.inference import _DEFRULES as _DR_sc, _INNER_RUN_BARRIER as _IRB_sc
            eng = self.engine
            for pat, cond in defn.rule:
                if pat is None or cond is None:
                    continue
                var_map: dict = {}
                pat_copy = copy_term(pat, var_map)
                cond_copy = copy_term(cond, var_map)
                mark = self.trail.mark()
                if not self.unify(t, pat_copy):
                    self.trail.undo_to(mark)
                    return False
                # A feature the pattern states as a meet holds the meet:
                # soap_opera's `wife => W:alcoholic & long_lost_sister(H)`
                # answers jane, not the conjunction that produced her.
                if not self._reduce_conjunctions(t):
                    self.trail.undo_to(mark)
                    return False
                cp_save, gs_save = eng.choice_stack, eng.goal_stack
                eng.goal_stack = None
                eng.push_goal(_GT_sc.PROVE, cond_copy.deref(), _DR_sc, None)
                old_ok = eng.main_loop_ok
                ok = eng.run(cs_barrier=cp_save if cp_save is not None else _IRB_sc)
                eng.main_loop_ok = old_ok
                eng.choice_stack, eng.goal_stack = cp_save, gs_save
                if not ok:
                    self.trail.undo_to(mark)
                    return False
            return True
        finally:
            self._proving_sort.clear()

    def _reduce_conjunctions(self, t: PsiTerm) -> bool:
        """Replace `A & B` features of t by the sort they meet at."""
        if self.engine is None or WL.and_sym is None:
            return True
        from wild_life.built_ins import _eval_and_conjunction as _eac_rc
        seen: set = set()
        stack = [t]
        while stack:
            node = stack.pop()
            node = node.deref()
            if id(node) in seen:
                continue
            seen.add(id(node))
            for key, ref in list(node.attr_list.items()):
                sub = ref.deref()
                if (sub.type is WL.and_sym and '1' in sub.attr_list
                        and '2' in sub.attr_list):
                    met = _eac_rc(sub, self.engine)
                    if met is None:
                        return False
                    met = met.deref()
                    # `W:alcoholic & long_lost_sister(H)` is still W, now
                    # narrowed, so the feature keeps pointing at W and stays
                    # the same node as the W in the characters list.
                    lhs = sub.attr_list['1'].deref()
                    if not self.unify(lhs, met):
                        return False
                    lhs = lhs.deref()
                    if lhs is not sub:
                        if sub.coref is None:
                            self.bind(sub, lhs)
                        else:
                            self.set_attr(node, key, lhs)
                    sub = lhs
                stack.append(sub)
        return True

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

        # Cycle detection for rational-tree (cyclic) unification.
        # If we are already in the process of unifying this exact pair of
        # canonical psi-terms (via a circular attr chain), assume they can be
        # unified and return True immediately to break the cycle.
        # Only a term with features can come round to itself, so a pair
        # without any needs no guarding against it.
        if not u.attr_list and not v.attr_list:
            return self._unify_impl_inner(u, v)
        _iu = id(u)
        _iv = id(v)
        _pair_key = (_iu, _iv) if _iu < _iv else (_iv, _iu)
        if _pair_key in self._unifying_pairs:
            return True
        self._unifying_pairs.add(_pair_key)
        try:
            return self._unify_impl_inner(u, v)
        finally:
            self._unifying_pairs.discard(_pair_key)

    def _unify_impl_inner(self, u: PsiTerm, v: PsiTerm) -> bool:
        """Actual unification body (called by _unify_impl after cycle detection)."""
        # Dot-access expression resolution: when either term is a dot-projection
        # (type.keyword.symbol == '.' AND has '1'/'2' args), resolve it to the
        # actual feature cell before unification.  This handles cases like
        # s(A.B, A.C) = s(A.D, A.E) where the dot-terms appear as sub-terms.
        # NOTE: a plain '.' atom (no args) must NOT be treated as a dot-access —
        # it is a legitimate operator name and must unify freely with variables.
        _dot_check = _is_dot_access
        if _is_dot_access(u) or _is_dot_access(v):
            if self.engine is not None:
                from wild_life.built_ins import _resolve_dot_feat as _rdf
                if _dot_check(u):
                    _cell_u = _rdf(u, self.engine)
                    if _cell_u is None:
                        return False
                    u = _cell_u.deref()
                if _dot_check(v):
                    _cell_v = _rdf(v, self.engine)
                    if _cell_v is None:
                        return False
                    v = _cell_v.deref()
                if u is v:
                    return True
            else:
                # No engine: cannot set up residuations; treat as compound unification
                pass

        # Sort conjunction: a stored clause head can carry an unevaluated `&`
        # term — asserting `b(A & int,S).` with A bound to real keeps
        # `real & int` in the database — and unifying against it has to use the
        # meet of the two sides, so that b(C,D) answers C = int.
        _and_sym = WL.and_sym
        if (self.engine is not None and _and_sym is not None
                and (u.type is _and_sym or v.type is _and_sym)):
            from wild_life.data_structures import QUOTED_TRUE as _QT_CJ, \
                NON_STRICT_TERM as _NST_CJ
            from wild_life.built_ins import _eval_and_conjunction as _eac

            def _meet_conj(x):
                if (x.type is WL.and_sym and '1' in x.attr_list
                        and '2' in x.attr_list
                        and not (x.flags & (_QT_CJ | _NST_CJ))):
                    m = _eac(x, self.engine)
                    if m is not None:
                        m = m.deref()
                        # The conjunction is the meet, so what points at it
                        # points at the meet: soap_opera's wife answers _B,
                        # not `_B & long_lost_sister(_A)`.
                        if m is not x and x.coref is None:
                            self.bind(x, m)
                        return m
                return x

            u2, v2 = _meet_conj(u), _meet_conj(v)
            if u2 is not u or v2 is not v:
                u, v = u2, v2
                if u is v:
                    return True

        # 変数の処理
        u_is_var = (u.type is WL.top and not u.attr_list and not u.resid)
        v_is_var = (v.type is WL.top and not v.attr_list and not v.resid)

        # Sort-constrained variables (X:sort — marked SORT_VAR by the parser, or
        # X:ran where ran is a FUNCTION sort) are treated as bindable variables.
        _DefType_fn = DefType
        if not u_is_var and not v_is_var:
            # SORT_VAR flag: set by parser for any X:sort syntax.  A term
            # that has since been given features is no longer a variable:
            # binding it away would throw those features out, which is how
            # `q(X), X.1 = 1, X.2 = 2` lost its 1 and 2 to the head's `@`s.
            if u.flags & _SORT_VAR and not u.attr_list:
                u_is_var = True
            elif (u.value is None and not u.attr_list and not u.resid and
                    not (u.flags & _QUOTED_TRUE) and
                    u.type is not None and u.type.type == DefType.FUNCTION and
                    u.type._builtin_func is None):
                u_is_var = True
            if v.flags & _SORT_VAR and not v.attr_list:
                v_is_var = True
            elif (v.value is None and not v.attr_list and not v.resid and
                    not (v.flags & _QUOTED_TRUE) and
                    v.type is not None and v.type.type == DefType.FUNCTION and
                    v.type._builtin_func is None):
                v_is_var = True

        if u_is_var:
            # If u is a sort-constrained variable (type != WL.top) and v is a plain
            # top variable, bind v→u so that dereferencing v returns u (which
            # retains its sort constraint).  For FUNCTION sorts this preserves sort
            # information for _is_user_function checks; for regular SORT sorts it
            # ensures the sort constraint is visible after binding.
            _SORT_VAR_FLAG = _SORT_VAR
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
                # Two variables becoming one variable are waiting for
                # everything either of them waited for: `A = B + C, D = E + F,
                # A = D` leaves A waiting on two sums, and shows two tildes.
                self._carry_resids(v, u)
                self._wakeup_resid(u, v)
            elif (not v_is_var and u.value is None and not u.attr_list
                    and u.type is not None and u.type is not WL.top
                    and u.type.type == _DefType_fn.FUNCTION
                    and u.type._builtin_func is None and u.type.rule):
                # A call standing where its value belongs — `X:ran` meeting the
                # number ran comes to.  A function symbol is not a sort, so
                # there is nothing to check: the call becomes its value.
                self.bind(u, v)
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
                if (not v_is_var and v.type is WL.disjunction
                        and v.attr_list  # bare disj type markers have no elements
                        and self.engine is not None):
                    from wild_life.built_ins import _collect_disjunction as _cdisj
                    _elems = _cdisj(v, self.engine)
                    if not _elems:
                        return False
                    # Bind u (X) to v FIRST so that choice-point trail marks are
                    # saved AFTER X.coref=v is set.  Backtracking then preserves
                    # X→v while undoing only v.coref (the inner binding).
                    self.bind(u, v)
                    # Now push BIND_DIRECT choice points (trail mark AFTER u→v).
                    for _alt in reversed(_elems[1:]):
                        self.engine.push_choice_point(GoalType.BIND_DIRECT, v, _alt, None)
                    # Bind v (the disjunction node) to the first element.
                    self.bind(v, _elems[0])
                    self._wakeup_resid(u, v)
                    # Fire sort delay rules for the first element (same as
                    # BIND_DIRECT does for subsequent elements). This ensures
                    # :: SortName | goal fires even for the first alternative.
                    _elem0_d = _elems[0].deref() if _elems else None
                    if (_elem0_d is not None and WL.delay_rules and self.engine is not None
                            and _elem0_d.type is not None and _elem0_d.type is not WL.top
                            and not getattr(_elem0_d, '_delay_fired', False)):
                        _elem0_d._delay_fired = True
                        self._fire_delay_rules(_elem0_d, _elem0_d.type)
                    return True
                # Fix A: Empty disjunction (disj_nil = bottom type) cannot
                # be unified with any variable — unify with {} must fail.
                if v.type is WL.disj_nil:
                    return False
                self.bind(u, v)
                # Fix B: SORT_VAR daemon transfer.
                # When a SORT_VAR variable u has daemon residuations (from
                # such_that `val | cond`), and v is itself a free variable
                # (plain atom with no attrs yet), transfer the daemon resids to v
                # instead of firing them immediately. The daemon fires when v
                # is later unified with the sort-constraint term (Fix C).
                from wild_life.data_structures import SORT_VAR as _SV_B
                _u_has_daemon_b = (
                    bool(u.flags & _SV_B) and u.resid and
                    any(getattr(r, 'daemon', False) for r in u.resid)
                )
                if _u_has_daemon_b:
                    _v_deref_b = v.deref()
                    _daemon_resids_b = [r for r in u.resid if getattr(r, 'daemon', False)]
                    _other_resids_b = [r for r in u.resid if not getattr(r, 'daemon', False)]
                    # Transfer daemon resids to v (the newly bound target)
                    if _v_deref_b.resid is None:
                        self.trail.trail_psi(_v_deref_b, 'resid')
                        _v_deref_b.resid = list(_daemon_resids_b)
                    else:
                        self.trail.trail_copy(_v_deref_b, 'resid')
                        _v_deref_b.resid = list(_v_deref_b.resid) + list(_daemon_resids_b)
                    # Fire non-daemon resids immediately
                    if _other_resids_b:
                        self._wakeup_resid(u, v)
                else:
                    self._wakeup_resid(u, v)
                # Sort narrowing: when a plain variable is bound to a term with
                # attributes, check if the term's sort can be narrowed based on
                # :: Sort(attrs) prototype declarations (e.g. @(nose=>pretty) → cleopatra).
                _v_canon = v.deref()
                # Apply prototype attrs: if the bound term's sort has prototype_attrs
                # (declared with :: Sort(attrs).), merge them into the term.
                # e.g. module "a" has :: p(aha=>1). → A=p gives A = p(aha => 1).
                # A sort under delay_check(S) holds its prototype back while
                # the term carries no features: `A = a` with `:: a(x=>c).` and
                # `delay_check(a)?` answers a, not a(x => c).  The prototype
                # goes on once the term is modified (see _apply_deferred_check).
                _v_defers = (_v_canon.attr_list == {}
                             and defers_check(_v_canon.type))
                if (not _v_defers and not self._skip_prototypes
                        and _v_canon.type is not None
                        and _v_canon.type is not WL.top
                        and getattr(_v_canon.type, 'prototype_attrs', None)):
                    _proto = _v_canon.type.prototype_attrs
                    # Create fresh copies of ALL prototype attrs using a single
                    # shared var_map so variables shared across attrs (e.g. L in
                    # both length=>L and area=>L*S) remain consistently shared.
                    _var_map: dict = {}
                    _proto_copies = {k: copy_term(pv, _var_map)
                                     for k, pv in _proto.items()}
                    _proto_fresh: list = []
                    for _pk, _pc in _proto_copies.items():
                        if _pk in _v_canon.attr_list:
                            # Unify existing attr value with prototype copy to
                            # propagate constraints (e.g. width=4 → S=4 → L*4=16 → L=4)
                            _existing_ref = _v_canon.attr_list[_pk]
                            if (self.engine is not None
                                    and self._is_open_arith_proto(_pc)):
                                self._proto_arith_eq(_existing_ref, _pc)
                            else:
                                self.unify(_existing_ref, _pc)
                        else:
                            # Add missing attr from fresh prototype copy
                            self.set_attr(_v_canon, _pk, _pc)
                            self._settle_disjunction(_pc)
                            _proto_fresh.append(_pk)
                    # A prototype feature written as a sum — `:: person(age =>
                    # A, yob => Y, today => A + Y)` — is what a person's today
                    # comes to, not an expression the term carries around.
                    for _pk in _proto_fresh:
                        if not self._constrain_proto_arith(_v_canon, _pk):
                            return False
                    _v_canon._proto_applied = True
                # The sorts named further down the bound term get their
                # prototypes too: `X = f(titi)` hands X a titi with its arg on
                # it, the same as `X = titi` does.
                if _v_canon.attr_list:
                    for _sub_proto in list(_v_canon.attr_list.values()):
                        self.apply_prototypes_deep(_sub_proto)
                # Fire global delay rules for the sort of the term being bound to.
                # e.g. :: C:cons | write(C.1), nl. fires when a plain var is bound to a cons.
                if (WL.delay_rules and self.engine is not None and not _v_defers
                        and _v_canon.type is not None and _v_canon.type is not WL.top):
                    # Fire sub-terms first (bottom-up / post-order, matching C Wild Life behaviour).
                    self._fire_delay_rules_for_subterms(_v_canon)
                    if not getattr(_v_canon, '_delay_fired', False):
                        _v_canon._delay_fired = True
                        self._fire_delay_rules(_v_canon, _v_canon.type)
                if _v_canon.attr_list and _v_canon.type is not None and self.engine is not None:
                    self._try_sort_narrowing(_v_canon)
                # Binding a variable to a conditional sort has to satisfy that
                # sort's condition, the same as narrowing an existing term to it.
                if not self._prove_sort_condition(_v_canon):
                    return False
            return True

        if v_is_var:
            # Eagerly evaluate pure arithmetic expressions to prevent deeply-nested
            # expression chains in recursive predicates like loop(N-1).
            # Only apply when u is a compound arithmetic op (not a function sort or var).
            # Skip if engine is in non-strict call context (engine.no_arith_eval=True).
            from wild_life.data_structures import SORT_VAR as _SORT_VAR_FLAG
            v_is_sort_var = bool(v.flags & _SORT_VAR_FLAG) and v.type is not WL.top
            # A term frozen by a non-strict call keeps its shape: binding it
            # to a variable is how it reaches the rest of the clause, and
            # evaluating it here would undo the freeze.
            from wild_life.data_structures import NON_STRICT_TERM as _NST_BIND
            _skip_arith = (getattr(self.engine, 'no_arith_eval', False)
                           if self.engine else False) or bool(u.flags & _NST_BIND)
            if self.engine is not None and not u_is_var and not _skip_arith:
                _arith_ops = _ARITH_OP_SYMS
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
                if _sym2 in _ARITH_OP_SYMS:
                    from wild_life.inference import (
                        _arith_is_settled as _ais_nst)
                    if _ais_nst(u):
                        from wild_life.data_structures import (
                            NON_STRICT_TERM as _NST)
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
                # Bind v to u FIRST so choice-point marks are saved after v→u.
                self.bind(v, u)
                for _alt in reversed(_elems[1:]):
                    self.engine.push_choice_point(GoalType.BIND_DIRECT, u, _alt, None)
                self.bind(u, _elems[0])
                self._wakeup_resid(v, u)
                # Fire delay rules for the first element (mirrors BIND_DIRECT)
                _uelems0_d = _elems[0].deref() if _elems else None
                if (_uelems0_d is not None and WL.delay_rules and self.engine is not None
                        and _uelems0_d.type is not None and _uelems0_d.type is not WL.top
                        and not getattr(_uelems0_d, '_delay_fired', False)):
                    _uelems0_d._delay_fired = True
                    self._fire_delay_rules(_uelems0_d, _uelems0_d.type)
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
            # A term meeting a disjunction takes one of its alternatives, and
            # only one it fits: `pick_name(ursule)` against the head
            # `pick_name({alfred;…;gertrude})` has no alternative to take and
            # fails, where binding the disjunction to ursule outright let it
            # through and left nothing bound.
            from wild_life.built_ins import _collect_disjunction as _cdisj_v
            _elems_v = _cdisj_v(v, self.engine)
            if not _elems_v:
                self.trail.trail_psi(v, 'coref')
                v.coref = u
                self._wakeup_resid(v, v)
                return True
            _fits_v = []
            _eng_v = self.engine
            _cs_v = _eng_v.choice_stack
            _fd_v = getattr(_eng_v, '_in_fire_delay', False)
            _eng_v._in_fire_delay = True
            try:
                for _e_v in _elems_v:
                    _m_v = self.trail.mark()
                    try:
                        _ok_v = self.unify(u, _e_v)
                    except UnificationFailure:
                        _ok_v = False
                    self.trail.undo_to(_m_v)
                    _eng_v.choice_stack = _cs_v
                    if _ok_v:
                        _fits_v.append(_e_v)
            finally:
                _eng_v._in_fire_delay = _fd_v
                _eng_v.choice_stack = _cs_v
            if not _fits_v:
                return False
            for _alt_v in reversed(_fits_v[1:]):
                _eng_v.push_choice_point(GoalType.UNIFY, v, _alt_v, None)
            self.trail.trail_psi(v, 'coref')
            v.coref = _fits_v[0]
            self._wakeup_resid(v, v)
            return self.unify(u, _fits_v[0])

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
            # Arithmetic narrowing: if evaluation failed (one var unbound), try to
            # solve for the unbound variable using inverse arithmetic.
            # e.g. 16 = L * 4 → L = 16/4 = 4  (prototype attr constraint solving)
            try:
                if u_is_num and not v_is_num:
                    if self._try_arith_narrow(v, u):
                        return True
                elif v_is_num and not u_is_num:
                    if self._try_arith_narrow(u, v):
                        return True
            except Exception:
                pass

        # Two arithmetic expressions meet where a shared variable reaches both
        # slots of a clause head — `r3(X, 1+1, 2*1)` called as `r3(a,C,C)`.
        # Neither side is a number yet, so the evaluation above passed them by;
        # compare what they come to instead of their shape.
        if self.engine is not None:
            from wild_life.data_structures import NON_STRICT_TERM as _NST_AA
            from wild_life.built_ins import (_ARITH_OPS_SET as _AOS_AA,
                                             _eval_arith as _ea_aa,
                                             _make_number as _mn_aa)

            # A draw or a clock reads differently every time, so comparing
            # what two of them come to says nothing about the terms.
            _AA_EFFECTFUL = frozenset(('random', 'genint', 'cpu_time',
                                       'real_time'))

            from wild_life.built_ins import _is_user_function as _iuf_aa

            def _is_open_arith(t):
                sym = t.type.keyword.symbol if (t.type and t.type.keyword) else ''
                # A program may give an operator a meaning of its own —
                # overload writes `A:list + B:list -> append(A,B)` — and then
                # `+` is not arithmetic at all: what the two sides come to is
                # not a number, and asking for it matches the rule's head,
                # which brings two of them together again for ever.
                return (sym in _AOS_AA and sym not in _AA_EFFECTFUL
                        and t.value is None and bool(t.attr_list)
                        and not (t.flags & _NST_AA)
                        and not _iuf_aa(t))

            if _is_open_arith(u) and _is_open_arith(v):
                _ok_u_aa, _val_u_aa = _ea_aa(u, self.engine)
                _ok_v_aa, _val_v_aa = (_ea_aa(v, self.engine) if _ok_u_aa
                                       else (False, 0.0))
                if _ok_u_aa and _ok_v_aa:
                    if _val_u_aa != _val_v_aa:
                        return False
                    return self.unify(_mn_aa(self.engine, _val_u_aa),
                                      _mn_aa(self.engine, _val_v_aa))
                # Neither has a value yet — `p(X,Y,X+Y,X*Y)` called as
                # p(X,Y,Z,Z) before X and Y are known.  The equation suspends
                # on the variables holding it up and is settled once they are
                # bound, which is what lets lefun1 answer A = 2, B = 2, C = 4.
                from wild_life.built_ins import (
                    _collect_arith_vars as _cav_aa,
                    _attach_arith_resid as _aar_aa,
                )
                _vars_aa: list = []
                _seen_aa: set = set()
                _cav_aa(u, WL, _vars_aa, _seen_aa)
                _cav_aa(v, WL, _vars_aa, _seen_aa)
                if _vars_aa:
                    _eq_defn_aa = (getattr(WL, 'eqsym', None)
                                   or WL.syntax_module.symbol_table.get('='))
                    if _eq_defn_aa is not None:
                        _eq_aa = PsiTerm(type_def=_eq_defn_aa)
                        _eq_aa.attr_list = {'1': u, '2': v}
                        _eq_aa._resid_marker = True
                        _pend_aa = Goal(GoalType.PROVE, _eq_aa, None, None,
                                        pending=True)
                        for _v_aa in _vars_aa:
                            _aar_aa(_v_aa, WL, _pend_aa, self.engine)
                        return True

        # 型の単一化
        if not self._unify_types(u, v):
            return False

        # Narrowing a sort can apply a prototype, and that can merge one of
        # these two psi-terms into another node.  Re-read both before going on,
        # so the features below land on what the terms now are rather than on a
        # node nothing points at any more.
        u = u.deref()
        v = v.deref()
        if u is v:
            return True

        # 値の単一化 (数値・文字列)
        if not self._unify_values(u, v):
            return False

        # A term written with a sort that has a `:: Sort(attrs)` prototype
        # carries that prototype: `Joe = person(today => 1992)` is a person
        # with an age and a yob as much as the first `person(...)` was, and
        # says of this one too that its today is its age plus its yob.  Once
        # per term — saying it twice of the same term would state the same
        # equation twice over.
        if not self._skip_prototypes:
            for _side in (u, v):
                if (_side.attr_list and _side.type is not None
                        and _side.type is not WL.top
                        and not getattr(_side, '_proto_applied', False)
                        and getattr(_side.type, 'prototype_attrs', None)):
                    if not self._apply_prototype_attrs(_side):
                        return False
            u = u.deref()
            v = v.deref()
            if u is v:
                return True

        # 特性の単一化
        if not self._unify_attrs(u, v):
            return False

        # A term that has never run a delay rule is one the reader has just
        # built: `X = m(1)` on an X that is already an m owes another `mm`,
        # because the m(1) is its own term and `:: m | write(mm)` has not run
        # on it.  The two histories join, so nothing runs twice afterwards.
        # Asked once the two sides have been put together, so that what the
        # rule reads is the whole term and not half of it: `visualize(A:
        # activity, …)` would otherwise work out an activity's earliest start
        # from the requests the head has yet to be given.
        self._fire_fresh_sorts(u, v)

        # After successful structural unification, merge the two psi-terms by
        # binding v → u (via coref).  This preserves the sharing relationship
        # so that print_variables can detect when two variables refer to the
        # same canonical term and show e.g. "Y = X" instead of "Y = !".
        # A number is a term like any other here: `merge2([box(Id,A)|…],
        # [@|…[box(Id,B)|@]])` makes one box's number the other's, and boites
        # reads that back as `box(_A: 3,1)` in one list and `box(_A,-1)` in
        # the other.
        from wild_life.data_structures import NON_STRICT_TERM as _NST_merge
        # Bind u → v so deref(u) returns v (the canonical psi-term).
        # This matches C Wild Life's convention: the second argument (v) is preferred
        # as the canonical representative. For example, when unifying A.c (T_c) with A,
        # we bind T_c → A so A remains canonical and A.c = A shows the circular reference.
        if u.coref is None:
            # Propagate NON_STRICT_TERM from u to v before binding: if u is a frozen
            # arithmetic term (e.g. `+(23) with NST) and v is the new canonical
            # representative, the freeze must survive on v too.
            if (u.flags & _NST_merge) and not (v.flags & _NST_merge):
                self.trail.trail_psi(v, 'flags')
                v.flags |= _NST_merge
            self.bind(u, v)
            # Fix C: fire pending daemon resids after compound-compound unification.
            # When a psi-term u has daemon resids (from such_that), and u is
            # merged into v (compound-compound), wake them now so the daemon fires.
            self._wakeup_resid(u, v)
        elif (u.resid or v.resid) and self.engine is not None:
            # No merge — one of them holds a value, so it stays the term it is.
            # A goal waiting on it was waiting for the features it has just
            # gained: `X = 23` waiting to become 23(1) is woken by `X = @(1)`.
            self._wakeup_resid(u, v)

        # Sort narrowing from a :: Sort(attrs) prototype, e.g. a term that has
        # become person(nose => pretty) narrows to cleopatra.  This runs after
        # the merge above, on whichever psi-term is now the canonical one —
        # narrowing the other would be undone by the merge.
        u_canon = u.deref()
        if u_canon.type is not None and u_canon.attr_list and self.engine is not None:
            self._try_sort_narrowing(u_canon)

        # A sort under delay_check(S) held its prototype and delay rules back
        # while the term carried no features.  Modifying the term is what they
        # were waiting for, so they run now: `B = c` stays c, and `B = d(@)`
        # then answers d(@,w => 2,z => a).
        if (u_canon.type is not None and u_canon.attr_list
                and self.engine is not None and not self._in_deferred_check
                and defers_check(u_canon.type)):
            self._in_deferred_check = True
            try:
                if not self._apply_prototype_attrs(u_canon):
                    return False
                if WL.delay_rules and not getattr(u_canon, '_delay_fired', False):
                    u_canon._delay_fired = True
                    self._fire_delay_rules(u_canon, u_canon.type)
            finally:
                self._in_deferred_check = False

        return True

    def _fire_fresh_sorts(self, u: PsiTerm, v: PsiTerm) -> None:
        """Run the delay rules of a newly built term meeting an existing one."""
        if self.engine is None or not WL.delay_rules:
            return
        if getattr(self.engine, '_in_fire_delay', False):
            return
        for term, other in ((u, v), (v, u)):
            if term.type is None or term.type is WL.top:
                continue
            if term.value is not None:
                # A literal carries its own firing rules (see the deferred
                # literal pass in _fire_delay_rules_inner); a head's `0` meeting
                # a call's `4` must not announce itself before the match fails.
                continue
            if getattr(term, '_delay_rules_fired', None):
                continue
            if not getattr(other, '_delay_rules_fired', None):
                continue
            self._fire_delay_rules(term, term.type, use_fired_set=True)
        _share_fired_rules(u, v)

    def _fire_narrowed(self, term: PsiTerm, other: PsiTerm, new_sort) -> None:
        """Run the delay rules a sort narrowing has just made applicable.

        Unifying a b1 with a b2 makes the term an a1, and `:: a1 | write('A1')`
        is what that owes — `:: b1 |` and `:: b2 |` already ran on the two terms
        that went in, so the shared history of both sides is what we skip.
        """
        if self.engine is None or not WL.delay_rules:
            return
        if new_sort is None or new_sort is WL.top:
            return
        _share_fired_rules(term, other)
        self._fire_delay_rules(term, new_sort, use_fired_set=True)

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
        # Narrowing owes the sort's delay rules just as much here as it does
        # below: `X = @(nom => amedee)` becoming a typ has to run what `::
        # typ(nom => N:name) | constraint(inst(N))` says about a typ, the same
        # as a bare `X = typ` does.
        if du is WL.top:
            self.bind_type(u, dv)
            self._fire_narrowed(u, v, dv)
            return self._apply_prototype_attrs(u) and self._prove_sort_condition(u)
        if dv is WL.top:
            self.bind_type(v, du)
            self._fire_narrowed(v, u, du)
            return self._apply_prototype_attrs(v) and self._prove_sort_condition(v)

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
            self._fire_narrowed(v, u, du)
            return self._apply_prototype_attrs(v) and self._prove_sort_condition(v)
        if dv.is_subtype_of(du):
            self.bind_type(u, dv)   # u の型を dv (より特殊) に引き上げ
            # Numeric value compatibility: if u has a concrete numeric value,
            # verify that it is compatible with the narrowed type (dv).
            if u.value is not None and isinstance(u.value, float) and WL.integer is not None and dv.is_subtype_of(WL.integer):
                import math as _math_ut
                if not _math_ut.isfinite(u.value) or u.value != int(u.value):
                    return False
            self._fire_narrowed(u, v, dv)
            return self._apply_prototype_attrs(u) and self._prove_sort_condition(u)

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
                # The alternative is a stand-in for u, not a term the reader
                # built, so it inherits u's delay-rule history: coming back
                # here owes `:: a2 |`, not `:: b1 |` and `:: b2 |` again.
                alt_psi._delay_rules_fired = _fired_rules_of(u)
                # The alternative narrows u to the other greatest lower bound
                # and then redoes the whole unification, because everything
                # this call goes on to do — merging u and v among it — is
                # undone on the way back here.
                _redo = Goal(GoalType.UNIFY, u, v, None)
                _redo.next = self.engine.goal_stack
                _narrow = Goal(GoalType.UNIFY, u, alt_psi, None)
                _narrow.next = _redo
                self.engine.choice_stack = ChoicePoint(
                    undo_point=self.trail.mark(),
                    goal_stack=_narrow,
                    next=self.engine.choice_stack,
                )

        # 最初の GLB で進める
        glb = glbs[0]
        self.bind_type(u, glb)
        self.bind_type(v, glb)
        self._fire_narrowed(u, v, glb)
        return (self._apply_prototype_attrs(u) and self._apply_prototype_attrs(v)
                and self._prove_sort_condition(u) and self._prove_sort_condition(v))

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

    def _try_arith_narrow(self, expr: PsiTerm, val: PsiTerm) -> bool:
        """Arithmetic narrowing: solve for an unbound variable in `expr` given
        concrete `val`.

        Handles binary ops (+, -, *, /) where exactly one arg is an unbound
        sort-constrained variable and the other is concrete.  Uses inverse
        arithmetic to bind the unbound variable.

        For prototype attr constraints like ``:: rectangle(area => L*S)``:
          - After S=4: unify(16, L*4) → L = 16/4 = 4.
        """
        from wild_life.built_ins import _eval_arith as _ea, _make_number as _mn
        expr = expr.deref()
        val = val.deref()
        if expr.type is None or expr.type.keyword is None:
            return False
        sym = expr.type.keyword.symbol
        if sym not in ('+', '-', '*', '/'):
            return False
        a1_ref = expr.attr_list.get('1')
        a2_ref = expr.attr_list.get('2')
        if a1_ref is None or a2_ref is None:
            return False
        a1 = a1_ref.deref()
        a2 = a2_ref.deref()
        eng = self.engine
        ok1, v1 = _ea(a1, eng)
        ok2, v2 = _ea(a2, eng)
        target = val.value
        if target is None:
            return False
        # Need exactly one side evaluable, the other being an unbound variable
        if ok1 and not ok2:
            # arg2 is the unknown: solve val = arg1 <op> arg2
            if sym == '+':      result = target - v1       # v1 + arg2 = target
            elif sym == '-':    result = v1 - target       # v1 - arg2 = target → arg2 = v1-target
            elif sym == '*':
                if v1 == 0:
                    return False
                result = target / v1
            elif sym == '/':
                if target == 0:
                    return False
                result = v1 / target                       # v1 / arg2 = target → arg2 = v1/target
            else:
                return False
            result_term = _mn(eng, int(result) if result == int(result) else result)
            return self.unify(a2, result_term)
        elif ok2 and not ok1:
            # arg1 is the unknown: solve val = arg1 <op> arg2
            if sym == '+':      result = target - v2
            elif sym == '-':    result = target + v2       # arg1 - v2 = target
            elif sym == '*':
                if v2 == 0:
                    return False
                result = target / v2
            elif sym == '/':    result = target * v2       # arg1 / v2 = target
            else:
                return False
            result_term = _mn(eng, int(result) if result == int(result) else result)
            return self.unify(a1, result_term)
        return False

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
                    # A prototype feature worth nothing in particular —
                    # `:: p_c(_)` — says nothing about a term that has a
                    # first feature, so it is no reason to call that term a
                    # p_c.  Only a feature the declaration gives a sort or a
                    # value to is evidence.
                    _pv_ev = proto_val.deref()
                    if (_pv_ev.value is not None
                            or (_pv_ev.type is not None
                                and _pv_ev.type is not WL.top)):
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
        proto = {} if self._skip_prototypes else child.prototype_attrs
        for key, proto_val in proto.items():
            if key not in u.attr_list:
                self.set_attr(u, key, proto_val.deref())

        # Fire global delay rules for the new sort
        if self.engine is not None:
            self._fire_delay_rules(u, child)

        return True

    def _fire_delay_rules(self, u: PsiTerm, new_sort,
                          use_fired_set: bool = False) -> None:  # noqa: E501
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
            self._fire_delay_rules_inner(u, new_sort, deferred_literal_fires,
                                         use_fired_set)
        finally:
            self.engine._in_fire_delay = False
        # Fire deferred delays for concrete integer/real literals found in goal copies.
        self._literal_fire_depth += 1
        try:
            for _lit_term in deferred_literal_fires:
                self._fire_delay_rules(_lit_term, _lit_term.type)
        finally:
            self._literal_fire_depth -= 1

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
        # A backtick holds its term as it is written; the 4 in `` `4 `` is a
        # shape to compare against, not a number the rule has just been given.
        if t.type is not None and t.type.keyword is not None \
                and t.type.keyword.symbol == '`':
            return
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
                                deferred_literal_fires: list = None,
                                use_fired_set: bool = False) -> None:
        """_fire_delay_rules の実処理 (再入禁止ガード外側から呼ぶ)。"""
        wl = WL
        fired_set = _fired_rules_of(u)
        for rule_inner in _delay_rules_by_specificity(wl):
            if use_fired_set and id(rule_inner) in fired_set:
                # This term has already run this rule for an earlier, wider
                # sort of its own: narrowing b1 to a1 owes A1, not B1 again.
                continue
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
                # Whether a sort under delay_check/1 holds its rules back is
                # decided by the caller — a bare term of such a sort does not
                # fire them, a modified one does (see _unify_impl_inner) — so
                # there is nothing more to check on the pattern's sort here.
            if not pat_sort_ok:
                continue
            # Trailed: what a term has run is part of what backtracking takes
            # back.  `create_CP(X), X = typ, …, fail` gives X a second typ to
            # be, and that one is owed the rule as much as the first was.
            # Re-read the term's history: unifying an earlier rule's pattern
            # with u can have joined it with another term's, and adding to the
            # set captured before that would write where nothing reads.
            fired_set = _fired_rules_of(u)
            self.trail.trail_copy(u, '_delay_rules_fired')
            fired_set.add(id(rule_inner))

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
            pre_unify_literals: list = []
            if deferred_literal_fires is not None and \
                    self._literal_fire_depth < _LITERAL_FIRE_LIMIT:
                self._collect_literal_integers(goal_copy, pre_unify_literals,
                                               set())

            # TENTATIVE UNIFICATION: take a trail mark before pattern unification.
            # If the goal fails, we undo back here and the added attrs are removed.
            # This implements C Wild Life's "delay rule semantics": pattern attrs are
            # only committed when the delay goal succeeds (e.g. manual7: best_friend
            # is added only when get_along(P,Q) succeeds).
            _trial_mark = self.trail.mark()

            # Unify pattern_d_copy with u (e.g. person(best_friend=>Q) with cleopatra_pt)
            # This binds u's attrs from the pattern (adds best_friend=Q_fresh).
            unify_ok = self.unify(pattern_d_copy, u)
            if not unify_ok:
                self.trail.undo_to(_trial_mark)
                continue

            # A feature the rule's pattern brings as `{[];list}` is worth the
            # first of them, with the rest there to come back to: an activity
            # written without requests has an empty list of them, and its
            # earliest start is what latest([]) answers.  A feature the term
            # already stated is settled by the unification above and is no
            # longer a choice.
            self._settle_pattern_disjunctions(pattern_d_copy)

            # Prove the goal itself, not a copy: it shares the pattern's
            # variables, and that sharing is how the proof reaches the term.
            # `get_along(P,Q)` binding Q to julius is what gives the cleopatra
            # its best_friend; against a copy the binding would be discarded.
            from wild_life.data_structures import GoalType as _GT
            from wild_life.inference import _DEFRULES as _defrules_sentinel
            goal_materialized = goal_copy.deref()

            # Fire integer literal delays BEFORE the goal so they appear first in output.
            # In C Wild Life the integer feature key '1' in write(C.1) fires BEFORE
            # the element value is printed (e.g. '1 d' not 'd 1').
            # We temporarily release _in_fire_delay to allow _fire_delay_rules to run.
            if deferred_literal_fires is not None and pre_unify_literals:
                self.engine._in_fire_delay = False
                self._literal_fire_depth += 1
                try:
                    for _lit in pre_unify_literals:
                        _lit_d = _lit.deref()
                        if not getattr(_lit_d, '_delay_fired', False):
                            _lit_d._delay_fired = True
                            self._fire_delay_rules(_lit_d, _lit_d.type)
                finally:
                    self._literal_fire_depth -= 1
                    self.engine._in_fire_delay = True

            # Prove the goal synchronously via a nested inner run.
            # Saves and restores engine goal/choice stacks so the nested proof
            # is isolated from the outer computation.
            try:
                from wild_life.inference import _INNER_RUN_BARRIER as _IRB
            except ImportError:
                _IRB = object()
            _eng = self.engine
            _cp_save = _eng.choice_stack
            _gs_save = _eng.goal_stack
            _eng.goal_stack = None
            _eng.push_goal(_GT.PROVE, goal_materialized, _defrules_sentinel, None)
            _old_main_ok = _eng.main_loop_ok
            _barrier = _cp_save if _cp_save is not None else _IRB
            _goal_ok = _eng.run(cs_barrier=_barrier)
            _eng.main_loop_ok = _old_main_ok
            _eng.choice_stack = _cp_save
            _eng.goal_stack = _gs_save

            if not _goal_ok:
                # Goal failed: undo pattern unification (tentative semantics).
                # Remove attrs added by the pattern (e.g. best_friend => Q_fresh).
                self.trail.undo_to(_trial_mark)

    def _settle_pattern_disjunctions(self, t: PsiTerm, _seen: set = None) -> None:
        """Bind the disjunctions a delay rule's pattern left on a term."""
        if _seen is None:
            _seen = set()
        t = t.deref()
        if id(t) in _seen:
            return
        _seen.add(id(t))
        for _v in list(t.attr_list.values()):
            _vd = _v.deref()
            if _vd.type is WL.disjunction:
                self._settle_disjunction(_vd)
                _vd = _vd.deref()
            self._settle_pattern_disjunctions(_vd, _seen)

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

    def _carry_resids(self, src: PsiTerm, dst: PsiTerm) -> None:
        """Move what one variable is waiting for onto the one it becomes."""
        if not src.resid or src is dst:
            return
        if src.value is not None or src.attr_list:
            return
        _have = {id(r.goal) for r in (dst.resid or ()) if getattr(r, 'goal', None)}
        _new = [r for r in src.resid
                if getattr(r, 'goal', None) is not None and id(r.goal) not in _have]
        if not _new:
            return
        if dst.resid is None:
            self.trail.trail_psi(dst, 'resid')
            dst.resid = list(_new)
        else:
            self.trail.trail_copy(dst, 'resid')
            dst.resid = list(dst.resid) + _new

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


_ARITH_OP_SYMS = frozenset(('+', '-', '*', '/', '//', 'mod', '^',
                            'max', 'min', '/\\', '\\/', 'xor', '>>', '<<'))


def _is_dot_access(t: PsiTerm) -> bool:
    """Whether t is a `T.F` projection rather than a bare `.` atom."""
    _ty = t.type
    return (_ty is not None and _ty.keyword is not None
            and _ty.keyword.symbol == '.'
            and t.attr_list.get('1') is not None)


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
    if t.flags & _SORT_VAR:
        if t.coref is not None:
            # Already bound — deref and fall through to copy the concrete value
            t = t.deref()
        else:
            tid = id(t)
            new_var = var_map.get(tid)
            if new_var is None:
                new_var = PsiTerm()
                new_var.type = t.type  # same sort constraint
                new_var.flags = t.flags
                var_map[tid] = new_var
            return new_var

    while t.coref is not None:
        t = t.coref

    # Post-deref SORT_VAR check: handles proxy tokens (tok.coref = stored_X)
    # where the SORT_VAR flag is on stored_X, not on tok.
    if t.flags & _SORT_VAR:
        if t.coref is not None:
            # Already bound — deref and fall through to copy the concrete value
            t = t.deref()
        else:
            tid = id(t)
            new_var = var_map.get(tid)
            if new_var is None:
                new_var = PsiTerm()
                new_var.type = t.type
                new_var.flags = t.flags
                var_map[tid] = new_var
            return new_var

    _attrs = t.attr_list
    # 変数 (未束縛 top)
    if not _attrs and t.type is WL.top and not t.resid:
        tid = id(t)
        new_var = var_map.get(tid)
        if new_var is None:
            new_var = PsiTerm()
            new_var.type = WL.top
            var_map[tid] = new_var
        return new_var

    # 定数・アトム
    if not _attrs and t.value is not None:
        result = PsiTerm()
        result.type = t.type
        result.value = t.value
        result.flags = t.flags
        result.status = t.status
        # Copy delay-tracking flags so that goal copies don't re-fire delay rules.
        # Without this, _write_term → _eval_arith on a goal copy would fire delay again
        # for each fresh copy, causing infinite recursion.
        _td = t.__dict__
        if _td.get('_delay_fired', False):
            result._delay_fired = True
        if _td.get('_is_computed', False):
            result._is_computed = True
        return result

    # 複合項
    # Preserve structural sharing: if the same Python object appears at
    # multiple positions in a rule (e.g. an empty sort-typed term X:sort
    # acting as a shared variable, or any shared sub-structure), all
    # occurrences must map to the SAME fresh copy.  Register the result in
    # var_map *before* recursing so that circular structures are also safe.
    tid = id(t)
    result = var_map.get(tid)
    if result is not None:
        return result
    result = PsiTerm()
    var_map[tid] = result  # register before recursing
    result.type = t.type
    result.value = t.value
    result.flags = t.flags
    result.status = t.status

    _out = result.attr_list
    for key, val in _attrs.items():
        _out[key] = copy_term(val, var_map)

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
