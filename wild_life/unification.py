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
            u_is_fn_sort = (u.type is not WL.top)
            if u_is_fn_sort and v_is_var:
                self.bind(v, u)   # v.coref = u; v.deref() = u (sort kept)
                self._wakeup_resid(u, v)
            else:
                self.bind(u, v)
                self._wakeup_resid(u, v)
            return True

        if v_is_var:
            # Eagerly evaluate pure arithmetic expressions to prevent deeply-nested
            # expression chains in recursive predicates like loop(N-1).
            # Only apply when u is a compound arithmetic op (not a function sort or var).
            # Skip if engine is in non-strict call context (engine.no_arith_eval=True).
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
            self.bind(v, u)
            self._wakeup_resid(v, u)
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

        # After successful structural unification, merge the two psi-terms by
        # binding v → u (via coref).  This preserves the sharing relationship
        # so that print_variables can detect when two variables refer to the
        # same canonical term and show e.g. "Y = X" instead of "Y = !".
        # Only do this for non-numeric atoms (numbers are primitive values that
        # should remain separate; ChoicePoint values in '!' terms are OK to merge).
        from wild_life.data_structures import ChoicePoint as _CP_merge
        _u_prim = isinstance(u.value, (int, float, str)) if u.value is not None else False
        _v_prim = isinstance(v.value, (int, float, str)) if v.value is not None else False
        if not _u_prim and not _v_prim and v.coref is None:
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
            return True
        if dv.is_subtype_of(du):
            self.bind_type(u, dv)   # u の型を dv (より特殊) に引き上げ
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

        all_keys = set(u_attrs.keys()) | set(v_attrs.keys())

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

    def _wakeup_resid(self, var: PsiTerm, val: PsiTerm):
        """残留ゴールを覚醒させる
        変数が束縛されたときに呼ばれる
        C版の wakeup() に対応
        """
        # 残留ゴールは inference.py の実行エンジンが処理する
        # ここではフラグを設定するだけ
        pass

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
    from wild_life.data_structures import SORT_VAR
    if t.flags & SORT_VAR:
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
