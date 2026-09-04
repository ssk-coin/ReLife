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
    """abort 例外"""
    pass


class HaltException(Exception):
    """halt 例外"""
    def __init__(self, code: int = 0):
        self.code = code


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

    def __init__(self, trail: Trail):
        self.trail = trail

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

        if u_is_var:
            self.bind(u, v)
            # 残留ゴールの覚醒
            self._wakeup_resid(u, v)
            return True

        if v_is_var:
            self.bind(v, u)
            self._wakeup_resid(v, u)
            return True

        # 型の単一化
        if not self._unify_types(u, v):
            return False

        # 値の単一化 (数値・文字列)
        if not self._unify_values(u, v):
            return False

        # 特性の単一化
        if not self._unify_attrs(u, v):
            return False

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
        glb = compute_glb(du, dv)
        if glb is None:
            return False            # 共通サブタイプなし → 型が非互換

        # 共通サブタイプが見つかった → 両方の型を GLB に制約
        self.bind_type(u, glb)
        self.bind_type(v, glb)
        return True

    def _unify_values(self, u: PsiTerm, v: PsiTerm) -> bool:
        """値 (数値・文字列) を単一化する"""
        # 両方が値を持つ場合は等値チェック
        if u.value is not None and v.value is not None:
            if isinstance(u.value, (int, float)) and isinstance(v.value, (int, float)):
                return float(u.value) == float(v.value)
            return u.value == v.value

        # 片方だけが値を持つ場合
        if u.value is not None and v.value is None:
            self.bind_value(v, u.value)
            return True
        if v.value is not None and u.value is None:
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

    t = t.deref()

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
    result = PsiTerm()
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
