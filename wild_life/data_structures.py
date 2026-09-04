"""
data_structures.py - Wild Life のコアデータ構造 (Python版)

C版の対応ファイル: extern.h, trees.h, trees.c

psi-term (ψ項): LIFE言語の基本データ単位
  - 型（ソート）を持つ特性項 (feature term)
  - 属性（フィーチャー）の辞書
  - 変数束縛のための coref ポインタ
  - 残留ゴールリスト (resid)

definition: シンボル/キーワードの定義
  - 型階層（親・子）
  - ルール（節）リスト
  - 演算子情報

C版との対応:
  struct wl_psi_term  ->  PsiTerm クラス
  struct wl_definition -> Definition クラス
  struct wl_node      -> Python の dict (属性ツリー)
  struct wl_keyword   -> Keyword クラス
  struct wl_module    -> Module クラス
"""

from __future__ import annotations
from typing import Optional, Any, List, Dict, Callable
from enum import Enum, auto
import math


# ==================== 定数 ====================

# トークンの種別定数 (extern.h より)
FACT = 100   # 事実 (ルール)
QUERY = 200  # クエリ
ERROR = 999  # エラー

# フラグ定数
QUOTED_TRUE = 1
UNFOLDED_TRUE = 2

# 演算子優先度の最大値
MAX_PRECEDENCE = 1200

# ソートの上限 (top type)
# C版では NULL が top を表すが、Python では専用クラスを使う


# ==================== 演算子型 ====================

class OperatorType(Enum):
    """演算子の結合性・前置/中置/後置の種類
    C版の typedef enum { nop, xf, fx, yf, fy, xfx, xfy, yfx } operator;
    """
    NOP = auto()  # 演算子でない
    XF  = auto()  # 後置, 左結合  (xf)
    FX  = auto()  # 前置, 右非結合 (fx)
    YF  = auto()  # 後置, 右結合  (yf)
    FY  = auto()  # 前置, 右結合  (fy)
    XFX = auto()  # 中置, 非結合  (xfx)
    XFY = auto()  # 中置, 右結合  (xfy)
    YFX = auto()  # 中置, 左結合  (yfx)


class OperatorData:
    """演算子データ
    C版の struct wl_operator_data に対応
    """
    def __init__(self, op_type: OperatorType, precedence: int,
                 next: Optional['OperatorData'] = None):
        self.type = op_type
        self.precedence = precedence
        self.next = next  # 同名で複数の演算子定義が可能

    def __repr__(self):
        return f"OpData({self.type.name}, prec={self.precedence})"


# ==================== 定義の種類 ====================

class DefType(Enum):
    """シンボル定義の種類
    C版の typedef enum { undef, predicate, function, type, global } def_type;
    """
    UNDEF     = auto()  # 未定義
    PREDICATE = auto()  # 述語
    FUNCTION  = auto()  # 関数
    TYPE      = auto()  # 型 (ソート)
    GLOBAL    = auto()  # グローバル変数


# ==================== モジュール ====================

class Module:
    """モジュール (名前空間)
    C版の struct wl_module に対応
    """
    def __init__(self, name: str, source_file: str = ""):
        self.module_name = name
        self.source_file = source_file
        self.open_modules: List[Module] = []      # 開いているモジュール
        self.inherited_modules: List[Module] = [] # 継承モジュール
        self.symbol_table: Dict[str, 'Definition'] = {}  # ハッシュテーブル

    def __repr__(self):
        return f"Module({self.module_name!r})"

    def __hash__(self):
        return hash(self.module_name)

    def __eq__(self, other):
        return isinstance(other, Module) and self.module_name == other.module_name


# ==================== キーワード ====================

class Keyword:
    """シンボル/キーワード
    C版の struct wl_keyword に対応
    """
    def __init__(self, symbol: str, module: Optional[Module] = None,
                 public: bool = True, private_feature: bool = False):
        self.symbol = symbol
        self.module = module
        self.public = public
        self.private_feature = private_feature
        self.definition: Optional['Definition'] = None

    @property
    def combined_name(self) -> str:
        """モジュール修飾名 (module#symbol)"""
        if self.module:
            return f"{self.module.module_name}#{self.symbol}"
        return self.symbol

    def __repr__(self):
        return f"Keyword({self.combined_name!r})"

    def __hash__(self):
        return hash(self.combined_name)

    def __eq__(self, other):
        return isinstance(other, Keyword) and self.combined_name == other.combined_name


# ==================== Definition (型/述語/関数の定義) ====================

class Definition:
    """シンボル定義 - 型定義、述語定義、関数定義を含む
    C版の struct wl_definition に対応

    Wild Life の型階層:
      - parents: 親ソートのリスト (より一般的な型)
      - children: 子ソートのリスト (より特殊な型)
      - top: 最も一般的な型 (すべての型の親)
    """
    # 全定義のリスト (型エンコードに使用)
    all_definitions: List['Definition'] = []

    def __init__(self, keyword: Optional[Keyword] = None):
        self.keyword = keyword
        self.date: int = 0           # 最終更新日時
        self.rule = None             # ルールリスト (PairList)
        self.properties = None       # 型プロパティ (TripleList)
        self.code = None             # 組み込み関数インデックス
        self.parents: List['Definition'] = []    # 親ソート
        self.children: List['Definition'] = []   # 子ソート
        self.type: DefType = DefType.UNDEF
        self.always_check: bool = True
        self.protected: bool = True
        self.evaluate_args: bool = True
        self.already_loaded: bool = False
        self.op_data: Optional[OperatorData] = None
        self.global_value: Optional['PsiTerm'] = None  # グローバル変数値
        self.init_value: Optional['PsiTerm'] = None    # 初期値
        self._builtin_func: Optional[Callable] = None  # 組み込み関数

        # 型エンコード (推移閉包による高速な型チェック)
        self._type_code: Optional[set] = None

        Definition.all_definitions.append(self)

    @property
    def symbol(self) -> str:
        """シンボル名"""
        if self.keyword:
            return self.keyword.symbol
        return "<anonymous>"

    def is_subtype_of(self, other: 'Definition') -> bool:
        """selfがotherのサブタイプかどうかを判定
        型階層を上に向かって探索する。
        """
        if self is other:
            return True
        # BFS/DFS で parents を辿る
        visited = set()
        stack = list(self.parents)
        while stack:
            d = stack.pop()
            if d is other:
                return True
            if d not in visited:
                visited.add(d)
                stack.extend(d.parents)
        return False

    def __repr__(self):
        return f"Def({self.symbol!r})"

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


# ==================== PsiTerm (ψ項) ====================

class PsiTerm:
    """LIFE言語の基本データ構造 - ψ項 (psi-term)
    C版の struct wl_psi_term に対応

    ψ項は以下から成る:
      - type (Definition): ソート (型)
      - value: 定数値 (int/float/str/None)
      - attr_list: 特性辞書 {特性名 -> PsiTerm}
      - coref: 変数束縛 (他のPsiTermへのポインタ)
      - resid: 残留ゴールのリスト

    例:
      整数 42:     PsiTerm(type=integer, value=42.0)
      文字列 "hi": PsiTerm(type=quoted_string, value="hi")
      変数 X:      PsiTerm(type=variable)  [coref=None initially]
      top (@):     PsiTerm(type=top)
      リスト [a,b]: PsiTerm(type=alist, attr_list={"1":..., "2":...})
    """

    _id_counter = 0  # デバッグ用ID

    def __init__(self, type_def: Optional[Definition] = None,
                 value: Any = None,
                 attr_list: Optional[Dict[str, 'PsiTerm']] = None,
                 coref: Optional['PsiTerm'] = None,
                 resid=None):
        PsiTerm._id_counter += 1
        self._id = PsiTerm._id_counter

        self.type = type_def      # Definition (ソート)
        self.value = value        # 定数値: float, str, または None
        self.attr_list: Dict[str, 'PsiTerm'] = attr_list or {}
        self.coref = coref        # 変数束縛 (union-find)
        self.resid = resid        # 残留ゴールリスト
        self.status: int = 0      # 状態フラグ
        self.flags: int = 0       # QUOTED_TRUE, UNFOLDED_TRUE など

    def deref(self) -> 'PsiTerm':
        """変数束縛を辿り、最終的な項を返す
        C版の deref_ptr マクロに対応:
          #define deref_ptr(P) while(P->coref) P=P->coref
        """
        t = self
        while t.coref is not None:
            t = t.coref
        return t

    def is_variable(self) -> bool:
        """未束縛変数かどうか。
        変数は make_var() で type=WL.top、属性なし、resid なし、coref なしとして作られる。
        WL.variable (シンボル 'variable') は別物なので間違えないこと。
        """
        from wild_life.runtime import WL  # 遅延import (循環参照回避)
        t = self.deref()
        return (t.type is WL.top and not t.attr_list
                and not t.resid and t.coref is None)

    def is_constant(self) -> bool:
        """定数かどうか (値を持つが属性を持たない)
        C版の wl_const マクロに対応:
          #define wl_const(S) ((S).value==NULL && (S).type!=variable)
        """
        from wild_life.runtime import WL
        t = self.deref()
        return t.value is None and t.type is not WL.variable

    def is_integer(self) -> bool:
        """整数かどうか"""
        from wild_life.runtime import WL
        t = self.deref()
        return t.type is WL.integer

    def is_real(self) -> bool:
        """実数かどうか"""
        from wild_life.runtime import WL
        t = self.deref()
        return t.type is WL.real

    def is_number(self) -> bool:
        """数値かどうか"""
        return self.is_integer() or self.is_real()

    def is_string(self) -> bool:
        """文字列かどうか"""
        from wild_life.runtime import WL
        t = self.deref()
        return t.type is WL.quoted_string

    def is_list(self) -> bool:
        """リスト (cons cell) かどうか"""
        from wild_life.runtime import WL
        t = self.deref()
        return t.type is WL.alist

    def is_nil(self) -> bool:
        """空リストかどうか"""
        from wild_life.runtime import WL
        t = self.deref()
        return t.type is WL.nil

    def get_number(self) -> float:
        """数値を取得"""
        t = self.deref()
        if t.value is None:
            raise ValueError(f"Not a number: {t}")
        return float(t.value)

    def get_string(self) -> str:
        """文字列を取得"""
        t = self.deref()
        if t.value is None:
            return t.type.symbol if t.type else ""
        return str(t.value)

    def get_attr(self, key: str) -> Optional['PsiTerm']:
        """属性の値を取得"""
        t = self.deref()
        return t.attr_list.get(key)

    def set_attr(self, key: str, val: 'PsiTerm'):
        """属性を設定"""
        t = self.deref()
        t.attr_list[key] = val

    def to_python_list(self) -> List['PsiTerm']:
        """LIFEのリストをPythonのリストに変換"""
        from wild_life.runtime import WL
        result = []
        t = self.deref()
        while t.type is WL.alist:
            head = t.attr_list.get("1")
            if head is not None:
                result.append(head.deref())
            tail = t.attr_list.get("2")
            if tail is None:
                break
            t = tail.deref()
        return result

    def __repr__(self):
        t = self.deref()
        if t is not self:
            return f"PsiTerm->({t!r})"
        type_name = t.type.symbol if t.type else "?"
        if t.value is not None:
            return f"PsiTerm({type_name}, {t.value!r})"
        if t.attr_list:
            attrs = ", ".join(f"{k}={v!r}" for k, v in list(t.attr_list.items())[:3])
            return f"PsiTerm({type_name}({attrs}))"
        return f"PsiTerm({type_name})"

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self is other


# ==================== ルール (節) ====================

class Rule:
    """LIFE言語のルール (節)
    C版の struct wl_pair_list に対応

    LIFE のルール形式:
      head :- body.  (Prolog風)
      f(X) -> X+1.   (関数定義)
      p(X) <- body.  (述語定義)
    """
    def __init__(self, head: Optional[PsiTerm], body: Optional[PsiTerm],
                 next: Optional['Rule'] = None):
        self.head = head   # ルールの頭部
        self.body = body   # ルールの本体
        self.next = next   # 次のルール

    def __repr__(self):
        return f"Rule({self.head!r} :- {self.body!r})"


class TypeProperty:
    """型プロパティ (三要素リスト)
    C版の struct wl_triple_list に対応
    """
    def __init__(self, attrs: Optional[PsiTerm], constraint: Optional[PsiTerm],
                 orig_type: Optional[Definition], next: Optional['TypeProperty'] = None):
        self.attrs = attrs           # 属性
        self.constraint = constraint # 制約
        self.orig_type = orig_type   # 元の型
        self.next = next


# ==================== スタック (ゴール・選択点・アンドゥ) ====================

class GoalType(Enum):
    """ゴールの種類
    C版の typedef enum { fail, prove, unify, ... } goals; に対応
    """
    FAIL        = auto()
    PROVE       = auto()
    UNIFY       = auto()
    UNIFY_NOEVAL= auto()
    DISJ        = auto()
    WHAT_NEXT   = auto()
    EVAL        = auto()
    EVAL_CUT    = auto()
    FREEZE_CUT  = auto()
    IMPLIES_CUT = auto()
    GENERAL_CUT = auto()
    MATCH       = auto()
    TYPE_DISJ   = auto()
    CLAUSE      = auto()
    DEL_CLAUSE  = auto()
    RETRACT     = auto()
    LOAD        = auto()
    C_WHAT_NEXT = auto()


class Goal:
    """ゴールスタックのエントリ
    C版の struct wl_goal に対応
    """
    def __init__(self, goal_type: GoalType, a: Optional[PsiTerm] = None,
                 b: Optional[PsiTerm] = None, c: Any = None,
                 next: Optional['Goal'] = None, pending: bool = False):
        self.type = goal_type
        self.a = a           # 第一引数
        self.b = b           # 第二引数
        self.c = c           # 第三引数
        self.next = next     # 次のゴール
        self.pending = pending

    def __repr__(self):
        return f"Goal({self.type.name}, {self.a!r}, {self.b!r})"


class ChoicePoint:
    """バックトラック用選択点
    C版の struct wl_choice_point に対応
    """
    def __init__(self, undo_point, goal_stack: Optional[Goal],
                 next: Optional['ChoicePoint'] = None):
        self.undo_point = undo_point   # アンドゥスタックの保存位置
        self.goal_stack = goal_stack   # ゴールスタックの保存
        self.next = next               # 次の選択点
        self.stack_top = None          # スタックトップの保存


class UndoEntry:
    """アンドゥスタックのエントリ (トレイル)
    C版の struct wl_stack に対応
    """
    PSI_TERM_PTR = 0
    RESID_PTR    = 1
    INT_PTR      = 2
    DEF_PTR      = 3
    GOAL_PTR     = 5

    def __init__(self, entry_type: int, obj: Any, field_name: str,
                 old_value: Any, next: Optional['UndoEntry'] = None):
        self.entry_type = entry_type
        self.obj = obj              # 変更されたオブジェクト
        self.field_name = field_name  # 変更されたフィールド名
        self.old_value = old_value  # 元の値
        self.next = next

    def undo(self):
        """変更を元に戻す"""
        setattr(self.obj, self.field_name, self.old_value)


# ==================== 残留 (Residuation) ====================

class Residuation:
    """残留ゴール - 変数が束縛されたときに覚醒するゴール
    C版の struct wl_residuation に対応
    """
    def __init__(self, goal: Optional[Goal], bestsort=None,
                 value=None, next: Optional['Residuation'] = None):
        self.sortflag: bool = True  # bestsortがDefinitionかどうか
        self.bestsort = bestsort     # 最良ソート
        self.value = value
        self.goal = goal
        self.next = next


# ==================== 特性比較関数 ====================

def featcmp_key(key: str):
    """featcmp の Python版ソートキー

    C版の featcmp 関数に対応:
      整数文字列は文字列より小さい (整数順序)
      整数文字列同士は数値順
      非整数文字列同士は辞書順

    例:
      "1" < "2" < "10" < "a" < "b"
    """
    # 整数かどうかを判定
    s = key.lstrip('-') if key.startswith('-') else key
    if s and s.isdigit():
        try:
            n = int(key)
            return (0, n, '')  # 整数は先に来る
        except ValueError:
            pass
    return (1, 0, key)  # 非整数は後に来る


def featcmp(s1: str, s2: str) -> int:
    """特性名の比較
    C版の featcmp 関数に対応
    整数特性 < 文字列特性、整数同士は数値順
    """
    k1 = featcmp_key(s1)
    k2 = featcmp_key(s2)
    if k1 < k2:
        return -1
    elif k1 > k2:
        return 1
    return 0


def sort_attrs(attr_dict: Dict[str, PsiTerm]) -> Dict[str, PsiTerm]:
    """属性辞書を featcmp 順にソート"""
    return dict(sorted(attr_dict.items(), key=lambda x: featcmp_key(x[0])))


# ==================== ユーティリティ ====================

def make_real(value: float) -> PsiTerm:
    """実数または整数のPsiTermを生成するユーティリティ"""
    from wild_life.runtime import WL
    t = PsiTerm()
    if value == math.floor(value) and abs(value) < 9007199254740991.0:
        t.type = WL.integer
    else:
        t.type = WL.real
    t.value = value
    return t


def make_string(s: str) -> PsiTerm:
    """文字列PsiTermを生成するユーティリティ"""
    from wild_life.runtime import WL
    t = PsiTerm()
    t.type = WL.quoted_string
    t.value = s
    return t


def make_atom(name: str, module: Optional[Module] = None) -> PsiTerm:
    """アトム(定数)PsiTermを生成するユーティリティ"""
    from wild_life.runtime import WL
    defn = WL.update_symbol(module, name)
    t = PsiTerm()
    t.type = defn
    return t


def make_var() -> PsiTerm:
    """新しい未束縛変数を生成するユーティリティ"""
    from wild_life.runtime import WL
    t = PsiTerm()
    t.type = WL.top  # 未束縛変数は top 型
    return t


def make_list(items: List[PsiTerm]) -> PsiTerm:
    """PythonリストからLIFEリストを生成するユーティリティ"""
    from wild_life.runtime import WL
    result = PsiTerm()
    result.type = WL.nil
    for item in reversed(items):
        cons = PsiTerm()
        cons.type = WL.alist
        cons.attr_list = {"1": item, "2": result}
        result = cons
    return result
