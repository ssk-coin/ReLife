"""
runtime.py - Wild Life ランタイムのグローバル状態とシステム初期化

C版の対応ファイル: built_ins.c (init_built_in_types関数), extern.h, lib.c, modules.c

LIFE言語インタープリタのグローバルシングルトン (WL) を定義します。
組み込み型、演算子定義、モジュールシステムを初期化します。
"""

from __future__ import annotations
from typing import Optional, Dict, List, Any, Callable
import sys
import os

from wild_life.data_structures import (
    Definition, DefType, Keyword, Module, PsiTerm,
    OperatorData, OperatorType, Rule, featcmp_key
)


class WildLifeRuntime:
    """Wild Life インタープリタのグローバルランタイム状態

    C版では多数のグローバル変数として定義されていた (extern.h 参照)
    Python では1つのシングルトンクラスとして管理する。
    """

    def __init__(self):
        # ==================== モジュール ====================
        self.user_module: Optional[Module] = None
        self.current_module: Optional[Module] = None
        self.bi_module: Optional[Module] = None    # 組み込みモジュール
        self.syntax_module: Optional[Module] = None # 構文モジュール
        self.module_table: Dict[str, Module] = {}

        # ==================== 組み込み型の Definition ====================
        # C版の extern.h に宣言されているグローバル変数に対応
        self.top: Optional[Definition] = None
        self.variable: Optional[Definition] = None
        self.integer: Optional[Definition] = None
        self.real: Optional[Definition] = None
        self.quoted_string: Optional[Definition] = None
        self.nil: Optional[Definition] = None
        self.alist: Optional[Definition] = None
        self.boolean: Optional[Definition] = None
        self.true: Optional[Definition] = None
        self.false: Optional[Definition] = None
        self.succeed: Optional[Definition] = None
        self.and_sym: Optional[Definition] = None
        self.disjunction: Optional[Definition] = None
        self.disj_nil: Optional[Definition] = None
        self.comment: Optional[Definition] = None
        self.eof: Optional[Definition] = None
        self.nothing: Optional[Definition] = None
        self.constant: Optional[Definition] = None
        self.built_in: Optional[Definition] = None
        self.stream: Optional[Definition] = None
        self.functor: Optional[Definition] = None
        self.apply: Optional[Definition] = None
        self.quote: Optional[Definition] = None
        self.cut: Optional[Definition] = None
        self.iff: Optional[Definition] = None
        self.eqsym: Optional[Definition] = None
        self.leftarrowsym: Optional[Definition] = None
        self.commasym: Optional[Definition] = None
        self.colonsym: Optional[Definition] = None
        self.such_that: Optional[Definition] = None
        self.funcsym: Optional[Definition] = None
        self.predsym: Optional[Definition] = None
        self.typesym: Optional[Definition] = None
        self.opsym: Optional[Definition] = None
        self.loadsym: Optional[Definition] = None
        self.dynamicsym: Optional[Definition] = None
        self.staticsym: Optional[Definition] = None
        self.encodesym: Optional[Definition] = None
        self.listingsym: Optional[Definition] = None
        self.delay_checksym: Optional[Definition] = None
        self.eval_argsym: Optional[Definition] = None
        self.inputfilesym: Optional[Definition] = None
        self.call_handlersym: Optional[Definition] = None
        self.life_or: Optional[Definition] = None
        self.minus_symbol: Optional[Definition] = None
        self.timesym: Optional[Definition] = None
        self.tracesym: Optional[Definition] = None
        self.abortsym: Optional[Definition] = None
        self.aborthooksym: Optional[Definition] = None
        self.nullsym: Optional[Definition] = None
        self.boolpredsym: Optional[Definition] = None
        self.final_dot: Optional[Definition] = None
        self.final_question: Optional[Definition] = None
        self.xf_sym: Optional[Definition] = None
        self.fx_sym: Optional[Definition] = None
        self.yf_sym: Optional[Definition] = None
        self.fy_sym: Optional[Definition] = None
        self.xfx_sym: Optional[Definition] = None
        self.xfy_sym: Optional[Definition] = None
        self.yfx_sym: Optional[Definition] = None
        self.sys_bytedata: Optional[Definition] = None
        self.sys_bitvector: Optional[Definition] = None
        self.sys_regexp: Optional[Definition] = None
        self.sys_stream: Optional[Definition] = None
        self.sys_file_stream: Optional[Definition] = None
        self.sys_socket_stream: Optional[Definition] = None

        # ==================== 特殊属性名 ====================
        # C版の char *one, *two, *three; などに対応
        self.one = "1"
        self.two = "2"
        self.three = "3"
        self.year_attr = "year"
        self.month_attr = "month"
        self.day_attr = "day"
        self.hour_attr = "hour"
        self.minute_attr = "minute"
        self.second_attr = "second"
        self.weekday_attr = "weekday"

        # ==================== 実行状態 ====================
        self.noisy: bool = True         # プロンプト・メッセージ出力フラグ
        self.verbose: bool = False      # 詳細出力フラグ
        self.trace: bool = False        # トレースフラグ
        self.types_done: bool = False   # 型エンコード完了フラグ
        self.types_modified: bool = False
        self.interrupted: bool = False
        self.warningflag: bool = True
        self.quietflag: bool = False
        self.ignore_eff: bool = True
        self.goal_count: int = 0
        self.assert_first: bool = False
        self.assert_ok: bool = False
        self.file_date: int = 3
        self.var_occurred: bool = False
        self.parse_ok: bool = True

        # ==================== I/O ====================
        self.input_stream = sys.stdin
        self.output_stream = sys.stdout
        self.line_count: int = 0
        self.input_file_name: str = "stdin"
        self.prompt: str = "> "

        # ==================== 表示設定 ====================
        self.page_width: int = 80
        self.print_depth: int = 1000000000

        # ==================== 特殊PsiTerm ====================
        self.null_psi_term: Optional[PsiTerm] = None
        self.error_psi_term: Optional[PsiTerm] = None

        # ==================== 組み込み関数テーブル ====================
        self.builtin_table: Dict[Definition, Callable] = {}

        # ==================== 初期化 ====================
        self._initialized = False

    def initialize(self):
        """インタープリタを初期化する
        C版の init_built_in_types(), init_modules() などに対応
        """
        if self._initialized:
            return
        self._initialized = True

        # モジュールシステムの初期化
        self._init_modules()

        # 組み込み型の定義
        self._init_built_in_types()

        # 組み込み演算子の定義
        self._init_operators()

        # 注意: 組み込み述語・関数の登録は built_ins.register_all(wl) で行う。
        # main.py から WL.initialize() の後で呼ぶ。

    # ==================== モジュール初期化 ====================

    def _init_modules(self):
        """モジュールシステムの初期化
        C版の init_modules() に対応 (modules.c)

        C版では全モジュールの定義が一つのグローバルシンボルテーブルで管理され
        ていたが、Python版ではモジュールごとに symbol_table を持つ。
        open_modules に syntax / bi を加えることで、
        update_symbol(None, name) がそれらのシンボルを参照できるようにする。
        """
        self.user_module = self._create_module("user")
        self.bi_module = self._create_module("bi")
        self.syntax_module = self._create_module("syntax")
        self.current_module = self.user_module

        # user_module は bi / syntax を "open" する（演算子・組み込みを参照可能に）
        self.user_module.open_modules = [self.bi_module, self.syntax_module]
        # bi_module も syntax を参照できるようにする
        self.bi_module.open_modules = [self.syntax_module]

    def _create_module(self, name: str) -> Module:
        """新しいモジュールを作成して登録"""
        m = Module(name)
        self.module_table[name] = m
        return m

    def create_module(self, name: str) -> Module:
        """モジュールを取得または作成"""
        if name in self.module_table:
            return self.module_table[name]
        return self._create_module(name)

    def find_module(self, name: str) -> Optional[Module]:
        """モジュールを名前で検索"""
        return self.module_table.get(name)

    def _all_modules(self):
        """Return all registered modules (for alias transitivity)."""
        return self.module_table.values()

    def set_current_module(self, m: Module):
        """現在のモジュールを設定"""
        self.current_module = m

    # ==================== シンボルテーブル ====================

    def update_symbol(self, module: Optional[Module], name: str) -> Definition:
        """シンボルを検索または作成する
        C版の update_symbol() / modules.c に対応

        Args:
            module: 所属モジュール (None の場合は current_module)
            name: シンボル名

        Returns:
            対応する Definition
        """
        if module is None:
            module = self.current_module

        # モジュールのシンボルテーブルを検索
        if module and name in module.symbol_table:
            return module.symbol_table[name]

        # 開いているモジュールを検索
        if module:
            for open_mod in module.open_modules:
                if name in open_mod.symbol_table:
                    return open_mod.symbol_table[name]

        # 新規作成
        kw = Keyword(name, module)
        defn = Definition(kw)
        kw.definition = defn

        if module:
            module.symbol_table[name] = defn

        return defn

    def _new_type(self, module: Optional[Module], name: str,
                  parents: Optional[List[Definition]] = None) -> Definition:
        """新しい型定義を作成"""
        defn = self.update_symbol(module, name)
        defn.type = DefType.TYPE
        defn.always_check = True
        defn.protected = True
        if parents:
            for p in parents:
                self._make_type_link(defn, p)
        return defn

    def _make_type_link(self, child: Definition, parent: Definition):
        """型階層リンクを作成 (子->親)
        C版の make_type_link() に対応 (types.c)
        """
        if parent not in child.parents:
            child.parents.append(parent)
        if child not in parent.children:
            parent.children.append(child)

    # ==================== 組み込み型の初期化 ====================

    def _init_built_in_types(self):
        """組み込み型の定義
        C版の init_built_in_types() の前半部分に対応 (built_ins.c)

        LIFE の型階層:
          top (最も一般的)
            ├── boolean
            │     ├── true
            │     └── false
            ├── integer
            ├── real
            ├── quoted_string
            ├── alist (リスト)
            ├── nil  (空リスト)
            ├── disjunction
            ├── ...
        """
        bi = self.bi_module
        syn = self.syntax_module
        usr = self.user_module

        # --- 基本型 ---
        # top: すべての型の親 (最も一般的)
        self.top = self.update_symbol(bi, "@")
        self.top.type = DefType.TYPE
        self.top.always_check = True
        self.top.protected = True

        # variable: 変数の型
        self.variable = self.update_symbol(bi, "variable")
        self.variable.type = DefType.TYPE

        # nothing: パースエラーなどで使う
        self.nothing = self.update_symbol(bi, "nothing")
        self.nothing.type = DefType.TYPE

        # constant: 定数の基底型
        self.constant = self.update_symbol(bi, "constant")
        self.constant.type = DefType.TYPE
        self._make_type_link(self.constant, self.top)

        # integer, real
        self.integer = self.update_symbol(bi, "integer")
        self.integer.type = DefType.TYPE
        self._make_type_link(self.integer, self.top)

        self.real = self.update_symbol(bi, "real")
        self.real.type = DefType.TYPE
        self._make_type_link(self.real, self.top)

        # quoted_string
        self.quoted_string = self.update_symbol(bi, "quoted_string")
        self.quoted_string.type = DefType.TYPE
        self._make_type_link(self.quoted_string, self.top)

        # boolean
        self.boolean = self.update_symbol(bi, "boolean")
        self.boolean.type = DefType.TYPE
        self._make_type_link(self.boolean, self.top)

        # true, false (boolean のサブタイプ)
        self.true = self.update_symbol(bi, "true")
        self.true.type = DefType.TYPE
        self._make_type_link(self.true, self.boolean)

        self.false = self.update_symbol(bi, "false")
        self.false.type = DefType.TYPE
        self._make_type_link(self.false, self.boolean)

        # succeed
        self.succeed = self.update_symbol(bi, "succeed")
        self.succeed.type = DefType.TYPE
        self._make_type_link(self.succeed, self.top)

        # eof
        self.eof = self.update_symbol(bi, "eof")
        self.eof.type = DefType.TYPE

        # comment (コメントトークン)
        self.comment = self.update_symbol(bi, "comment")
        self.comment.type = DefType.TYPE

        # nil, alist (リスト)
        self.nil = self.update_symbol(bi, "nil")
        self.nil.type = DefType.TYPE
        self._make_type_link(self.nil, self.top)

        self.alist = self.update_symbol(bi, "alist")
        self.alist.type = DefType.TYPE
        self._make_type_link(self.alist, self.top)

        # disjunction (論理和 {a;b;c})
        self.disjunction = self.update_symbol(bi, "disjunction")
        self.disjunction.type = DefType.TYPE
        self._make_type_link(self.disjunction, self.top)

        self.disj_nil = self.update_symbol(bi, "disj_nil")
        self.disj_nil.type = DefType.TYPE
        self._make_type_link(self.disj_nil, self.disjunction)

        # stream
        self.stream = self.update_symbol(bi, "stream")
        self.stream.type = DefType.TYPE

        # --- 特殊シンボル ---
        self.and_sym = self.update_symbol(syn, "&")
        self.and_sym.type = DefType.TYPE

        self.quote = self.update_symbol(syn, "quote")
        self.quote.type = DefType.PREDICATE

        self.cut = self.update_symbol(syn, "!")
        self.cut.type = DefType.PREDICATE

        self.iff = self.update_symbol(syn, ":-")
        self.iff.type = DefType.PREDICATE

        self.eqsym = self.update_symbol(syn, "=")
        self.eqsym.type = DefType.PREDICATE

        self.leftarrowsym = self.update_symbol(syn, "<-")
        self.leftarrowsym.type = DefType.PREDICATE

        self.commasym = self.update_symbol(syn, ",")
        self.commasym.type = DefType.TYPE

        self.colonsym = self.update_symbol(syn, ":")
        self.colonsym.type = DefType.TYPE

        self.such_that = self.update_symbol(syn, "|")
        self.such_that.type = DefType.TYPE

        self.funcsym = self.update_symbol(syn, "->")
        self.funcsym.type = DefType.TYPE

        self.predsym = self.update_symbol(syn, "<-")
        self.predsym.type = DefType.TYPE

        self.typesym = self.update_symbol(syn, "type")
        self.typesym.type = DefType.TYPE

        self.opsym = self.update_symbol(syn, "op")
        self.opsym.type = DefType.TYPE

        self.loadsym = self.update_symbol(syn, "import")
        self.loadsym.type = DefType.PREDICATE

        self.dynamicsym = self.update_symbol(bi, "dynamic")
        self.staticsym = self.update_symbol(bi, "static")
        self.encodesym = self.update_symbol(bi, "encode")
        self.listingsym = self.update_symbol(bi, "listing")
        self.delay_checksym = self.update_symbol(bi, "delay_check")
        self.eval_argsym = self.update_symbol(bi, "eval_args")
        self.inputfilesym = self.update_symbol(bi, "input_file")
        self.call_handlersym = self.update_symbol(bi, "call_handler")

        self.life_or = self.update_symbol(syn, ";")
        self.life_or.type = DefType.TYPE

        self.minus_symbol = self.update_symbol(syn, "-")

        self.timesym = self.update_symbol(syn, "*")
        self.tracesym = self.update_symbol(bi, "trace")
        self.abortsym = self.update_symbol(bi, "abort")
        self.aborthooksym = self.update_symbol(bi, "abort_hook")
        self.nullsym = self.update_symbol(bi, "null")
        self.boolpredsym = self.update_symbol(bi, "bool_pred")

        # 終端トークン用の専用 Definition（演算子 '.' や '?' とは別オブジェクト）
        # C版では tokenizer が EOF/空白の後の . / ? を special token として区別する
        self.final_dot = self._new_type(bi, "__END_DOT__")
        self.final_question = self._new_type(bi, "__END_QUERY__")

        self.functor = self.update_symbol(bi, "functor")
        self.apply = self.update_symbol(bi, "apply")
        self.apply.type = DefType.FUNCTION

        self.built_in = self.update_symbol(bi, "built_in")
        self.built_in.type = DefType.TYPE

        self.xf_sym = self.update_symbol(bi, "xf")
        self.yf_sym = self.update_symbol(bi, "yf")
        self.fx_sym = self.update_symbol(bi, "fx")
        self.fy_sym = self.update_symbol(bi, "fy")
        self.xfx_sym = self.update_symbol(bi, "xfx")
        self.xfy_sym = self.update_symbol(bi, "xfy")
        self.yfx_sym = self.update_symbol(bi, "yfx")

        # システム型
        self.sys_bytedata = self.update_symbol(bi, "sys_bytedata")
        self.sys_bitvector = self.update_symbol(bi, "sys_bitvector")
        self.sys_regexp = self.update_symbol(bi, "sys_regexp")
        self.sys_stream = self.update_symbol(bi, "sys_stream")
        self.sys_file_stream = self.update_symbol(bi, "sys_file_stream")
        self.sys_socket_stream = self.update_symbol(bi, "sys_socket_stream")

        # --- 特殊PsiTerm ---
        self.null_psi_term = PsiTerm()
        self.null_psi_term.type = self.nothing

        self.error_psi_term = PsiTerm()
        self.error_psi_term.type = self.nothing

    # ==================== 演算子の初期化 ====================

    def _init_operators(self):
        """組み込み演算子の定義
        C版の init_built_in_types() 内の op() 呼び出しに対応

        LIFE言語の演算子 (Prolog互換 + 拡張):
        """
        def op(prec: int, op_type: OperatorType, name: str,
               module: Optional[Module] = None):
            """演算子を定義する"""
            defn = self.update_symbol(module or self.syntax_module, name)
            od = OperatorData(op_type, prec, defn.op_data)
            defn.op_data = od

        OT = OperatorType

        # --- 標準 Prolog 演算子 (優先度順) ---
        # xfx: 非結合中置演算子
        op(1200, OT.XFX, ":-")
        op(1200, OT.XFX, "-->")
        op(1200, OT.FX,  ":-")
        op(1200, OT.FX,  "?-")

        op(1100, OT.XFY, ";")
        op(1050, OT.XFY, "->")
        op(1000, OT.XFY, ",")

        op(900,  OT.FY,  "not")
        op(900,  OT.FY,  "\\+")

        op(900,  OT.XFX, "=")
        op(900,  OT.XFX, "\\=")
        op(900,  OT.XFX, "==")
        op(900,  OT.XFX, "\\==")
        op(900,  OT.XFX, "===")
        op(900,  OT.XFX, "\\===")
        op(900,  OT.XFX, "is")
        op(900,  OT.XFX, "=..")
        op(900,  OT.XFX, "<")
        op(900,  OT.XFX, ">")
        op(900,  OT.XFX, "=<")
        op(900,  OT.XFX, ">=")
        op(900,  OT.XFX, "=\\=")
        op(900,  OT.XFX, "=:=")

        # LIFE特有演算子
        op(1150, OT.XFX, "<|")   # サブタイプ宣言 (sort declaration)
        op(900,  OT.XFX, "<-")   # 述語定義
        op(900,  OT.XFX, "=>")   # 特性割り当て

        op(700,  OT.XFX, ":")    # 型制約
        op(600,  OT.XFY, "|")    # such that / リスト分割

        op(500,  OT.YFX, "+")
        op(500,  OT.YFX, "-")
        op(500,  OT.YFX, "/\\")
        op(500,  OT.YFX, "\\/")
        op(500,  OT.FX,  "+")
        op(500,  OT.FX,  "-")

        op(400,  OT.YFX, "*")
        op(400,  OT.YFX, "/")
        op(400,  OT.YFX, "//")
        op(400,  OT.YFX, "mod")
        op(400,  OT.YFX, "rem")
        op(400,  OT.YFX, "<<")
        op(400,  OT.YFX, ">>")
        op(400,  OT.YFX, "xor")

        op(200,  OT.XFX, "**")
        op(200,  OT.XFY, "^")
        op(200,  OT.FY,  "\\")

        # 特殊演算子
        op(100,  OT.YFX, ".")    # 特性アクセス
        op(200,  OT.FX,  "type") # 型宣言
        op(200,  OT.FX,  "fun")  # 関数宣言
        op(200,  OT.FX,  "pred") # 述語宣言

    # ==================== 組み込み述語・関数の初期化 ====================

    def _init_built_ins(self):
        """組み込み述語・関数を登録する
        C版の init_built_in_types() の後半部分に対応 (built_ins.c)
        各組み込み関数は built_ins.py に実装されている。
        """
        from wild_life import built_ins as bi

        def new_bi(module: Module, name: str, def_type: DefType,
                   func: Callable):
            """組み込み関数を登録"""
            defn = self.update_symbol(module, name)
            defn.type = def_type
            defn._builtin_func = func
            defn.evaluate_args = True
            self.builtin_table[defn] = func

        bim = self.bi_module
        syn = self.syntax_module

        # --- データベース操作 ---
        new_bi(bim, "dynamic",   DefType.PREDICATE, bi.c_dynamic)
        new_bi(bim, "static",    DefType.PREDICATE, bi.c_static)
        new_bi(bim, "assert",    DefType.PREDICATE, bi.c_assert_last)
        new_bi(bim, "asserta",   DefType.PREDICATE, bi.c_assert_first)
        new_bi(bim, "clause",    DefType.PREDICATE, bi.c_clause)
        new_bi(bim, "retract",   DefType.PREDICATE, bi.c_retract)
        new_bi(bim, "setq",      DefType.PREDICATE, bi.c_setq)
        new_bi(bim, "listing",   DefType.PREDICATE, bi.c_listing)

        # --- I/O ---
        new_bi(bim, "get",         DefType.PREDICATE, bi.c_get)
        new_bi(bim, "put",         DefType.PREDICATE, bi.c_put)
        new_bi(bim, "write",       DefType.PREDICATE, bi.c_write)
        new_bi(bim, "writeq",      DefType.PREDICATE, bi.c_writeq)
        new_bi(bim, "write_err",   DefType.PREDICATE, bi.c_write_err)
        new_bi(bim, "nl",          DefType.PREDICATE, bi.c_nl)
        new_bi(bim, "read",        DefType.PREDICATE, bi.c_read_psi)
        new_bi(bim, "open_in",     DefType.PREDICATE, bi.c_open_in)
        new_bi(bim, "open_out",    DefType.PREDICATE, bi.c_open_out)
        new_bi(bim, "close",       DefType.PREDICATE, bi.c_close)
        new_bi(bim, "parse",       DefType.FUNCTION,  bi.c_parse)

        # --- 算術 ---
        new_bi(syn, "+",   DefType.FUNCTION, bi.c_plus)
        new_bi(syn, "-",   DefType.FUNCTION, bi.c_minus)
        new_bi(syn, "*",   DefType.FUNCTION, bi.c_times)
        new_bi(syn, "/",   DefType.FUNCTION, bi.c_div)
        new_bi(syn, "//",  DefType.FUNCTION, bi.c_idiv)
        new_bi(syn, "mod", DefType.FUNCTION, bi.c_mod)
        new_bi(syn, "**",  DefType.FUNCTION, bi.c_power)
        new_bi(syn, "^",   DefType.FUNCTION, bi.c_power)
        new_bi(bim, "sqrt",  DefType.FUNCTION, bi.c_sqrt)
        new_bi(bim, "sin",   DefType.FUNCTION, bi.c_sin)
        new_bi(bim, "cos",   DefType.FUNCTION, bi.c_cos)
        new_bi(bim, "tan",   DefType.FUNCTION, bi.c_tan)
        new_bi(bim, "exp",   DefType.FUNCTION, bi.c_exp)
        new_bi(bim, "log",   DefType.FUNCTION, bi.c_log)
        new_bi(bim, "abs",   DefType.FUNCTION, bi.c_abs)
        new_bi(bim, "sign",  DefType.FUNCTION, bi.c_sign)
        new_bi(bim, "floor", DefType.FUNCTION, bi.c_floor)
        new_bi(bim, "ceiling", DefType.FUNCTION, bi.c_ceiling)
        new_bi(bim, "round", DefType.FUNCTION, bi.c_round)
        new_bi(bim, "truncate", DefType.FUNCTION, bi.c_truncate)
        new_bi(bim, "max",   DefType.FUNCTION, bi.c_max)
        new_bi(bim, "min",   DefType.FUNCTION, bi.c_min)
        new_bi(bim, "gcd",   DefType.FUNCTION, bi.c_gcd)

        # --- 比較 ---
        new_bi(syn, "<",   DefType.FUNCTION, bi.c_lt)
        new_bi(syn, "=<",  DefType.FUNCTION, bi.c_ltoe)
        new_bi(syn, ">",   DefType.FUNCTION, bi.c_gt)
        new_bi(syn, ">=",  DefType.FUNCTION, bi.c_gtoe)
        new_bi(syn, "=\\=", DefType.FUNCTION, bi.c_diff)
        new_bi(syn, "=:=", DefType.FUNCTION, bi.c_equal)
        new_bi(syn, "and", DefType.FUNCTION, bi.c_and)
        new_bi(syn, "or",  DefType.FUNCTION, bi.c_or)
        new_bi(syn, "not", DefType.FUNCTION, bi.c_not)
        new_bi(syn, "xor", DefType.FUNCTION, bi.c_xor)
        new_bi(syn, "===", DefType.FUNCTION, bi.c_same_address)
        new_bi(syn, "\\===", DefType.FUNCTION, bi.c_diff_address)

        # --- 型・特性 ---
        new_bi(bim, "nonvar",        DefType.FUNCTION,  bi.c_nonvar)
        new_bi(bim, "var",           DefType.FUNCTION,  bi.c_var)
        new_bi(bim, "is_function",   DefType.FUNCTION,  bi.c_is_function)
        new_bi(bim, "is_predicate",  DefType.FUNCTION,  bi.c_is_predicate)
        new_bi(bim, "is_sort",       DefType.FUNCTION,  bi.c_is_sort)
        new_bi(bim, "features",      DefType.FUNCTION,  bi.c_features)
        new_bi(bim, "root_sort",     DefType.FUNCTION,  bi.c_rootsort)
        new_bi(bim, "strip",         DefType.FUNCTION,  bi.c_strip)
        new_bi(bim, "sort",          DefType.FUNCTION,  bi.c_sort_of)
        new_bi(syn, ".",             DefType.FUNCTION,  bi.c_project)

        # --- リスト操作 ---
        new_bi(bim, "length",     DefType.FUNCTION,  bi.c_length)
        new_bi(bim, "append",     DefType.PREDICATE, bi.c_append)
        new_bi(bim, "member",     DefType.PREDICATE, bi.c_member)
        new_bi(bim, "last",       DefType.FUNCTION,  bi.c_last)
        new_bi(bim, "nth",        DefType.FUNCTION,  bi.c_nth)
        new_bi(bim, "reverse",    DefType.FUNCTION,  bi.c_reverse)
        new_bi(bim, "list_sort",  DefType.FUNCTION,  bi.c_list_sort)

        # --- 文字列 ---
        new_bi(bim, "string_to_atom",  DefType.FUNCTION, bi.c_string_to_atom)
        new_bi(bim, "atom_to_string",  DefType.FUNCTION, bi.c_atom_to_string)
        new_bi(bim, "number_to_string", DefType.FUNCTION, bi.c_number_to_string)
        new_bi(bim, "string_length",   DefType.FUNCTION, bi.c_string_length)
        new_bi(bim, "concat",          DefType.FUNCTION, bi.c_concat)
        new_bi(bim, "substring",       DefType.FUNCTION, bi.c_substring)

        # --- 制御 ---
        new_bi(bim, "true",   DefType.PREDICATE, bi.c_true)
        new_bi(bim, "fail",   DefType.PREDICATE, bi.c_fail)
        new_bi(bim, "halt",   DefType.PREDICATE, bi.c_halt)
        new_bi(bim, "abort",  DefType.PREDICATE, bi.c_abort)
        new_bi(bim, "call",   DefType.PREDICATE, bi.c_call)
        new_bi(bim, "once",   DefType.PREDICATE, bi.c_once)
        new_bi(bim, "findall", DefType.PREDICATE, bi.c_findall)
        new_bi(bim, "bagof",  DefType.PREDICATE, bi.c_bagof)
        new_bi(bim, "setof",  DefType.PREDICATE, bi.c_setof)

        # --- モジュール ---
        new_bi(bim, "import",    DefType.PREDICATE, bi.c_import)
        new_bi(bim, "module",    DefType.PREDICATE, bi.c_module)

        # --- システム ---
        new_bi(bim, "time",      DefType.FUNCTION,  bi.c_time)
        new_bi(bim, "date",      DefType.FUNCTION,  bi.c_date)
        new_bi(bim, "argv",      DefType.FUNCTION,  bi.c_argv)
        new_bi(bim, "getenv",    DefType.FUNCTION,  bi.c_getenv)

    # ==================== PsiTerm 生成ユーティリティ ====================

    def make_integer(self, n: float) -> PsiTerm:
        """整数PsiTermを生成"""
        t = PsiTerm()
        t.type = self.integer
        t.value = float(n)
        t.status = 4
        t.flags = QUOTED_TRUE
        return t

    def make_real(self, f: float) -> PsiTerm:
        """実数PsiTermを生成"""
        t = PsiTerm()
        t.type = self.real
        t.value = f
        t.status = 4
        t.flags = QUOTED_TRUE
        return t

    def make_number(self, f: float) -> PsiTerm:
        """数値PsiTermを生成 (整数か実数か自動判定)"""
        import math
        if f == math.floor(f) and abs(f) < 9007199254740991.0:
            return self.make_integer(f)
        return self.make_real(f)

    def make_string(self, s: str) -> PsiTerm:
        """文字列PsiTermを生成"""
        t = PsiTerm()
        t.type = self.quoted_string
        t.value = s
        t.status = 4
        t.flags = QUOTED_TRUE
        return t

    def make_atom(self, name: str,
                  module: Optional[Module] = None) -> PsiTerm:
        """アトムPsiTermを生成"""
        defn = self.update_symbol(module, name)
        t = PsiTerm()
        t.type = defn
        t.status = 4
        t.flags = QUOTED_TRUE
        return t

    def make_var(self) -> PsiTerm:
        """新しい未束縛変数を生成"""
        t = PsiTerm()
        t.type = self.top
        t.status = 0
        t.flags = 0
        return t

    def make_nil(self) -> PsiTerm:
        """空リストを生成"""
        t = PsiTerm()
        t.type = self.nil
        t.status = 4
        t.flags = QUOTED_TRUE
        return t

    def make_cons(self, head: PsiTerm, tail: PsiTerm) -> PsiTerm:
        """cons セルを生成"""
        t = PsiTerm()
        t.type = self.alist
        t.attr_list = {"1": head, "2": tail}
        t.status = 4
        t.flags = QUOTED_TRUE
        return t

    def make_list(self, items: List[PsiTerm]) -> PsiTerm:
        """PythonリストからLIFEリストを生成"""
        result = self.make_nil()
        for item in reversed(items):
            result = self.make_cons(item, result)
        return result

    def make_pair(self, left: PsiTerm, right: PsiTerm) -> PsiTerm:
        """& (and) ペアを生成"""
        t = PsiTerm()
        t.type = self.and_sym
        t.attr_list = {"1": left, "2": right}
        return t

    def make_true(self) -> PsiTerm:
        """true を生成"""
        t = PsiTerm()
        t.type = self.true
        t.status = 4
        t.flags = QUOTED_TRUE
        return t

    def make_false(self) -> PsiTerm:
        """false を生成"""
        t = PsiTerm()
        t.type = self.false
        t.status = 4
        t.flags = QUOTED_TRUE
        return t

    # ==================== 設定関連 ====================

    QUOTED_TRUE = 1

    def get_two_args(self, attr_list: dict) -> tuple:
        """属性リストから引数1,2を取得
        C版の get_two_args() に対応
        """
        a = attr_list.get("1")
        b = attr_list.get("2")
        return a, b

    def get_one_arg(self, attr_list: dict) -> Optional[PsiTerm]:
        """属性リストから引数1を取得
        C版の get_one_arg() に対応
        """
        return attr_list.get("1")

    # ==================== 外部から組み込みを登録するAPI ====================

    def new_built_in(self, name: str, func: Callable,
                     def_type: "DefType" = None,
                     module: "Module" = None) -> "Definition":
        """組み込み述語/関数を登録する公開API

        built_ins.py の register_all() から呼ばれる。
        module を省略すると bi_module を使用する。
        def_type を省略すると PREDICATE を使用する。
        """
        if def_type is None:
            def_type = DefType.PREDICATE
        if module is None:
            module = self.bi_module
        defn = self.update_symbol(module, name)
        defn.type = def_type
        defn._builtin_func = func
        defn.evaluate_args = True
        self.builtin_table[defn] = func
        return defn

    def add_operator(self, prec: int, op_type: "OperatorType",
                     name: str, module: "Module" = None) -> None:
        """演算子を動的に追加する公開API (bi_op から呼ばれる)"""
        if module is None:
            module = self.syntax_module
        defn = self.update_symbol(module, name)
        # 既存のチェーンを先頭に繋ぐ（同名シンボルに複数の演算子種別が許される）
        op = OperatorData(op_type, prec, defn.op_data)
        defn.op_data = op


# ==================== グローバルシングルトン ====================

# C版では多数のグローバル変数として定義されていたが、
# Python版では1つのシングルトンとしてまとめる
WL = WildLifeRuntime()

# 定数のエクスポート
QUOTED_TRUE = WildLifeRuntime.QUOTED_TRUE


def init():
    """インタープリタを初期化する (起動時に呼ぶ)"""
    WL.initialize()
