"""
parser_.py - Wild Life パーサ (Python版)

C版の対応ファイル: parser.c, parser.h

LIFE言語のパーサ:
  演算子優先度パーサを使用して LIFE 項を読み込む
  C版と同じアルゴリズム:
    - read_psi_term(): 複合項を読む
    - parse_list(): リスト/論理和を読む
    - read_life_form(): 演算子優先度パーサ (状態機械)
    - parse(): トップレベルパーサ

演算子優先度パーサのアルゴリズム:
  C版の parser.c の crunch(), read_life_form() に対応
  状態 0: 項を期待 (前置演算子またはアトム)
  状態 1: 演算子を期待 (中置/後置演算子)
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Any
import sys

from wild_life.data_structures import (
    PsiTerm, Definition, OperatorData, OperatorType, Rule,
    featcmp_key, QUOTED_TRUE
)
from wild_life.runtime import WL, init
from wild_life.tokenizer import TokenizerState

# 演算子でない場合の優先度
NOP = 2000


class ParseError(Exception):
    """パースエラー"""
    pass


# ==================== パーサスタックの要素 ====================

class StackEntry:
    """パーサスタックの要素
    C版の psi_term_stack[], int_stack[], op_stack[] に対応
    """
    def __init__(self, term: PsiTerm, prec: int,
                 op: OperatorType = OperatorType.NOP):
        self.term = term     # psi-term
        self.prec = prec     # 優先度
        self.op = op         # 演算子の種類


# ==================== パーサ ====================

class Parser:
    """LIFE言語のパーサ
    C版の parser.c の関数群をクラスにまとめたもの
    """

    def __init__(self, tokenizer: TokenizerState):
        self.ts = tokenizer          # トークナイザ
        self.stack: List[StackEntry] = []  # パーサスタック
        self.parse_ok: bool = True   # パースエラーフラグ

        if not WL._initialized:
            init()

    # ==================== スタック操作 ====================

    def push(self, term: PsiTerm, prec: int,
             op: OperatorType = OperatorType.NOP):
        """スタックにプッシュ (C版の push() に対応)"""
        self.stack.append(StackEntry(term, prec, op))

    def pop(self) -> StackEntry:
        """スタックからポップ (C版の pop() に対応)"""
        if not self.stack:
            self.parse_ok = False
            entry = StackEntry(WL.error_psi_term or PsiTerm(), 0)
            return entry
        return self.stack.pop()

    def look_prec(self) -> int:
        """スタックトップの優先度を見る (C版の look() に対応)"""
        if not self.stack:
            return 0
        return self.stack[-1].prec

    # ==================== 演算子優先度の取得 ====================

    def precedence(self, tok: PsiTerm, op_type: OperatorType) -> int:
        """トークンが指定の演算子種別を持つ場合の優先度を返す
        C版の precedence() に対応
        Returns NOP if tok is not an operator of type op_type.
        """
        if tok.type is None:
            return NOP
        od = tok.type.op_data
        while od is not None:
            if od.type == op_type:
                return od.precedence
            od = od.next
        return NOP

    # ==================== 補助関数 ====================

    def equ_tok(self, tok: PsiTerm, name: str) -> bool:
        """トークンが指定の名前かどうか"""
        if tok.type is None:
            return False
        return tok.type.symbol == name

    def equ_tokch(self, tok: PsiTerm, c: str) -> bool:
        """トークンが指定の1文字かどうか"""
        if tok.type is None:
            return False
        sym = tok.type.symbol
        return len(sym) == 1 and sym[0] == c

    def equ_tokc(self, tok: PsiTerm, c: Optional[str]) -> bool:
        """トークンが指定の文字かどうか (c=None の場合は空文字列)"""
        if tok.type is None:
            return False
        sym = tok.type.symbol
        if c is None:
            return sym == ''
        return len(sym) == 1 and sym[0] == c

    def is_wl_const(self, tok: PsiTerm) -> bool:
        """定数かどうか (C版の wl_const マクロに対応)"""
        return tok.value is None and tok.type is not WL.variable

    def bad_psi_term(self, tok: PsiTerm) -> bool:
        """パーサが定数として扱えない項かどうか
        C版の bad_psi_term() に対応
        """
        if tok.type is WL.final_dot or tok.type is WL.final_question:
            return True
        sym = tok.type.symbol if tok.type else ""
        return (len(sym) == 1 and
                sym[0] in "()[]{}") if sym else False

    # ==================== 項の作成 ====================

    def make_life_form(self, tok: PsiTerm, arg1: Optional[PsiTerm],
                       arg2: Optional[PsiTerm]) -> PsiTerm:
        """演算子適用項を作成
        C版の make_life_form() に対応

        tok(arg1, arg2) または tok(arg1) の項を作成する。
        ':' の場合は変数制約を作成。
        '-' の場合は負数の処理。
        """
        # tok を dereference
        t = tok.deref() if tok else tok
        t.attr_list = {}
        t.resid = None

        # ':' (型制約) の特別処理
        if t.type is WL.colonsym and arg1 and arg2:
            a1 = arg1.deref()
            a2 = arg2.deref()
            if a1 is not a2:
                if a1.type is WL.top and not a1.attr_list and not a1.resid:
                    # a1 が未束縛変数 -> a2 に束縛
                    a1.coref = arg2
                    result = arg1
                    result.attr_list = {}
                    result.resid = None
                    return result
                elif a2.type is WL.top and not a2.attr_list and not a2.resid:
                    # a2 が未束縛変数 -> a1 に束縛
                    a2.coref = arg1
                    result = arg2
                    result.attr_list = {}
                    result.resid = None
                    return result
                else:
                    # エラー: ':' は '&' が必要
                    self.syntax_error("':' occurs where '&' required")
                    return WL.error_psi_term or PsiTerm()
            else:
                return arg1

        # '-' (負数) の特別処理
        if (t.type is WL.minus_symbol and arg1 and not arg2 and
                arg1.value is not None and
                (arg1.type is WL.integer or arg1.type is WL.real)):
            result = PsiTerm()
            result.type = arg1.type
            result.value = -arg1.value
            return result

        # 一般的な演算子適用
        if arg1:
            t.attr_list["1"] = arg1
        if arg2:
            t.attr_list["2"] = arg2

        return t

    # ==================== CRUNCH (スタック縮小) ====================

    def crunch(self, prec: int, limit: int):
        """スタック上の演算子を適用してスタックを縮小する
        C版の crunch() に対応

        prec 以下の優先度の演算子をスタックから取り出し、
        対応するpsi-termを作成してスタックに戻す。
        """
        while (self.parse_ok and
               prec >= self.look_prec() and
               len(self.stack) > limit):

            e1 = self.pop()
            t1 = e1.term
            op1 = e1.op

            if op1 == OperatorType.NOP:
                # スタックトップが項 -> 前の演算子を取り出す
                if len(self.stack) <= limit:
                    # 直前に演算子がない（項単独）→ 押し戻して終了
                    self.stack.append(e1)
                    break
                e2 = self.pop()
                t2 = e2.term
                op2 = e2.op

                if op2 == OperatorType.FX:
                    # 前置演算子: t2(t1)
                    t = self.make_life_form(t2, t1, None)
                elif op2 == OperatorType.XFX:
                    # 中置演算子: t2(e3, t1)
                    e3 = self.pop()
                    t3 = e3.term
                    op3 = e3.op
                    if op3 == OperatorType.NOP:
                        t = self.make_life_form(t2, t3, t1)
                    else:
                        self.parse_ok = False
                        t = WL.error_psi_term or PsiTerm()
                else:
                    self.parse_ok = False
                    t = WL.error_psi_term or PsiTerm()

            elif op1 == OperatorType.XF:
                # 後置演算子: t1(e2)
                e2 = self.pop()
                t2 = e2.term
                op2 = e2.op
                if op2 == OperatorType.NOP:
                    t = self.make_life_form(t1, t2, None)
                else:
                    self.parse_ok = False
                    t = WL.error_psi_term or PsiTerm()
            else:
                self.parse_ok = False
                t = WL.error_psi_term or PsiTerm()

            # 縮小した結果をスタックに戻す
            self.push(t, self.look_prec(), OperatorType.NOP)

    # ==================== リストパーサ ====================

    def list_nil(self, typ: Definition) -> PsiTerm:
        """リストの終端 nil を作成
        C版の list_nil() に対応
        """
        nihil = PsiTerm()
        if typ is WL.disjunction:
            nihil.type = WL.disj_nil
        else:
            nihil.type = WL.nil
        nihil.status = 0
        nihil.flags = 0
        nihil.attr_list = {}
        nihil.resid = None
        nihil.value = None
        nihil.coref = None
        return nihil

    def parse_list(self, typ: Definition, end_char: str,
                   sep_char: str) -> PsiTerm:
        """リストまたは論理和を読む
        C版の parse_list() に対応

        例:
          [a,b,c]    -> alist(1=>a, 2=>alist(1=>b, 2=>alist(1=>c, 2=>nil)))
          {a;b;c}    -> disjunction(1=>a, 2=>disjunction(1=>b, 2=>disj_nil))
          [a,b|T]    -> alist(1=>a, 2=>alist(1=>b, 2=>T))
          []         -> nil
        """
        result = self.list_nil(typ)

        if not self.parse_ok:
            return result

        t = self.ts.read_token()

        if not self.equ_tokc(t, end_char):
            # CAR の読み込み
            self.ts.put_back_token(t)
            car = self.read_life_form(sep_char, '|')

            # CDR の読み込み
            t = self.ts.read_token()
            if self.equ_tokch(t, sep_char):
                cdr = self.parse_list(typ, end_char, sep_char)
            elif self.equ_tokch(t, end_char):
                cdr = self.list_nil(typ)
            elif self.equ_tokch(t, '|'):
                # [a,b|Tail] 形式
                cdr = self.read_life_form(end_char, chr(0))
                t2 = self.ts.read_token()
                if not self.equ_tokch(t2, end_char):
                    if not self.ts.string_parse:
                        sys.stderr.write(
                            f"*** Syntax error: bad end of list\n"
                        )
                    else:
                        self.parse_ok = False
            else:
                if self.ts.string_parse:
                    self.parse_ok = False
                else:
                    sys.stderr.write(
                        f"*** Syntax error: bad symbol in list\n"
                    )
                cdr = self.list_nil(typ)

            result.type = typ
            result.attr_list["1"] = car
            result.attr_list["2"] = cdr

        return result

    # ==================== 複合項パーサ ====================

    def read_psi_term(self) -> PsiTerm:
        """複合psi-termを読む
        C版の read_psi_term() に対応

        [A,B,C] -> alist
        {0;1;2} -> disjunction
        f(X,Y)  -> f(1=>X, 2=>Y) または f(x=>V, y=>W) など
        """
        if not self.parse_ok:
            return WL.error_psi_term or PsiTerm()

        t = self.ts.read_token()

        # リストの読み込み
        if self.equ_tokch(t, '['):
            t = self.parse_list(WL.alist, ']', ',')
        elif self.equ_tokch(t, '{'):
            t = self.parse_list(WL.disjunction, '}', ';')

        if not self.parse_ok:
            return WL.error_psi_term or PsiTerm()

        # 引数リストの読み込み ( ... )
        if (t.type is not WL.eof and
                not self.bad_psi_term(t)):

            t2 = self.ts.read_token()
            if self.equ_tokch(t2, '('):
                count = 0
                while True:
                    f2 = True
                    t2 = self.ts.read_token()

                    # ラベル付き引数: key => value
                    if self.is_wl_const(t2) and not self.bad_psi_term(t2):
                        t3 = self.ts.read_token()
                        if self.equ_tok(t3, "=>"):
                            t3 = self.read_life_form(',', ')')
                            # 特性を挿入
                            key = t2.type.keyword
                            if key and key.private_feature:
                                feat_name = key.combined_name
                            else:
                                feat_name = t2.type.symbol if t2.type else str(count)
                            self._feature_insert(feat_name, t.attr_list, t3)
                            f2 = False
                        else:
                            self.ts.put_back_token(t3)

                    # 整数ラベル付き引数: 1 => value
                    if self.parse_ok and t2.type is WL.integer and t2.value is not None:
                        t3 = self.ts.read_token()
                        if self.equ_tok(t3, "=>"):
                            t3 = self.read_life_form(',', ')')
                            v = int(t2.value)
                            feat_name = str(v)
                            self._feature_insert(feat_name, t.attr_list, t3)
                            f2 = False
                        else:
                            self.ts.put_back_token(t3)

                    # 位置引数
                    if f2:
                        self.ts.put_back_token(t2)
                        t2 = self.read_life_form(',', ')')
                        count += 1
                        self._feature_insert(str(count), t.attr_list, t2)

                    t2 = self.ts.read_token()
                    if self.equ_tokch(t2, ')'):
                        break
                    elif not self.equ_tokch(t2, ','):
                        if self.ts.string_parse:
                            self.parse_ok = False
                        else:
                            sys.stderr.write(
                                "*** Syntax error: ',' expected in argument list\n"
                            )
                        break

                    if not self.parse_ok:
                        break
            else:
                self.ts.put_back_token(t2)

        # 変数に引数がある場合は apply に変換
        if t.type is WL.variable and t.attr_list:
            t2 = t
            t = PsiTerm()
            t.type = WL.apply
            t.value = None
            t.coref = None
            t.resid = None
            t.attr_list = dict(t2.attr_list)
            t.attr_list[WL.functor.symbol] = t2

        return t

    def _feature_insert(self, key: str, attr_list: dict,
                        psi: PsiTerm):
        """属性を attr_list に挿入する
        C版の feature_insert() に対応
        重複特性はエラー
        """
        if key in attr_list:
            sys.stderr.write(
                f"*** Syntax error: duplicate feature {key}\n"
            )
        else:
            attr_list[key] = psi

    # ==================== 演算子優先度パーサ ====================

    def read_life_form(self, stop1: Optional[str],
                       stop2: Optional[str]) -> PsiTerm:
        """演算子優先度パーサのメイン関数
        C版の read_life_form() に対応

        Args:
            stop1: 停止文字1 (例: ',' はリスト要素の区切り)
            stop2: 停止文字2 (例: '|' はリストの | 区切り)

        Returns:
            読み込んだ psi-term
        """
        limit = len(self.stack)  # 現在のスタック底
        state = 0                # 0: 項を期待, 1: 演算子を期待
        prec = 0
        fin = False

        if not self.parse_ok:
            return WL.error_psi_term or PsiTerm()

        while not fin and self.parse_ok:
            if state == 1:
                t = self.ts.read_token()
            else:
                t = self.read_psi_term()

            if not fin:
                if state == 1:
                    # 演算子を期待する状態
                    if (stop1 and self.equ_tokch(t, stop1)) or \
                       (stop2 and self.equ_tokch(t, stop2)):
                        fin = True
                        self.ts.put_back_token(t)
                    else:
                        # 後置演算子チェック
                        pr_op = self.precedence(t, OperatorType.XF)
                        pr_1 = pr_op - 1

                        if pr_op == NOP:
                            pr_op = self.precedence(t, OperatorType.YF)
                            pr_1 = pr_op

                        if pr_op == NOP:
                            # 中置演算子チェック
                            pr_op = self.precedence(t, OperatorType.XFX)
                            pr_1 = pr_op - 1
                            pr_2 = pr_op - 1

                            if pr_op == NOP:
                                pr_op = self.precedence(t, OperatorType.XFY)
                                pr_1 = pr_op - 1
                                pr_2 = pr_op

                            if pr_op == NOP:
                                pr_op = self.precedence(t, OperatorType.YFX)
                                pr_1 = pr_op
                                pr_2 = pr_op - 1

                            if pr_op == NOP:
                                # 演算子でない -> 終了
                                fin = True
                                self.ts.put_back_token(t)
                            else:
                                self.crunch(pr_1, limit)
                                self.push(t, pr_2, OperatorType.XFX)
                                prec = pr_2
                                state = 0
                        else:
                            # 後置演算子を適用
                            self.crunch(pr_1, limit)
                            self.push(t, pr_1, OperatorType.XF)
                            prec = pr_1

                else:
                    # 項を期待する状態
                    if t.attr_list:
                        pr_op = NOP
                    else:
                        # 前置演算子チェック
                        pr_op = self.precedence(t, OperatorType.FX)
                        pr_2 = pr_op - 1

                        if pr_op == NOP:
                            pr_op = self.precedence(t, OperatorType.FY)
                            pr_2 = pr_op

                    if pr_op == NOP:
                        # 前置演算子でない
                        if self.equ_tokch(t, '('):
                            # 括弧で囲まれた式
                            t2 = self.read_life_form(')', None)
                            if self.parse_ok:
                                self.push(t2, prec, OperatorType.NOP)
                                t2 = self.ts.read_token()
                                if not self.equ_tokch(t2, ')'):
                                    if self.ts.string_parse:
                                        self.parse_ok = False
                                    else:
                                        sys.stderr.write(
                                            "*** Syntax error: ')' missing\n"
                                        )
                                    self.ts.put_back_token(t2)
                                state = 1
                        elif self.bad_psi_term(t):
                            self.ts.put_back_token(t)
                            fin = True
                        else:
                            self.push(t, prec, OperatorType.NOP)
                            state = 1
                    else:
                        # 前置演算子
                        self.push(t, pr_2, OperatorType.FX)
                        prec = pr_2

        # 最終的なスタック縮小
        if state == 1:
            self.crunch(NOP, limit)

        # スタックチェック
        if self.parse_ok and len(self.stack) != limit + 1:
            if self.ts.string_parse:
                self.parse_ok = False
            else:
                sys.stderr.write("*** Syntax error: bad expression\n")

        if self.parse_ok and self.stack:
            e = self.pop()
            result = e.term
        else:
            result = WL.error_psi_term or PsiTerm()

        # スタックを limit に戻す
        while len(self.stack) > limit:
            self.stack.pop()

        return result

    # ==================== トップレベルパーサ ====================

    def parse(self) -> Tuple[PsiTerm, int]:
        """一つのクエリまたはファクトを読む
        C版の parse() に対応

        Returns:
            (parsed_term, kind) where kind is FACT, QUERY, or ERROR
        """
        from wild_life.data_structures import FACT, QUERY, ERROR

        self.stack = []
        self.parse_ok = True
        self.ts.init_var_tree()

        # 式を読む
        s = self.read_life_form(None, None)

        kind = ERROR
        if self.parse_ok:
            if s.type is not WL.eof:
                t = self.ts.read_token()
                if t.type is WL.final_question:
                    kind = QUERY
                elif t.type is WL.final_dot:
                    kind = FACT
                else:
                    if not self.ts.string_parse:
                        sys.stderr.write(
                            f"*** Syntax error: expected '?' or '.' "
                            f"but got '{t.type.symbol if t.type else '?'}'\n"
                        )
                    kind = ERROR
            else:
                kind = QUERY  # EOF は暗黙の終了

        if not self.parse_ok:
            # エラー回復: 次の '.' か '?' まで読み飛ばす
            while True:
                tok = self.ts.read_token()
                if (tok.type is WL.final_dot or
                        tok.type is WL.final_question or
                        tok.type is WL.eof):
                    break
            kind = ERROR

        return s, kind

    def syntax_error(self, msg: str):
        """構文エラーを報告"""
        sys.stderr.write(
            f"*** Syntax error near line {self.ts.line_count}: {msg}\n"
        )
        self.parse_ok = False


# ==================== 文字列からパース ====================

def parse_string(s: str) -> Tuple[Optional[PsiTerm], int]:
    """文字列を LIFE 項としてパースする

    Args:
        s: パースする文字列 (例: "f(X,Y)?")

    Returns:
        (parsed_term, kind) or (None, ERROR) on failure
    """
    if not WL._initialized:
        init()
    from wild_life.tokenizer import tokenizer_from_string
    from wild_life.data_structures import ERROR

    ts = tokenizer_from_string(s)
    p = Parser(ts)
    term, kind = p.parse()
    if not p.parse_ok:
        return None, ERROR
    return term, kind


def parse_term_string(s: str) -> Optional[PsiTerm]:
    """文字列をLIFE項としてパースする (クエリ終端不要)

    Args:
        s: パースする文字列 (例: "f(X,Y)")

    Returns:
        parsed PsiTerm or None on failure
    """
    if not WL._initialized:
        init()
    from wild_life.tokenizer import tokenizer_from_string

    ts = tokenizer_from_string(s + ".")
    p = Parser(ts)
    term, kind = p.parse()
    if not p.parse_ok:
        return None
    return term
