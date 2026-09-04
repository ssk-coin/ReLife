"""
tokenizer.py - Wild Life トークナイザ (Python版)

C版の対応ファイル: token.c, token.h

LIFE言語のトークナイザ:
  - 文字ストリームからトークン列を生成する
  - トークンはPsiTermとして表現される

トークンの種類:
  - 変数: 大文字またはアンダースコアで始まる (Variable, _)
  - 定数/アトム: 小文字で始まる (foo, bar)
  - 記号: +-*/=<>など (=, +, :-, ...)
  - 整数: 42, -3
  - 実数: 3.14, 1.0e-5
  - 文字列: "hello", 'atom'
  - コメント: % から行末まで
  - 特殊: (, ), [, ], {, }, ,, ., ?, ;, @, !
"""

from __future__ import annotations
from typing import Optional, IO, List
import sys
import math
import os

from wild_life.data_structures import PsiTerm, Definition, Module
from wild_life.runtime import WL, init


# ==================== 文字分類 ====================

def is_digit(c: str) -> bool:
    """数字かどうか (C版: DIGIT マクロ)"""
    return len(c) == 1 and '0' <= c <= '9'


def is_upper(c: str) -> bool:
    """大文字またはアンダースコアかどうか (C版: UPPER マクロ)"""
    return len(c) == 1 and (('A' <= c <= 'Z') or c == '_')


def is_lower(c: str) -> bool:
    """小文字かどうか (C版: LOWER マクロ)"""
    return len(c) == 1 and ('a' <= c <= 'z')


def is_alpha(c: str) -> bool:
    """英数字またはアンダースコアかどうか (C版: ISALPHA マクロ)"""
    return is_digit(c) or is_upper(c) or is_lower(c)


def is_single(c: str) -> bool:
    """単一文字トークンかどうか (C版: SINGLE マクロ)
    ( ) [ ] { } , . ; @ ! `
    """
    return c in "()[]{},.;@!`"


def is_symbol(c: str) -> bool:
    """記号文字かどうか (C版: SYMBOL マクロ)
    複数文字トークンの構成要素になれる文字
    """
    return c in "#$%&*+->/:<=~^|\\."


def legal_in_name(c: str) -> bool:
    """名前に使える文字かどうか (大文字・小文字・数字)"""
    return is_upper(c) or is_lower(c) or is_digit(c)


def is_symbolic(c: str) -> bool:
    """記号かどうか (is_symbol と同義)"""
    return is_symbol(c)


# ==================== トークナイザ状態 ====================

class TokenizerState:
    """トークナイザの状態
    C版のグローバル変数群に対応:
      input_stream, line_count, saved_char, saved_psi_term, etc.
    """

    EOLN = '\n'
    EOF_CHAR = ''   # Python では EOF を空文字列で表現

    def __init__(self, stream: IO = None, filename: str = "stdin"):
        self.stream: IO = stream or sys.stdin
        self.filename: str = filename
        self.line_count: int = 0
        self.start_of_line: bool = True

        # 1文字先読みバッファ (最大2文字)
        self._saved_char: Optional[str] = None
        self._old_saved_char: Optional[str] = None

        # 1トークン先読みバッファ (最大2トークン)
        self._saved_token: Optional[PsiTerm] = None
        self._old_saved_token: Optional[PsiTerm] = None

        self.eof_flag: bool = False
        self.parse_ok: bool = True

        # 文字列パースモード
        self.string_parse: bool = False
        self.string_input: str = ""
        self.string_pos: int = 0

        # 変数テーブル (現在のクエリ内の変数)
        self.var_tree: dict = {}  # name -> PsiTerm
        self.var_occurred: bool = False

    def read_char(self) -> Optional[str]:
        """1文字読み込む
        C版の read_char() に対応
        Returns:
            読み込んだ文字、またはEOFの場合 None
        """
        # 先読みバッファから読む
        if self._saved_char is not None:
            c = self._saved_char
            self._saved_char = self._old_saved_char
            self._old_saved_char = None
            return c

        # 文字列パースモード
        if self.string_parse:
            if self.string_pos < len(self.string_input):
                c = self.string_input[self.string_pos]
                self.string_pos += 1
                return c
            else:
                return None  # EOF

        # ファイル/ストリームから読む
        if self.eof_flag:
            return None

        if self.start_of_line:
            self.start_of_line = False
            self.line_count += 1
            if self.stream == sys.stdin and WL.noisy:
                print(WL.prompt, end='', flush=True)

        try:
            c = self.stream.read(1)
        except Exception:
            return None

        if not c:  # EOF
            self.eof_flag = True
            return None

        if c == self.EOLN:
            self.start_of_line = True

        return c

    def put_back_char(self, c: Optional[str]):
        """1文字戻す (最大2文字まで)
        C版の put_back_char() に対応
        """
        if c is None:
            return
        if self._old_saved_char is not None:
            # エラー: 3文字戻そうとしている
            pass  # 無視
        self._old_saved_char = self._saved_char
        self._saved_char = c

    def put_back_token(self, t: PsiTerm):
        """1トークン戻す (最大2トークンまで)
        C版の put_back_token() に対応
        """
        if self._old_saved_token is not None:
            pass  # エラー: 3トークン戻そうとしている
        self._old_saved_token = self._saved_token
        self._saved_token = t

    def read_comment(self) -> PsiTerm:
        """コメントを読み飛ばす (% から行末まで)
        C版の read_comment() に対応
        """
        while True:
            c = self.read_char()
            if c is None or c == self.EOLN:
                break
        tok = PsiTerm()
        tok.type = WL.comment
        return tok

    def _read_string_escape(self) -> Optional[str]:
        """文字列中のエスケープシーケンスを処理"""
        c = self.read_char()
        if c is None:
            return None
        escape_map = {
            'a': '\a', 'b': '\b', 'f': '\f', 'n': '\n',
            'r': '\r', 't': '\t', 'v': '\v', '\\': '\\',
            '"': '"', "'": "'"
        }
        if c in escape_map:
            return escape_map[c]
        elif c == 'x':
            # 16進数エスケープ
            c1 = self.read_char()
            if c1 is None or not c1 in '0123456789abcdefABCDEF':
                return None
            n = int(c1, 16)
            c2 = self.read_char()
            if c2 and c2 in '0123456789abcdefABCDEF':
                n = 16 * n + int(c2, 16)
            else:
                if c2:
                    self.put_back_char(c2)
            return chr(n)
        elif '0' <= c <= '7':
            # 8進数エスケープ
            n = ord(c) - ord('0')
            for _ in range(2):
                c2 = self.read_char()
                if c2 and '0' <= c2 <= '7':
                    n = 8 * n + (ord(c2) - ord('0'))
                else:
                    if c2:
                        self.put_back_char(c2)
                    break
            return chr(n)
        else:
            return c

    def read_string(self, tok: PsiTerm, end_char: str):
        """引用符付き文字列またはアトムを読む
        C版の read_string() に対応

        Args:
            tok: 読み込んだトークンを格納する PsiTerm
            end_char: 終端文字 ('"' または "'")
        """
        chars = []
        while True:
            c = self.read_char()
            if c is None:
                # EOF - エラー
                self.parse_ok = False
                break
            elif end_char == '"' and c == '\\':
                # エスケープシーケンス
                ec = self._read_string_escape()
                if ec is not None:
                    chars.append(ec)
            elif c == end_char:
                # 終端文字が2つ並んだ場合はエスケープ
                c2 = self.read_char()
                if c2 == end_char:
                    chars.append(end_char)
                else:
                    if c2:
                        self.put_back_char(c2)
                    break
            else:
                chars.append(c)

        result = ''.join(chars)

        if end_char == '"':
            tok.type = WL.quoted_string
            tok.value = result
        else:
            # 'atom' 形式 - アトムとして扱う
            tok.type = WL.update_symbol(None, result)
            tok.value = None

    def read_number(self, tok: PsiTerm, first_char: str):
        """数値を読む
        C版の read_number() に対応

        Syntax: digit+ [ '.' digit+ ] [ ('e'|'E') ['+'-'] digit+ ]
        """
        f = float(ord(first_char) - ord('0'))
        c = self.read_char()

        # 整数部を読む
        while c and is_digit(c):
            f = f * 10.0 + float(ord(c) - ord('0'))
            c = self.read_char()

        is_real = False

        # 小数部
        if c == '.':
            c2 = self.read_char()
            if c2 and is_digit(c2):
                is_real = True
                p = 10.0
                while c2 and is_digit(c2):
                    f += float(ord(c2) - ord('0')) / p
                    p *= 10.0
                    c2 = self.read_char()
                self.put_back_char(c2)
                c = self.read_char()
            else:
                # '.' の後が数字でない -> 元に戻す
                self.put_back_char(c2)
                self.put_back_char(c)
                c = self.read_char()

        # 指数部
        if c and (c == 'e' or c == 'E'):
            c2 = self.read_char()
            if c2 and (c2 == '+' or c2 == '-' or is_digit(c2)):
                is_real = True
                pos_flag = (c2 == '+' or is_digit(c2))
                if not is_digit(c2):
                    c2 = self.read_char()
                pwr = 0
                while c2 and is_digit(c2):
                    pwr = pwr * 10 + (ord(c2) - ord('0'))
                    c2 = self.read_char()
                self.put_back_char(c2)
                p = 10.0 ** pwr if pos_flag else 10.0 ** (-pwr)
                f *= p
                c = self.read_char()
            else:
                self.put_back_char(c2)
                c = self.read_char()

        if c:
            self.put_back_char(c)

        tok.value = f
        # 整数かどうかを判定 (C版: if(f==floor(f)) tok->type=integer)
        if f == math.floor(f) and not is_real:
            tok.type = WL.integer
        else:
            tok.type = WL.real

    def read_name(self, tok: PsiTerm, first_char: str,
                  char_test, typ: Definition):
        """名前(変数/定数/記号)を読む
        C版の read_name() に対応

        Args:
            tok: 読み込んだトークンを格納する PsiTerm
            first_char: 最初の文字
            char_test: 継続文字の判定関数
            typ: トークンの種類 (variable または constant)
        """
        chars = [first_char]
        module = None

        while True:
            c = self.read_char()
            if c is None:
                break

            # モジュール修飾 (Module#Symbol 形式)
            if (c == '#' and char_test is legal_in_name and
                    len(chars) > 0 and module is None):
                # モジュール名を読んだ
                mod_name = ''.join(chars)
                module = WL.create_module(mod_name)
                chars = []
                # 次の文字の種類を見て char_test を変える
                c2 = self.read_char()
                if c2 and is_symbol(c2):
                    char_test = is_symbolic
                if c2:
                    self.put_back_char(c2)
                continue

            if char_test(c):
                chars.append(c)
            else:
                self.put_back_char(c)
                break

        # モジュール名だけで終わった場合
        if module and not chars:
            name = module.module_name
            self.put_back_char('#')
            module = None
        else:
            name = ''.join(chars)

        tok.coref = None
        tok.resid = None
        tok.attr_list = {}

        if typ is WL.variable:
            tok.type = WL.variable
            if name == '_':
                # 匿名変数 '_' は top になる
                tok.type = WL.update_symbol(WL.current_module, "@")
                tok.value = None
            else:
                # 変数テーブルに登録/検索
                self.var_occurred = True
                if name not in self.var_tree:
                    # 新しい変数を作成
                    var_psi = PsiTerm()
                    var_psi.type = WL.top
                    self.var_tree[name] = var_psi
                tok.coref = self.var_tree[name]
                tok.value = name  # 変数名を保存
        else:
            # 定数/アトム
            tok.type = WL.update_symbol(module, name)
            tok.value = None

            # グローバル変数の場合
            if tok.type.type == 'global':
                self.var_occurred = True

    def read_token_main(self, for_parser: bool = True) -> PsiTerm:
        """メイントークン読み取りルーティン
        C版の read_token_main() に対応

        Args:
            for_parser: True の場合はパーサ用 (プロンプト変更あり)
        """
        # 先読みバッファからトークンを返す
        if for_parser and self._saved_token is not None:
            tok = self._saved_token
            self._saved_token = self._old_saved_token
            self._old_saved_token = None
            return tok

        # 空白をスキップ
        while True:
            c = self.read_char()
            if c is None:
                tok = PsiTerm()
                tok.type = WL.eof
                tok.value = None
                return tok
            if ord(c) > 32:
                break

        tok = PsiTerm()
        tok.status = 0
        tok.flags = 0
        tok.resid = None
        tok.attr_list = {}

        # '.' と '?' の特別処理 (C版の RM: Jul 7 1993 対応)
        if c == '.' or c == '?':
            c2 = self.read_char()
            if c2:
                self.put_back_char(c2)
            if c2 is None or (c2 and ord(c2) <= 32):
                # 終端ドット/クエスチョン
                if c == '.':
                    tok.type = WL.final_dot
                else:
                    tok.type = WL.final_question
                tok.value = None
            else:
                self.read_name(tok, c, is_symbolic, WL.constant)
        elif c == '%':
            tok = self.read_comment()
        elif c == '"':
            self.read_string(tok, c)
            tok.type = WL.quoted_string
        elif c == "'":
            self.read_string(tok, c)
        elif is_digit(c):
            self.read_number(tok, c)
        elif is_upper(c):
            self.read_name(tok, c, legal_in_name, WL.variable)
        elif is_lower(c):
            self.read_name(tok, c, legal_in_name, WL.constant)
        elif is_symbol(c):
            # 記号列を読む
            self.read_name(tok, c, is_symbolic, WL.constant)
        elif is_single(c):
            # 単一文字トークン
            tok.type = WL.update_symbol(WL.current_module, c)
            tok.value = None
        else:
            # 不正な文字 - エラー
            WL.output_stream.write(
                f"*** Error: illegal character {ord(c)} in input\n"
            )
            return self.read_token_main(for_parser)

        # コメントはスキップして再帰
        if tok.type is WL.comment:
            return self.read_token_main(for_parser)

        # カットトークンの特別処理
        if tok.type is WL.cut:
            tok.value = None  # choice_stack の位置は inference.py で処理

        # 変数でない場合はcorefをNULLに
        if tok.type is not WL.variable:
            tok.coref = None

        tok.attr_list = {}
        tok.status = 0
        tok.flags = 0
        tok.resid = None

        if for_parser:
            # 行末の空白スキップ (プロンプト変更のため)
            while True:
                c = self.read_char()
                if c is None:
                    break
                if c == self.EOLN:
                    self.put_back_char(c)
                    break
                elif ord(c) <= 32:
                    continue
                else:
                    self.put_back_char(c)
                    break

        return tok

    def read_token(self) -> PsiTerm:
        """パーサ用トークン読み取り (C版の read_token() に対応)"""
        return self.read_token_main(for_parser=True)

    def read_token_b(self) -> PsiTerm:
        """組み込み用トークン読み取り (C版の read_token_b() に対応)"""
        return self.read_token_main(for_parser=False)

    def init_var_tree(self):
        """変数テーブルを初期化 (各クエリの先頭で呼ぶ)"""
        self.var_tree = {}
        self.var_occurred = False

    def save_state(self) -> dict:
        """パーサ状態を保存 (C版の save_parse_state() に対応)"""
        return {
            'line_count': self.line_count,
            'start_of_line': self.start_of_line,
            'saved_char': self._saved_char,
            'old_saved_char': self._old_saved_char,
            'saved_token': self._saved_token,
            'old_saved_token': self._old_saved_token,
            'eof_flag': self.eof_flag,
        }

    def restore_state(self, state: dict):
        """パーサ状態を復元 (C版の restore_parse_state() に対応)"""
        self.line_count = state['line_count']
        self.start_of_line = state['start_of_line']
        self._saved_char = state['saved_char']
        self._old_saved_char = state['old_saved_char']
        self._saved_token = state['saved_token']
        self._old_saved_token = state['old_saved_token']
        self.eof_flag = state['eof_flag']


# ==================== ファイル・文字列からのトークナイザ生成 ====================

def tokenizer_from_file(filename: str) -> TokenizerState:
    """ファイルからトークナイザを作成"""
    if not WL._initialized:
        init()
    f = open(filename, 'r', encoding='utf-8', errors='replace')
    return TokenizerState(f, filename)


def tokenizer_from_string(s: str) -> TokenizerState:
    """文字列からトークナイザを作成"""
    if not WL._initialized:
        init()
    ts = TokenizerState(sys.stdin, "<string>")
    ts.string_parse = True
    ts.string_input = s
    ts.string_pos = 0
    return ts


def tokenizer_from_stdin() -> TokenizerState:
    """標準入力からトークナイザを作成"""
    if not WL._initialized:
        init()
    return TokenizerState(sys.stdin, "stdin")
