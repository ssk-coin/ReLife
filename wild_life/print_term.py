"""
print_term.py — Term printing for the Wild Life interpreter.
Corresponds to print.c in the original C source.
"""

from __future__ import annotations
import sys
from typing import Optional, Dict, Set, List, Tuple, IO

# Avoid circular imports at module level
from wild_life.data_structures import (
    PsiTerm, Definition, OperatorType
)

PRINT_DEPTH = 10
MAX_PRECEDENCE = 1200

DOTDOT = ": "

# ─────────────────────────────────────────────────────────────────────────────
# Character classification helpers (mirror of print.c's helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _is_lower(c: str) -> bool:
    return len(c) == 1 and c.islower()

def _is_upper(c: str) -> bool:
    return len(c) == 1 and c.isupper()

def _is_digit(c: str) -> bool:
    return len(c) == 1 and c.isdigit()

def _is_alpha(c: str) -> bool:
    return len(c) == 1 and (c.isalnum() or c == '_')

_SINGLE_CHARS = set("!;,|[]{}()")

def _is_single(c: str) -> bool:
    return c in _SINGLE_CHARS

_SYMBOL_CHARS = set("#&*+-./:<=>?@\\^`~")

def _is_symbol(c: str) -> bool:
    return c in _SYMBOL_CHARS

def _all_symbol(s: str) -> bool:
    return bool(s) and all(_is_symbol(c) for c in s)

def _no_quote(s: str) -> bool:
    """Return True if s does not need to be quoted."""
    if not s:
        return False
    if s[0] == '%':
        return False
    if _is_single(s[0]) and len(s) == 1:
        return True
    if s[0] == '_' and len(s) == 1:
        return False
    if _all_symbol(s):
        return True
    if not _is_lower(s[0]):
        return False
    return all(_is_alpha(c) for c in s[1:])

def _needs_quoting(s: str) -> bool:
    return not _no_quote(s)


# ─────────────────────────────────────────────────────────────────────────────
# Printer state class
# ─────────────────────────────────────────────────────────────────────────────

class PrintState:
    """Holds all per-print-call state (replaces C's global print vars)."""

    def __init__(self, outfile: IO = None):
        self.outfile: IO = outfile or sys.stdout
        self.print_depth: int = PRINT_DEPTH
        self.const_quote: bool = True
        self.write_resids: bool = False
        self.write_canon: bool = False
        self.write_corefs: bool = True
        self.listing_flag: bool = False
        self.indent: bool = False
        self.gen_sym_counter: int = 0
        # Maps id(psiterm) -> name string or None
        self.pointer_names: Dict[int, Optional[str]] = {}
        # Maps id(psiterm) -> name string (already printed)
        self.printed_pointers: Dict[int, str] = {}
        # Buffer for indenting mode
        self._buf: List[str] = []

    # ─── output helpers ────────────────────────────────────────────────────

    def write(self, s: str) -> None:
        if self.indent:
            self._buf.append(s)
        else:
            self.outfile.write(s)

    def flush(self) -> None:
        if self.indent:
            text = ''.join(self._buf)
            self.outfile.write(text)
            self._buf.clear()
        self.outfile.flush()

    # ─── naming helpers ─────────────────────────────────────────────────────

    def _nice_name(self) -> str:
        self.gen_sym_counter += 1
        g = self.gen_sym_counter
        parts = []
        while True:
            g -= 1
            parts.append(chr(g % 26 + ord('A')))
            g //= 26
            if g == 0:
                break
        return '_' + ''.join(reversed(parts))

    def _unique_name(self, var_tree: dict) -> str:
        while True:
            name = self._nice_name()
            if name not in var_tree:
                return name

    def go_through(self, t: 'PsiTerm', var_tree: dict = None) -> None:
        """Walk t to find shared sub-terms that need variable names."""
        if var_tree is None:
            var_tree = {}
        self._go_through_term(t)

    def _go_through_term(self, t: Optional['PsiTerm']) -> None:
        if t is None:
            return
        t = t.deref()
        tid = id(t)
        if tid in self.pointer_names:
            self.pointer_names[tid] = 'SHARED'  # needs a name
            return
        self.pointer_names[tid] = None  # seen once
        for val in t.attr_list.values():
            self._go_through_term(val)

    def insert_variables(self, var_tree: dict, force: bool) -> None:
        """Map variable names from var_tree into pointer_names."""
        for name, pterm in var_tree.items():
            if pterm is None:
                continue
            t = pterm.deref()
            tid = id(t)
            if tid in self.pointer_names:
                if self.pointer_names[tid] is not None or force:
                    self.pointer_names[tid] = name

    def forbid_variables(self, var_tree: dict) -> None:
        """Pre-register top-level variables in printed_pointers."""
        for name, pterm in var_tree.items():
            if pterm is None:
                continue
            t = pterm.deref()
            self.printed_pointers[id(t)] = name


# ─────────────────────────────────────────────────────────────────────────────
# Core printing logic
# ─────────────────────────────────────────────────────────────────────────────

def _str_to_int(s: str) -> int:
    """Return int value of s if s is a non-negative integer string, else -1."""
    if not s:
        return -1
    try:
        v = int(s)
        if str(v) == s and v >= 0:
            return v
        return -1
    except ValueError:
        return -1


def _print_symbol_quoted(ps: PrintState, sym: str, quote: bool) -> None:
    """Print atom sym, quoting if needed."""
    if quote and _needs_quoting(sym):
        ps.write("'")
        ps.write(sym.replace("'", "''"))
        ps.write("'")
    else:
        ps.write(sym)


def _print_symbol(ps: PrintState, kw) -> None:
    """Print a keyword symbol (module-aware)."""
    if kw is None:
        return
    sym = kw.symbol if hasattr(kw, 'symbol') else str(kw)
    ps.write(sym)


def _print_symbol_q(ps: PrintState, kw) -> None:
    """Print a keyword symbol with quoting if needed."""
    if kw is None:
        return
    sym = kw.symbol if hasattr(kw, 'symbol') else str(kw)
    _print_symbol_quoted(ps, sym, ps.const_quote)


def _check_opargs(attr_list: dict) -> int:
    """Return bitmask: bit0=has '1', bit1=has '2', bit2=has other."""
    result = 0
    for k in attr_list:
        if k == '1':
            result |= 1
        elif k == '2':
            result |= 2
        else:
            result |= 4
    return result


NOTOP = 0
INFIX = 1
PREFIX = 2
POSTFIX = 3


def _opcheck(t: 'PsiTerm') -> Tuple[int, int, OperatorType]:
    """Return (kind, prec, op_type) for operator printing."""
    from wild_life.data_structures import OperatorType
    defn = t.type
    if defn is None or defn.op_data is None:
        return NOTOP, 0, OperatorType.NOP
    numarg = _check_opargs(t.attr_list)
    if numarg not in (1, 3):
        return NOTOP, 0, OperatorType.NOP
    op_data = defn.op_data
    while op_data is not None:
        op = op_data.type if hasattr(op_data, 'type') else OperatorType.NOP
        if numarg == 1:
            if op in (OperatorType.XF, OperatorType.YF):
                return POSTFIX, op_data.precedence, op
            if op in (OperatorType.FX, OperatorType.FY):
                return PREFIX, op_data.precedence, op
        if numarg == 3:
            if op in (OperatorType.XFX, OperatorType.XFY, OperatorType.YFX):
                return INFIX, op_data.precedence, op
        op_data = op_data.next if hasattr(op_data, 'next') else None
    return NOTOP, 0, OperatorType.NOP


def _get_two_args(attr_list: dict):
    """Return (arg1, arg2) from attr_list, either or both may be None."""
    return attr_list.get('1'), attr_list.get('2')


def _pretty_psi_with_ops(ps: PrintState, t: 'PsiTerm', sprec: int, depth: int) -> bool:
    """Try to print t as an operator expression. Return True if done."""
    if ps.write_canon:
        return False
    from wild_life.data_structures import OperatorType
    tkind, tprec, ttype = _opcheck(t)
    if tkind == NOTOP:
        return False

    surround = tkind in (INFIX, PREFIX, POSTFIX) and tprec >= sprec
    if surround:
        ps.write("(")

    if tkind == INFIX:
        arg1, arg2 = _get_two_args(t.attr_list)
        if arg1:
            arg1 = arg1.deref()
        if arg2:
            arg2 = arg2.deref()
        a1kind, a1prec, a1type = _opcheck(arg1) if arg1 else (NOTOP, 0, OperatorType.NOP)
        a2kind, a2prec, a2type = _opcheck(arg2) if arg2 else (NOTOP, 0, OperatorType.NOP)

        # p1: whether arg1 needs parentheses
        if a1prec > tprec:
            p1 = True
        elif a1prec < tprec:
            p1 = False
        else:
            if ttype in (OperatorType.XFY, OperatorType.XFX):
                p1 = True
            elif a1type in (OperatorType.YFX, OperatorType.FX, OperatorType.FY):
                p1 = False
            else:
                p1 = True

        # p2: whether arg2 needs parentheses
        if a2prec > tprec:
            p2 = True
        elif a2prec < tprec:
            p2 = False
        else:
            if ttype in (OperatorType.YFX, OperatorType.XFX):
                p2 = True
            elif a2type in (OperatorType.XFY, OperatorType.XF, OperatorType.YF):
                p2 = False
            else:
                p2 = True

        sym = t.type.keyword.symbol if t.type and t.type.keyword else ''
        if p1:
            ps.write("(")
        _pretty_tag_or_psi_term(ps, arg1, MAX_PRECEDENCE + 1, depth)
        if p1:
            ps.write(")")
        if not p1 and sym != ',':
            ps.write(" ")
        _print_symbol_q(ps, t.type.keyword)
        if ps.listing_flag and sym in (',', ':-'):
            ps.write("\n        ")
        else:
            if not p2 and sym != '.':
                ps.write(" ")
        if p2:
            ps.write("(")
        _pretty_tag_or_psi_term(ps, arg2, MAX_PRECEDENCE + 1, depth)
        if p2:
            ps.write(")")

    elif tkind == PREFIX:
        arg1, _ = _get_two_args(t.attr_list)
        if arg1:
            arg1 = arg1.deref()
        a1kind, a1prec, a1type = _opcheck(arg1) if arg1 else (NOTOP, 0, OperatorType.NOP)
        if a1type in (OperatorType.FX, OperatorType.FY):
            p1 = False
        else:
            p1 = tprec <= a1prec
        _print_symbol_q(ps, t.type.keyword)
        if not p1:
            ps.write(" ")
        if p1:
            ps.write("(")
        _pretty_tag_or_psi_term(ps, arg1, MAX_PRECEDENCE + 1, depth)
        if p1:
            ps.write(")")

    elif tkind == POSTFIX:
        arg1, _ = _get_two_args(t.attr_list)
        if arg1:
            arg1 = arg1.deref()
        a1kind, a1prec, a1type = _opcheck(arg1) if arg1 else (NOTOP, 0, OperatorType.NOP)
        if a1type in (OperatorType.XF, OperatorType.YF):
            p1 = False
        else:
            p1 = tprec <= a1prec
        if p1:
            ps.write("(")
        _pretty_tag_or_psi_term(ps, arg1, MAX_PRECEDENCE + 1, depth)
        if p1:
            ps.write(")")
        if not p1:
            ps.write(" ")
        _print_symbol_q(ps, t.type.keyword)

    if surround:
        ps.write(")")
    return True


def _count_features(attr_list: dict) -> int:
    return len(attr_list)


def _check_legal_cons(t: 'PsiTerm', t_type) -> bool:
    """Return True if t is a proper cons (has exactly '1' and '2')."""
    return (
        t.type == t_type
        and _count_features(t.attr_list) == 2
        and '1' in t.attr_list
        and '2' in t.attr_list
    )


def _pretty_list(ps: PrintState, t: 'PsiTerm', depth: int, wl) -> None:
    """Pretty-print a list or disjunction."""
    t_type = t.type

    if t_type == wl.alist or (wl.alist and t_type and
            t_type.is_subtype_of(wl.alist)):
        if t_type != wl.alist:
            _print_symbol(ps, t_type.keyword)
            ps.write(DOTDOT)
        ps.write("[")
        sep = ","
        end = "]"
    elif t_type == wl.disjunction:
        ps.write("{")
        sep = ";"
        end = "}"
    else:
        ps.write("[")
        sep = ","
        end = "]"

    list_depth = 0
    done = False
    while not done:
        if list_depth == ps.print_depth:
            ps.write("...")
        arg1, arg2 = _get_two_args(t.attr_list)
        if arg1:
            arg1 = arg1.deref()
        if arg2:
            arg2 = arg2.deref()

        if list_depth < ps.print_depth:
            _pretty_tag_or_psi_term(ps, arg1, 999, depth)

        if arg2 is None:
            done = True
        else:
            tid2 = id(arg2)
            if tid2 in ps.pointer_names and ps.pointer_names[tid2]:
                ps.write("|")
                _pretty_tag_or_psi_term(ps, arg2, MAX_PRECEDENCE + 1, depth)
                done = True
            elif (arg2.type == wl.nil and not arg2.attr_list) or \
                 (arg2.type == wl.disj_nil and not arg2.attr_list):
                done = True
            elif not _check_legal_cons(arg2, t_type):
                ps.write("|")
                _pretty_tag_or_psi_term(ps, arg2, MAX_PRECEDENCE + 1, depth)
                done = True
            else:
                if list_depth < ps.print_depth:
                    ps.write(sep)
                t = arg2

        list_depth += 1

    ps.write(end)


def _pretty_tag_or_psi_term(ps: PrintState, p: Optional['PsiTerm'],
                              sprec: int, depth: int, wl=None) -> None:
    """Print p, using variable name tag if shared."""
    if wl is None:
        from wild_life.runtime import WL as wl
    if p is None:
        ps.write("<VOID>")
        return
    p = p.deref()
    tid = id(p)

    name = ps.pointer_names.get(tid)
    if name is not None:
        if name == 'SHARED':
            name = ps._unique_name({})
            ps.pointer_names[tid] = name
        n2 = ps.printed_pointers.get(tid)
        if n2 is None:
            ps.write(name)
            ps.printed_pointers[tid] = name
            if not _is_top(p, wl):
                ps.write(DOTDOT)
                _pretty_psi_term(ps, p, 0, depth, wl)
        else:
            ps.write(n2)
    else:
        _pretty_psi_term(ps, p, sprec, depth, wl)


def _is_top(t: 'PsiTerm', wl) -> bool:
    """Return True if t is the 'top' type with no value/attrs."""
    return t.type == wl.top and t.value is None and not t.attr_list


def _pretty_psi_term(ps: PrintState, t: Optional['PsiTerm'],
                      sprec: int, depth: int, wl=None) -> None:
    """Core recursive pretty-printer."""
    if wl is None:
        from wild_life.runtime import WL as wl
    if t is None:
        return
    t = t.deref()

    # List / disjunction sugar
    if (t.type == wl.alist or t.type == wl.disjunction):
        if _check_legal_cons(t, t.type):
            _pretty_list(ps, t, depth + 1, wl)
            _maybe_resid(ps, t)
            return

    if t.type == wl.nil and not t.attr_list:
        ps.write("[]")
        _maybe_resid(ps, t)
        return
    if wl.disj_nil and t.type == wl.disj_nil and not t.attr_list:
        ps.write("{}")
        _maybe_resid(ps, t)
        return

    args_written = False
    if t.value is not None:
        _print_value(ps, t, wl)
        args_written = True
    else:
        if depth < ps.print_depth:
            args_written = _pretty_psi_with_ops(ps, t, sprec, depth + 1)
        if not args_written:
            _print_symbol_q(ps, t.type.keyword if t.type else None)

    if not args_written and t.attr_list and depth < ps.print_depth:
        _pretty_attr(ps, t.attr_list, depth + 1, wl)
    elif not args_written and t.attr_list and depth >= ps.print_depth:
        ps.write("(...)")

    _maybe_resid(ps, t)


def _print_value(ps: PrintState, t: 'PsiTerm', wl) -> None:
    """Print the value of a psi-term that has a concrete value."""
    defn = t.type
    if defn is None:
        ps.write(repr(t.value))
        return

    # Check integer / real
    if wl.integer and defn.is_subtype_of(wl.integer):
        v = t.value
        if isinstance(v, float):
            if v == int(v):
                ps.write(str(int(v)))
            else:
                ps.write(repr(v))
        else:
            ps.write(str(v))
        if defn != wl.integer:
            ps.write(DOTDOT)
            _print_symbol(ps, defn.keyword)
        return

    if wl.real and defn.is_subtype_of(wl.real):
        v = t.value
        ps.write(f"{v:g}")
        if defn != wl.real and defn != wl.integer:
            ps.write(DOTDOT)
            _print_symbol(ps, defn.keyword)
        return

    if wl.quoted_string and defn.is_subtype_of(wl.quoted_string):
        if ps.const_quote:
            ps.write('"')
            ps.write(str(t.value).replace('"', '""'))
            ps.write('"')
        else:
            ps.write(str(t.value))
        if defn != wl.quoted_string:
            ps.write(DOTDOT)
            _print_symbol_q(ps, defn.keyword)
        return

    if defn == wl.stream:
        ps.write(f"stream({t.value!r})")
        return

    if defn == wl.eof:
        _print_symbol_q(ps, defn.keyword)
        return

    if defn == wl.cut:
        _print_symbol_q(ps, defn.keyword)
        return

    # fallback
    ps.write(repr(t.value))


def _pretty_attr(ps: PrintState, attr_list: dict, depth: int, wl) -> None:
    """Print attribute list in parenthesized form."""
    ps.write("(")
    # Sort features: integers first (numerically), then strings
    from wild_life.data_structures import featcmp_key
    keys = sorted(attr_list.keys(), key=featcmp_key)
    cnt = [1]  # mutable counter

    first = True
    for k in keys:
        if not first:
            ps.write(",")
        first = False
        v = attr_list[k]
        iv = _str_to_int(k)
        if iv < 0:
            # Named feature
            _print_symbol_quoted(ps, k, ps.const_quote)
            ps.write(" => ")
        elif iv == cnt[0]:
            cnt[0] += 1
            # positional — no label
        else:
            ps.write(str(iv))
            ps.write(" => ")
        if v:
            _pretty_tag_or_psi_term(ps, v, 999, depth, wl)
        else:
            ps.write("<null>")
    ps.write(")")


def _maybe_resid(ps: PrintState, t: 'PsiTerm') -> None:
    """Print residuation markers if any."""
    if t.resid:
        for r in t.resid:
            if getattr(r, 'goal', None) and getattr(r.goal, 'pending', False):
                ps.write("~")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def term_to_string(t: Optional['PsiTerm'], quoted: bool = True,
                   print_depth: int = PRINT_DEPTH,
                   var_tree: dict = None, wl=None) -> str:
    """Convert a PsiTerm to its string representation."""
    if wl is None:
        from wild_life.runtime import WL as wl
    import io
    buf = io.StringIO()
    ps = PrintState(outfile=buf)
    ps.print_depth = print_depth
    ps.const_quote = quoted
    ps.indent = False

    vt = var_tree or {}
    ps.go_through(t, vt)
    ps.insert_variables(vt, False)

    _pretty_tag_or_psi_term(ps, t, MAX_PRECEDENCE + 1, 0, wl)
    return buf.getvalue()


def write_term(t: Optional['PsiTerm'], outfile: IO = None,
               quoted: bool = True, print_depth: int = PRINT_DEPTH,
               var_tree: dict = None, wl=None) -> None:
    """Write a term to outfile (default stdout)."""
    if wl is None:
        from wild_life.runtime import WL as wl
    if outfile is None:
        outfile = sys.stdout
    ps = PrintState(outfile=outfile)
    ps.print_depth = print_depth
    ps.const_quote = quoted
    ps.indent = False

    vt = var_tree or {}
    ps.go_through(t, vt)
    ps.insert_variables(vt, False)

    _pretty_tag_or_psi_term(ps, t, MAX_PRECEDENCE + 1, 0, wl)


def print_variables(var_tree: dict, outfile: IO = None,
                    print_depth: int = PRINT_DEPTH, wl=None) -> bool:
    """
    Print all query variables in the form 'X = value'.
    Returns True if there were any variables.
    """
    if wl is None:
        from wild_life.runtime import WL as wl
    if not var_tree:
        return False
    if outfile is None:
        outfile = sys.stdout

    ps = PrintState(outfile=outfile)
    ps.print_depth = print_depth
    ps.const_quote = True
    ps.write_resids = True
    ps.indent = False

    # Scan all variables
    for name, pterm in var_tree.items():
        if pterm is not None:
            ps._go_through_term(pterm.deref())
    ps.insert_variables(var_tree, True)
    ps.forbid_variables(var_tree)

    first = True
    for name in sorted(var_tree.keys()):
        pterm = var_tree[name]
        if pterm is None:
            continue
        t = pterm.deref()
        if not first:
            outfile.write(", ")
        first = False
        outfile.write(name)
        outfile.write(" = ")

        n2 = ps.printed_pointers.get(id(t))
        if n2 and n2 < name:
            outfile.write(n2)
        else:
            _pretty_psi_term(ps, t, MAX_PRECEDENCE + 1, 0, wl)
            ps.flush()

    if not first:
        outfile.write(".")
    return not first


def display_psi_term(t: Optional['PsiTerm'], outfile: IO = None,
                     wl=None) -> None:
    """Simple display of a psi-term (no variable tracking)."""
    if wl is None:
        from wild_life.runtime import WL as wl
    if outfile is None:
        outfile = sys.stdout
    s = term_to_string(t, quoted=True, wl=wl)
    outfile.write(s)
