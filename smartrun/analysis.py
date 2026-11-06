"""Static analysis: split source into blocks and work out what each block
reads, writes, mutates, whether it is pure, which files it reads, whether it
is a CPU-bound Python loop, and which libraries it touches."""
from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass, field

BUILTINS = frozenset(dir(builtins))
GPU_LIBS = frozenset({"torch", "cupy", "jax", "cudf", "cuml", "tensorflow"})
NUMERIC_LIBS = frozenset({
    "numpy", "pandas", "scipy", "sklearn", "cv2", "PIL", "polars", "pyarrow", "numexpr",
    "xgboost", "lightgbm"}) | GPU_LIBS

# method names that (very likely) mutate their receiver
MUTATING_METHODS = frozenset({
    "append", "extend", "insert", "pop", "remove", "clear", "update", "add", "discard",
    "sort", "reverse", "setdefault", "popitem", "fill", "resize", "put", "itemset",
    "appendleft", "extendleft", "popleft", "rotate", "difference_update",
    "intersection_update", "symmetric_difference_update", "push", "write", "writelines",
    "seek", "truncate", "shuffle", "drop_duplicates_inplace", "__setitem__", "__delitem__",
    "setflags", "partition", "sort_values_inplace", "fillna_inplace", "copyto"})

# calls with side effects or non-deterministic results -> block is never cached
IMPURE_LAST = frozenset({
    "print", "input", "sleep", "system", "popen", "urandom", "getenv", "putenv", "exec",
    "eval", "exit", "quit", "breakpoint", "now", "today", "utcnow", "time", "time_ns",
    "perf_counter", "monotonic", "process_time", "uuid1", "uuid4", "mkdir", "makedirs",
    "remove", "unlink", "rename", "rmtree", "copyfile", "move", "touch", "write",
    "writelines", "write_text", "write_bytes", "save", "savez", "savez_compressed",
    "savetxt", "savefig", "dump", "to_csv", "to_parquet", "to_pickle", "to_json",
    "to_excel", "to_feather", "to_hdf", "to_sql", "flush", "read", "readline", "readlines",
    "globals", "locals", "__import__", "show", "display", "seek", "tell"})
IMPURE_MODULES = frozenset({
    "time", "subprocess", "socket", "requests", "urllib", "http", "uuid", "secrets",
    "threading", "multiprocessing", "shutil", "tempfile", "logging", "sys", "glob",
    "asyncio", "sqlite3", "smtplib", "ftplib"})
RANDOM_FUNCS = frozenset({"shuffle", "permutation", "choice", "uniform", "normal",
                          "sample", "randperm", "bernoulli", "multinomial", "dropout"})
SEEDED_CTORS = frozenset({"default_rng", "RandomState", "Random"})
READ_FUNCS = frozenset({"load", "loadtxt", "genfromtxt", "fromfile", "imread", "read_text",
                        "read_bytes", "open", "loadmat"})
PURE_BUILTINS = frozenset({
    "len", "sum", "min", "max", "sorted", "range", "abs", "round", "int", "float", "str",
    "bool", "list", "tuple", "dict", "set", "frozenset", "enumerate", "zip", "map", "filter",
    "any", "all", "isinstance", "type", "repr", "hash", "id", "reversed", "iter", "next",
    "divmod", "pow", "chr", "ord", "hex", "bin", "oct", "format", "bytes"})

_MARK = re.compile(r"#\s*smartrun:\s*([A-Za-z_, ]+)")


# ------------------------------------------------------------------ helpers
def dotted(n):
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
        return ".".join(reversed(parts))
    return None


def base_name(n):
    while isinstance(n, (ast.Attribute, ast.Subscript, ast.Starred)):
        n = n.value
    return n.id if isinstance(n, ast.Name) else None


def _arg_names(a: ast.arguments):
    out = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        out.add(a.vararg.arg)
    if a.kwarg:
        out.add(a.kwarg.arg)
    return out


def _scope_walk(nodes):
    """Walk nodes without descending into nested function/class/lambda bodies."""
    stack = list(nodes)
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def _scope_locals(nodes):
    out, declared = set(), set()
    for n in _scope_walk(nodes):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            declared.update(n.names)
    return out - declared


def _simple_expr(e):
    """Expression we can safely re-evaluate to find a file path (no calls)."""
    return not any(isinstance(x, (ast.Call, ast.Lambda, ast.Await, ast.Yield, ast.NamedExpr))
                   for x in ast.walk(e))


# ------------------------------------------------------------------ scanner
class Scan(ast.NodeVisitor):
    def __init__(self, aliases, defs_mut, strict=False):
        self.aliases, self.defs_mut, self.strict = aliases, defs_mut, strict
        self.loads, self.stores, self.muts, self.all_muts = set(), set(), set(), set()
        self.names, self.now_loads = set(), set()
        self.impure = self.loops = False
        self.files, self.file_exprs = [], []
        self.scopes, self.depth = [], 0

    # -- bookkeeping
    def _local(self, n):
        return any(n in s for s in self.scopes)

    def _load(self, n):
        self.names.add(n)
        if not self._local(n):
            self.loads.add(n)

    def _store(self, n):
        self.names.add(n)
        if not self._local(n):
            self.stores.add(n)

    def _mut(self, expr):
        b = base_name(expr)
        if b:
            self.all_muts.add(b)
            if not self._local(b):
                self.muts.add(b)

    def _mut_target(self, t):
        if isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._mut_target(e)
        elif isinstance(t, ast.Starred):
            self._mut_target(t.value)
        elif isinstance(t, (ast.Subscript, ast.Attribute)):
            self._mut(t.value)

    def _now(self, nodes):
        if self.depth == 0:
            sub = Scan(self.aliases, self.defs_mut, self.strict)
            for n in nodes:
                sub.visit(n)
            self.now_loads |= sub.loads

    # -- names / assignments
    def visit_Name(self, n):
        (self._load if isinstance(n.ctx, ast.Load) else self._store)(n.id)

    def visit_Attribute(self, n):
        d = dotted(n)
        if d and d.startswith(("os.environ", "sys.argv", "sys.stdin", "sys.stdout", "sys.stderr")):
            self.impure = True
        self.generic_visit(n)

    def visit_Assign(self, n):
        for t in n.targets:
            self._mut_target(t)
        self.generic_visit(n)

    def visit_AnnAssign(self, n):
        self._mut_target(n.target)
        self.generic_visit(n)

    def visit_AugAssign(self, n):
        self._mut_target(n.target)
        if isinstance(n.target, ast.Name):
            self._load(n.target.id)
        self.generic_visit(n)

    def visit_Delete(self, n):
        for t in n.targets:
            self._mut_target(t)
        self.generic_visit(n)

    def visit_For(self, n):
        self.loops = True
        self._mut_target(n.target)
        self.generic_visit(n)

    visit_AsyncFor = visit_For

    def visit_While(self, n):
        self.loops = True
        self.generic_visit(n)

    def visit_Global(self, n):
        self.impure = True

    visit_Nonlocal = visit_Global

    def visit_Import(self, n):
        for a in n.names:
            self._store((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            if a.name == "*":
                self.impure = True
            else:
                self._store(a.asname or a.name)

    # -- scopes
    def _comp(self, n):
        self.loops = True
        names = {t.id for g in n.generators for t in ast.walk(g.target) if isinstance(t, ast.Name)}
        self.scopes.append(names)
        self.generic_visit(n)
        self.scopes.pop()

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

    def visit_Lambda(self, n):
        for d in n.args.defaults + [x for x in n.args.kw_defaults if x]:
            self.visit(d)
        self.scopes.append(_arg_names(n.args))
        self.depth += 1
        self.visit(n.body)
        self.depth -= 1
        self.scopes.pop()

    def visit_FunctionDef(self, n):
        a = n.args
        defaults = a.defaults + [x for x in a.kw_defaults if x]
        annots = [x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs if x.annotation]
        annots += [x.annotation for x in (a.vararg, a.kwarg) if x is not None and x.annotation]
        if n.returns:
            annots.append(n.returns)
        exprs = list(n.decorator_list) + defaults + annots
        self._now(exprs)
        if self.depth == 0:
            self._store(n.name)
        else:
            self.names.add(n.name)
        for e in exprs:
            self.visit(e)
        self.scopes.append((_arg_names(a) | _scope_locals(n.body)))
        self.depth += 1
        for s in n.body:
            self.visit(s)
        self.depth -= 1
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        exprs = list(n.decorator_list) + list(n.bases) + [k.value for k in n.keywords]
        body_now = [s for s in n.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self._now(exprs + body_now)
        if self.depth == 0:
            self._store(n.name)
        else:
            self.names.add(n.name)
        for e in exprs:
            self.visit(e)
        locs = _scope_locals(n.body)
        self.depth += 1
        for s in n.body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(s)
            else:
                self.scopes.append(locs)
                self.visit(s)
                self.scopes.pop()
        self.depth -= 1

    # -- calls: mutation, purity, file reads
    def visit_Call(self, n):
        f = n.func
        d = dotted(f)
        last = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        segs = d.split(".") if d else ([last] if last else [])
        root = segs[0] if segs else None
        mod = self.aliases.get(root, root)

        if isinstance(f, ast.Attribute) and f.attr in MUTATING_METHODS:
            self._mut(f.value)
        if isinstance(f, ast.Name):
            if f.id in ("setattr", "delattr") and n.args:
                self._mut(n.args[0])
            if f.id in self.defs_mut:
                params, mutated = self.defs_mut[f.id]
                for i, a in enumerate(n.args):
                    if i < len(params) and params[i] in mutated:
                        self._mut(a)
                for k in n.keywords:
                    if k.arg in mutated:
                        self._mut(k.value)
        if self.strict:
            self._strict(n, f, root)
        if last:
            self._purity(n, last, segs, mod)
        self.generic_visit(n)

    def _strict(self, n, f, root):
        if isinstance(f, ast.Name) and (f.id in PURE_BUILTINS or f.id in self.defs_mut):
            return
        if isinstance(f, ast.Attribute) and root in self.aliases:
            return
        for a in list(n.args) + [k.value for k in n.keywords]:
            if isinstance(a, ast.Name):
                self._mut(a)
        if isinstance(f, ast.Attribute):
            self._mut(f.value)

    def _purity(self, n, last, segs, mod):
        if last in IMPURE_LAST:
            self.impure = True
        if mod in IMPURE_MODULES:
            self.impure = True
        elif mod == "os" and not (len(segs) > 1 and segs[1] == "path"):
            self.impure = True
        seeded = last in SEEDED_CTORS and bool(n.args or n.keywords)
        if not seeded and ("random" in segs or mod == "random" or last.startswith("rand")
                           or last in RANDOM_FUNCS):
            self.impure = True
        is_open = last == "open" and isinstance(n.func, ast.Name)
        if is_open or last in READ_FUNCS or last.startswith("read_"):
            if is_open:
                mode = n.args[1] if len(n.args) > 1 else next(
                    (k.value for k in n.keywords if k.arg == "mode"), None)
                if mode is not None and not (isinstance(mode, ast.Constant)
                                             and isinstance(mode.value, str)
                                             and not set(mode.value) & set("wax+")):
                    self.impure = True
            arg0 = n.args[0] if n.args else None
            if arg0 is None:
                self.impure = True
            elif isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                self.files.append(arg0.value)
            elif not self.scopes and _simple_expr(arg0):
                self.file_exprs.append(arg0)
            else:
                self.impure = True


# ------------------------------------------------------------------ blocks
@dataclass
class Block:
    idx: int
    node: ast.stmt
    src: str
    kind: str                      # 'stmt' | 'def' | 'import'
    reads: set = field(default_factory=set)
    writes: set = field(default_factory=set)
    now_reads: set = field(default_factory=set)
    hidden_writes: set = field(default_factory=set)  # globals a def mutates when *called*
    eff: set = field(default_factory=set)         # reads incl. hidden reads of functions used
    files: set = field(default_factory=set)
    file_exprs: list = field(default_factory=list)
    libs: set = field(default_factory=set)
    pure: bool = True
    loops: bool = False
    is_callable: bool = False
    markers: set = field(default_factory=set)
    sources: dict = field(default_factory=dict)   # name -> idx of producing block
    deps: set = field(default_factory=set)
    dependents: set = field(default_factory=set)
    sig: str = ""
    # filled in during a run
    key: str = ""
    out_hash: str = ""
    code: object = None


def split_blocks(code: str, strict: bool = False):
    tree = ast.parse(code)
    lines = code.splitlines()

    aliases = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                aliases[(a.asname or a.name).split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            for a in n.names:
                aliases[a.asname or a.name] = n.module.split(".")[0]

    # pass 1: which user functions mutate which parameters
    defs_mut = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sc = Scan(aliases, {}, strict)
            sc.visit(n)
            a = n.args
            params = [x.arg for x in a.posonlyargs + a.args]
            mutated = sc.all_muts & (set(params) | {x.arg for x in a.kwonlyargs})
            defs_mut[n.name] = (params, mutated)

    # pass 2: scan each top-level statement
    blocks, scans = [], []
    for i, node in enumerate(tree.body):
        sc = Scan(aliases, defs_mut, strict)
        sc.visit(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "def"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            kind = "import"
        else:
            kind = "stmt"
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        chunk = lines[max(first - 2, 0):node.end_lineno]
        b = Block(i, node, ast.unparse(node), kind)
        for m in _MARK.finditer("\n".join(chunk)):
            b.markers |= {x.strip().lower() for x in m.group(1).split(",") if x.strip()}
        if kind == "def":
            b.writes = {node.name}
            b.hidden_writes = (set(sc.stores) | set(sc.muts)) - {node.name}
        else:
            b.writes = set(sc.stores) | set(sc.muts)
        b.pure = not sc.impure
        b.loops = sc.loops
        b.files = set(sc.files)
        b.file_exprs = sc.file_exprs if kind == "stmt" else []
        b.now_reads = set(sc.now_loads)
        b.libs = {aliases.get(n, n) for n in sc.names}
        b.is_callable = (isinstance(node, ast.Assign)
                         and any(isinstance(x, ast.Lambda) for x in ast.walk(node.value)))
        b.sig = hashlib_sig(node)
        blocks.append(b)
        scans.append(sc)

    defined = set().union(*(b.writes for b in blocks)) if blocks else set()
    compound = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
    for b, sc in zip(blocks, scans):
        defined |= b.hidden_writes
        b.reads = {n for n in sc.loads if n not in BUILTINS or n in defined}
        b.now_reads = {n for n in b.now_reads if n not in BUILTINS or n in defined}
        b.reads |= sc.muts | b.hidden_writes
        if isinstance(b.node, compound):
            b.reads |= sc.stores
        if "pure" in b.markers:
            b.pure = True
        if "impure" in b.markers:
            b.pure = False

    # hidden dependencies: a block calling a function inherits that function's reads,
    # impurity, file reads, loop-ness and library use (transitively, fixpoint)
    callables = {}
    for b in blocks:
        if b.kind == "def":
            callables.setdefault(b.node.name, []).append(b.idx)
        elif b.is_callable:
            for t in b.node.targets:
                if isinstance(t, ast.Name):
                    callables.setdefault(t.id, []).append(b.idx)
    for b in blocks:
        b.eff = set(b.reads)
    changed = True
    while changed:
        changed = False
        for b in blocks:
            for n in list(b.eff):
                for d in callables.get(n, ()):
                    if d == b.idx:
                        continue
                    o = blocks[d]
                    if not o.eff <= b.eff:
                        b.eff |= o.eff; changed = True
                    if not o.hidden_writes <= b.hidden_writes and b.kind == "def":
                        b.hidden_writes |= o.hidden_writes; changed = True
                    if b.kind != "def" and not o.hidden_writes <= b.writes:
                        b.writes |= o.hidden_writes
                        b.eff |= o.hidden_writes; changed = True
                    if not o.files <= b.files:
                        b.files |= o.files; changed = True
                    if not o.libs <= b.libs:
                        b.libs |= o.libs; changed = True
                    if o.loops and not b.loops:
                        b.loops = True; changed = True
                    if not o.pure and b.pure and b.kind == "stmt" and "pure" not in b.markers:
                        b.pure = False; changed = True

    # dependency edges: read-after-write, write-after-write, write-after-read
    last_writer, readers = {}, {}
    for b in blocks:
        for n in b.eff:
            s = last_writer.get(n)
            if s is not None and s != b.idx:
                b.sources[n] = s
                b.deps.add(s)
        for n in b.writes:
            s = last_writer.get(n)
            if s is not None:
                b.deps.add(s)
            b.deps |= readers.get(n, set())
        for n in b.eff:
            readers.setdefault(n, set()).add(b.idx)
        for n in b.writes:
            last_writer[n] = b.idx
            readers[n] = set()
        b.deps.discard(b.idx)
    for b in blocks:
        for d in b.deps:
            blocks[d].dependents.add(b.idx)
    return blocks, aliases, last_writer


def hashlib_sig(node):
    import hashlib
    return hashlib.blake2b(ast.dump(node).encode(), digest_size=8).hexdigest()
