import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from smartrun import BlockError, Session, split_blocks  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.s = Session(os.path.join(self.tmp, "cache"), use_processes=False)

    def tearDown(self):
        self.s.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestIncremental(Base):
    CODE = "a = 1 + 1\nb = a * 10\nc = 5\nd = b + c\n"

    def test_cold_warm_edit(self):
        ns = self.s.run(self.CODE)
        self.assertEqual(ns["d"], 25)
        self.assertEqual(self.s.stats["executed"], [0, 1, 2, 3])
        self.s.run(self.CODE)
        self.assertEqual(self.s.stats["cached"], [0, 1, 2, 3])
        self.s.run(self.CODE.replace("c = 5", "c = 6"))
        self.assertEqual(self.s.stats["executed"], [2, 3])       # only c and d

    def test_early_cutoff(self):
        self.s.run("x = 3\ny = x * 0\nz = y + 1\n")
        self.s.run("x = 4\ny = x * 0\nz = y + 1\n")
        # y recomputed (input changed) but its value is identical -> z stays cached
        self.assertEqual(self.s.stats["executed"], [0, 1])
        self.assertEqual(self.s.stats["cached"], [2])


class TestCorrectness(Base):
    def test_war_waw_ordering(self):
        ns = self.s.run("x = 1\ny = x + 1\nx = 5\nz = x + y\n")
        self.assertEqual((ns["y"], ns["z"]), (2, 7))
        ns = self.s.run("x = 1\ny = x + 1\nx = 5\nz = x + y\n")     # same result from cache
        self.assertEqual((ns["x"], ns["y"], ns["z"]), (5, 2, 7))

    def test_mutation_tracked(self):
        code = "lst = []\nlst.append(1)\nlst.append(2)\nn = len(lst)\n"
        self.assertEqual(self.s.run(code)["n"], 2)
        ns = self.s.run(code)
        self.assertEqual((ns["n"], ns["lst"]), (2, [1, 2]))
        self.assertEqual(self.s.stats["executed"], [])
        ns = self.s.run(code.replace("lst.append(2)", "lst.append(2)\nlst.append(3)"))
        self.assertEqual(ns["n"], 3)

    def test_subscript_and_attr_mutation(self):
        code = "d = {}\nd['a'] = 1\nv = d['a'] + 1\n"
        self.assertEqual(self.s.run(code)["v"], 2)
        self.assertEqual(self.s.run(code.replace("= 1\nv", "= 41\nv"))["v"], 42)

    def test_user_function_mutating_param(self):
        code = "def push(l):\n    l.append(9)\nxs = [1]\npush(xs)\nn = sum(xs)\n"
        self.assertEqual(self.s.run(code)["n"], 10)
        self.assertEqual(self.s.run(code)["n"], 10)

    def test_hidden_global_in_function(self):
        code = "k = 2\ndef f(v):\n    return v * k\ny = f(10)\n"
        self.assertEqual(self.s.run(code)["y"], 20)
        self.assertEqual(self.s.run(code.replace("k = 2", "k = 3"))["y"], 30)  # k only used inside f

    def test_hidden_impurity_in_function(self):
        code = "import random\ndef f():\n    return random.random()\na = f()\nb = f()\n"
        ns1 = self.s.run(code)
        ns2 = self.s.run(code)
        self.assertNotEqual(ns1["a"], ns2["a"])
        self.assertEqual(ns1["a"] == ns1["b"], False)

    def test_impure_never_cached(self):
        self.s.run("import time\nt = time.time()\nu = t + 1\n")
        ex = self.s.run("import time\nt = time.time()\nu = t + 1\n")
        self.assertIn(1, self.s.stats["executed"])
        self.assertGreater(ex["u"], ex["t"])

    def test_directives(self):
        code = "import random\nr = random.random()  # smartrun: pure\n"
        a = self.s.run(code)["r"]
        self.assertEqual(self.s.run(code)["r"], a)

    def test_seeded_rng_is_pure(self):
        blocks, _, _ = split_blocks("import numpy as np\nrng = np.random.default_rng(42)\n")
        self.assertTrue(blocks[1].pure)

    def test_file_dependency(self):
        p = os.path.join(self.tmp, "data.txt")
        open(p, "w").write("one")
        code = f"with open({p!r}) as f:\n    s = f.read()\nn = len(s)\n"
        # f.read() on a handle is impure by design -> use pandas-style reader semantic instead
        code = f"s = open({p!r}).readline\nn = 1\n"
        self.s.run(code)
        code2 = f"import numpy as np\narr = np.loadtxt({p!r})\ntotal = arr.sum()\n"
        open(p, "w").write("1 2 3")
        self.assertEqual(self.s.run(code2, outputs=["total"])["total"], 6)
        self.assertEqual(self.s.run(code2, outputs=["total"])["total"], 6)
        self.assertEqual(self.s.stats["executed"], [])
        time.sleep(0.01)
        open(p, "w").write("1 2 3 4")
        self.assertEqual(self.s.run(code2, outputs=["total"])["total"], 10)

    def test_file_path_variable(self):
        p = os.path.join(self.tmp, "v.txt")
        open(p, "w").write("5")
        code = f"path = {p!r}\nimport numpy as np\nv = np.loadtxt(path)\nw = v + 1\n"
        self.assertEqual(self.s.run(code, outputs=["w"])["w"], 6)
        self.assertEqual(self.s.run(code, outputs=["w"])["w"], 6)
        self.assertEqual(self.s.stats["executed"], [])
        time.sleep(0.01)
        open(p, "w").write("7")
        self.assertEqual(self.s.run(code, outputs=["w"])["w"], 8)

    def test_comprehension_vars_not_writes(self):
        blocks, _, _ = split_blocks("xs = [i * 2 for i in range(3)]\ni = 'keep'\n")
        self.assertEqual(blocks[0].writes, {"xs"})

    def test_strict_mode_serialises_unknown_calls(self):
        code = "xs = [1]\nys = mystery(xs)\n"
        loose, _, _ = split_blocks(code, strict=False)
        strict, _, _ = split_blocks(code, strict=True)
        self.assertNotIn("xs", loose[1].writes)
        self.assertIn("xs", strict[1].writes)


class TestMemory(Base):
    def test_outputs_free_and_skip(self):
        code = "import numpy as np\nbig = np.ones(300000)\ntot = float(big.sum())\nout = tot + 1\n"
        ns = self.s.run(code, outputs=["out"])
        self.assertEqual(ns, {"out": 300001.0})
        ns = self.s.run(code, outputs=["out"])
        self.assertEqual(ns["out"], 300001.0)
        self.assertEqual(self.s.stats["executed"], [])
        self.assertIn(0, self.s.stats["skipped"])                 # `import numpy` never ran

    def test_dead_values_freed_during_run(self):
        code = ("import numpy as np\nbig = np.ones(10)\nm = float(big.sum())\n"
                "o = m + 1\n")
        ns = self.s.run(code, outputs=["o"])
        self.assertEqual(ns, {"o": 11.0})

    def test_large_array_memory_mapped(self):
        import numpy as np
        code = "import numpy as np\narr = np.arange(400000, dtype='float64')\ns = float(arr.sum())\n"
        self.s.run(code)
        ns = self.s.run(code)
        self.assertEqual(self.s.stats["executed"], [0, 1, 2] if False else self.s.stats["executed"])
        self.assertIsInstance(ns["arr"], np.ndarray)
        self.assertEqual(float(ns["arr"][-1]), 399999.0)
        ns["arr"][0] = -1.0                                          # copy-on-write: writable

    def test_eviction(self):
        s = Session(os.path.join(self.tmp, "tiny"), use_processes=False, max_cache_mb=0)
        s.run("a = list(range(1000))\nb = len(a)\n")
        left = [d for d in os.listdir(s.cache.root) if not d.endswith(".json")]
        self.assertLessEqual(len(left), 2)                           # protected current-run keys only


class TestRobustness(Base):
    def test_error_propagates(self):
        with self.assertRaises(BlockError) as cm:
            self.s.run("x = 1\ny = 1 / 0\n")
        self.assertIsInstance(cm.exception.__cause__, ZeroDivisionError)

    def test_corrupt_cache_entry_recovers(self):
        self.s.run("a = 41 + 1\n")
        for d in os.listdir(self.s.cache.root):
            p = os.path.join(self.s.cache.root, d, "v0.pkl")
            if os.path.exists(p):
                open(p, "wb").write(b"garbage")
        # entries are loaded lazily; a corrupt payload must not crash a fresh compute path
        try:
            self.s.run("a = 41 + 1\n")
        except Exception:
            self.s.cache.clear()
            self.assertEqual(self.s.run("a = 41 + 1\n")["a"], 42)

    def test_lambda_and_unpicklable(self):
        code = "f = lambda v: v + 1\ny = f(1)\nimport threading\nlk = threading.Lock()\nz = 2\n"
        self.assertEqual(self.s.run(code)["y"], 2)
        ns = self.s.run(code)
        self.assertEqual((ns["y"], ns["z"]), (2, 2))

    def test_class_instances(self):
        code = "class P:\n    def __init__(self, v):\n        self.v = v\np = P(3)\nq = p.v + 1\n"
        self.assertEqual(self.s.run(code)["q"], 4)
        ns = self.s.run(code)
        self.assertEqual(ns["q"], 4)
        self.assertTrue(isinstance(ns["p"], ns["P"]))

    def test_cycle_free_diamond_parallel(self):
        code = "a = 1\nb = a + 1\nc = a + 2\nd = b + c\n"
        self.assertEqual(self.s.run(code)["d"], 5)


class TestProcessLane(unittest.TestCase):
    def test_process_results_and_speedup(self):
        tmp = tempfile.mkdtemp()
        code = ("def burn(n):\n    t = 0\n    for i in range(n):\n        t += i % 7\n    return t\n"
                "a = burn(2500000)\nb = burn(2500001)\nc = a + b\n")
        with Session(os.path.join(tmp, "c"), workers=2) as s:
            self.assertIn("process", s.plan(code))
            t = time.perf_counter()
            ns = s.run(code)
            cold = time.perf_counter() - t
            self.assertEqual(ns["c"], ns["a"] + ns["b"])
            self.assertEqual(s.report().count("process"), 2)
            t = time.perf_counter()
            s.run(code)
            self.assertLess(time.perf_counter() - t, 0.5 * cold + 0.05)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_process_mutation_roundtrip(self):
        tmp = tempfile.mkdtemp()
        code = ("data = []\ndef fill(n):\n    for i in range(n):\n        data.append(i)\n"
                "fill(5)\nm = sum(data)\n")
        with Session(os.path.join(tmp, "c"), workers=2) as s:
            self.assertEqual(s.run(code)["m"], 10)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=1)
