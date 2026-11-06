"""smartrun - incremental, parallel, cached execution of ordinary Python code."""
from ....smartrun.session import Session, BlockError, run
from ....smartrun.analysis import split_blocks

__version__ = "0.2.0"
__all__ = ["Session", "BlockError", "run", "split_blocks", "load_ipython_extension"]


def load_ipython_extension(ip):
    """%load_ext smartrun  ->  %%smartrun cell magic (cells run as cached, parallel blocks)."""
    sess = Session()

    def smartrun(line, cell):
        sess.run(cell, ns=ip.user_ns)
        print(sess.report())

    ip.register_magic_function(smartrun, "cell", "smartrun")
