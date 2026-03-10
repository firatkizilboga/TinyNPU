import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
compiler_root = os.path.join(project_root, "software", "compiler")
if compiler_root not in sys.path:
    sys.path.append(compiler_root)

from software.workload.jit_test_gen import build_simple_chain_artifact


if __name__ == "__main__":
    test_case = build_simple_chain_artifact(seed=7, dim=16)
    print(test_case["artifact"].inspect(test_case["inputs"]))
