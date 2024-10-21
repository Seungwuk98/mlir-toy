import os

import lit.formats

from lit.llvm import llvm_config

config.name = "Toy MLIR Test"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [
        '.mlir',
        '.toy',
        ]


config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.toy_obj_root, 'test', 'LitTests')


llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.toy_tools_dir, append_path=True)


