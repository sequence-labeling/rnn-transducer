"""setup.py script for rnn-transducer TensorFlow wrapper"""

from __future__ import print_function

import os
import platform
import re
import setuptools
import sys
import unittest
from setuptools.command.build_ext import build_ext as orig_build_ext

# We need to import tensorflow to find where its include directory is.
from distutils.version import LooseVersion
try:
    import tensorflow as tf
except ImportError:
    raise RuntimeError("Tensorflow must be installed to build the tensorflow wrapper.")

if "CUDA_HOME"  in os.environ:
    print("CUDA_HOME is found in the environment,but we have not implement a gpu version",
          file=sys.stderr)
#    enable_gpu = False
#else:
 #   enable_gpu = True


if "TENSORFLOW_SRC_PATH" not in os.environ:
    print("Please define the TENSORFLOW_SRC_PATH environment variable.\n"
          "This should be a path to the Tensorflow source directory.",
          file=sys.stderr)
    sys.exit(1)

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

rnn_transducer_path = "../build"
if "RNN_TRANSDUCER_PATH" in os.environ:
    rnn_transducer_path = os.environ["RNN_TRANSDUCER_PATH"]
print(os.path.join(rnn_transducer_path, "libtrasducer"+lib_ext))
if not os.path.exists(os.path.join(rnn_transducer_path, "libtransducer"+lib_ext)):
    print(("Could not find libtransducer.so in {}.\n"
           "Build rnn-transducer and set RNN_TRANSDUCER_PATH to the location of"
           " libtransducer.so (default is '../build')").format(rnn_transducer_path),
          file=sys.stderr)
    sys.exit(1)

root_path = os.path.realpath(os.path.dirname(__file__))

tf_include = tf.sysconfig.get_include()
tf_src_dir = os.environ["TENSORFLOW_SRC_PATH"]
tf_includes = [tf_include, tf_src_dir]
rnn_transducer_includes = [os.path.join(root_path, '../include')]
include_dirs = tf_includes + rnn_transducer_includes
library_dirs =[rnn_transducer_path]
libraries = ['transducer']
extra_compile_args = ['-std=c++11', '-fPIC']
# current tensorflow code triggers return type errors, silence those for now
extra_compile_args += ['-Wno-return-type']
if LooseVersion(tf.__version__)>=LooseVersion("1.4.0"):
   include_dirs+=[tf_include+"//external/nsync/public"]
   library_dirs+=[tf.sysconfig.get_lib()]
   libraries+=["tensorflow_framework"]   
   extra_compile_args += [ '-D_GLIBCXX_USE_CXX11_ABI=0']
   print("tensorflow version >= 1.4.0")     
"""
if (enable_gpu):
    extra_compile_args += ['-DWARPCTC_ENABLE_GPU']
    include_dirs += [os.path.join(os.environ["CUDA_HOME"], 'include')]

    # mimic tensorflow cuda include setup so that their include command work
    if not os.path.exists(os.path.join(root_path, "include")):
        os.mkdir(os.path.join(root_path, "include"))

    cuda_inc_path = os.path.join(root_path, "include/cuda")
    if not os.path.exists(cuda_inc_path) or os.readlink(cuda_inc_path) != os.environ["CUDA_HOME"]:
        if os.path.exists(cuda_inc_path):
            os.remove(cuda_inc_path)
        os.symlink(os.environ["CUDA_HOME"], cuda_inc_path)
    include_dirs += [os.path.join(root_path, 'include')]
"""
# Ensure that all expected files and directories exist.
for loc in include_dirs:
    if not os.path.exists(loc):
        print(("Could not find file or directory {}.\n"
               "Check your environment variables and paths?").format(loc),
              file=sys.stderr)
        sys.exit(1)

lib_srcs =['src/transducer_op.cc']

ext = setuptools.Extension('transducer_tensorflow.kernels',
                           sources = lib_srcs,
                           language = 'c++',
                           include_dirs = include_dirs,
                           library_dirs =library_dirs,
                           runtime_library_dirs = [os.path.realpath(rnn_transducer_path)],
                           libraries = libraries,
                           extra_compile_args = extra_compile_args)

class build_tf_ext(orig_build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        orig_build_ext.build_extensions(self)

def discover_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

# Read the README.md file for the long description. This lets us avoid
# duplicating the package description in multiple places in the source.
README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
with open(README_PATH, "r") as handle:
    # Extract everything between the first set of ## headlines
    LONG_DESCRIPTION = re.search("#.*([^#]*)##", handle.read()).group(1).strip()

setuptools.setup(
    name = "rnn_transucer_tensorflow",
    version = "0.1",
    description = "TensorFlow wrapper for rnn-transducer",
    long_description = LONG_DESCRIPTION,
    author = "dumiao",
    url = "https://github.com/sequence-labeling/rnn-transducer",
    license = "Apache",
    packages = ["transducer_tensorflow"],
    ext_modules = [ext],
    cmdclass = {'build_ext': build_tf_ext},
    test_suite = 'setup.discover_test_suite',
)
