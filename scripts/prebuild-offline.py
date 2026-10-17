"""Compile for the known GB10 target without initializing a CUDA device."""
import runpy
from tensorfold.cuda import build
build.arch_flags = lambda need=build.MIN_CAPABILITY: ['-gencode=arch=compute_121,code=sm_121']
runpy.run_path('/prebuild.py',run_name='__main__')
