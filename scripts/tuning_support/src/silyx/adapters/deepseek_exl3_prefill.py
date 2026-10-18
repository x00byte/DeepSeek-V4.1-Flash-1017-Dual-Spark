"""Default-off NT2/MTL2 geometry over Jay Leaton's exact EXL3 x3pf kernel.

Only the dispatcher instantiation and private symbol namespace change. The
original device templates, K ranges, half operands, MMA chains and warp-order
FP32 reductions are retained verbatim when generating the isolated extension.
No backend import, environment mutation or compilation occurs on import.

The original family files retain Jay Leaton's MIT/SPDX notice; TensorFold and
ExLlamaV3 supply the included EXL3 helpers. See docs/deepseek-transfer-audit.md.
This candidate remains disabled. The bounded actual-layer GPU experiment
passed component word comparisons but regressed latency on layers 3 and 20;
see results/2026-10-18/deepseek-exl3-nt2-mtl2/summary.json. Full-model quality
and serving throughput are unqualified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

MODEL = 'LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw'
REVISION = 'afc7aa0aab98ee735c0f54e0a122019af1ac08f1'
METADATA_SHA256 = {
    'config.json':'eb48f2216645b36f400036fa2d67d8ad6b48233b0f4a1bcb7e99fa6450dad253',
    'model.safetensors.index.json':'778baeaf9fbcd351800536dc090ff1edddb72d102472b08de0edd9a52cb96656',
}
SOURCE_SHA256 = {
    'families/deepseek_v41/cuda/x3pf.cu':'5aedfee754a31031e5fe956f2f95ef5f121deb43c2b90acd745bb59c4246b35b',
    'families/deepseek_v41/cuda/x3pf.cpp':'a85498df311ec2c7fc7109428d4842e8fe0c5dd84facfc972a8e5e671059c93c',
    'cuda/exl3/experts_grouped.cuh':'cb2a016df23375db577ca2d9199ba145eb176e71a22295bdf386caba1cebff22',
    'cuda/exl3/experts.py':'ed980055a0c039a0f6f46a3853f813545512329f4e022a0e966858f4ceff4525',
}
OLD_DISPATCH = '''    if (nt == 4 && mtl == 4) ok = dispatch_ranges<4, 4>(a, stream, (int)lo, (int)hi);
    else if (nt == 8 && mtl == 2) ok = dispatch_ranges<8, 2>(a, stream, (int)lo, (int)hi);'''
NEW_DISPATCH = '    if (nt == 2 && mtl == 2) ok = dispatch_ranges<2, 2>(a, stream, (int)lo, (int)hi);'


@dataclass(frozen=True)
class Geometry:
    nt: int = 2
    member_tiles: int = 2
    warps: int = 4

    def __post_init__(self):
        if (type(self.nt)is not int or type(self.member_tiles)is not int
                or type(self.warps)is not int or (self.nt,self.member_tiles,self.warps)!=(2,2,4)):
            raise ValueError('only the isolated exact NT2/MTL2/four-K-warp candidate is supported')

    @property
    def accumulator_scalars_per_thread(self):
        return self.member_tiles*self.nt*2*4

    @property
    def declared_shared_bytes(self):
        # Source rows_sh[MTL*16] + red[W][16][NT*16], before compiler padding.
        return self.member_tiles*16*4+self.warps*16*self.nt*16*4


def verify_sources(root):
    root=Path(root)
    actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_SHA256}
    if actual!=SOURCE_SHA256:raise ValueError('pinned patched TensorFold EXL3/x3pf sources required')
    return actual


def verify_checkpoint(model):
    model=Path(model)
    actual={name:hashlib.sha256((model/name).read_bytes()).hexdigest() for name in METADATA_SHA256}
    if actual!=METADATA_SHA256:raise ValueError('exact1017 checkpoint config/index required')
    config=json.loads((model/'config.json').read_bytes())
    fields=dict(hidden_size=5120,num_hidden_layers=40,n_routed_experts=384,
                num_experts_per_tok=6,moe_intermediate_size=2304,n_shared_experts=1)
    text=config.get('text_config',config)
    if any(text.get(key)!=value for key,value in fields.items()):
        raise ValueError('reviewed1017 expert geometry required')
    return actual


def specialize(cuda_source,binding_source):
    """Return private extension sources, preserving the original device body."""
    paths=('families/deepseek_v41/cuda/x3pf.cu','families/deepseek_v41/cuda/x3pf.cpp')
    for source,path in zip((cuda_source,binding_source),paths):
        if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA256[path]:
            raise ValueError('unreviewed x3pf source or binding')
    if cuda_source.count(OLD_DISPATCH)!=1:
        raise ValueError('exact sole x3pf dispatcher required')
    cuda_source=cuda_source.replace(OLD_DISPATCH,NEW_DISPATCH)
    # Keep the public TensorFold extension untouched and avoid interposition
    # between differently instantiated DSOs in a process.
    cuda_source=cuda_source.replace('namespace dsv41_x3pf','namespace silyx_dsv41_x3pf_nt2_mtl2')
    cuda_source=cuda_source.replace('using namespace dsv41_x3pf','using namespace silyx_dsv41_x3pf_nt2_mtl2')
    symbol='dsv41_x3pf_grouped_cuda'
    return (cuda_source.replace(symbol,'silyx_'+symbol+'_nt2_mtl2'),
            binding_source.replace(symbol,'silyx_'+symbol+'_nt2_mtl2'))


def k_ranges(inputs,splits):
    if type(inputs)is not int or type(splits)is not int or splits<1 or inputs%(16*splits*4):
        raise ValueError('original four-K-warp split geometry required')
    count=inputs//16//splits//4
    return tuple(tuple(range((split*4+warp)*count,(split*4+warp+1)*count))
                 for split in range(splits) for warp in range(4))


def output_ownership(members,*,max_rows,slots,columns,matrices,splits):
    """CPU candidate grid ownership only, not a floating-point oracle.

    Members use the original row<<5|slot encoding. Yields each live Z address
    once, with its pair row and output column, following the NT2/MTL2 grid.
    """
    if (type(max_rows)is not int or not 1<=max_rows<=1024 or slots!=7
            or columns%32 or columns<32 or matrices not in (1,2) or splits<1):
        raise ValueError('bounded1024-row top6+shared candidate geometry required')
    pairs=max_rows*slots
    for block_column in range(columns//32):
        for matrix in range(matrices):
            for split in range(splits):
                for first in range(0,max_rows,32):
                    for mt in range(2):
                        for lane_row in range(16):
                            position=first+mt*16+lane_row
                            code=members[position] if position<len(members) else -1
                            if code<0:continue
                            row,slot=code>>5,code&31
                            if row>=max_rows or slot>=slots:raise ValueError('invalid original member code')
                            pair=row*slots+slot
                            for offset in range(32):
                                column=block_column*32+offset
                                address=((matrix*splits+split)*pairs+pair)*columns+column
                                yield address,pair,column,matrix,split


def memory_plan(*,rows=1024,experts=384,bits=3):
    """Conservative full-layer owned-byte plan; no device allocation/probe."""
    if (type(rows)is not int or rows!=1024 or experts!=384 or bits not in (2,3)):
        raise ValueError('full384-expert1024-row target component required')
    dims,width,slots=5120,1152,7
    pairs=rows*slots
    # Three TP-sliced trellises per expert plus one5-bit shared expert.
    trellis=3*dims*width*(experts*bits+5)//8
    # Per-expert six fp16 Hadamard scale vectors, conservative whole-D/I sum.
    scales=(experts+1)*3*(dims+width)*2
    # Original fixed configs: gate/up split4, down split1. Keep one scratch
    # shared by all variants, with explicit reference partial/activation copies.
    scratch=(2*pairs*dims*2+pairs*width*2+
             max(2*4*width,dims)*pairs*4+pairs*dims*4+
             (experts+1)*rows*4)
    reference=max(2*4*width,dims)*pairs*4+pairs*width*2+2*rows*dims*4
    return dict(trellis_bytes=trellis,scale_bytes=scales,scratch_bytes=scratch,
                reference_and_output_bytes=reference,
                owned_peak_estimate_bytes=trellis+scales+scratch+reference+128*1024**2,
                allocator_cap_bytes=5*1024**3,
                initial_cuda_and_host_reserve_bytes=8*1024**3,
                final_cuda_and_host_reserve_bytes=2*1024**3,
                estimate_excludes_context_and_CPU_loader_peak=True,
                admit_only_from_observed_after_context_memory=True)


def prepare(tf_root,model,cache,*,offline_compile=False):
    """Explicit private extension compile; no serving or upstream mutation."""
    tf_root,cache=Path(tf_root),Path(cache)
    verify_sources(tf_root);verify_checkpoint(model)
    if offline_compile and (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('MAX_JOBS')!='1'):
        raise RuntimeError('offline compile requires hidden CUDA and MAX_JOBS=1')
    import torch
    if offline_compile:
        if torch.cuda.is_initialized():raise RuntimeError('offline compiler must not initialize CUDA')
    elif torch.cuda.is_current_stream_capturing():
        raise RuntimeError('explicit extension preparation before capture required')
    family=tf_root/'families/deepseek_v41/cuda'
    cuda,binding=specialize((family/'x3pf.cu').read_text(),(family/'x3pf.cpp').read_text())
    identity=hashlib.sha256((cuda+binding).encode()).hexdigest()
    directory=cache/('silyx_exl3_nt2_mtl2_'+identity[:16]);directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name,text in (('kernel.cu',cuda),('binding.cpp',binding)):
        path=directory/name
        if path.exists() and path.read_text()!=text:raise RuntimeError('private generated source changed')
        path.write_text(text);paths.append(str(path))
    flags=['-O3','-lineinfo','-Xptxas=-v']
    if offline_compile:
        from torch.utils.cpp_extension import load
        module=load(name='silyx_exl3_nt2_mtl2_'+identity[:16],sources=paths,
            extra_include_paths=[str(tf_root/'cuda/exl3')],extra_cuda_cflags=flags+[
                '-gencode=arch=compute_121,code=sm_121'],verbose=True)
        if torch.cuda.is_initialized():raise RuntimeError('offline compile initialized CUDA')
    else:
        from tensorfold.cuda.build import load
        module=load(name='silyx_exl3_nt2_mtl2_'+identity[:16],sources=paths,
            extra_include_paths=[str(tf_root/'cuda/exl3')],extra_cuda_cflags=flags,verbose=False)
    module.silyx_identity=dict(source_sha256=SOURCE_SHA256,generated_source_sha256=identity,
        model=MODEL,revision=REVISION,nt=2,member_tiles=2,warps=4,
        checkpoint_metadata_sha256=METADATA_SHA256,gpu_qualified=False)
    return module
