"""Optional host-only telemetry for the pinned TensorFold DeepSeek 1017 batcher.

Records the *existing* token-list queue puts, not retokenized SSE text. It does
not alter plans, sampling, queue items, model operations or rank communication.
The upstream modules remain unmodified; install before constructing the engine.
Jay Leaton's MIT family implementation and TensorFold are the observed backend.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import json
from pathlib import Path
import threading
import time
import sys
import weakref

PREFIX = 'tensorfold.'
PINS = {
    'families.deepseek_v41.cuda.batch': '040f41e9ce4ef8ff3ffd20e19e20d4648d1978d3beccdeb1b0b3481ae205d3ca',
    'families.deepseek_v41.cuda.rounds': 'c85302535305ae1046bf0aaffa368f0e74212ff21e7f8039606f010b281c0acb',
    'families.glm5_next.spark.health': '6bd74b6970f83629bb770841e07d0428bd48d31f1c34f6168394b2fe218fc688',
}
SCHEMA = 1
COUNTERS = ('prefill_tokens', 'prefill_finalizations', 'prefill_rounds',
            'decode_rounds', 'verified_rows', 'dspark_windows', 'lookup_windows', 'serial_windows',
            'rounds_started', 'rounds_completed', 'round_errors')


def source_identity(modules):
    actual = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
              for name, module in modules.items()}
    if actual != PINS:
        raise ValueError('exact DeepSeek 1017 batch/rounds/health sources required')
    return actual


def check_tree(root):
    root = Path(root)
    actual = {name: hashlib.sha256((root/(name.replace('.', '/')+'.py')).read_bytes()).hexdigest()
              for name in PINS}
    if actual != PINS:
        raise ValueError('exact installed 1017 phase-observer sources required')
    return actual


class State:
    def __init__(self, *, clock=time.monotonic):
        self.lock = threading.Lock()
        self.clock = clock
        self.values = dict.fromkeys(COUNTERS, 0)
        self.active = False
        self.rows_histogram = {}
        self.draft_slots_histogram = {}

    def snapshot(self):
        with self.lock:
            return {'schema': SCHEMA, **self.values, 'executor_active': self.active,
                    'verified_rows_histogram': dict(self.rows_histogram),
                    'dspark_padded_slots_histogram': dict(self.draft_slots_histogram)}

    def execute(self, original, executor, plan):
        # Count conservatively before execution. A failed/restored piece can
        # overcount; it can never silently make a mixed interval look pure.
        pieces = list(plan.pieces)
        tokens = sum(int(end)-int(start) for _, start, end in pieces)
        finals = len(plan.finals)
        if tokens < 0:
            raise ValueError('negative prefill piece')
        with self.lock:
            if self.active:
                raise RuntimeError('unexpected concurrent Executor.run')
            self.active = True
            self.values['rounds_started'] += 1
            self.values['prefill_tokens'] += tokens
            self.values['prefill_finalizations'] += finals
            self.values['prefill_rounds'] += int(bool(pieces or finals))
        try:
            result = original(executor, plan)
        except BaseException:
            with self.lock:
                self.values['round_errors'] += 1
            raise
        else:
            with self.lock:
                count = sum(len(row.tokens) for row in result.rows)
                self.values['decode_rounds'] += int(bool(result.rows))
                self.values['verified_rows'] += count
                if count:
                    key = str(count)
                    self.rows_histogram[key] = self.rows_histogram.get(key, 0)+1
                for window in plan.windows:
                    kind = 'dspark_windows' if window.depth > 0 else 'lookup_windows' if window.drafts else 'serial_windows'
                    self.values[kind] += 1
                slots = sum(window.depth > 0 for window in plan.windows)
                if slots:
                    key = str(1 << (slots-1).bit_length())
                    self.draft_slots_histogram[key] = self.draft_slots_histogram.get(key, 0)+1
                self.values['rounds_completed'] += 1
            return result
        finally:
            with self.lock:
                self.active = False


class ObservedQueue:
    """Transparent queue proxy. The original object receives the identical item."""
    def __init__(self, original, state, stats, *, limit=4096, sessions_disabled=False):
        self.original, self.state, self.stats = original, state, stats
        self.limit = limit
        self.emitted = 0
        self.trace = []
        self.overflow = False
        self.sessions_disabled = sessions_disabled

    def put(self, item, *args, **kwargs):
        if isinstance(item, list) and item and all(type(token) is int for token in item):
            self.emitted += len(item)
            if len(self.trace) < self.limit:
                self.trace.append({'at_s': self.state.clock(), 'count': len(item),
                                   'emitted': self.emitted, 'phase': self.state.snapshot()})
            else:
                self.overflow = True
        if item is None:
            self.stats['deepseek_phase'] = {
                'schema': SCHEMA, 'source_sha256': dict(PINS),
                'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'clock': 'server_time.monotonic', 'emitted_tokens': self.emitted,
                'session_store_disabled': self.sessions_disabled,
                'truncated': self.overflow, 'emissions': list(self.trace),
                'end': self.state.snapshot()}
        return self.original.put(item, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


def graph_view(executor):
    """Copy existing host metadata; no graph access/replay or CUDA query."""
    fwd = getattr(executor, 'fwd', None)
    drafter = getattr(fwd, 'drafter', None)
    result = {}
    for name, graphs in (('target_graphs', getattr(fwd, 'graphs', None)),
                         ('draft_graphs', getattr(drafter, 'graphs', None))):
        cache = getattr(graphs, 'cache', None)
        if cache is not None:
            graph_keys = dict(cache.graphs)
            result[name] = {'stats': dict(cache.stats), 'on': bool(cache.on),
                            'held': len(graph_keys), 'keys': sorted(map(repr, graph_keys)),
                            'max_slots': getattr(graphs, 'max_slots', getattr(cache, 'slots', None))}
    spec = getattr(fwd, 'spec', None)
    if spec is not None:
        result['speculative_draft'] = dict(spec.stats)
    return result


class Context:
    """Process-lifetime startup hook; exit only after the engine has stopped."""
    def __init__(self, modules=None, *, trace_limit=4096, clock=time.monotonic):
        if type(trace_limit) is not int or not 1 <= trace_limit <= 4096:
            raise ValueError('trace_limit must be in 1..4096')
        self.modules = modules
        self.trace_limit, self.clock = trace_limit, clock
        self.states = weakref.WeakKeyDictionary()
        self.lock = threading.Lock()
        self.saved = []

    def state(self, executor):
        with self.lock:
            if executor not in self.states:
                self.states[executor] = State(clock=self.clock)
            return self.states[executor]

    def __enter__(self):
        if self.saved:
            raise RuntimeError('observer already installed')
        modules = self.modules or {name: importlib.import_module(PREFIX+name) for name in PINS}
        source_identity(modules)
        batch = modules['families.deepseek_v41.cuda.batch'].Batcher
        executor = modules['families.deepseek_v41.cuda.rounds'].Executor
        health = modules['families.glm5_next.spark.health']
        original_run, original_submit, original_view = executor.run, batch.submit, health.engine_view
        if any(getattr(fn, '_deepseek_phase_observer', False)
               for fn in (original_run, original_submit, original_view)):
            raise RuntimeError('another phase observer is installed')

        @functools.wraps(original_run)
        def run(ex, plan):
            return self.state(ex).execute(original_run, ex, plan)

        @functools.wraps(original_submit)
        def submit(owner, job):
            if isinstance(job.out, ObservedQueue):
                raise RuntimeError('job queue already observed')
            old = job.out
            observed = ObservedQueue(old, self.state(owner.ex), job.stats, limit=self.trace_limit,
                                     sessions_disabled=owner.store is None)
            job.out = observed
            try:
                return original_submit(owner, job)
            except BaseException:
                job.out = old
                raise

        @functools.wraps(original_view)
        def view(app):
            out = original_view(app)
            owner = getattr(getattr(app, 'engine', None), 'batch', None)
            ex = getattr(owner, 'ex', None)
            if ex is not None:
                out['deepseek_phase'] = {'source_sha256': dict(PINS),
                                         'session_store_disabled': owner.store is None,
                                         **self.state(ex).snapshot(), **graph_view(ex)}
            return out

        for obj, name, replacement in ((executor, 'run', run), (batch, 'submit', submit),
                                       (health, 'engine_view', view)):
            replacement._deepseek_phase_observer = True
            self.saved.append((obj, name, getattr(obj, name), replacement))
            setattr(obj, name, replacement)
        return self

    def __exit__(self, exc_type, exc, tb):
        if any(state.snapshot()['executor_active'] for state in self.states.values()):
            raise RuntimeError('stop the engine before removing the phase observer')
        foreign = any(getattr(obj, name) is not replacement for obj, name, _, replacement in self.saved)
        for obj, name, original, replacement in reversed(self.saved):
            if getattr(obj, name) is replacement:
                setattr(obj, name, original)
        self.saved.clear()
        if foreign:
            raise RuntimeError('foreign phase-hook replacement; it was preserved')


def main():
    if sys.argv[1:2] == ['--check-sources']:
        if len(sys.argv) != 3:
            raise ValueError('--check-sources requires the installed tensorfold package directory')
        print(json.dumps({'source_sha256': check_tree(sys.argv[2]), 'gpu_execution': False}))
        return 0
    # Same CLI/parser and original graph/engine lifecycle as python -m tensorfold.
    with Context():
        from tensorfold.cli import main as upstream
        return upstream()


if __name__ == '__main__':
    raise SystemExit(main())
