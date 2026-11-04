"""SDK-free bounded numerical analysis in a separate interpreter process.

The accepted language is the existing AST subset, not arbitrary Python. No
physical bridge is constructed. A result is a candidate, never an authority.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

from astra_robotics_agent.physical_keyboard.records.learning_contracts import (
    PROTOCOLS, LearningContractError, clone, encoded,
)


class AnalysisError(LearningContractError):
    def __init__(self, category, message):
        self.category = category
        super().__init__(category + ': ' + str(message)[:1024])


@dataclass(frozen=True)
class AnalysisLimits:
    max_input_bytes: int = 1024 * 1024
    max_source_bytes: int = 65536
    max_result_bytes: int = 32768
    max_steps: int = 200000
    max_iterations: int = 20000
    max_container_items: int = 65536
    max_string_chars: int = 65536
    wall_time_s: float = 5.0
    memory_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        maxima = dict(max_input_bytes=1024*1024, max_source_bytes=65536,
                      max_result_bytes=32768, max_steps=2000000,
                      max_iterations=200000, max_container_items=65536,
                      max_string_chars=65536, memory_bytes=1024*1024*1024)
        for name, maximum in maxima.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise AnalysisError('invalid_budget', name)
        if (type(self.wall_time_s) not in (int, float)
                or not math.isfinite(self.wall_time_s) or not 0 < self.wall_time_s <= 60):
            raise AnalysisError('invalid_budget', 'wall_time_s')


_WRAPPER = '\ndef main(robot, profile):\n    return analyze(profile["data"], profile["parameters"])\n'
_RESERVED = {'robot', 'profile', 'main', 'ToolError'}


def _source(lines, limits):
    if type(lines) is not list or not lines or any(type(line) is not str or '\n' in line or '\r' in line for line in lines):
        raise AnalysisError('invalid_source', 'source_lines must be individual lines')
    source = '\n'.join(lines) + '\n'
    try:
        if len((source + _WRAPPER).encode('utf-8')) > limits.max_source_bytes:
            raise AnalysisError('source_limit', 'source including wrapper exceeds byte budget')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError('invalid_source', str(exc)) from exc
    analyses = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze']
    if len(analyses) != 1 or [n.arg for n in analyses[0].args.args] != ['data', 'parameters']:
        raise AnalysisError('invalid_source', 'define analyze(data, parameters)')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise AnalysisError('forbidden_capability', 'imports are unavailable')
        name = node.id if isinstance(node, ast.Name) else node.arg if isinstance(node, ast.arg) else node.name if isinstance(node, ast.FunctionDef) else None
        if name in _RESERVED:
            raise AnalysisError('forbidden_capability', 'reserved host name: ' + name)
        if isinstance(node, ast.Attribute) and not (isinstance(node.value, ast.Name) and node.value.id == 'math'):
            raise AnalysisError('forbidden_capability', 'only listed math attributes are available')
    return source


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    encoded(value, 2*1024*1024)
    return value


def run_analysis(source_lines, data, parameters, *, limits=None):
    """Analyze finite host-selected JSON, or raise a classified AnalysisError.

    Callers bind source IDs to the result; analysis output cannot claim them.
    Inputs are cloned before spawning. No job is retried or enlarged.
    """
    if limits is None:
        limits = AnalysisLimits()
    elif type(limits) is dict:
        try:
            limits = AnalysisLimits(**limits)
        except TypeError as exc:
            raise AnalysisError('invalid_budget', str(exc)) from exc
    if not isinstance(limits, AnalysisLimits):
        raise AnalysisError('invalid_budget', 'AnalysisLimits or explicit fields required')
    source = _source(source_lines, limits)
    try:
        input_raw = encoded(data, limits.max_input_bytes)
        parameters_raw = encoded(parameters, limits.max_input_bytes)
        payload = encoded({'data': data, 'parameters': parameters}, limits.max_input_bytes)
        # Fail before launching if the interpreter input nesting/container guard
        # cannot accept the request. This does not change physical Limits.
        from astra_robotics_agent.physical_keyboard.runtime.skill_language import Limits, _Interpreter
        _Interpreter({}, Limits(max_container_items=limits.max_container_items,
                               max_string_chars=limits.max_string_chars)).check({'data': data, 'parameters': parameters})
    except (ValueError, TypeError, RecursionError) as exc:
        category = 'container_limit' if 'item budget' in str(exc) else 'input_limit'
        raise AnalysisError(category, str(exc)) from exc
    job = encoded({'source_lines': list(source_lines), 'input': _loads(payload), 'limits': asdict(limits)}, 2*1024*1024)
    src_root = str(Path(__file__).resolve().parents[3])
    bootstrap = ('import sys;sys.path.insert(0,sys.argv[1]);'
                 'from astra_robotics_agent.physical_keyboard.runtime.analysis_worker import _child;_child()')
    try:
        completed = subprocess.run([sys.executable, '-I', '-c', bootstrap, src_root], input=job,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=limits.wall_time_s,
                                   check=False, cwd=Path(__file__).resolve().parent)
    except subprocess.TimeoutExpired as exc:
        raise AnalysisError('timeout', 'analysis wall deadline reached; no retry') from exc
    if len(completed.stdout) > limits.max_result_bytes + 16384 or len(completed.stderr) > 16384:
        raise AnalysisError('worker_output_limit', 'worker output exceeds envelope budget')
    try:
        message = _loads(completed.stdout)
    except (ValueError, UnicodeError, RecursionError) as exc:
        category = 'memory_limit' if completed.returncode < 0 else 'worker_failure'
        raise AnalysisError(category, 'worker ended without a valid result') from exc
    if message.get('status') != 'ok' or completed.returncode != 0:
        raise AnalysisError(message.get('category', 'worker_failure'), message.get('message', 'worker failed'))
    result_raw = encoded(message['result'], limits.max_result_bytes)
    return dict(protocol=PROTOCOLS['analysis'], result=message['result'],
                input_sha256=_hash(input_raw), parameters_sha256=_hash(parameters_raw),
                source_sha256=_hash(source.encode()), result_sha256=_hash(result_raw),
                budgets=asdict(limits), environment=message['environment'], authority='candidate_only')


def _child():
    """Fixed trusted subprocess entry; user code never runs Python bytecode."""
    from astra_robotics_agent.physical_keyboard.runtime.skill_language import (
        BudgetExceeded, Limits, SkillError, ValidationError, run_skill,
    )
    try:
        raw = sys.stdin.buffer.read(2*1024*1024 + 1)
        if len(raw) > 2*1024*1024:
            raise AnalysisError('input_limit', 'worker input exceeds budget')
        job = _loads(raw)
        limits = AnalysisLimits(**job['limits'])
        source = _source(job['source_lines'], limits)
        import resource
        import os
        memory_mechanism = 'RLIMIT_AS'
        if sys.platform == 'darwin':
            # Darwin rejects a small RLIMIT_AS below Python's mapped address
            # space. Bound resident memory in this otherwise finite language
            # with a process watchdog, and disclose its sampling granularity.
            import threading
            import time
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limits.memory_bytes:
                raise MemoryError('analysis budget is below the interpreter resident baseline')
            memory_mechanism = 'resident_watchdog_10ms'
            def watch_memory():
                while True:
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limits.memory_bytes:
                        os.write(1, b'{"status":"error","category":"memory_limit","message":"resident memory budget reached"}')
                        os._exit(71)
                    time.sleep(.01)
            threading.Thread(target=watch_memory, daemon=True).start()
        else:
            # RLIMIT_AS constrains new allocations; lowering it below already
            # mapped memory does not by itself kill a tiny allocation-free job.
            # Verify the Linux baseline before installing the kernel limit.
            if sys.platform.startswith('linux'):
                mapped = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
                if mapped > limits.memory_bytes:
                    raise MemoryError('analysis budget is below the interpreter address-space baseline')
            resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(limits.wall_time_s)), max(2, math.ceil(limits.wall_time_s)+1)))
        environment = dict(python=platform.python_version(), platform=sys.platform,
                           interpreter='bounded-python-ast', worker_pid=__import__('os').getpid(),
                           memory_limit=memory_mechanism, memory_enforcement='kernel_limit' if sys.platform != 'darwin' else 'sampled_resident_limit',
                           sdk_imported=False, callbacks=[], wrapper_sha256=_hash(_WRAPPER.encode()))
        values = clone(job['input'], limits.max_input_bytes)
        result = run_skill([('analysis.py', source), ('analysis-entry.py', _WRAPPER)], tools={},
                           profile=values, limits=Limits(max_steps=limits.max_steps,
                           max_iterations=limits.max_iterations, max_container_items=limits.max_container_items,
                           max_string_chars=limits.max_string_chars, max_source_bytes=limits.max_source_bytes))
        encoded(result, limits.max_result_bytes)
        message = dict(status='ok', result=result, environment=environment)
    except MemoryError:
        message = dict(status='error', category='memory_limit', message='analysis process memory budget reached')
    except BudgetExceeded as exc:
        category = ('output_limit' if 'result' in str(exc) else 'container_limit' if 'Container' in str(exc)
                    else 'iteration_limit' if 'iteration' in str(exc).lower() else 'step_limit')
        message = dict(status='error', category=category, message=str(exc))
    except AnalysisError as exc:
        message = dict(status='error', category=exc.category, message=str(exc))
    except (ValidationError, SkillError, ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        category = ('output_limit' if isinstance(exc, LearningContractError) and 'byte budget' in str(exc)
                    else 'invalid_source' if isinstance(exc, ValidationError) else 'analysis_failure')
        message = dict(status='error', category=category, message=str(exc)[:1024])
    sys.stdout.buffer.write(encoded(message, 49152))
    sys.stdout.buffer.flush()
