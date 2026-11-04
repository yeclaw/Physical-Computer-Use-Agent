"""Convenient assembly of the existing RSI loop and an external device adapter.

This owns evidence registration and lifecycle records, not action/lesson choices.
No robot, model account or device-specific prompt is required for the mock demo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile
import uuid

from ..runtime.control_protocol import encoded, write_new
from ..runtime.device_adapter import json_copy, load_adapter, load_factory
from ..runtime.program_runtime import ProgramRuntime
from ..records.run_recording import freeze_python_tree
from ..records.redaction import redact_text
from ..skills.knowledge import KnowledgeStore
from .instructions import learning_instructions
from .learning_loop import run_episode


class _ClaimedOutput:
    """Internal ownership token; external paths still require exclusive creation."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=False)


class AgentRunner:
    """Bring a session and device; get the normal loop and persistent knowledge.

    ``session.run_turn(prompt, callback, images=...)`` uses the existing reply
    shape: {turn_id, final_text, usage}. Provider startup is owned by the caller;
    run() always records provider/device cleanup, including on exceptions.
    Environment/application callbacks are explicit keyword arguments to run().
    """

    def __init__(self, *, session, device, knowledge, scope, output, task, run_id=None):
        if type(task) is not str or not task.strip():
            raise ValueError('A nonempty task is required')
        claim = output if isinstance(output, _ClaimedOutput) else _ClaimedOutput(output)
        self.root = claim.path
        self.run_id = run_id or ('run-' + uuid.uuid4().hex)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', self.run_id):
            raise ValueError('Invalid run_id')
        self.session, self.device, self.task = session, device, task
        self.store = KnowledgeStore(Path(knowledge).resolve(), json_copy(scope), self.run_id)
        self.runtime = ProgramRuntime(device, self.root / 'runtime', run_id=self.run_id)
        self.output = self.root / 'agent'
        self.output.mkdir()
        import astra_robotics_agent
        package = Path(next(iter(astra_robotics_agent.__path__))).resolve()
        self.source = freeze_python_tree(package, self.root / 'source')
        write_new(self.root / 'run.json', encoded(dict(protocol='physical-agent-run-v1',
            run_id=self.run_id, task=task, scope=scope, adapter=self.runtime.description,
            source=self.source, task_success_verified=False)), readonly=True)

    def observe(self):
        result = self.runtime.observe()
        self.store.register_evidence(result['evidence_id'], result.pop('evidence_files'))
        result['contentItems'].insert(0, dict(type='inputText', text='Current task: ' + self.task))
        return result

    def execute(self, arguments):
        result = self.runtime.execute(arguments)
        self.store.register_evidence(result['evidence_id'], result.pop('evidence_files'))
        return result

    def run(self, **loop_options):
        # No fixed total decisions/tokens and no hidden resetting of knowledge.
        # Explicit caller limits remain available for offline fixtures.
        error = None
        try:
            options = dict(skill_development=True, continual_learning=True)
            options.update(loop_options)
            if set(options) & {'observe', 'execute', 'store', 'output', 'session'}:
                raise ValueError('Core callbacks are owned by AgentRunner')
            report = run_episode(session=self.session, observe=self.observe, execute=self.execute,
                store=self.store, output=self.output, **options)
            write_new(self.root / 'report.json', encoded(report), readonly=True)
            return report
        except BaseException as exc:
            error = redact_text(repr(exc))
            raise
        finally:
            provider_cleanup = None
            try:
                close = getattr(self.session, 'close', None)
                if close is not None:
                    close()
            except BaseException as exc:
                provider_cleanup = redact_text(repr(exc))
            cleanup = self.runtime.close()
            write_new(self.root / 'cleanup.json', encoded(dict(
                original_error=error, provider_cleanup_error=provider_cleanup, runtime=cleanup)), readonly=True)


def run_config(config, *, output):
    """Operator-selected trusted factories; no dynamic imports chosen by models."""
    if (type(config) is not dict or set(config) != {'task', 'scope', 'knowledge', 'device', 'model'}):
        raise ValueError('Config requires task/scope/knowledge/device/model')
    for key in ('device', 'model'):
        value = config[key]
        if type(value) is not dict or set(value) != {'factory', 'settings'} or type(value['settings']) is not dict:
            raise ValueError(key + ' requires factory and settings')
    claim = _ClaimedOutput(output)
    root = claim.path
    device = None
    session = None
    runner = None
    original_error = None
    workspace = tempfile.TemporaryDirectory(prefix='physical-agent-provider-')
    try:
        device = load_adapter(config['device']['factory'], config['device']['settings'])
        instructions = learning_instructions(skill_development=True, continual_learning=True)
        session = load_factory(config['model']['factory'])(json_copy(config['model']['settings']),
            instructions=instructions, cwd=Path(workspace.name).resolve(), output=root/'provider')
        runner = AgentRunner(session=session, device=device, knowledge=config['knowledge'],
                             scope=config['scope'], output=claim, task=config['task'])
        return runner.run()
    except BaseException as exc:
        original_error = exc
        raise
    finally:
        # run() closes normal ownership. Also handle provider/constructor failure.
        if runner is None:
            failures = []
            for obj in (session, device):
                if obj is not None:
                    try:
                        obj.close()
                    except BaseException as exc:
                        failures.append(redact_text(repr(exc)))
            if failures:
                if original_error is not None:
                    original_error.add_note('Startup cleanup also failed: ' + '; '.join(failures))
                else:
                    raise RuntimeError('Startup cleanup failed: ' + '; '.join(failures))
            if original_error is not None:
                try:
                    root.mkdir(parents=True, exist_ok=True)
                    write_new(root / 'startup-failure.json', encoded(dict(
                        error=redact_text(repr(original_error)), cleanup_errors=failures)), readonly=True)
                except BaseException as record_error:
                    original_error.add_note('Startup failure recording also failed: ' + redact_text(repr(record_error)))
        workspace.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--knowledge', type=Path, help='Override the persistent knowledge directory')
    args = parser.parse_args(argv)
    from ..runtime.transport import strict_loads
    config = strict_loads(args.config.read_bytes())
    if args.knowledge is not None:
        config['knowledge'] = str(args.knowledge)
    result = run_config(config, output=args.output)
    print(json.dumps(dict(output=str(args.output.resolve()),
        finished_by_model=result.get('finished_by_model'), stable_promoted=result.get('stable_promoted'))))


if __name__ == '__main__':
    main()
