"""Reproducible synthetic ledger acceptance, with no model or robot execution."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from astra_robotics_agent.physical_keyboard import EvaluationPlan, Ledger, Scope, Usage
from astra_robotics_agent.physical_keyboard.contracts import canonical, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    source_paths = [*sorted((repository / 'src/astra_robotics_agent/physical_keyboard').rglob('*.py')),
                    repository / 'tests/test_keyboard_rsi_ledger.py', Path(__file__).resolve()]
    manifest = {'experiment': 'EXP-008', 'protocol': 'physical-keyboard-rsi-ledger-v1',
                'evidence_mode': 'synthetic', 'physical_improvement_claim': False,
                'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
                'command': sys.argv, 'seed': 0, 'random_decisions': False,
                'model': None, 'model_requests': 0, 'model_tokens': 0, 'hardware_initialized': False,
                'physical_actions': 0, 'source_sha256': {}, 'passed': False}
    for source in source_paths:
        relative = source.relative_to(repository)
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest['source_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_file = output / 'manifest.json'
    def save():
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    save()
    started = time.monotonic()
    try:
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p',
                   'test_keyboard_rsi_ledger.py', '-v']
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(repository / 'src')
        manifest['tests_command'] = command
        with (output / 'tests.log').open('x') as stream:
            tests = subprocess.run(command, cwd=repository, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        manifest['tests_exit_code'] = tests.returncode
        save()
        if tests.returncode:
            raise RuntimeError('Software tests failed; artifacts retained')

        scope = Scope('synthetic', 'fixture-keyboard', 'fixture-layout',
                      *[digest(s.encode()) for s in ['synthetic configuration', 'synthetic protocol',
                                                    'literal-text comparator v1', 'synthetic English',
                                                    'no-hardware safety']])
        with Ledger(output / 'ledger') as ledger:
            seed = ledger.freeze_bundle(label='synthetic-seed', kind='seed', scope=scope,
                files={'skill.py': b'# Synthetic seed fixture only; never executed.\n',
                       'profile.json': canonical({'synthetic': True, 'version': 0})},
                provenance={'origin': 'original', 'purpose': 'software acceptance fixture'})
            ledger.bootstrap(seed, reason='Software seed, not an acquired robotic skill')
            candidate = ledger.freeze_bundle(label='synthetic-candidate', kind='skill_program', scope=scope,
                files={'skill.py': b'# Synthetic candidate fixture only; never executed.\n',
                       'profile.json': canonical({'synthetic': True, 'version': 1})}, parent=seed,
                provenance={'origin': 'original', 'purpose': 'software acceptance fixture'})
            plan = EvaluationPlan({'probe-a': 'cat', 'probe-b': 'dog'}, 2, 1, 0)
            round_id = 'SOFTWARE-ONLY-008'
            ledger.open_round(round_id, candidate=candidate, plan=plan, generation_usage=Usage.zero(),
                metadata={'experiment_id': 'EXP-008', 'evidence_mode': 'synthetic', 'model': None,
                          'note': 'Manually constructed results test bookkeeping, not learning',
                          'code_revision': manifest['source_sha256']})
            for case_id, expected in plan.cases.items():
                for repetition in range(plan.repetitions):
                    for variant in ['parent', 'candidate']:
                        trial_id = f'{case_id}-{repetition}-{variant}'
                        before = ledger.put_evidence(canonical({'synthetic': True, 'stage': 'before', 'trial': trial_id}))
                        observed = expected if variant == 'candidate' else '?'
                        after = ledger.put_evidence(canonical({'synthetic': True, 'stage': 'after',
                                                               'trial': trial_id, 'observed': observed}))
                        ledger.start_trial(round_id, trial_id, variant=variant, case_id=case_id,
                            repetition=repetition, condition_id=f'{case_id}-{repetition}', before_evidence=[before])
                        ledger.finish_trial(round_id, trial_id, status='observed', observed_text=observed,
                            after_evidence=[after], execution_id='fixture-execution-' + trial_id, usage=Usage.zero())
            decision = ledger.decide(round_id)
            assert decision['verdict'] == 'accepted'
            assert ledger.verify()['stable'][scope.key] == candidate
            manifest['after_promotion'] = ledger.verify()
            ledger.rollback(scope=scope, target=seed, expected_current=candidate,
                            reason='Explicit software rollback acceptance', event_id='software-rollback')
            manifest['seed_bundle'] = seed
            manifest['candidate_bundle'] = candidate
            manifest['scope'] = asdict(scope)
            manifest['comparison_plan'] = asdict(plan)
        with Ledger(output / 'ledger') as reopened:
            report = reopened.export(output / 'export')
            assert report['stable'][scope.key] == seed
            assert report['rounds'][round_id]['decision']['verdict'] == 'accepted'
            assert report['overall_usage']['total_input_plus_output'] == 0
            manifest['reopen_verified'] = True
            manifest['final_report'] = report
        manifest['passed'] = True
    except BaseException as exc:
        manifest['error'] = repr(exc)
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest['files'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(output.rglob('*')) if p.is_file() and p != manifest_file}
        save()
    print(json.dumps({'passed': manifest['passed'], 'output': str(output),
                      'tests_exit_code': manifest.get('tests_exit_code'),
                      'events': manifest['final_report']['event_count'], 'synthetic_trials': 8,
                      'model_requests': 0, 'physical_actions': 0,
                      'manifest_sha256': hashlib.sha256(manifest_file.read_bytes()).hexdigest()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
