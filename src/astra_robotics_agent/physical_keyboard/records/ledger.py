"""Independent append-only RSI ledger with atomic decisions and replayed stable state.

This is an integrity/audit component, not an agent sandbox or physical evaluator.
It accepts observations from a trusted future evaluator; it cannot prove a key moved.
"""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile

from astra_robotics_agent.physical_keyboard.records.contracts import PROTOCOL, EvaluationPlan, Scope, Usage, aggregate_usage, canonical, digest, identifier, integer, sha256


class LedgerError(ValueError):
    pass


class IntegrityError(LedgerError):
    pass


class StaleParentError(LedgerError):
    pass


class Ledger:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / 'objects'
        if self.objects.is_symlink():
            raise IntegrityError('Object directory cannot be a symlink')
        self.objects.mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / 'events.sqlite3', timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL, payload TEXT NOT NULL, timestamp TEXT NOT NULL,
                previous_sha256 TEXT NOT NULL, sha256 TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT, 'append-only events'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT, 'append-only events'); END;
        ''')
        self.verify()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def put_evidence(self, content):
        if not isinstance(content, bytes):
            raise TypeError('Artifacts must be explicit bytes')
        key = digest(content)
        target = self.objects / key
        if target.exists() or target.is_symlink():
            if self.read_object(key) != content:
                raise IntegrityError('Conflicting object')
            return key
        descriptor, temporary = tempfile.mkstemp(prefix='.pending-', dir=self.objects)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)  # install completely written bytes, never overwrite
            except FileExistsError:
                self.read_object(key)
            directory = os.open(self.objects, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return key

    def read_object(self, key):
        sha256(key)
        path = self.objects / key
        if path.is_symlink() or not path.is_file():
            raise IntegrityError('Missing or non-regular object: ' + key)
        content = path.read_bytes()
        if digest(content) != key:
            raise IntegrityError('Artifact hash mismatch: ' + key)
        return content

    @contextmanager
    def _transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            state = self._replay()
            yield state
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _existing(self, event_id, kind, payload):
        row = self.db.execute('SELECT kind,payload FROM events WHERE event_id=?', (event_id,)).fetchone()
        if row is None:
            return False
        if row['kind'] != kind or row['payload'].encode() != canonical(payload):
            raise LedgerError('Conflicting idempotency key: ' + event_id)
        return True

    def _append(self, event_id, kind, payload):
        if self._existing(event_id, kind, payload):
            return
        last = self.db.execute('SELECT seq,sha256 FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        event = {'seq': 1 if last is None else last['seq'] + 1, 'event_id': event_id,
                 'kind': kind, 'payload': payload, 'timestamp': datetime.now(timezone.utc).isoformat(),
                 'previous_sha256': '0' * 64 if last is None else last['sha256']}
        self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',
                        (event['seq'], event_id, kind, canonical(payload).decode(), event['timestamp'],
                         event['previous_sha256'], digest(canonical(event))))

    def _bundle(self, bundle_id):
        try:
            value = json.loads(self.read_object(bundle_id))
            if value['protocol'] != PROTOCOL:
                raise IntegrityError('Unknown bundle protocol')
            Scope(**value['scope'])
            for name, entry in value['files'].items():
                self._filename(name)
                if len(self.read_object(entry['sha256'])) != entry['bytes']:
                    raise IntegrityError('Artifact size mismatch')
            return value
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise IntegrityError('Malformed bundle') from exc

    @staticmethod
    def _filename(name):
        if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
                or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                or str(PurePosixPath(name)) != name or name == '.'):
            raise LedgerError('Unsafe artifact path')

    def _replay(self):
        state = {'bundles': {}, 'rounds': {}, 'stable': {}, 'stable_history': {},
                 'execution_ids': set(), 'event_count': 0, 'head_sha256': '0' * 64}
        try:
            for row in self.db.execute('SELECT * FROM events ORDER BY seq'):
                event = {k: row[k] for k in ['seq', 'event_id', 'kind', 'timestamp', 'previous_sha256']}
                payload = json.loads(row['payload'])
                event['payload'] = payload
                if (row['seq'] != state['event_count'] + 1
                        or row['previous_sha256'] != state['head_sha256']
                        or canonical(payload).decode() != row['payload']
                        or digest(canonical(event)) != row['sha256']):
                    raise IntegrityError('Event chain mismatch')
                kind = row['kind']
                if kind == 'bundle_frozen':
                    state['bundles'][payload['bundle_id']] = self._bundle(payload['bundle_id'])
                elif kind == 'stable_bootstrap':
                    scope = payload['scope']
                    state['stable'][scope] = payload['bundle_id']
                    state['stable_history'].setdefault(scope, []).append(payload['bundle_id'])
                elif kind == 'round_opened':
                    state['rounds'][payload['round_id']] = {**payload, 'trials': {}, 'decision': None}
                elif kind == 'trial_started':
                    for ref in payload['before_evidence']:
                        self.read_object(ref)
                    state['rounds'][payload['round_id']]['trials'][payload['trial_id']] = {
                        **payload, 'result': None}
                elif kind == 'trial_finished':
                    for ref in payload['after_evidence']:
                        self.read_object(ref)
                    trial = state['rounds'][payload['round_id']]['trials'][payload['trial_id']]
                    trial['result'] = payload
                    state['execution_ids'].add(payload['execution_id'])
                elif kind == 'round_decided':
                    r = state['rounds'][payload['round_id']]
                    r['decision'] = payload
                    if payload['verdict'] == 'accepted':
                        if state['stable'].get(r['scope']) != r['parent']:
                            raise IntegrityError('Accepted event has stale parent')
                        state['stable'][r['scope']] = r['candidate']
                        state['stable_history'][r['scope']].append(r['candidate'])
                elif kind == 'stable_rollback':
                    if state['stable'].get(payload['scope']) != payload['expected_current']:
                        raise IntegrityError('Rollback chain mismatch')
                    state['stable'][payload['scope']] = payload['target']
                    state['stable_history'][payload['scope']].append(payload['target'])
                else:
                    raise IntegrityError('Unknown event kind')
                state['event_count'] += 1
                state['head_sha256'] = row['sha256']
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise IntegrityError('Malformed event history') from exc
        return state

    def freeze_bundle(self, *, label, kind, scope, files, provenance, parent=None):
        identifier(label)
        if kind not in {'seed', 'device_adaptation', 'skill_program', 'experiment_strategy'}:
            raise LedgerError('Unknown change kind')
        if not isinstance(scope, Scope) or not files or not isinstance(provenance, dict):
            raise LedgerError('Missing scope/files/provenance')
        origin = provenance.get('origin')
        if origin not in {'original', 'inspired', 'adapted', 'reused'}:
            raise LedgerError('Explicit provenance required')
        if origin in {'adapted', 'reused'} and not all(provenance.get(k) for k in ['source', 'revision', 'license']):
            raise LedgerError('Code reuse needs source, revision and license metadata')
        entries = {}
        for name, content in files.items():
            self._filename(name)
            entries[name] = {'sha256': self.put_evidence(content), 'bytes': len(content)}
        manifest = {'protocol': PROTOCOL, 'label': label, 'kind': kind, 'scope': asdict(scope),
                    'files': entries, 'provenance': provenance, 'parent': parent}
        bundle_id = self.put_evidence(canonical(manifest))
        with self._transaction() as state:
            if parent is not None:
                if parent not in state['bundles'] or state['bundles'][parent]['scope'] != asdict(scope):
                    raise LedgerError('Parent scope mismatch or missing parent')
            if kind != 'seed' and parent is None:
                raise LedgerError('Candidate needs a frozen parent')
            self._append('bundle:' + bundle_id, 'bundle_frozen', {'bundle_id': bundle_id})
        return bundle_id

    def bootstrap(self, bundle_id, *, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise LedgerError('Bootstrap reason required')
        with self._transaction() as state:
            b = state['bundles'][bundle_id]
            scope = Scope(**b['scope']).key
            payload = {'scope': scope, 'bundle_id': bundle_id, 'reason': reason}
            event_id = 'bootstrap:' + scope
            if self._existing(event_id, 'stable_bootstrap', payload):
                return
            if b['kind'] != 'seed' or b['parent'] is not None or scope in state['stable']:
                raise LedgerError('Only explicit initial seed may bootstrap')
            self._append(event_id, 'stable_bootstrap', payload)

    def open_round(self, round_id, *, candidate, plan, generation_usage, metadata):
        identifier(round_id)
        if not isinstance(plan, EvaluationPlan) or not isinstance(generation_usage, Usage):
            raise LedgerError('Explicit comparison plan and generation usage required')
        if not isinstance(metadata, dict) or metadata.get('evidence_mode') not in {'synthetic', 'physical'}:
            raise LedgerError('Explicit evidence mode required')
        plan_payload = asdict(plan)
        EvaluationPlan(**plan_payload)  # caller may have mutated the nested cases dict
        with self._transaction() as state:
            b = state['bundles'][candidate]
            if metadata['evidence_mode'] != b['scope']['evidence_mode']:
                raise LedgerError('Synthetic and physical evidence cannot share a stable lineage')
            scope = Scope(**b['scope']).key
            payload = {'round_id': round_id, 'candidate': candidate, 'parent': b['parent'],
                       'scope': scope, 'plan': plan_payload, 'plan_sha256': digest(canonical(plan_payload)),
                       'generation_usage': asdict(generation_usage), 'metadata': metadata}
            if self._existing('round:' + round_id, 'round_opened', payload):
                return
            if any(r['candidate'] == candidate for r in state['rounds'].values()):
                raise LedgerError('Frozen candidate already has a comparison round')
            if b['parent'] is None or state['stable'].get(scope) != b['parent']:
                raise StaleParentError('Comparison must start from current stable parent')
            self._append('round:' + round_id, 'round_opened', payload)

    def start_trial(self, round_id, trial_id, *, variant, case_id, repetition,
                    condition_id, before_evidence):
        identifier(trial_id)
        identifier(condition_id)
        integer(repetition, 'repetition')
        before_evidence = list(before_evidence)
        payload = dict(round_id=round_id, trial_id=trial_id, variant=variant, case_id=case_id,
                       repetition=repetition, condition_id=condition_id, before_evidence=list(before_evidence))
        with self._transaction() as state:
            event_id = 'start:' + round_id + ':' + trial_id
            if self._existing(event_id, 'trial_started', payload):
                return
            r = state['rounds'][round_id]
            if r['decision'] is not None:
                raise LedgerError('Round already decided')
            if variant not in {'parent', 'candidate'} or case_id not in r['plan']['cases'] or repetition >= r['plan']['repetitions']:
                raise LedgerError('Trial outside frozen schedule')
            if any((t['variant'], t['case_id'], t['repetition']) == (variant, case_id, repetition)
                   for t in r['trials'].values()):
                raise LedgerError('Scheduled slot already attempted; do not hide its failure')
            if not before_evidence:
                raise LedgerError('Initial evidence required')
            for ref in before_evidence:
                self.read_object(ref)
            self._append(event_id, 'trial_started', payload)

    def finish_trial(self, round_id, trial_id, *, status, observed_text,
                     after_evidence, execution_id, usage):
        identifier(execution_id)
        after_evidence = list(after_evidence)
        if status not in {'observed', 'uncertain', 'blocked', 'aborted'}:
            raise LedgerError('Unknown result status')
        if status == 'observed' and not isinstance(observed_text, str):
            raise LedgerError('Observed result needs literal text, not success=true')
        if status != 'observed' and observed_text is not None:
            raise LedgerError('Unknown/blocked results cannot assert observed text')
        if not isinstance(usage, Usage) or not after_evidence:
            raise LedgerError('Explicit usage and outcome evidence required')
        payload = dict(round_id=round_id, trial_id=trial_id, status=status,
                       observed_text=observed_text, after_evidence=list(after_evidence),
                       execution_id=execution_id, usage=asdict(usage))
        with self._transaction() as state:
            event_id = 'finish:' + round_id + ':' + trial_id
            if self._existing(event_id, 'trial_finished', payload):
                return
            r = state['rounds'][round_id]
            if r['decision'] is not None or trial_id not in r['trials']:
                raise LedgerError('Missing trial start or round closed')
            if execution_id in state['execution_ids']:
                raise LedgerError('Execution receipt already used')
            for ref in after_evidence:
                self.read_object(ref)
            self._append(event_id, 'trial_finished', payload)

    def execution_context(self, round_id, variant):
        """Verified frozen inputs for the trusted coordinator, never caller file paths.

        A query is not an execution grant. The RPC owner checks the still-unfinished
        durable trial again when granting and before each synthetic action.
        """
        if variant not in {'parent', 'candidate'}:
            raise LedgerError('Unknown execution variant')
        with self._transaction() as state:
            r = state['rounds'][round_id]
            bundle_id = r[variant]
            inflight = [dict(t) for other in state['rounds'].values()
                        if other['scope'] == r['scope']
                        for t in other['trials'].values() if t['result'] is None]
            return {'round_id': round_id, 'variant': variant,
                    'bundle_id': bundle_id, 'bundle': state['bundles'][bundle_id],
                    'scope_key': r['scope'], 'round_decided': r['decision'] is not None,
                    'inflight_trials': inflight}

    def trial_state(self, round_id, trial_id):
        """Return verified start/result and the exact committed start event anchor."""
        with self._transaction() as state:
            trial = state['rounds'][round_id]['trials'].get(trial_id)
            if trial is None:
                return None
            row = self.db.execute('SELECT sha256 FROM events WHERE event_id=?',
                                  ('start:' + round_id + ':' + trial_id,)).fetchone()
            if row is None:
                raise IntegrityError('Missing durable start anchor')
            return {**trial, 'start_event_sha256': row['sha256']}

    @staticmethod
    def _comparison(r):
        plan = r['plan']
        expected = 2 * len(plan['cases']) * plan['repetitions']
        trials = list(r['trials'].values())
        finished = [t for t in trials if t['result'] is not None]
        usage = aggregate_usage([r['generation_usage']] + [
            t['result']['usage'] if t['result'] is not None else Usage(None, None, None, None)
            for t in trials])
        reasons = []
        if len(trials) != expected or len(finished) != expected:
            reasons.append('incomplete_schedule')
        if any(t['result']['status'] != 'observed' for t in finished):
            reasons.append('unverified_or_interrupted_result')
        if not usage['complete']:
            reasons.append('unknown_usage')
        if (plan['max_total_tokens'] is not None and usage['total_input_plus_output'] is not None
                and usage['total_input_plus_output'] > plan['max_total_tokens']):
            reasons.append('token_budget_exceeded')
        pairs = {}
        scores = {v: {c: 0 for c in plan['cases']} for v in ['parent', 'candidate']}
        for t in finished:
            result = t['result']
            pairs.setdefault((t['case_id'], t['repetition']), {})[t['variant']] = t
            if result['status'] == 'observed' and result['observed_text'] == plan['cases'][t['case_id']]:
                scores[t['variant']][t['case_id']] += 1
        for pair in pairs.values():
            if set(pair) == {'parent', 'candidate'} and pair['parent']['condition_id'] != pair['candidate']['condition_id']:
                reasons.append('initial_condition_mismatch')
        totals = {v: sum(s.values()) for v, s in scores.items()}
        verdict = 'inconclusive' if reasons else 'rejected'
        if not reasons and (totals['candidate'] - totals['parent'] >= plan['min_success_gain']
                            and all(scores['candidate'][c] >= scores['parent'][c] for c in plan['cases'])):
            verdict = 'accepted'
        return {'verdict': verdict, 'reasons': sorted(set(reasons)), 'expected_trials': expected,
                'started_trials': len(trials), 'finished_trials': len(finished), 'correct': totals,
                'per_case_correct': scores, 'usage': usage,
                'evidence_mode': r['metadata']['evidence_mode'], 'physical_truth_verified_by_ledger': False}

    def decide(self, round_id):
        with self._transaction() as state:
            r = state['rounds'][round_id]
            if r['decision'] is not None:
                return r['decision']
            if state['stable'].get(r['scope']) != r['parent']:
                raise StaleParentError('Stable changed; compare again from new parent')
            if any(t['result'] is None for t in r['trials'].values()):
                raise LedgerError('Finish in-flight attempts before deciding; preserve late receipts and unknown cost')
            result = {'round_id': round_id, **self._comparison(r)}
            self._append('decision:' + round_id, 'round_decided', result)
            return result

    def rollback(self, *, scope, target, expected_current, reason, event_id):
        identifier(event_id)
        if not isinstance(reason, str) or not reason.strip():
            raise LedgerError('Rollback needs a reason')
        payload = {'scope': scope.key, 'target': target, 'expected_current': expected_current, 'reason': reason}
        with self._transaction() as state:
            if self._existing('rollback:' + event_id, 'stable_rollback', payload):
                return
            if state['stable'].get(scope.key) != expected_current:
                raise StaleParentError('Rollback has stale expected_current')
            if target not in state['stable_history'].get(scope.key, []):
                raise LedgerError('Rollback target was never stable in this scope')
            self._append('rollback:' + event_id, 'stable_rollback', payload)

    def verify(self):
        with self._transaction() as state:
            return {'protocol': PROTOCOL, 'event_count': state['event_count'],
                    'head_sha256': state['head_sha256'], 'stable': dict(state['stable']),
                    'overall_usage': self._overall_usage(state),
                    'rounds': {k: {'comparison': self._comparison(r), 'decision': r['decision']}
                               for k, r in state['rounds'].items()}}

    @staticmethod
    def _overall_usage(state):
        values = []
        for r in state['rounds'].values():
            values.append(r['generation_usage'])
            values.extend(t['result']['usage'] if t['result'] is not None
                          else Usage(None, None, None, None) for t in r['trials'].values())
        return aggregate_usage(values)

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        with self._transaction() as state:
            events = [dict(row) for row in self.db.execute('SELECT * FROM events ORDER BY seq')]
            report = {'protocol': PROTOCOL, 'head_sha256': state['head_sha256'],
                      'event_count': state['event_count'], 'stable': state['stable'],
                      'overall_usage': self._overall_usage(state),
                      'rounds': {k: {'comparison': self._comparison(r), 'decision': r['decision']}
                                 for k, r in state['rounds'].items()}}
        (directory / 'events.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in events))
        (directory / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        lines = ['# RSI software ledger export', '', 'This ledger records evidence; it does not verify physical truth.', '',
                 f'Events: {report["event_count"]}; head SHA-256: {report["head_sha256"]}', '']
        for name, r in report['rounds'].items():
            c = r['comparison']
            lines.extend([f'## {name}', '', f'Evidence mode: {c["evidence_mode"]}; decision: '
                          + ('pending' if r['decision'] is None else r['decision']['verdict']),
                          f'Started/finished/expected: {c["started_trials"]}/{c["finished_trials"]}/{c["expected_trials"]}',
                          f'Correct: {c["correct"]}; usage: {c["usage"]}', f'Reasons: {c["reasons"]}', ''])
        (directory / 'SUMMARY.md').write_text('\n'.join(lines))
        return report
