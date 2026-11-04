"""Real temporary stores and bounded programs; no model, SDK, network or motion."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from astra_robotics_agent.physical_keyboard.runtime.skill_language import run_skill
from astra_robotics_agent.physical_keyboard.skills.development import SkillDevelopment
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeError, KnowledgeStore


SCOPE = {'task_id': 'fixture', 'device_id': 'fixture_robot', 'layout_id': 'fixture_layout'}
SOURCE = ['def main(robot, profile):', '    return {"text": profile["text"]}']


def contract():
    return {'description': 'A model-authored parameterized demonstration fixture',
            'input_schema': {'type': 'object', 'properties': {'text': {'type': 'string', 'minLength': 1}},
                             'required': ['text'], 'additionalProperties': False},
            'preconditions': ['The model must check the applicable starting condition.'],
            'postconditions': ['Inspect fresh evidence to assess the intended effect.']}


class SkillDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'first')
        self.evidence = self.root / 'actual-observation.json'
        self.evidence.write_text('{"observed":"fixture evidence, not actual physical success"}')
        self.store.register_evidence('obs-first', [self.evidence])
        self.library = SkillDevelopment(self.store)

    def package_args(self, identifier='parameterized'):
        return dict(id=identifier, source_lines=list(SOURCE), profile={'text': 'default'},
                    contract=contract(), applicability=['fixture layout and checked start'],
                    limitations=['No physical verification'], evidence_ids=['obs-first'])

    def legacy_skill(self, identifier='original'):
        return self.store.propose('skill', dict(id=identifier, source_lines=list(SOURCE), profile={'text': 'old'},
            applicability=['Only a model-reported starting condition'], limitations=['Not a stable skill'],
            evidence_ids=['obs-first']))

    def issue_args(self, skill_id, status='open'):
        return dict(id='side_effect', summary='The result needed a compensating edit',
                    rationale='Investigate reuse cost using actual evidence', status=status,
                    skill_ids=[skill_id], evidence_ids=['obs-first'])

    def outcome_args(self, skill_id):
        return dict(id='attempt', skill_ids=[skill_id], evidence_ids=['obs-first'],
                    observation_evidence_id='obs-first', task_result='success', correction_performed=True,
                    remaining_issues=['Underlying side effect is not shown repaired'],
                    summary='A model report that current output was corrected')

    def test_old_history_retrieval_is_not_limited_to_recent_eight(self):
        old = self.store.propose('lesson', dict(id='old_localization', claim='Buried localization insight',
            applicability=['Full original condition'], limitations=['Full original uncertainty'],
            evidence_ids=['obs-first']))
        for index in range(12):
            self.store.propose('lesson', dict(id='recent_' + str(index), claim='Unrelated later record',
                applicability=['Other'], limitations=['Unknown'], evidence_ids=['obs-first']))
        self.assertNotIn(old['candidate_id'], self.store.context_text())
        results = self.library.search('localization', kind='lesson')['results']
        self.assertEqual(results[0]['candidate_id'], old['candidate_id'])
        full = self.library.inspect(old['candidate_id'])
        self.assertEqual(full['claim'], 'Buried localization insight')
        self.assertEqual(full['applicability'], ['Full original condition'])
        self.assertNotIn('image_base64', json.dumps(results))

    def test_contract_and_profile_are_model_declarations_and_parameterized_across_words(self):
        saved = self.library.save_package(self.package_args())
        self.assertEqual(saved['contract'], contract())
        self.assertFalse(saved['stable'])
        full = self.library.inspect(saved['package_id'])
        self.assertEqual(full['origin'], 'astra_generated')
        self.assertEqual(full['source_lines'], SOURCE)
        self.assertEqual(full['contract']['preconditions'], contract()['preconditions'])
        for word in ('unseen', 'different'):
            program = self.library.resolve_call({'package_id': saved['package_id'], 'arguments': {'text': word}})
            result = run_skill([('skill.py', program['source'])], tools={}, profile=program['profile'])
            self.assertEqual(result, {'text': word})
        self.assertEqual(self.store.get_skill(saved['candidate_id'])['profile']['text'], 'default')

    def test_bad_call_input_is_rejected_without_changing_any_records(self):
        saved = self.library.save_package(self.package_args())
        before = self.library.index.read_bytes()
        for arguments in ({}, {'text': 3}, {'text': 'ok', 'extra': True}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.library.resolve_call({'package_id': saved['package_id'], 'arguments': arguments})
        self.assertEqual(before, self.library.index.read_bytes())
        with self.assertRaises(ValueError):
            self.library.resolve_call({'package_id': saved['candidate_id'], 'arguments': {'text': 'x'}})

    def test_old_candidates_unchanged_and_model_parent_lineage_is_explicit(self):
        old = self.legacy_skill()
        old_folder = self.store.namespace / 'candidates' / old['candidate_id']
        old_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in old_folder.iterdir()}
        args = self.package_args()
        args['parent_id'] = old['candidate_id']
        first = self.library.save_package(args)
        args['parent_id'] = first['package_id']
        args['source_lines'] = ['def main(robot, profile):', '    return {"text": profile["text"], "revision": 2}']
        second = self.library.save_package(args)
        self.assertEqual(second['version'], 2)
        self.assertEqual(self.library.inspect(second['package_id'])['parent_id'], first['package_id'])
        self.assertEqual(self.library.inspect(second['package_id'])['origin'], 'astra_generated')
        self.assertEqual(old_hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in old_folder.iterdir()})

    def test_unknown_parent_or_evidence_reject_before_candidate_append(self):
        before = self.store._events()
        args = self.package_args()
        args['parent_id'] = 'skill-missing-' + '0' * 64
        with self.assertRaises(ValueError):
            self.library.save_package(args)
        args.pop('parent_id')
        args['evidence_ids'] = ['obs-does-not-exist']
        with self.assertRaises(ValueError):
            self.library.save_package(args)
        self.assertEqual(before, self.store._events())

    def test_dependency_linking_uses_pinned_programs_and_one_compiled_entry(self):
        dependency = self.legacy_skill()
        args = self.package_args('combined')
        args['dependencies'] = {'step': dependency['candidate_id']}
        args['source_lines'] = ['def main(robot, profile):',
                                '    first = step(robot, {"text": profile["text"]})',
                                '    second = step(robot, {"text": "second"})',
                                '    return [first, second]']
        saved = self.library.save_package(args)
        linked = self.library.resolve_call({'package_id': saved['package_id'], 'arguments': {'text': 'new'}})
        self.assertEqual(linked['dependency_pins'][0]['candidate_id'], dependency['candidate_id'])
        result = run_skill([('skill.py', linked['source'])], tools={}, profile=linked['profile'])
        self.assertEqual(result, [{'text': 'new'}, {'text': 'second'}])

    def test_compensation_report_does_not_close_the_issue_or_promote(self):
        saved = self.library.save_package(self.package_args())
        issue = self.library.record_improvement(self.issue_args(saved['package_id']))
        outcome = self.library.record_outcome(self.outcome_args(saved['package_id']))
        self.assertEqual(outcome['basis'], 'model_report')
        self.assertFalse(outcome['stable'])
        self.assertFalse(self.library.inspect(outcome['record_id'])['independently_verified'])
        context = self.library.context()
        self.assertEqual(context['pending_improvements'][0]['record_id'], issue['record_id'])
        self.assertEqual(context['pending_improvements'][0]['status'], 'open')
        self.assertTrue(context['recent_outcomes'][0]['correction_performed'])
        self.assertFalse(list((self.store.namespace / 'verifications').iterdir()))

    def test_improvement_revisions_and_new_run_persist_without_overwriting(self):
        saved = self.library.save_package(self.package_args())
        first = self.library.record_improvement(self.issue_args(saved['package_id']))
        second = self.library.record_improvement(self.issue_args(saved['package_id'], 'resolved'))
        self.assertEqual(second['version'], 2)
        self.assertEqual(self.library.inspect(first['record_id'])['status'], 'open')
        self.assertFalse(self.library.context()['pending_improvements'])
        later = SkillDevelopment(KnowledgeStore(self.root / 'knowledge', SCOPE, 'later'))
        reopened = later.record_improvement(self.issue_args(saved['package_id'], 'reopened'))
        self.assertEqual(reopened['version'], 3)
        self.assertEqual(later.context()['pending_count'], 1)
        self.assertEqual(later.inspect(saved['package_id'])['contract'], contract())

    def test_evidence_default_metadata_and_opt_in_bounded_text_images(self):
        text_files = []
        for index in range(3):
            p = self.root / ('long-' + str(index) + '.txt')
            p.write_text('x' * 12000)
            text_files.append(p)
        image = self.root / 'historical.jpg'
        image.write_bytes(b'\xff\xd8\xff' + b'fixture-not-a-real-camera')
        self.store.register_evidence('history', text_files + [image])
        metadata = self.library.inspect_evidence({'evidence_id': 'history'})
        self.assertTrue(metadata['historical'])
        self.assertTrue(all(set(item) == {'index', 'name', 'sha256', 'bytes'} for item in metadata['files']))
        self.assertNotIn('text', metadata['files'][0])
        self.assertNotIn('image_base64', metadata['files'][-1])
        selected = self.library.inspect_evidence({'evidence_id': 'history', 'file_indices': [0, 1, 2, 3], 'include_images': True})
        self.assertEqual(sum(len(item.get('text', '')) for item in selected['files']), 16000)
        self.assertEqual(base64.b64decode(selected['files'][3]['image_base64']), image.read_bytes())
        with self.assertRaises(ValueError):
            self.library.inspect_evidence({'evidence_id': 'history', 'file_indices': [-1]})
        with self.assertRaises(ValueError):
            self.library.inspect_evidence({'evidence_id': '../elsewhere'})

    def large_execution_evidence(self):
        payload = {'execution': {'result': {
            'padding': '中文🙂' * 4000,
            'dispatched_count': 472,
            'sequence': [{'index': 471, 'source': 'v'}, {'index': 472, 'source': 'v→e', 'sent': False}]}}}
        raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        path = self.root / 'execution-response.json'
        path.write_text(raw, encoding='utf-8')
        registered = self.store.register_evidence('execution-response', [path])
        self.assertGreater(len(raw), 11442)
        self.assertGreater(len(raw.encode('utf-8')), 11442)
        return payload, raw, registered

    def test_evidence_text_pages_recover_complete_large_utf8_response_tail(self):
        payload, raw, registered = self.large_execution_evidence()
        before = (self.store.namespace / 'index.jsonl').read_bytes()
        legacy = self.library.inspect_evidence({'evidence_id': 'execution-response', 'file_indices': [0]})
        self.assertEqual(set(legacy), {'evidence_id', 'scope', 'run_id', 'historical', 'files',
                                       'images_requested', 'meaning'})
        self.assertEqual(set(legacy['files'][0]), {'index', 'name', 'sha256', 'bytes', 'text', 'truncated'})
        self.assertEqual(legacy['files'][0]['text'], raw[:8000])
        self.assertNotIn('dispatched_count', legacy['files'][0]['text'])
        first = self.library.inspect_evidence({'evidence_id': 'execution-response', 'file_indices': [0],
                                              'text_offset': 0})['files'][0]
        second_result = self.library.inspect_evidence({'evidence_id': 'execution-response', 'file_indices': [0],
                                                       'text_offset': first['next_offset']})
        second = second_result['files'][0]
        self.assertEqual(first['text'], raw[:8000])
        self.assertEqual(first['offset_unit'], 'unicode_code_points')
        self.assertEqual(first['next_offset'], 8000)
        self.assertTrue(first['truncated'])
        self.assertEqual(first['total'], len(raw))
        self.assertNotEqual(first['total'], registered['files'][0]['bytes'])
        self.assertEqual(json.loads(first['text'] + second['text']), payload)
        self.assertIn('dispatched_count', second['text'])
        self.assertIn('sequence', second['text'])
        self.assertFalse(second['truncated'])
        self.assertIsNone(second['next_offset'])
        self.assertTrue(second_result['historical'])
        self.assertNotIn('current_observation_id', second_result)
        end = self.library.inspect_evidence({'evidence_id': 'execution-response', 'file_indices': [0],
                                            'text_offset': len(raw), 'text_limit': 1})['files'][0]
        self.assertEqual(end['text'], '')
        self.assertIsNone(end['next_offset'])
        self.assertEqual(before, (self.store.namespace / 'index.jsonl').read_bytes())

    def test_evidence_json_pointer_selects_nested_tail_and_pages_selected_value(self):
        payload, _, _ = self.large_execution_evidence()
        base = {'evidence_id': 'execution-response', 'file_indices': [0]}
        count = self.library.inspect_evidence(dict(base, json_pointer='/execution/result/dispatched_count'))['files'][0]
        self.assertEqual(count['text'], '472')
        self.assertEqual(count['total'], 3)
        self.assertEqual(count['text_format'], 'canonical_json')
        selected = self.library.inspect_evidence(dict(base, json_pointer='/execution/result/sequence/1'))['files'][0]
        self.assertEqual(json.loads(selected['text']), payload['execution']['result']['sequence'][1])
        self.assertEqual(selected['text'], '{"index":472,"sent":false,"source":"v→e"}')
        pages, offset = [], 0
        while offset is not None:
            entry = self.library.inspect_evidence(dict(base, json_pointer='/execution/result',
                                                       text_offset=offset, text_limit=7000))['files'][0]
            self.assertEqual(entry['json_pointer'], '/execution/result')
            self.assertLessEqual(len(entry['text']), 7000)
            pages.append(entry['text'])
            offset = entry['next_offset']
        canonical = json.dumps(payload['execution']['result'], sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        self.assertEqual(''.join(pages), canonical)
        self.assertEqual(json.loads(''.join(pages)), payload['execution']['result'])

    def test_evidence_json_pointer_rfc6901_escaping_arrays_scalars_and_root(self):
        payload = {'': {'a/b': {'m~n': [False, None, 1.25, '中文🙂'], '~1': {'0': 'object key'}}}}
        path = self.root / 'pointer.json'
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        self.store.register_evidence('pointer', [path])
        base = {'evidence_id': 'pointer', 'file_indices': [0]}
        for pointer, expected in (('', payload), ('//a~1b/m~0n/0', False), ('//a~1b/m~0n/1', None),
                                  ('//a~1b/m~0n/2', 1.25), ('//a~1b/m~0n/3', '中文🙂'),
                                  ('//a~1b/~01/0', 'object key')):
            with self.subTest(pointer=pointer):
                entry = self.library.inspect_evidence(dict(base, json_pointer=pointer))['files'][0]
                self.assertEqual(json.loads(entry['text']), expected)
        string_json = '"中文🙂"'
        page = self.library.inspect_evidence(dict(base, json_pointer='//a~1b/m~0n/3',
                                                  text_offset=1, text_limit=2))['files'][0]
        self.assertEqual(page['text'], '中文')
        self.assertEqual(page['total'], len(string_json))
        self.assertEqual(page['next_offset'], 3)
        tail = self.library.inspect_evidence(dict(base, json_pointer='//a~1b/m~0n/3',
                                                  text_offset=3, text_limit=2))['files'][0]
        self.assertEqual(tail['text'], '🙂"')
        self.assertIsNone(tail['next_offset'])

    def test_evidence_paging_rejects_invalid_requests_without_writing(self):
        base = {'evidence_id': 'obs-first', 'file_indices': [0]}
        before = (self.store.namespace / 'index.jsonl').read_bytes()
        invalid = [
            {'text_offset': -1}, {'text_offset': True}, {'text_offset': 1.5}, {'text_offset': 10000},
            {'text_limit': 0}, {'text_limit': 8001}, {'text_limit': False}, {'text_limit': '3'},
            {'text_limit': 3, 'file_indices': []}, {'text_limit': 3, 'file_indices': [0, 0]},
            {'text_limit': 3, 'file_indices': [-1]}, {'text_limit': 3, 'file_indices': [1]},
            {'text_limit': 3, 'file_indices': [True]}, {'text_limit': 3, 'file_indices': '0'},
            {'text_limit': 3, 'include_images': 1}, {'text_limit': 3, 'path': '/tmp/unregistered'},
            {'json_pointer': None}, {'json_pointer': []}, {'json_pointer': '#/observed'},
            {'json_pointer': 'observed'}, {'json_pointer': '/~2'}, {'json_pointer': '/~'},
            {'json_pointer': '/missing'}, {'json_pointer': '/observed/0'},
            {'json_pointer': '/' + 'x' * 1024}, {'json_pointer': '/' * 17}, {'json_pointer': '/\ud800'},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(KnowledgeError):
                self.library.inspect_evidence(dict(base, **changes))
        with self.assertRaises(KnowledgeError):
            self.library.inspect_evidence({'evidence_id': 'obs-first', 'text_offset': 0})
        self.assertEqual(before, (self.store.namespace / 'index.jsonl').read_bytes())

    def test_evidence_json_pointer_rejects_invalid_array_indices_and_non_json(self):
        path = self.root / 'array.json'
        path.write_text('{"items":[1,2]}')
        self.store.register_evidence('array', [path])
        for token in ('-1', '-', '01', '+1', '2', '1.0', '', '9' * 1000):
            with self.subTest(token=token), self.assertRaises(KnowledgeError):
                self.library.inspect_evidence({'evidence_id': 'array', 'file_indices': [0],
                                               'json_pointer': '/items/' + token})
        for index, raw in enumerate((b'not json', b'{"x":NaN}', b'{"x":1,"x":2}', b'{"x":1e999}',
                                     b'{"x":"\\ud800"}', b'\xffbinary', b'\xff\xd8\xffimage')):
            path = self.root / ('invalid-' + str(index))
            path.write_bytes(raw)
            identifier = 'invalid-' + str(index)
            self.store.register_evidence(identifier, [path])
            with self.subTest(raw=raw), self.assertRaises(KnowledgeError):
                self.library.inspect_evidence({'evidence_id': identifier, 'file_indices': [0], 'json_pointer': ''})
            if index >= 5:
                with self.subTest(raw=raw), self.assertRaises(KnowledgeError):
                    self.library.inspect_evidence({'evidence_id': identifier, 'file_indices': [0], 'text_offset': 0})

    def test_evidence_paging_cannot_bypass_registration_scope_or_hash(self):
        _, _, registered = self.large_execution_evidence()
        request = {'evidence_id': 'execution-response', 'file_indices': [0],
                   'json_pointer': '/execution/result/dispatched_count'}
        other = SkillDevelopment(KnowledgeStore(self.root / 'knowledge', dict(SCOPE, layout_id='other'), 'other'))
        with self.assertRaises(KnowledgeError):
            other.inspect_evidence(request)
        for identifier in ('not-registered', '/tmp/execution-response.json', 'https://example.com/response.json'):
            with self.subTest(identifier=identifier), self.assertRaises(KnowledgeError):
                self.library.inspect_evidence(dict(request, evidence_id=identifier))
        obj = self.store.namespace / 'objects' / registered['files'][0]['sha256']
        obj.write_bytes(obj.read_bytes() + b' ')
        with self.assertRaisesRegex(KnowledgeError, 'hash mismatch'):
            self.library.inspect_evidence(request)

    def test_evidence_page_requires_one_file_and_preserves_legacy_image_limits(self):
        text = self.root / 'another.txt'
        text.write_text('another')
        image = self.root / 'large.jpg'
        image.write_bytes(b'\xff\xd8\xff' + b'x' * (4 * 1024 * 1024 - 3))
        self.store.register_evidence('mixed', [self.evidence, text, image])
        for indices in ([0, 1], [0, 2]):
            with self.subTest(indices=indices), self.assertRaises(KnowledgeError):
                self.library.inspect_evidence({'evidence_id': 'mixed', 'file_indices': indices, 'text_limit': 5})
        accepted = self.library.inspect_evidence({'evidence_id': 'mixed', 'file_indices': [2], 'include_images': True})
        self.assertEqual(len(base64.b64decode(accepted['files'][2]['image_base64'])), 4 * 1024 * 1024)
        image.write_bytes(image.read_bytes() + b'x')
        self.store.register_evidence('oversize-image', [image])
        metadata = self.library.inspect_evidence({'evidence_id': 'oversize-image', 'file_indices': [0]})
        self.assertNotIn('image_base64', metadata['files'][0])
        with self.assertRaisesRegex(KnowledgeError, 'image budget'):
            self.library.inspect_evidence({'evidence_id': 'oversize-image', 'file_indices': [0], 'include_images': True})

    def test_outcome_requires_registered_observation_and_scope_references(self):
        saved = self.library.save_package(self.package_args())
        args = self.outcome_args(saved['package_id'])
        args['observation_evidence_id'] = 'other'
        with self.assertRaises(ValueError):
            self.library.record_outcome(args)
        other = SkillDevelopment(KnowledgeStore(self.root / 'knowledge', dict(SCOPE, layout_id='other'), 'other'))
        with self.assertRaises(ValueError):
            other.inspect(saved['package_id'])
        with self.assertRaises(ValueError):
            other.inspect_evidence({'evidence_id': 'obs-first'})

    def test_corrupt_development_object_and_dependency_are_detected(self):
        saved = self.library.save_package(self.package_args())
        row = json.loads(self.library.index.read_text().splitlines()[0])
        path = self.library.objects / (row['sha256'] + '.json')
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            self.library.inspect(saved['package_id'])

    def test_corrupt_dependency_cannot_be_used_via_a_previously_linked_package(self):
        dependency = self.legacy_skill()
        args = self.package_args('linked')
        args['dependencies'] = {'step': dependency['candidate_id']}
        args['source_lines'] = ['def main(robot, profile):', '    return step(robot, {"text": profile["text"]})']
        saved = self.library.save_package(args)
        path = self.store.namespace / 'candidates' / dependency['candidate_id'] / 'source.py'
        path.write_text(path.read_text() + '\n')
        with self.assertRaises(ValueError):
            self.library.resolve_call({'package_id': saved['package_id'], 'arguments': {'text': 'x'}})

    def test_optional_capability_disconnection_preserves_inspect_but_rejects_call(self):
        enabled_store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'enabled', demo_tools=True)
        enabled = SkillDevelopment(enabled_store)
        args = self.package_args('optional')
        args['source_lines'] = ['def main(robot, profile):', '    return robot.trajectory_catalog()']
        saved = enabled.save_package(args)
        disabled = SkillDevelopment(KnowledgeStore(self.root / 'knowledge', SCOPE, 'disabled'))
        self.assertEqual(disabled.inspect(saved['package_id'])['candidate_id'], saved['candidate_id'])
        with self.assertRaises(ValueError):
            disabled.resolve_call({'package_id': saved['package_id'], 'arguments': {'text': 'x'}})

    def test_symbolic_link_routes_cannot_read_or_write_an_external_index(self):
        old = self.legacy_skill()
        sink = self.root / 'outside.jsonl'
        sink.write_text('untouched')
        self.library.index.symlink_to(sink)
        with self.assertRaises(ValueError):
            self.library.record_improvement(self.issue_args(old['candidate_id']))
        self.assertEqual(sink.read_text(), 'untouched')
        other_store = KnowledgeStore(self.root / 'knowledge', dict(SCOPE, layout_id='linked'), 'other')
        (other_store.namespace / 'development').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            SkillDevelopment(other_store)

    def test_historical_image_count_is_bounded_and_inspection_is_read_only(self):
        images = []
        for index in range(3):
            p = self.root / ('image-' + str(index) + '.jpg')
            p.write_bytes(b'\xff\xd8\xfffixture')
            images.append(p)
        self.store.register_evidence('three-images', images)
        index_before = (self.store.namespace / 'index.jsonl').read_bytes()
        accepted = self.library.inspect_evidence({'evidence_id': 'three-images', 'file_indices': [0, 1], 'include_images': True})
        self.assertEqual(sum('image_base64' in item for item in accepted['files']), 2)
        with self.assertRaises(ValueError):
            self.library.inspect_evidence({'evidence_id': 'three-images', 'file_indices': [0, 1, 2], 'include_images': True})
        metadata = self.library.inspect_evidence({'evidence_id': 'three-images', 'file_indices': [0]})
        self.assertEqual(metadata['files'][0]['mime_type'], 'image/jpeg')
        self.assertNotIn('image_base64', metadata['files'][0])
        self.assertEqual(index_before, (self.store.namespace / 'index.jsonl').read_bytes())

    def test_large_context_is_small_and_contains_no_programs_or_images(self):
        saved = self.library.save_package(self.package_args())
        for index in range(16):
            args = self.issue_args(saved['package_id'])
            args.update(id='issue_' + str(index), summary='long ' * 300, rationale='evidence ' * 100)
            self.library.record_improvement(args)
        context = self.library.context(limit=16)
        raw = json.dumps(context, ensure_ascii=False)
        self.assertLessEqual(len(raw.encode()), 8192)
        self.assertTrue(context['summary_truncated'])
        self.assertEqual(context['pending_count'], 16)
        self.assertNotIn('source_lines', raw)
        self.assertNotIn('image_base64', raw)

    def test_invalid_types_fail_as_validation_not_unhandled_type_errors(self):
        with self.assertRaises(ValueError):
            self.library.search(kind={})
        saved = self.library.save_package(self.package_args())
        args = self.issue_args(saved['package_id'])
        args['status'] = {}
        with self.assertRaises(ValueError):
            self.library.record_improvement(args)
        args = self.outcome_args(saved['package_id'])
        args['task_result'] = []
        with self.assertRaises(ValueError):
            self.library.record_outcome(args)

    def test_returned_scope_does_not_mutate_library_identity(self):
        scope = self.library.context()['scope']
        scope['layout_id'] = 'mutated'
        self.assertEqual(self.library.scope, SCOPE)
        saved = self.library.save_package(self.package_args())
        self.assertEqual(self.library.inspect(saved['package_id'])['scope'], SCOPE)

    def test_corrupt_extension_rejected_before_new_legacy_candidate(self):
        self.library.save_package(self.package_args('first'))
        self.library.index.write_text(self.library.index.read_text() + '{}\n')
        before = (self.store.namespace / 'index.jsonl').read_bytes()
        with self.assertRaises(ValueError):
            self.library.save_package(self.package_args('second'))
        self.assertEqual(before, (self.store.namespace / 'index.jsonl').read_bytes())


if __name__ == '__main__':
    unittest.main()
