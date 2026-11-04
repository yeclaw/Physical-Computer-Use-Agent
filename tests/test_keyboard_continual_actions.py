"""Explicit fixture declarations exercise learning routes without model or SDK."""
import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard.agent.development import DevelopmentActions
from astra_robotics_agent.physical_keyboard.agent.learning import ContinualLearning, LEARNING_ACTIONS
from astra_robotics_agent.physical_keyboard.adapters.skill_sources.learning_sources import LearningSources
from astra_robotics_agent.physical_keyboard.records.learning_access import LearningAccess
from astra_robotics_agent.physical_keyboard.records.learning_contracts import PROTOCOLS, digest, encoded
from astra_robotics_agent.physical_keyboard.runtime.skill_language import run_skill
from astra_robotics_agent.physical_keyboard.skills.knowledge import KnowledgeStore
from astra_robotics_agent.physical_keyboard.skills.learning_changes import LearningChanges


SCOPE = dict(task_id='route_fixture', device_id='fixture_robot', layout_id='fixture_layout')


class Session:
    model, effort, service_tier = 'fixture-model', 'high', 'default'
    thread_id = 'fixture-thread'
    total_usage, total_usage_complete = {}, False

    def __init__(self):
        self.calls = []

    def compact_context(self):
        self.calls.append('local-capability-check')
        return dict(status='unsupported', requires_explicit_rebuild_choice=True)

    def close(self):
        self.calls.append('close')


class ContinualActionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.access = LearningAccess(SCOPE)
        self.store = KnowledgeStore(self.root / 'knowledge', SCOPE, 'fixture-run', learning_access=self.access)
        self.development = DevelopmentActions(self.store)
        self.session = Session()
        self.sources = LearningSources(SCOPE, access=self.access)
        self.sources.register('numeric', kind='robot_attempt', data=dict(samples=[dict(q=[.12345678912345678])]))
        folder = self.root / 'observation'; folder.mkdir()
        self.response = folder / 'response.json'
        self.response.write_text(json.dumps(dict(state={'joint': [.12345678912345678]}, snapshot={'fixture': True})))
        self.image = folder / 'right_wrist.jpg'
        self.image.write_bytes(b'\xff\xd8\xfffixture-jpeg-registered-by-host')
        self.store.register_evidence('obs', [self.response, self.image])
        self.learning = ContinualLearning(self.development, self.session, output=self.root / 'learning',
                                         sources=self.sources, access=self.access)

    def facts(self, **updates):
        result = dict(task_contract_sha256='a' * 64, observation_evidence_id='obs', generation=3,
            last_execution=None, motion_available=False, inflight_rpc=False, inflight_action=False,
            inflight_model=False, stop_requested=False, assistance=None, fault=None,
            source_policy_sha256=digest(self.access.descriptor()))
        result.update(updates)
        return result

    def handle(self, action, args, **updates):
        return self.learning.handle(action, args, facts=self.facts(**updates))

    def package(self, **updates):
        result = dict(id='route-package', source_lines=['def main(robot, profile):', '    return profile["text"]'],
            profile={'text': 'legacy-default'}, contract=dict(description='Explicit fixture capability',
                input_schema={'type': 'object', 'properties': {}, 'additionalProperties': False},
                preconditions=['Check a neutral fixture start'], postconditions=['Observe fresh evidence']),
            applicability=['Fixture scope'], limitations=['Not physical verification'], evidence_ids=['obs'])
        result.update(updates)
        return result

    def analysis(self, result_line=None, source_id='numeric'):
        return dict(id='computed-parameters', source_lines=['def analyze(data, parameters):',
            result_line or '    return {"text": "exact", "computed": data["sample"]["samples"][0]["q"][0] + parameters["delta"]}'],
            inputs={'sample': dict(source_id=source_id)}, parameters={'delta': .1}, reason='A falsifiable fixture calculation')

    def test_construction_context_and_descriptor_do_not_open_all_evidence_or_call_provider(self):
        with patch.object(self.store, '_evidence', side_effect=AssertionError('startup must not expand evidence')):
            instance = ContinualLearning(self.development, self.session, output=self.root / 'second', access=self.access)
            descriptor, context = instance.descriptor(), instance.context('obs')
        self.assertEqual(set(descriptor['actions']), LEARNING_ACTIONS)
        self.assertEqual(descriptor['mandatory_model_roundtrips'], 0)
        self.assertLessEqual(len(encoded(context)), 8192)
        self.assertNotIn('base64', json.dumps(context))
        self.assertEqual(self.session.calls, [])
        self.assertFalse(descriptor['execution_authority'])

    def test_optional_routes_do_not_change_old_development_actions_when_unused(self):
        other_store = KnowledgeStore(self.root / 'disabled', SCOPE, 'old-run')
        DevelopmentActions(other_store)
        self.assertFalse((other_store.namespace / 'learning').exists())
        self.assertEqual(self.session.calls, [])

    def test_read_analyze_exact_resource_explicit_package_materialization_and_call(self):
        listing, extra = self.handle('list_learning_sources', {'origin': 'sources'})
        self.assertEqual(listing['result']['sources'][0]['source_id'], 'numeric')
        self.assertEqual(extra, [])
        brief, _ = self.handle('inspect_learning_sources', {'source_id': 'numeric'})
        self.assertEqual(brief['result']['data']['samples']['type'], 'array')
        analysis, _ = self.handle('analyze_learning_sources', self.analysis())
        descriptor = analysis['result']['resource']
        self.assertEqual(descriptor['kind'], 'analysis_result')
        self.assertEqual(descriptor['source_ids'], ['numeric'])
        self.assertNotIn('result', analysis['result']['analysis'])
        self.assertNotIn('samples', json.dumps(analysis['result']['result_shape']))
        self.assertFalse(analysis['result']['dispatched'])
        resource_id = descriptor['resource_id']
        brief_resource, _ = self.handle('read_learning_resource', {'resource_id': resource_id})
        self.assertNotIn('value', brief_resource['result']['data'])
        full, _ = self.handle('read_learning_resource', {'resource_id': resource_id, 'view': 'full'})
        self.assertEqual(full['result']['data']['value']['computed'], .12345678912345678 + .1)
        args = self.package(profile_resource_id=resource_id); del args['profile']
        saved = self.learning.save_package(args)
        program = self.library_call(saved['package_id'])
        self.assertEqual(program['profile']['computed'], full['result']['data']['value']['computed'])
        self.assertEqual(run_skill([('fixture.py', program['source'])], tools={}, profile=program['profile']), 'exact')
        self.assertEqual(self.learning.resources.get(resource_id)['kind'], 'analysis_result')
        self.assertEqual(self.session.calls, [])

    def library_call(self, package_id):
        return self.development.library.resolve_call({'package_id': package_id, 'arguments': {}})

    def test_analysis_sources_are_host_bound_and_arrays_or_authority_cannot_become_profile(self):
        payloads = ('    return [1, 2]', '    return {"speed_percent": 0.9}')
        for index, line in enumerate(payloads):
            args = self.analysis(line); args['id'] += str(index)
            reply, _ = self.handle('analyze_learning_sources', args)
            resource_id = reply['result']['resource']['resource_id']
            package = self.package(profile_resource_id=resource_id); del package['profile']
            with self.subTest(line=line), self.assertRaises(ValueError):
                self.learning.save_package(package)
        forged, _ = self.handle('analyze_learning_sources', self.analysis('    return {"source_ids": ["untrusted"]}'))
        self.assertEqual(forged['result']['resource']['source_ids'], ['numeric'])

    def test_missing_source_or_arbitrary_path_and_worker_io_are_rejected_without_api(self):
        for source_id in ('missing', '/tmp/secret.json'):
            with self.subTest(source_id=source_id), self.assertRaises(ValueError):
                self.handle('analyze_learning_sources', self.analysis(source_id=source_id))
        with self.assertRaises(ValueError):
            self.handle('analyze_learning_sources', self.analysis('    return open("/tmp/secret")'))
        with self.assertRaises(ValueError):
            self.handle('archive_evidence', {'evidence_ids': ['obs'], 'path': '/tmp/secret'})
        self.assertEqual(self.session.calls, [])

    def test_own_evidence_is_listed_metadata_only_and_registered_only_on_demand(self):
        own = ContinualLearning(self.development, self.session, output=self.root / 'own', access=self.access)
        self.assertEqual(own.sources.list_sources()['sources'], [])
        reply, _ = own.handle('list_learning_sources', {'origin': 'evidence'}, facts=self.facts())
        items = reply['result']['registered_evidence']['items']
        self.assertEqual(items[0]['source_id'], 'evidence:obs')
        self.assertNotIn('"joint"', json.dumps(items))
        self.assertEqual(own.sources.list_sources()['sources'], [])
        read, _ = own.handle('inspect_learning_sources', {'source_id': 'evidence:obs'}, facts=self.facts())
        self.assertEqual(read['result']['provenance']['current_observation_unchanged'], True)
        self.assertEqual(own.sources.list_sources()['sources'][0]['source_id'], 'evidence:obs')

    def test_historical_images_use_separate_channel_with_verified_bytes_and_no_state_update(self):
        reply, extra = self.handle('inspect_learning_sources', dict(source_id='evidence:obs', image_indices=[1]))
        image = next(item for item in extra if item['type'] == 'inputImage')
        self.assertEqual(base64.b64decode(image['imageUrl'].partition(',')[2]), self.image.read_bytes())
        self.assertIn('HISTORICAL', extra[0]['text'])
        self.assertNotIn('base64', json.dumps(reply))
        self.assertNotIn('path', reply['result']['images'][0])
        self.assertTrue(reply['observation_unchanged'])
        self.assertEqual(reply['receipt']['historical_image_count'], 1)
        self.assertEqual(reply['receipt']['result_json_bytes'], len(encoded(reply['result'])))
        self.assertEqual(reply['receipt']['extra_content_json_bytes'], len(encoded(extra)))

    def test_audit_only_scoring_evidence_is_not_a_learning_source(self):
        self.response.write_text(json.dumps(dict(state={}, snapshot={}, typing_buffer='hidden')))
        self.store.register_evidence('audit-only', [self.response])
        with self.assertRaises(ValueError):
            self.handle('inspect_learning_sources', {'source_id': 'evidence:audit-only'})

    def test_old_profile_save_read_byte_receipts_and_pending_ref_guard(self):
        saved = self.learning.save_package(self.package())
        self.assertEqual(self.library_call(saved['package_id'])['profile']['text'], 'legacy-default')
        request = {'identifier': saved['package_id']}
        first, _ = self.handle('read_knowledge', request)
        second, _ = self.handle('read_knowledge', request)
        self.assertFalse(first['result']['receipt']['repeat_read'])
        self.assertTrue(second['result']['receipt']['repeat_read'])
        self.assertEqual(first['result']['receipt']['data_bytes'], len(encoded(first['result']['data'])))
        self.assertNotIn('source_lines', first['result']['data'])
        args = self.package(id='pending', source_ids=self.access.evidence_sources('obs'))
        def fail(stage):
            if stage == 'before_final_commit': raise RuntimeError('Interrupted fixture')
        self.learning.changes.failure_hook = fail
        plan = self.learning.changes.preflight({'packages': [args]})
        with self.assertRaises(RuntimeError):
            self.handle('commit_learning', {'packages': [args]})
        pending = plan['operations'][0]['identifier']
        with self.assertRaises(ValueError):
            self.handle('read_knowledge', {'identifier': pending})
        with self.assertRaises(ValueError):
            self.learning.save_package(self.package(id='child', dependencies={'part': pending}))

    def test_work_checkpoint_and_context_request_preserve_stop_and_never_move(self):
        declaration = dict(summary='Fixture task still pending', knowledge_ids=[], evidence_ids=['obs'], pending_issues=[])
        saved, _ = self.handle('save_working_context', declaration)
        args = dict(checkpoint_id=saved['result']['checkpoint_id'], method='native_compact', reason='Explicit boundary')
        result, _ = self.handle('manage_working_context', args)
        self.assertEqual(result['result']['status'], 'unsupported')
        self.assertTrue(result['result']['fresh_observation_required'])
        self.assertFalse(result['result']['motion_available'])
        self.assertEqual(self.session.calls, ['local-capability-check'])
        stopped, _ = self.handle('save_working_context', declaration, stop_requested=True)
        args['checkpoint_id'] = stopped['result']['checkpoint_id']
        with self.assertRaises(ValueError):
            self.handle('manage_working_context', args, stop_requested=True)
        self.assertEqual(self.session.calls, ['local-capability-check'])

    def test_archive_accepts_only_registered_evidence_and_restores_same_bytes(self):
        raw = self.image.read_bytes()
        reply, _ = self.handle('archive_evidence', {'evidence_ids': ['obs'], 'name': 'fixture copy'})
        descriptor = reply['result']
        self.assertNotIn('archive_path', descriptor)
        self.assertTrue(descriptor['originals_retained'])
        restored = self.learning.archives.read(descriptor['archive_id'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(restored, raw)
        self.assertEqual(self.image.read_bytes(), raw)
        with self.assertRaises(ValueError):
            self.handle('archive_evidence', {'evidence_ids': ['not-registered']})

    def test_source_policy_denies_transitive_source_reads_and_analysis(self):
        access = LearningAccess(SCOPE, dict(protocol=PROTOCOLS['access'], scope=SCOPE,
            allowed_source_ids=[], allowed_knowledge_ids=[], allowed_evidence_ids=[], allow_own_experience=True))
        restricted = ContinualLearning(self.development, self.session, output=self.root / 'restricted',
            sources=self.sources, access=access)
        with self.assertRaises(ValueError):
            restricted.handle('analyze_learning_sources', self.analysis(), facts=self.facts())
        with self.assertRaises(ValueError):
            restricted.handle('inspect_learning_sources', {'source_id': 'evidence:obs'}, facts=self.facts())
        listed, _ = restricted.handle('list_learning_sources', {'origin': 'sources'}, facts=self.facts())
        self.assertEqual(listed['result']['sources'], [])

    def test_old_inline_images_are_same_evidence_references_in_text_and_numerical_inputs(self):
        payload = base64.b64encode(self.image.read_bytes()).decode('ascii')
        self.sources.register('embedded', kind='robot_attempt', data={'raw': {'images_base64': {'right': payload}}})
        reply, extra = self.handle('inspect_learning_sources', {'source_id': 'embedded', 'fields': ['raw']})
        self.assertNotIn(payload, json.dumps(reply))
        self.assertEqual(extra, [])
        reference = reply['result']['data']['raw']['image_references']['right']
        self.assertEqual(reference['sha256'], hashlib.sha256(self.image.read_bytes()).hexdigest())
        self.assertFalse(reply['result']['encoding_projection']['new_visual_evidence'])
        args = self.analysis('    return {"reference_bytes": data["sample"]["raw"]["image_references"]["right"]["bytes"]}',
                             source_id='embedded')
        computed, _ = self.handle('analyze_learning_sources', args)
        self.assertNotIn(payload, json.dumps(computed))
        self.assertEqual(self.learning.resources.get(computed['result']['resource']['resource_id'])['value'],
                         {'reference_bytes': len(self.image.read_bytes())})

    def test_restricted_changes_rebinds_broad_resource_service_without_leaking_contents(self):
        broad = self.learning.resources
        hidden = broad.put('hidden', {'hidden_parameter': 0.123456789}, source_ids=['hidden-demo'])
        restricted = LearningAccess(SCOPE, dict(protocol=PROTOCOLS['access'], scope=SCOPE,
            allowed_source_ids=[], allowed_knowledge_ids=[], allowed_evidence_ids=[], allow_own_experience=True))
        changes = LearningChanges(self.development.library, broad, access=restricted)
        with self.assertRaises(ValueError):
            changes.source_ids(hidden['resource_id'])
        with self.assertRaises(ValueError):
            changes.resources.get(hidden['resource_id'])
        self.assertEqual(broad.get(hidden['resource_id'])['value'], {'hidden_parameter': 0.123456789})

    def test_lost_pending_intent_never_reclassifies_new_resource_as_legacy(self):
        changes = self.learning.changes
        args = {'resources': [dict(id='unpublished', value={'pending_fixture': 1}, source_ids=['source-a'])]}
        plan = changes.preflight(args)
        def fail(stage):
            if stage == 'before_final_commit': raise RuntimeError('Synthetic interrupted publication')
        changes.failure_hook = fail
        with self.assertRaises(RuntimeError):
            changes.commit(args)
        resource_id = plan['record_ids'][0]
        self.assertFalse(changes.is_committed(resource_id))
        intent = next((changes.root / 'intents').glob('*.json'))
        intent.rename(intent.with_suffix('.missing'))
        self.assertFalse(changes.is_committed(resource_id))
        with self.assertRaises(ValueError):
            self.handle('read_learning_resource', {'resource_id': resource_id, 'view': 'full'})
        restored = LearningChanges(self.development.library, self.learning.resources)
        self.assertFalse(restored.is_committed(resource_id))

    def test_actions_schema_has_no_motion_routes_or_execution_imports(self):
        for action in ('run_skill', 'call_skill', [], None):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.handle(action, {})
        with self.assertRaises(ValueError):
            self.handle('knowledge_directory', {'hypothesis': 'Unexpected top-level model field'})
        tree = ast.parse((Path(__file__).parents[1] / 'src/astra_robotics_agent/physical_keyboard/agent/learning.py').read_text())
        imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any('startouch' in name or 'transport' in name or 'rpc' in name for name in imports))


if __name__ == '__main__':
    unittest.main()
