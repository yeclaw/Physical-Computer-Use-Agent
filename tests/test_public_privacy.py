"""Synthetic credential diagnostics: no accounts, servers or model calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_robotics_agent.codex import _selected_error
from astra_robotics_agent.physical_keyboard.agent.runner import AgentRunner, run_config
from astra_robotics_agent.physical_keyboard.records.redaction import redact_text
from astra_robotics_agent.physical_keyboard.runtime.mock_device import MockDevice
from astra_robotics_agent.physical_keyboard.runtime.program_runtime import ProgramRuntime


SCOPE = {'task_id': 'privacy-test', 'device_id': 'synthetic-device', 'layout_id': 'one'}


class PublicPrivacyTests(unittest.TestCase):
    def test_structured_auth_fields_are_removed_without_losing_diagnostics(self):
        fields = ['id_token', 'session-token', 'clientSecret', 'auth_token', 'X-Api-Key',
                  'account_id', 'private_key', 'secret_key', 'cookies',
                  'aws_secret_access_key', 'chatgptAccountId']
        details = {key: 'unusable-fixture-' + str(n) for n, key in enumerate(fields)}
        details.update(retryAfterSeconds=2, totalTokens=17, reason='invalid_request')
        result = _selected_error({'code': 401, 'additionalDetails': {'nested': details}})
        text = json.dumps(result)
        self.assertNotIn('unusable-fixture', text)
        self.assertEqual(result['additionalDetails']['nested'],
                         {'retryAfterSeconds': 2, 'totalTokens': 17, 'reason': 'invalid_request'})
        self.assertEqual(result['code'], 401)

    def test_header_assignment_url_and_cookie_credentials_are_redacted(self):
        samples = [
            ('Authorization: Basic ZmFrZS11c2VyOmZha2UtcGFzcw==', 'ZmFrZS11c2VyOmZha2UtcGFzcw=='),
            ('Authorization: Bearer unusable-bearer-fixture', 'unusable-bearer-fixture'),
            ('client_secret="unusable fixture with spaces"', 'unusable fixture with spaces'),
            ("{'session_token': 'unusable-session-fixture'}", 'unusable-session-fixture'),
            ('https://fake-user:fake-password@example.invalid/path', 'fake-password'),
            ('https://example.invalid/path?X-Amz-Signature=unusable-signature-fixture', 'unusable-signature-fixture'),
            ('Cookie: first=unusable-first; second=unusable-second', 'unusable-second'),
            ("{'private_key': 'unusable-private-fixture'}", 'unusable-private-fixture'),
            ("{'secret_key': 'unusable-secret-fixture'}", 'unusable-secret-fixture'),
            ("{'cookies': 'unusable-cookies-fixture'}", 'unusable-cookies-fixture'),
            ('AWS_SECRET_ACCESS_KEY=unusable-access-fixture', 'unusable-access-fixture'),
        ]
        for value, secret in samples:
            with self.subTest(form=value.split(':', 1)[0]):
                clean = redact_text(value)
                self.assertNotIn(secret, clean)
                self.assertIn('[REDACTED]', clean)
                self.assertEqual(redact_text(clean), clean)

    def test_noncredential_robot_diagnostics_remain_useful(self):
        value = 'IK rejected at xyz=[0.21, 0.02, 0.15], joint=3, code=42; totalTokens=17'
        self.assertEqual(redact_text(value), value)

    def test_runtime_unknown_dispatch_redacts_receipts_and_keeps_nonreplay(self):
        class Broken(MockDevice):
            def dispatch(self, prepared):
                self.calls = getattr(self, 'calls', 0) + 1
                raise RuntimeError('Authorization: Basic ZmFrZS11c2VyOmZha2UtcGFzcw==')
        with tempfile.TemporaryDirectory() as directory:
            device = Broken()
            runtime = ProgramRuntime(device, Path(directory) / 'runtime', run_id='privacy')
            try:
                reply = runtime.execute({'source': 'def main(robot, profile):\n    return robot.move_tcp([0.1, 0., 0.2], [0., 0., 0., 1.])', 'profile': {}, 'label': 'fixture'})
                self.assertTrue(reply['response']['result']['motion_outcome_uncertain'])
                with self.assertRaises(RuntimeError):
                    runtime.execute({'source': 'def main(robot, profile):\n    return robot.observe()', 'profile': {}, 'label': 'no_replay'})
                self.assertEqual(device.calls, 1)
                text = '\n'.join(p.read_text() for p in Path(directory).rglob('*.json*'))
                self.assertNotIn('ZmFrZS11c2VyOmZha2UtcGFzcw==', text)
                self.assertIn('[REDACTED]', text)
            finally:
                runtime.close()

    def test_runner_provider_and_cleanup_errors_are_redacted(self):
        class BrokenSession:
            def run_turn(self, *args, **kwargs):
                raise RuntimeError('client_secret=unusable-provider-fixture')
            def close(self):
                raise RuntimeError('id_token=unusable-cleanup-fixture')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = AgentRunner(session=BrokenSession(), device=MockDevice(), task='Observe only',
                                 knowledge=root / 'knowledge', output=root / 'run', scope=SCOPE)
            with self.assertRaises(RuntimeError):
                runner.run()
            text = (root / 'run/cleanup.json').read_text()
            self.assertNotIn('unusable-provider-fixture', text)
            self.assertNotIn('unusable-cleanup-fixture', text)
            self.assertIn('[REDACTED]', text)

    def test_factory_startup_failure_record_is_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'task': 'Observe only', 'scope': SCOPE,
                'knowledge': str(root / 'knowledge'),
                'device': {'factory': 'astra_robotics_agent.physical_keyboard.runtime.mock_device:factory', 'settings': {}},
                'model': {'factory': 'unused.factory:create', 'settings': {}}}))
            def broken(*args, **kwargs):
                raise RuntimeError('auth_token=unusable-startup-fixture')
            with patch('astra_robotics_agent.physical_keyboard.agent.runner.load_factory', return_value=broken):
                with self.assertRaises(RuntimeError):
                    run_config(json.loads(config.read_text()), output=root / 'run')
            text = (root / 'run/startup-failure.json').read_text()
            self.assertNotIn('unusable-startup-fixture', text)
            self.assertIn('[REDACTED]', text)


if __name__ == '__main__':
    unittest.main()
