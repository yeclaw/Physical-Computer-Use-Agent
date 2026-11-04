"""Opt-in local read boundary for experiments with withheld prior knowledge.

This protects enumerated local paths in the app-server and its children. It is
not a claim that every possible provider-side tool or external service is an
allowlisted tool. The default Codex client remains unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


class ReadIsolation:
    def __init__(self, document):
        if (type(document) is not dict or set(document) != {'protocol', 'deny_paths', 'probe_paths'}
                or document['protocol'] != 'codex-local-read-isolation-v1'):
            raise ValueError('invalid read isolation policy')
        def paths(values):
            if type(values) is not list or not values or len(values) > 128:
                raise ValueError('explicit bounded isolation paths required')
            result = []
            for value in values:
                if type(value) is not str or not Path(value).is_absolute() or '\x00' in value:
                    raise ValueError('absolute isolation paths required')
                path = Path(value).resolve()
                if path == Path('/') or path in result:
                    raise ValueError('duplicate or root isolation path')
                result.append(path)
            return result
        self.deny_paths = paths(document['deny_paths'])
        self.probe_paths = paths(document['probe_paths'])
        if any(not any(p == r or r in p.parents for r in self.deny_paths) for p in self.probe_paths):
            raise ValueError('each read probe must be within a denied path')
        self.document = json.loads(json.dumps(document))
        self._frozen_config = None
        self.sha256 = hashlib.sha256(json.dumps(document, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        self.profile = '(version 1)(allow default)' + ''.join(
            '(deny file-read* (subpath %s))' % json.dumps(str(p)) for p in self.deny_paths)

    def prefix(self, cwd):
        cwd = Path(cwd).resolve()
        if any(cwd == p or p in cwd.parents for p in self.deny_paths):
            raise ValueError('model workspace is within withheld paths')
        if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').is_file():
            raise RuntimeError('configured local read boundary is unavailable; no fallback')
        return ['/usr/bin/sandbox-exec', '-p', self.profile]

    def config_overrides(self):
        # Read names only. Never copy/log credentials or edit the user config.
        if self._frozen_config is not None:
            return dict(self._frozen_config)
        config = {'developer_instructions': '', 'project_doc_max_bytes': 0}
        codex_root = Path(os.environ.get('CODEX_HOME') or (Path.home() / '.codex')).resolve()
        for name in ('sessions', 'archived_sessions', 'memories', 'history.jsonl', 'AGENTS.md', 'skills', 'plugins'):
            protected = codex_root / name
            if not any(protected == p or p in protected.parents for p in self.deny_paths):
                raise ValueError('actual Codex context paths require explicit isolation policy coverage')
        path = codex_root / 'config.toml'
        if path.is_file():
            data = tomllib.loads(path.read_text())
            def custom_instructions(value):
                if type(value) is dict:
                    return any((key in {'model_instructions_file', 'experimental_instructions_file'} and item)
                               or custom_instructions(item) for key, item in value.items())
                return type(value) is list and any(custom_instructions(v) for v in value)
            if custom_instructions(data):
                raise ValueError('custom model instructions require separate input review')
            for name in data.get('mcp_servers', {}):
                if not name or not all(c.isalnum() or c in '_-' for c in name):
                    raise ValueError('MCP name cannot be safely overridden')
                config['mcp_servers.' + name + '.enabled'] = False
        return config

    def preflight(self, cwd):
        """No Codex/model launch: prove actual denied opens and one allowed read."""
        prefix = self.prefix(cwd)
        overrides = self.config_overrides()
        results = []
        for path in self.probe_paths:
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
                raise ValueError('read probe must be an existing small regular file')
            result = subprocess.run(prefix + ['/bin/cat', str(path)], cwd=str(cwd),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=10, check=False)
            # sandbox_apply failure alone must never count as read denial.
            denied = result.returncode != 0 and b'Operation not permitted' in result.stderr and b'sandbox_apply' not in result.stderr
            results.append(dict(path=str(path), denied=denied, returncode=result.returncode))
            if not denied:
                raise RuntimeError('local read boundary did not prove denial: ' + str(path))
        control = Path(cwd) / 'read-isolation-control.txt'
        with control.open('xb') as stream:
            stream.write(b'allowed-control-only\n')
        try:
            result = subprocess.run(prefix + ['/bin/cat', str(control)], cwd=str(cwd),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=10, check=False)
            if result.returncode != 0 or result.stdout != b'allowed-control-only\n':
                raise RuntimeError('local read boundary blocked the clean workspace')
        finally:
            control.unlink()
        self._frozen_config = dict(overrides)
        return dict(policy_sha256=self.sha256, local_read_probes=results,
                    validated_config_override_keys=sorted(overrides),
                    clean_workspace_read=True, model_started=False,
                    meaning='enumerated local path denial only; not a provider tool allowlist')
