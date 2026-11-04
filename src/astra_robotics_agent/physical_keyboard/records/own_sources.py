"""Host-only own-origin journal, bound to one root and starting source policy.

This grants no legacy knowledge IDs. A copied journal from another root or a
broader starting condition is rejected; model-supplied origins are not input.
"""
import json
from pathlib import Path

from .learning_contracts import LearningContractError, digest, encoded, exact, identifier
from .storage import read_bytes, write_new, append_bytes


class OwnSourceLedger:
    def __init__(self, path, knowledge_root, base_policy):
        self.path = Path(path).absolute()
        root = Path(knowledge_root).resolve()
        if self.path.parent != root or type(base_policy) is not dict:
            raise LearningContractError('own ledger requires this knowledge root and an explicit base policy')
        root.mkdir(parents=True, exist_ok=True)
        self.base_policy = base_policy
        self.header = dict(protocol='physical-keyboard-own-sources-v1', knowledge_root=str(root),
                           base_policy_sha256=digest(base_policy))
        if not self.path.exists() and not self.path.is_symlink():
            write_new(self.path, encoded(self.header) + b'\n')
        raw = read_bytes(self.path, maximum=4 * 1024 * 1024)
        if not raw.endswith(b'\n'):
            raise LearningContractError('own source journal has an incomplete tail')
        lines = raw.splitlines()
        if not lines or lines[0] != encoded(self.header):
            raise LearningContractError('own source journal belongs to another root or starting policy')
        self.evidence_ids = set()
        self.tail = digest(self.header)
        for line in lines[1:]:
            value = json.loads(line)
            exact(value, {'evidence_id', 'previous', 'sha256'})
            identifier(value['evidence_id'])
            unsigned = dict(evidence_id=value['evidence_id'], previous=value['previous'])
            if (value['previous'] != self.tail or value['sha256'] != digest(unsigned)
                    or value['evidence_id'] in self.evidence_ids or line != encoded(value)):
                raise LearningContractError('own source journal integrity mismatch')
            self.evidence_ids.add(value['evidence_id']); self.tail = value['sha256']

    def record(self, evidence_id):
        identifier(evidence_id)
        if evidence_id not in self.evidence_ids:
            value = dict(evidence_id=evidence_id, previous=self.tail)
            value['sha256'] = digest(value)
            append_bytes(self.path, encoded(value) + b'\n')
            self.evidence_ids.add(evidence_id); self.tail = value['sha256']
