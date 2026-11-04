"""Frozen goals and nonterminal learning checkpoints, without robot/model imports.

Completion here is a structured *model visual confirmation*, not an independent
vision evaluator, skill promotion, or permission to execute another action.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from astra_robotics_agent.physical_keyboard.records.source_inputs import _read_source


TASK_CONTRACT_PROTOCOL = "physical-keyboard-task-contract-v1"
CHAT_CONTRACT_PROTOCOL = "physical-keyboard-chat-contract-v1"
CHAT_COMPLETION_REQUIREMENTS = (
    "new_outgoing_message_visible", "compose_box_empty", "released_and_input_stable",
)
COMPLETION_REQUIREMENTS = (
    "exact_visible_text", "released_and_input_stable", "reusable_skill_saved",
)
_SCOPE_KEYS = {"task_id", "device_id", "layout_id"}
_CONTRACT_KEYS = {
    "protocol", "id", "version", "scope", "objective", "target_text",
    "completion_requirements",
}
_CHAT_CONTRACT_KEYS = (_CONTRACT_KEYS - {"target_text"}) | {
    "application", "recipient_scope", "send_key",
}
_PARTIAL_STATUSES = {"partial", "blocked", "view_ready", "calibrated_partial"}
_ASSISTANCE_KINDS = {
    "physical_safety", "hardware_fault", "operator_action",
    "missing_external_capability",
}


def _text(value, label, maximum=4096):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label}: a nonempty string of at most {maximum} characters is required")
    return value


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label}: exact keys required: {sorted(expected)}")


def _scope(value, expected=None):
    _keys(value, _SCOPE_KEYS, "scope")
    for key in _SCOPE_KEYS:
        _text(value[key], "scope." + key, 128)
    if expected is not None and value != expected:
        raise ValueError("task/device/layout scope does not match this campaign")


def _copy(value):
    """Copy JSON data without permitting nonfinite numbers or custom objects."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("campaign data must be finite JSON") from exc


@dataclass(frozen=True)
class TaskContract:
    """Validated source bytes plus immutable provenance; document access copies."""

    raw: bytes
    sha256: str
    _source_path: str
    _document_json: str

    @property
    def document(self):
        return json.loads(self._document_json)

    @property
    def is_chat(self):
        return self.document["protocol"] == CHAT_CONTRACT_PROTOCOL

    @property
    def is_continuous_chat(self):
        return self.is_chat and self.document.get('session_mode') == 'continuous'

    @property
    def is_responsive_chat(self):
        return self.is_continuous_chat and self.document.get('conversation_policy') == 'responsive_v1'

    @property
    def is_skill_development_training(self):
        from astra_robotics_agent.physical_keyboard.adapters.environments.skill_training import PROTOCOL
        return self.document['protocol'] == PROTOCOL

    @property
    def text(self):
        if self.is_skill_development_training:
            from astra_robotics_agent.physical_keyboard.adapters.environments.skill_training import contract_text
            return contract_text(self._document_json)
        if self.is_continuous_chat:
            return (
                'PERSISTENT PHYSICAL CHAT SESSION — UNTIL THE OPERATOR STOPS IT\n'
                + self._document_json
                + '\nA sent message, saved Skill/Lesson, ordinary assistance or phase '
                'summary does not end this session. Keep the same conversation, '
                'knowledge and cumulative costs. A completed finish records one '
                'nonterminal chat phase; wait_for_update waits without model calls. '
                'Use preserved send evidence even if a later incoming message scrolls '
                'that bubble out of view. Do not resend or request scrolling merely '
                'to keep historical evidence visible. You own text, caret, relevance, '
                'response choices and RSI. The runtime does not track typed text. '
                'Physical safety, hardware faults and unknown execution still stop.\n'
            )
        if self.is_chat:
            return (
                "FROZEN PHYSICAL CHAT CONTRACT — DRAFT TEXT IS NOT A SENT MESSAGE\n"
                + self._document_json
                + "\nThe operator selected the conversation authorized by recipient_scope. Stay in that "
                "conversation; do not select another recipient. You own reply content, "
                "text/caret understanding and physical corrections. Verify the draft in "
                "an external-camera observation before a separate send action, then "
                "verify the new outgoing bubble relative to an earlier observation. "
                "The framework does not read or track text and cannot independently "
                "verify recipient identity or sending. Save useful knowledge when "
                "justified; completion does not require creating a new Skill. "
                "Training reset is unavailable. Partial summaries remain checkpoints. "
                "Unknown execution/communication outcomes still stop without replay.\n"
            )
        return (
            "FROZEN OVERALL TASK CONTRACT — PHASE SUMMARIES ARE NOT COMPLETION\n"
            + self._document_json
            + "\nContinue autonomous exploration toward this overall objective after "
            "partial, blocked, view_ready, or calibrated_partial phase summaries. "
            "A completed claim requires a fresh visual completion review and an "
            "existing scoped reusable Skill candidate; candidate existence alone "
            "does not prove reliability. Confirmation is a model visual assessment, "
            "not an independent evaluator. The contract does not expand execution "
            "authority or override unknown-motion and safety stops.\n"
        )

    @property
    def metadata(self):
        data = self.document
        return {
            "protocol": data["protocol"], "id": data["id"],
            "version": data["version"], "scope": data["scope"],
            "source_path": self._source_path, "sha256": self.sha256,
            "bytes": len(self.raw), "origin": "frozen_task_contract",
        }


def load_task_contract(path, expected_scope: dict) -> TaskContract:
    """Read strict UTF-8 JSON using the existing no-symlink, 32 KiB reader.

    Duplicate JSON members, nonfinite values, symlinked path components,
    nonregular files and concurrent source changes are rejected by that reader.
    The returned document/metadata never expose internal mutable dictionaries.
    """
    _scope(expected_scope)
    source_path, raw, document = _read_source(path)
    from astra_robotics_agent.physical_keyboard.adapters.environments.skill_training import PROTOCOL, validate_document
    if type(document) is dict and document.get('protocol') == PROTOCOL:
        validate_document(document, expected_scope)
        return TaskContract(raw=raw, sha256=hashlib.sha256(raw).hexdigest(),
            _source_path=source_path,
            _document_json=json.dumps(document, ensure_ascii=False, sort_keys=True,
                                      indent=2, allow_nan=False))
    if (type(document) is not dict or type(document.get("protocol")) is not str
            or document.get("protocol") not in {
                TASK_CONTRACT_PROTOCOL, CHAT_CONTRACT_PROTOCOL}):
        raise ValueError("unexpected task contract protocol")
    chat = document["protocol"] == CHAT_CONTRACT_PROTOCOL
    expected_keys = _CHAT_CONTRACT_KEYS if chat else _CONTRACT_KEYS
    if chat and 'session_mode' in document:
        expected_keys = expected_keys | {'session_mode'}
        if document['session_mode'] not in ('continuous', 'single_exchange'):
            raise ValueError('chat session_mode must be continuous or single_exchange')
    if chat and 'conversation_policy' in document:
        expected_keys = expected_keys | {'conversation_policy'}
        if (document['conversation_policy'] != 'responsive_v1'
                or document.get('session_mode') != 'continuous'):
            raise ValueError('conversation_policy responsive_v1 requires continuous chat')
    _keys(document, expected_keys, "task contract")
    if (type(document["id"]) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", document["id"]) is None):
        raise ValueError("contract id must be 1..64 ASCII letters/digits/_/-")
    if type(document["version"]) is not int or document["version"] < 1:
        raise ValueError("contract version must be a positive integer")
    _scope(document["scope"], expected_scope)
    _text(document["objective"], "objective")
    if chat:
        if (document["application"] != "wechat"
                or document["recipient_scope"] not in ("operator_selected_self_account", "operator_selected_group", "operator_selected_direct_chat")
                or document["send_key"] not in ("unconfirmed", "Return")):
            raise ValueError("chat requires WeChat, an explicitly operator-selected conversation, and a declared send key")
    else:
        target = _text(document["target_text"], "target_text", 256)
        if re.fullmatch(r"[a-z ]+", target) is None:
            raise ValueError("target_text may contain only lowercase English letters and spaces")
    requirements = CHAT_COMPLETION_REQUIREMENTS if chat else COMPLETION_REQUIREMENTS
    if document["completion_requirements"] != list(requirements):
        raise ValueError("completion_requirements must exactly match the three frozen requirements")
    return TaskContract(
        raw=raw, sha256=hashlib.sha256(raw).hexdigest(),
        _source_path=source_path,
        _document_json=json.dumps(document, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False),
    )


class CampaignLifecycle:
    """Pure bookkeeping for one goal; no implicit reset, actions or retries."""

    def __init__(self, contract: TaskContract):
        if not isinstance(contract, TaskContract):
            raise ValueError("a validated TaskContract is required")
        self.contract = contract
        self.checkpoint_count = 0
        self._pending_completion = None
        self._observations = []

    def record_observation(self, evidence_id):
        """Track evidence provenance only, never text, keys or caret state."""
        if self.contract.is_chat:
            _text(evidence_id, "observation_evidence_id", 256)
            if evidence_id in self._observations:
                raise ValueError("chat observation evidence IDs must be new")
            self._observations.append(evidence_id)

    @property
    def pending_completion(self):
        return _copy(self._pending_completion)

    def checkpoint(self, result, requested_action="checkpoint") -> dict:
        if type(result) is not dict:
            raise ValueError("checkpoint result must be an object")
        if type(requested_action) is not str or requested_action not in {"checkpoint", "finish"}:
            raise ValueError("checkpoint requested_action must be checkpoint or finish")
        if type(result.get("status")) is not str or result["status"] not in _PARTIAL_STATUSES:
            raise ValueError("checkpoint requires a nonterminal partial/blocked/view status")
        result = _copy(result)
        self.checkpoint_count += 1
        return {
            "phase": "checkpoint", "terminal": False,
            "checkpoint_number": self.checkpoint_count,
            "requested_action": requested_action, "result": result,
            "task_contract": self.contract.document,
            "instruction": (
                "This records a learning phase, not the end of the overall task. "
                "Keep the same conversation, knowledge and cumulative costs; "
                "choose your next autonomous action toward the frozen objective. "
                "Ordinary uncertainty or failed probes are not external blockers."
            ),
        }

    def begin_completion_review(self, result, observation_evidence_id) -> dict:
        if type(result) is not dict or result.get("status") != "completed":
            raise ValueError("only status=completed may request a completion review")
        result = _copy(result)
        evidence_id = _text(observation_evidence_id, "observation_evidence_id", 256)
        self._pending_completion = {
            "result": result, "review_observation_evidence_id": evidence_id,
        }
        instruction = (
            "Review the newly supplied camera observation. Confirm a NEW outgoing "
            "message bubble in the selected authorized conversation relative to a registered "
            "earlier observation, the empty compose box and released typing petal. "
            "A draft, old bubble or SDK success alone does not show sending. This is "
            "your model visual claim, not an independent evaluator. Continue if unsure."
        ) if self.contract.is_chat else (
            "Review the newly supplied camera observation, not the earlier "
            "completed claim. Confirm the exact target text, released and "
            "stable input, and existing reusable scoped Skill candidate IDs. "
            "A saved candidate alone is not proof of a reliable skill. "
            "If any requirement is unmet, continue learning instead. "
            "Confirmation is your visual assessment, not an independent evaluator."
        )
        return {
            "phase": "completion_review", "terminal": False,
            "result": _copy(result),
            "review_observation_evidence_id": evidence_id,
            "task_contract": self.contract.document,
            "instruction": instruction,
        }

    def confirm_completion(self, arguments, current_observation_evidence_id, get_skill) -> dict:
        if self.contract.is_chat:
            return self._confirm_chat(arguments, current_observation_evidence_id)
        _keys(arguments, {
            "review_observation_evidence_id", "observed_text",
            "released_and_input_stable", "skill_candidate_ids",
        }, "completion confirmation")
        if self._pending_completion is None:
            raise ValueError("no pending fresh completion review")
        expected_id = self._pending_completion["review_observation_evidence_id"]
        if (arguments["review_observation_evidence_id"] != expected_id
                or current_observation_evidence_id != expected_id):
            raise ValueError("completion review requires the current fresh observation ID")
        if arguments["observed_text"] != self.contract.document["target_text"]:
            raise ValueError("observed_text does not exactly match target_text")
        if arguments["released_and_input_stable"] is not True:
            raise ValueError("released_and_input_stable must be true")
        identifiers = arguments["skill_candidate_ids"]
        if type(identifiers) is not list or not identifiers or len(identifiers) > 64:
            raise ValueError("1..64 saved Skill candidate IDs are required")
        for identifier in identifiers:
            _text(identifier, "skill_candidate_id", 256)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Skill candidate IDs must be unique")
        scope = self.contract.document["scope"]
        for identifier in identifiers:
            # The real store verifies registration, scope, manifest, source and
            # profile checksums before returning a usable candidate program.
            skill = get_skill(identifier)
            if (type(skill) is not dict or skill.get("candidate_id") != identifier
                    or type(skill.get("profile")) is not dict):
                raise ValueError("scoped Skill lookup returned an invalid candidate")
            _text(skill.get("source"), "skill.source", 64 * 1024)
            _text(skill.get("label"), "skill.label", 256)
            _copy(skill["profile"])
            if "scope" in skill:
                _scope(skill["scope"], scope)
            if "kind" in skill and skill["kind"] != "skill":
                raise ValueError("completion requires a Skill candidate, not another kind")
        result = _copy(self._pending_completion["result"])
        result["status"] = "completed"
        result["completion_review"] = {
            **_copy(arguments), "basis": "model_visual_confirmation",
            "independently_verified": False,
        }
        self._pending_completion = None
        return result

    def _confirm_chat(self, arguments, current_observation_evidence_id):
        _keys(arguments, {
            "review_observation_evidence_id", "before_observation_evidence_id",
            "recipient_label", "observed_outgoing_text", "new_outgoing_message_visible",
            "compose_box_empty", "released_and_input_stable",
        }, "chat completion confirmation")
        if self._pending_completion is None:
            raise ValueError("no pending fresh completion review")
        review = self._pending_completion["review_observation_evidence_id"]
        if (arguments["review_observation_evidence_id"] != review
                or current_observation_evidence_id != review
                or not self._observations or self._observations[-1] != review):
            raise ValueError("chat completion requires the current fresh review observation")
        before = arguments["before_observation_evidence_id"]
        if type(before) is not str or before not in self._observations[:-1]:
            raise ValueError("chat completion requires a registered earlier observation")
        _text(arguments["recipient_label"], "recipient_label", 256)
        text = _text(arguments["observed_outgoing_text"], "observed_outgoing_text", 256)
        if re.fullmatch(r"[a-z0-9]+(?: [a-z0-9]+)*", text) is None:
            raise ValueError("chat reply must use supported lowercase English, digits and spaces")
        for field in CHAT_COMPLETION_REQUIREMENTS:
            if arguments[field] is not True:
                raise ValueError(field + " must be true; a draft is not a sent message")
        result = _copy(self._pending_completion["result"])
        result["status"] = "completed"
        result["completion_review"] = {
            **_copy(arguments), "basis": "model_visual_confirmation",
            "independently_verified": False,
        }
        self._pending_completion = None
        return result

    def invalidate_completion_review(self):
        """Caller invokes this after action/reset/observation changes."""
        self._pending_completion = None

    def validate_assistance(self, arguments, current_observation_evidence_id) -> dict:
        _keys(arguments, {
            "kind", "reason", "required_human_action", "observation_evidence_id",
        }, "external assistance")
        if type(arguments["kind"]) is not str or arguments["kind"] not in _ASSISTANCE_KINDS:
            raise ValueError("assistance must name an allowed external/safety blocker")
        _text(arguments["reason"], "reason")
        _text(arguments["required_human_action"], "required_human_action")
        evidence_id = _text(arguments["observation_evidence_id"], "observation_evidence_id", 256)
        if evidence_id != current_observation_evidence_id:
            raise ValueError("external assistance requires the current observation ID")
        return {
            **_copy(arguments), "status": "assistance_required", "terminal": True,
            "basis": "model_requested_external_assistance",
        }
