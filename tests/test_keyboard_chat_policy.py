"""Pure-schema coverage for bounded, auditable responsive chat decisions."""
import copy
import math
import unittest

from astra_robotics_agent.physical_keyboard.chat_policy import (
    POLICY_ID, RESPONSIVE_CHAT_INSTRUCTIONS, validate_wait_assessment,
)


def assessment(**changes):
    result = dict(observation_evidence_id="run-obs-0027",
                  summary="A visible question quotes another person's earlier message.",
                  reply_target="visible quoted sender; own identity unclear",
                  addressing="unclear",
                  reason="No new direct question is established in this observation.")
    result.update(changes)
    return result


class WaitAssessmentTests(unittest.TestCase):
    def test_visible_task_report_is_preserved_in_a_detached_dictionary(self):
        original = assessment(summary="来信引用了另一条消息，回复对象不确定。")
        before = copy.deepcopy(original)
        result = validate_wait_assessment(original)
        self.assertEqual(result, before)
        self.assertIsNot(result, original)
        original["reason"] = "caller modified input"
        self.assertEqual(result["reason"], before["reason"])
        result["summary"] = "caller modified returned data"
        self.assertEqual(original["summary"], before["summary"])

    def test_all_declared_addressing_values_are_accepted(self):
        for value in ("to_me", "general", "unclear", "none"):
            with self.subTest(value=value):
                self.assertEqual(validate_wait_assessment(assessment(addressing=value))["addressing"], value)

    def test_exact_keys_prevent_hidden_state_or_extra_instruction_channels(self):
        for missing in assessment():
            value = assessment(); del value[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                validate_wait_assessment(value)
        for field, extra in (("chain_of_thought", "private reasoning"),
                             ("expected_text", "not a framework text tracker"),
                             ("caret", 4), ("new_authority", {"move": True}),
                             ("extra", math.nan)):
            with self.subTest(extra=field), self.assertRaises(ValueError):
                validate_wait_assessment(assessment(**{field: extra}))

    def test_container_must_be_an_exact_dictionary(self):
        class CustomDict(dict):
            pass
        for value in (None, True, 1, "{}", [], list(assessment().items()), CustomDict(assessment())):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                validate_wait_assessment(value)

    def test_keys_must_be_exact_strings_not_string_subclasses(self):
        class CustomStr(str):
            pass
        value = assessment()
        value[CustomStr("summary")] = value.pop("summary")
        with self.assertRaises(ValueError):
            validate_wait_assessment(value)
        value = assessment(); value[1] = value.pop("summary")
        with self.assertRaises(ValueError):
            validate_wait_assessment(value)

    def test_all_fields_require_exact_strings_and_reject_nonfinite_and_nested_values(self):
        class CustomStr(str):
            pass
        for field in assessment():
            for value in (None, False, True, 0, 1.5, math.nan, math.inf, -math.inf,
                          [], ["message"], {}, {"summary": "nested"},
                          CustomStr("unclear"), object()):
                with self.subTest(field=field, value_type=type(value).__name__), self.assertRaises(ValueError):
                    validate_wait_assessment(assessment(**{field: value}))

    def test_empty_or_whitespace_fields_are_rejected(self):
        for field in assessment():
            for value in ("", " ", "\n\t", "\u3000"):
                with self.subTest(field=field, value=repr(value)), self.assertRaises(ValueError):
                    validate_wait_assessment(assessment(**{field: value}))

    def test_length_boundaries_preserve_compact_audit_records(self):
        limits = {"observation_evidence_id": 256, "summary": 400,
                  "reply_target": 160, "reason": 400}
        for field, maximum in limits.items():
            with self.subTest(field=field):
                self.assertEqual(validate_wait_assessment(assessment(**{field: "x" * maximum}))[field],
                                 "x" * maximum)
                with self.assertRaises(ValueError):
                    validate_wait_assessment(assessment(**{field: "x" * (maximum + 1)}))

    def test_unknown_or_case_changed_addressing_is_not_silently_normalized(self):
        for value in ("self", "TO_ME", "to_me ", "reply", "to_me|general|unclear|none", "x" * 17):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_wait_assessment(assessment(addressing=value))

    def test_evidence_id_is_reported_without_claiming_freshness_or_independent_truth(self):
        # A syntactically valid stale/foreign id reaches the calling loop's
        # current-evidence check; this pure validator cannot establish freshness.
        value = assessment(observation_evidence_id="foreign-obs-0001")
        self.assertEqual(validate_wait_assessment(value)["observation_evidence_id"],
                         "foreign-obs-0001")


class ResponsivePolicyContractTests(unittest.TestCase):
    def test_policy_exposes_existing_tools_and_responsive_audit_contract(self):
        self.assertEqual(POLICY_ID, "responsive_v1")
        for required in ("fresh incoming human messages", "arguments.assessment",
                         "observation_evidence_id", "reply_target", "addressing",
                         "existing hypothesis field", "not hidden reasoning"):
            with self.subTest(required=required):
                self.assertIn(required, RESPONSIVE_CHAT_INSTRUCTIONS)

    def test_continuous_typing_retains_scoped_source_evidence_and_send_boundary(self):
        for required in ("robot.play_sequence", "h/e/l/o/w/r/d/Space",
                         "longest reliable", "fall back locally", "parameterized Skill",
                         "press-and-release", "source trajectory identity/SHA",
                         "original SDK protections", "no optional", "separate confirmed Return"):
            with self.subTest(required=required):
                self.assertIn(required, RESPONSIVE_CHAT_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
