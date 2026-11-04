"""Typing demand protocol tests: no model, service, GUI or robot is involved."""

import copy
import hashlib
import json
import random
import unittest

from astra_robotics_agent.physical_keyboard.adapters.environments.typing_cases import (
    ALPHABET, CONFIG_PROTOCOL, DEFAULT_CONFIG, LENGTH_BINS,
    generate_manifest, manifest_sha256, validate_manifest,
)


def config(seed=0, cases_per_bin=4):
    return {"protocol": CONFIG_PROTOCOL, "seed": seed, "cases_per_bin": cases_per_bin}


class TypingManifestGenerationTests(unittest.TestCase):
    def test_default_batch_and_honest_character_coverage(self):
        manifest = generate_manifest(copy.deepcopy(DEFAULT_CONFIG))
        self.assertEqual(len(manifest["base_cases"]), 16)
        self.assertEqual(len(manifest["occurrences"]), 48)
        targets = manifest["base_cases"]
        self.assertEqual(len({case["target_text"] for case in targets}), 16)
        coverage = manifest["coverage"]
        self.assertEqual(coverage["alphabet"], ALPHABET)
        self.assertEqual(coverage["counts_basis"], "base_cases")
        self.assertEqual(coverage["missing_chars"], [])
        self.assertEqual(set(coverage["requested_counts"]), set(ALPHABET))
        for char in ALPHABET:
            self.assertEqual(coverage["requested_counts"][char], sum(
                case["target_text"].count(char) for case in targets))
            self.assertGreater(coverage["requested_counts"][char], 0)
        self.assertEqual(coverage["space_count"], sum(
            case["target_text"].count(" ") for case in targets))
        for name in LENGTH_BINS:
            group = [case for case in targets if case["length_bin"] == name]
            text = "".join(case["target_text"] for case in group)
            row = coverage["by_bin"][name]
            self.assertEqual(row["requested_counts"], {
                char: text.count(char) for char in ALPHABET})
            self.assertEqual(row["missing_chars"], [char for char in ALPHABET if char not in text])
            self.assertEqual(row["space_count"], text.count(" "))
            self.assertEqual(len(group), 4)
        # Tiny targets cannot cover 36 characters in four strings of length <=3.
        self.assertGreaterEqual(len(coverage["by_bin"]["tiny"]["missing_chars"]), 24)

    def test_lengths_families_and_schedule_for_all_batch_sizes_and_edge_seeds(self):
        for seed in (0, 1, 25, 2 ** 32 - 1):
            for count in range(1, 17):
                with self.subTest(seed=seed, count=count):
                    manifest = generate_manifest(config(seed, count))
                    self.assertEqual(validate_manifest(manifest), manifest)
                    registered = {case["case_id"]: case for case in manifest["base_cases"]}
                    self.assertEqual(len(registered), 4 * count)
                    for case in registered.values():
                        target = case["target_text"]
                        low, high = LENGTH_BINS[case["length_bin"]]
                        self.assertTrue(low <= len(target) <= high)
                        self.assertEqual(target, target.strip())
                        self.assertTrue(set(target) <= set(ALPHABET + " "))
                    positions = {case_id: [] for case_id in registered}
                    occurrence_ids = set()
                    for index, occurrence in enumerate(manifest["occurrences"]):
                        occurrence_ids.add(occurrence["occurrence_id"])
                        positions[occurrence["case_id"]].append(index)
                        repeat = len(positions[occurrence["case_id"]])
                        self.assertEqual(occurrence["repeat_index"], repeat)
                        self.assertEqual(occurrence["role"], ("initial", "immediate", "delayed")[repeat - 1])
                    self.assertEqual(len(occurrence_ids), 12 * count)
                    for first, immediate, delayed in positions.values():
                        self.assertEqual(immediate, first + 1)
                        self.assertTrue(2 <= delayed - immediate - 1 <= 5)
                    if count >= 4:
                        self.assertEqual({case["content_type"] for case in registered.values()},
                                         {"natural", "coverage_combination", "letter_digit_mix",
                                          "repeated_alternating"})

    def test_deterministic_seed_changes_and_no_shared_mutation_or_global_rng(self):
        source = config()
        original = copy.deepcopy(source)
        global_rng_state = random.getstate()
        first = generate_manifest(source)
        second = generate_manifest(source)
        self.assertEqual(first, second)
        self.assertEqual(source, original)
        self.assertEqual(random.getstate(), global_rng_state)
        self.assertNotEqual(first["batch_id"], generate_manifest(config(seed=1))["batch_id"])
        self.assertNotEqual(first["base_cases"], generate_manifest(config(seed=1))["base_cases"])
        self.assertNotEqual(first["occurrences"], generate_manifest(config(seed=1))["occurrences"])
        validated = validate_manifest(first)
        validated["base_cases"][0]["target_text"] = "changed"
        validated["coverage"]["requested_counts"]["a"] = -1
        second["config"]["seed"] = 99
        self.assertEqual(first, generate_manifest(original))

    def test_hash_is_exact_canonical_json_without_newline_and_key_order_independent(self):
        manifest = generate_manifest(config())
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
        expected = hashlib.sha256(raw).hexdigest()
        self.assertEqual(manifest_sha256(manifest), expected)
        self.assertNotEqual(expected, hashlib.sha256(raw + b"\n").hexdigest())
        reordered = dict(reversed(list(manifest.items())))
        self.assertEqual(manifest_sha256(reordered), expected)
        self.assertEqual(manifest_sha256(json.loads(raw)), expected)


class TypingManifestRejectionTests(unittest.TestCase):
    def test_config_exact_fields_ranges_and_types(self):
        bad = [None, [], (), {}, dict(config(), extra=1)]
        for field in config():
            value = config()
            del value[field]
            bad.append(value)
        for value in (None, True, False, -1, 2 ** 32, 0.0, "0", [], {}):
            bad.append(dict(config(), seed=value))
        for value in (None, True, False, 0, -1, 17, 4.0, "4", [], {}):
            bad.append(dict(config(), cases_per_bin=value))
        for value in (None, True, 1, [], "other-protocol"):
            bad.append(dict(config(), protocol=value))
        for value in bad:
            with self.subTest(config=value), self.assertRaises(ValueError):
                generate_manifest(value)

    def assert_invalid(self, change):
        manifest = generate_manifest(config())
        change(manifest)
        before = copy.deepcopy(manifest)
        with self.assertRaises(ValueError):
            validate_manifest(manifest)
        self.assertEqual(manifest, before)
        with self.assertRaises(ValueError):
            manifest_sha256(manifest)

    def test_manifest_schema_and_references(self):
        for value in (None, [], (), {}, {"protocol": "other"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_manifest(value)
        mutations = [
            lambda m: m.update(extra=True),
            lambda m: m.update(protocol="other"),
            lambda m: m.update(batch_id=""),
            lambda m: m.update(base_cases=tuple(m["base_cases"])),
            lambda m: m["base_cases"].pop(),
            lambda m: m["base_cases"][0].update(extra=True),
            lambda m: m["base_cases"][0].update(case_id=[]),
            lambda m: m["base_cases"][0].update(content_type=""),
            lambda m: m["base_cases"][0].update(length_bin=[]),
            lambda m: m["base_cases"][0].update(length_bin="long"),
            lambda m: m.update(occurrences=tuple(m["occurrences"])),
            lambda m: m["occurrences"].pop(),
            lambda m: m["occurrences"][0].update(extra=True),
            lambda m: m["occurrences"][0].update(case_id="unknown-case"),
            lambda m: m["occurrences"][0].update(occurrence_id=None),
            lambda m: m["occurrences"][1].update(occurrence_id=m["occurrences"][0]["occurrence_id"]),
            lambda m: m["base_cases"][1].update(case_id=m["base_cases"][0]["case_id"]),
            lambda m: m["base_cases"][1].update(target_text=m["base_cases"][0]["target_text"]),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.assert_invalid(mutation)

    def test_target_restrictions_and_repeat_integrity(self):
        for text in ("", " A", "A", "a!", "\na", "a\n", "a\t", "中文", " a", "a ", 1, None):
            with self.subTest(text=text):
                self.assert_invalid(lambda m: m["base_cases"][0].update(target_text=text))
        for value in (True, "1", 1.0, 0, 2, 4):
            with self.subTest(repeat=value):
                self.assert_invalid(lambda m: m["occurrences"][0].update(repeat_index=value))
        self.assert_invalid(lambda m: m["occurrences"][0].update(role="delayed"))

        def break_immediate(m):
            m["occurrences"][1], m["occurrences"][2] = m["occurrences"][2], m["occurrences"][1]

        def make_gap_too_short(m):
            delayed = m["occurrences"].pop(6)
            m["occurrences"].insert(2, delayed)

        def make_gap_too_long(m):
            delayed = m["occurrences"].pop(6)
            m["occurrences"].append(delayed)

        for mutation in (break_immediate, make_gap_too_short, make_gap_too_long):
            with self.subTest(mutation=mutation.__name__):
                self.assert_invalid(mutation)

    def test_coverage_and_valid_looking_provenance_tampering(self):
        mutations = [
            lambda m: m["coverage"].update(extra=True),
            lambda m: m["coverage"].update(counts_basis="successes"),
            lambda m: m["coverage"].update(alphabet=ALPHABET[::-1]),
            lambda m: m["coverage"].update(space_count=True),
            lambda m: m["coverage"].update(missing_chars=["a"]),
            lambda m: m["coverage"].update(missing_chars=()),
            lambda m: m["coverage"]["requested_counts"].update(a=999),
            lambda m: m["coverage"]["requested_counts"].update(a=True),
            lambda m: m["coverage"]["requested_counts"].update(a=1.0),
            lambda m: m["coverage"]["by_bin"].pop("tiny"),
            lambda m: m["coverage"]["by_bin"]["tiny"]["missing_chars"].clear(),
            lambda m: m["config"].update(seed=1),
            lambda m: m.update(batch_id="different-valid-id"),
            lambda m: m["base_cases"][0].update(content_type="different-family"),
            lambda m: m["base_cases"].reverse(),
            lambda m: m.update(occurrences=m["occurrences"][12:] + m["occurrences"][:12]),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.assert_invalid(mutation)


if __name__ == "__main__":
    unittest.main()
