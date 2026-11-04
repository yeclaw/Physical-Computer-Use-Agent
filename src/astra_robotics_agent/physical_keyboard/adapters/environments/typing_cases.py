"""Deterministic, offline typing demand; this module never evaluates success.

Lengths include spaces. Coverage counts each base target once, so repetition
does not disguise a gap in the requested character distribution. No files,
models, corpora, application state or robot interfaces are accessed here.
"""

import copy
import hashlib
import json
import random
import re


CONFIG_PROTOCOL = "physical-keyboard-typing-case-config-v1"
PROTOCOL = "physical-keyboard-typing-cases-v1"
SUBSET_CONFIG_PROTOCOL = "physical-keyboard-typing-subset-config-v1"
SUBSET_PROTOCOL = "physical-keyboard-typing-cases-subset-v1"
CURRICULUM_PROTOCOL = "physical-keyboard-typing-curriculum-v1"
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
LENGTH_BINS = {"tiny": (1, 3), "short": (4, 10),
               "medium": (11, 25), "long": (26, 60)}
DEFAULT_CONFIG = {"protocol": CONFIG_PROTOCOL, "seed": 0, "cases_per_bin": 4}

_FAMILIES = ("coverage_combination", "natural", "letter_digit_mix",
             "repeated_alternating")
# Small original examples, rather than a downloaded or exhaustive corpus.
_PHRASES = {
    "tiny": ("go", "up", "hi", "yes"),
    "short": ("we go", "go home", "look here", "move up"),
    "medium": ("we bring the box", "put the cup here", "we can try again",
               "bring the red bag"),
    "long": ("bring the blue box to the quiet room",
             "we can carry the small bag to the desk",
             "put the green cup beside the brown box",
             "bring our new map and wait by the door"),
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
_MANIFEST_FIELDS = {"protocol", "config", "batch_id", "base_cases",
                    "occurrences", "coverage"}
_CASE_FIELDS = {"case_id", "target_text", "length_bin", "content_type"}
_OCCURRENCE_FIELDS = {"case_id", "occurrence_id", "repeat_index", "role"}
_COVERAGE_FIELDS = {"requested_counts", "space_count", "missing_chars"}


def _object(value, fields, label):
    if type(value) is not dict or set(value) != fields:
        raise ValueError("%s must be an object with exactly %s" %
                         (label, ", ".join(sorted(fields))))


def _identifier(value, label):
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("%s must be a bounded identifier" % label)


def _config(value):
    _object(value, {"protocol", "seed", "cases_per_bin"}, "config")
    if type(value["protocol"]) is not str or value["protocol"] != CONFIG_PROTOCOL:
        raise ValueError("unsupported typing case config protocol")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2 ** 32:
        raise ValueError("seed must be an integer in 0..2**32-1")
    if (type(value["cases_per_bin"]) is not int
            or not 1 <= value["cases_per_bin"] <= 16):
        raise ValueError("cases_per_bin must be an integer in 1..16")
    return copy.deepcopy(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _coverage(base_cases):
    def counts(cases):
        text = "".join(case["target_text"] for case in cases)
        requested = {char: text.count(char) for char in ALPHABET}
        return {"requested_counts": requested, "space_count": text.count(" "),
                "missing_chars": [char for char in ALPHABET if not requested[char]]}

    result = counts(base_cases)
    result.update(alphabet=ALPHABET, counts_basis="base_cases", by_bin={
        name: counts([case for case in base_cases if case["length_bin"] == name])
        for name in LENGTH_BINS})
    return result


def _text(rng, length_bin, family, index, natural):
    lower, upper = LENGTH_BINS[length_bin]
    if family == "natural":
        return natural[index // len(_FAMILIES)]
    if family == "coverage_combination":
        # One long coverage target contains all 36 characters, even in a
        # small batch; the other bins honestly retain their coverage gaps.
        width = rng.randint(max(lower, 36) if length_bin == "long" else lower, upper)
        cycle = list(ALPHABET)
        rng.shuffle(cycle)
        return ("".join(cycle) * ((width + 35) // 36))[:width]
    if family == "letter_digit_mix":
        width = rng.randint(max(2, lower), upper)
        chars = [rng.choice(ALPHABET[:26]), rng.choice(ALPHABET[26:])]
        chars.extend(rng.choice(ALPHABET) for unused in range(width - 2))
        rng.shuffle(chars)
        return "".join(chars)
    width = rng.randint(max(2, lower), upper)
    pair = rng.sample(ALPHABET, 2)
    if rng.randrange(2):
        pair[1] = pair[0]
    return ("".join(pair) * ((width + 1) // 2))[:width]


def _build_manifest(config):
    rng = random.Random(config["seed"])
    base_cases = []
    used = set()
    for length_bin in LENGTH_BINS:
        natural = list(_PHRASES[length_bin])
        rng.shuffle(natural)
        for index in range(config["cases_per_bin"]):
            family = _FAMILIES[index % len(_FAMILIES)]
            for unused in range(256):
                target = _text(rng, length_bin, family, index, natural)
                if (target not in used
                        and (family == "natural" or target not in _PHRASES[length_bin])):
                    break
            else:
                raise ValueError("cannot produce unique targets for this config")
            used.add(target)
            case = {"target_text": target, "length_bin": length_bin,
                    "content_type": family}
            case["case_id"] = "case-" + _digest(case)[:24]
            base_cases.append(case)

    shuffled = list(base_cases)
    rng.shuffle(shuffled)
    occurrences = []
    # Each four-case block is feasible without borrowing occurrences from
    # another block. Delayed gaps are respectively 4, 5, 4 and 2 occurrences.
    pattern = (0, 0, 1, 1, 2, 2, 0, 3, 3, 1, 2, 3)
    for start in range(0, len(shuffled), 4):
        repeats = [0] * 4
        for offset in pattern:
            case_id = shuffled[start + offset]["case_id"]
            repeats[offset] += 1
            repeat = repeats[offset]
            occurrences.append({"case_id": case_id,
                                "occurrence_id": case_id + "-r%d" % repeat,
                                "repeat_index": repeat,
                                "role": ("initial", "immediate", "delayed")[repeat - 1]})
    return {"protocol": PROTOCOL, "config": copy.deepcopy(config),
            "batch_id": "typing-%d-%d-%s" % (config["seed"], config["cases_per_bin"],
                                              _digest(base_cases)[:16]),
            "base_cases": base_cases, "occurrences": occurrences,
            "coverage": _coverage(base_cases)}


def generate_manifest(config):
    """Return a reproducible manifest from an exact three-field config.

    ``DEFAULT_CONFIG`` specifies seed 0 and four cases per length bin.
    Invalid types (including bool integers), fields or ranges raise ValueError.
    """
    return _build_manifest(_config(config))


def _subset_config(value):
    """A self-contained, pinned selection from exactly one original batch."""
    _object(value, {"protocol", "parent_manifest", "parent_manifest_sha256", "selected_case_ids"},
            "subset config")
    if value["protocol"] != SUBSET_CONFIG_PROTOCOL:
        raise ValueError("unsupported subset config protocol")
    parent = value["parent_manifest"]
    if type(parent) is not dict or parent.get("protocol") != PROTOCOL:
        raise ValueError("subset parent must be an original generated v1 manifest")
    parent = validate_manifest(parent)
    pin = value["parent_manifest_sha256"]
    if type(pin) is not str or pin != _digest(parent):
        raise ValueError("subset parent manifest SHA-256 mismatch")
    selected = value["selected_case_ids"]
    if (type(selected) is not list or len(selected) != 3
            or any(type(item) is not str for item in selected) or len(set(selected)) != 3):
        raise ValueError("subset requires exactly three unique ordered case IDs")
    registered = {item["case_id"] for item in parent["base_cases"]}
    for item in selected:
        _identifier(item, "selected case_id")
        if item not in registered:
            raise ValueError("subset case is not in its pinned parent")
    return copy.deepcopy(value)


def _build_subset_manifest(config):
    registered = {item["case_id"]: item for item in config["parent_manifest"]["base_cases"]}
    cases = [copy.deepcopy(registered[item]) for item in config["selected_case_ids"]]
    short_pin = _digest(config)[:12]
    repeats = [0, 0, 0]
    occurrences = []
    # All three immediate revisits stay adjacent. Their delayed revisits have
    # exactly 4, 3 and 2 intervening occurrences, without adding filler demand.
    for index in (0, 0, 1, 1, 2, 2, 0, 1, 2):
        repeats[index] += 1
        repeat, case_id = repeats[index], cases[index]["case_id"]
        occurrences.append(dict(case_id=case_id,
            occurrence_id=case_id + "-subset-" + short_pin + "-r%d" % repeat,
            repeat_index=repeat, role=("initial", "immediate", "delayed")[repeat - 1]))
    return dict(protocol=SUBSET_PROTOCOL, config=copy.deepcopy(config),
        batch_id="typing-subset-" + _digest(config)[:24], base_cases=cases,
        occurrences=occurrences, coverage=_coverage(cases))


def generate_subset_manifest(parent_manifest, selected_case_ids):
    """Generate three original targets with first/immediate/delayed visits.

    Parent case IDs and exact demands remain unchanged. New occurrence/batch
    identities distinguish this reduced-coverage schedule from the original.
    This neither creates a cold-start split nor freezes ongoing skill learning.
    """
    if type(parent_manifest) is not dict or parent_manifest.get("protocol") != PROTOCOL:
        raise ValueError("subset parent must be an original generated v1 manifest")
    parent = validate_manifest(parent_manifest)
    config = _subset_config(dict(protocol=SUBSET_CONFIG_PROTOCOL, parent_manifest=parent,
        parent_manifest_sha256=_digest(parent), selected_case_ids=selected_case_ids))
    return _build_subset_manifest(config)


def _validate_coverage(value, base_cases):
    _object(value, _COVERAGE_FIELDS | {"alphabet", "counts_basis", "by_bin"}, "coverage")
    _object(value["by_bin"], set(LENGTH_BINS), "coverage.by_bin")
    for name, group in [("coverage", value)] + list(value["by_bin"].items()):
        if name != "coverage":
            _object(group, _COVERAGE_FIELDS, "coverage.by_bin." + name)
        _object(group["requested_counts"], set(ALPHABET), name + ".requested_counts")
        counts = list(group["requested_counts"].values()) + [group["space_count"]]
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("coverage counts must be nonnegative integers")
        if (type(group["missing_chars"]) is not list
                or any(type(char) is not str for char in group["missing_chars"])):
            raise ValueError("coverage missing_chars must be a list of characters")
    if value != _coverage(base_cases):
        raise ValueError("coverage does not match recomputed base target demand")


def validate_manifest(manifest):
    """Validate schema, references, schedule and deterministic provenance.

    Return a detached copy. Changing a generated target, identity, config or
    order requires regeneration; updating only its coverage is insufficient.
    No caller-owned object is changed, even when validation fails.
    """
    _object(manifest, _MANIFEST_FIELDS, "manifest")
    if manifest["protocol"] == CURRICULUM_PROTOCOL:
        config = manifest['config']
        _object(config, {'curriculum_id', 'targets', 'schedule'}, 'curriculum config')
        expected = generate_curriculum(**config)
        if manifest != expected:
            raise ValueError('curriculum differs from its explicit targets and schedule')
        return copy.deepcopy(expected)
    if manifest["protocol"] == SUBSET_PROTOCOL:
        config = _subset_config(manifest["config"])
        expected = _build_subset_manifest(config)
        if manifest != expected:
            raise ValueError("subset manifest differs from its pinned parent selection and schedule")
        return copy.deepcopy(expected)
    config = _config(manifest["config"])
    if type(manifest["protocol"]) is not str or manifest["protocol"] != PROTOCOL:
        raise ValueError("unsupported typing manifest protocol")
    _identifier(manifest["batch_id"], "batch_id")
    cases = manifest["base_cases"]
    expected_count = 4 * config["cases_per_bin"]
    if type(cases) is not list or len(cases) != expected_count:
        raise ValueError("base_cases count must equal four times cases_per_bin")
    registered = {}
    targets = set()
    bin_counts = dict.fromkeys(LENGTH_BINS, 0)
    for case in cases:
        _object(case, _CASE_FIELDS, "base case")
        _identifier(case["case_id"], "case_id")
        _identifier(case["content_type"], "content_type")
        text = case["target_text"]
        if (type(text) is not str or not 1 <= len(text) <= 60 or text != text.strip(" ")
                or any(char not in ALPHABET + " " for char in text)):
            raise ValueError("target_text must contain only a-z/0-9/internal spaces")
        name = case["length_bin"]
        if type(name) is not str or name not in LENGTH_BINS:
            raise ValueError("unknown length_bin")
        lower, upper = LENGTH_BINS[name]
        if not lower <= len(text) <= upper:
            raise ValueError("target_text is outside its length bin")
        if case["case_id"] in registered or text in targets:
            raise ValueError("base case identifiers and targets must be unique")
        registered[case["case_id"]] = []
        targets.add(text)
        bin_counts[name] += 1
    if any(count != config["cases_per_bin"] for count in bin_counts.values()):
        raise ValueError("each length bin must have exactly cases_per_bin targets")
    occurrences = manifest["occurrences"]
    if type(occurrences) is not list or len(occurrences) != 3 * expected_count:
        raise ValueError("each base case must have exactly three occurrences")
    occurrence_ids = set()
    for position, occurrence in enumerate(occurrences):
        _object(occurrence, _OCCURRENCE_FIELDS, "occurrence")
        _identifier(occurrence["case_id"], "occurrence case_id")
        _identifier(occurrence["occurrence_id"], "occurrence_id")
        if occurrence["case_id"] not in registered:
            raise ValueError("occurrence references an unknown case_id")
        if occurrence["occurrence_id"] in occurrence_ids:
            raise ValueError("occurrence_id must be unique")
        occurrence_ids.add(occurrence["occurrence_id"])
        positions = registered[occurrence["case_id"]]
        repeat = len(positions) + 1
        if (type(occurrence["repeat_index"]) is not int or repeat > 3
                or occurrence["repeat_index"] != repeat
                or occurrence["role"] != ("initial", "immediate", "delayed")[repeat - 1]):
            raise ValueError("occurrence repeat_index/role/order is invalid")
        positions.append(position)
    for positions in registered.values():
        if (len(positions) != 3 or positions[1] != positions[0] + 1
                or not 2 <= positions[2] - positions[1] - 1 <= 5):
            raise ValueError("repeat schedule requires immediate and 2..5 occurrence delayed visits")
    _validate_coverage(manifest["coverage"], cases)
    if manifest != _build_manifest(config):
        raise ValueError("manifest differs from deterministic generation for its config")
    return copy.deepcopy(manifest)


def manifest_sha256(manifest):
    """Hash a validated manifest as sorted compact UTF-8 JSON, without newline."""
    return _digest(validate_manifest(manifest))


def generate_curriculum(curriculum_id, targets, schedule):
    """Explicit online-learning demand; no hidden actions or keyboard geometry.

    Original coverage batches remain unchanged. Each target has first,
    immediate and delayed use, keeping the existing scoring/identity contract.
    """
    _identifier(curriculum_id, 'curriculum_id')
    if (type(targets) is not list or not 1 <= len(targets) <= 64
            or any(type(t) is not str or not 1 <= len(t) <= 60 or t != t.strip(' ')
                   or any(c not in ALPHABET + ' ' for c in t) for t in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError('unique bounded typing targets required')
    if (type(schedule) is not list or len(schedule) != 3 * len(targets)
            or any(type(i) is not int or not 0 <= i < len(targets) for i in schedule)):
        raise ValueError('explicit three-visit schedule required')
    cases, occurrences = [], []
    for target in targets:
        label = next(name for name, (low, high) in LENGTH_BINS.items() if low <= len(target) <= high)
        case = dict(target_text=target, length_bin=label, content_type='explicit_learning_demand')
        case['case_id'] = 'case-' + _digest(case)[:24]
        cases.append(case)
    for index, case in enumerate(cases):
        positions = [n for n, value in enumerate(schedule) if value == index]
        if (len(positions) != 3 or positions[1] != positions[0] + 1
                or not 2 <= positions[2] - positions[1] - 1 <= 5):
            raise ValueError('curriculum requires adjacent second and spaced third visits')
    counts = [0] * len(cases)
    for index in schedule:
        counts[index] += 1
        repeat = counts[index]
        case_id = cases[index]['case_id']
        occurrences.append(dict(case_id=case_id, occurrence_id=case_id + '-r%d' % repeat,
            repeat_index=repeat, role=('initial', 'immediate', 'delayed')[repeat - 1]))
    config = dict(curriculum_id=curriculum_id, targets=targets, schedule=schedule)
    return dict(protocol=CURRICULUM_PROTOCOL, config=copy.deepcopy(config),
        batch_id='typing-curriculum-' + _digest(config)[:24], base_cases=cases,
        occurrences=occurrences, coverage=_coverage(cases))
