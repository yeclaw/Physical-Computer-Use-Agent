"""Skill declaration/linking checks with the real bounded interpreter and fake tools."""
import copy
import hashlib
import json
import unittest

from astra_robotics_agent.physical_keyboard.skills.programs import (
    ProgramError, compile_program, validate_arguments, validate_contract,
)
from astra_robotics_agent.physical_keyboard.runtime.skill_language import run_skill


def contract(schema=None):
    return {
        'description': 'A model-authored reusable operation.',
        'input_schema': schema or {'type': 'object', 'properties': {}},
        'preconditions': ['The current layout matches the observed layout.'],
        'postconditions': ['The model checks the resulting observation.'],
        'side_effects': [],
    }


def execute(bundle, tools=None, *, demo_tools=False):
    return run_skill([('linked.py', '\n'.join(bundle['source_lines']) + '\n')],
                     tools=tools or {}, profile=bundle['profile'], demo_tools=demo_tools)


class ContractTests(unittest.TestCase):
    def test_declarations_are_copied_without_inventing_semantic_conditions(self):
        original = contract()
        result = validate_contract(original)
        self.assertEqual(result, original)
        result['preconditions'].append('changed')
        self.assertEqual(len(original['preconditions']), 1)
        self.assertNotIn('stable', result)
        self.assertNotIn('independently_verified', result)

    def test_unseen_parameter_values_follow_one_model_declared_interface(self):
        declaration = contract({'type': 'object', 'properties': {
            'payload': {'type': 'string', 'minLength': 1, 'maxLength': 32},
        }, 'required': ['payload'], 'additionalProperties': False})
        lines = [
            'def main(robot, profile):',
            '    output = ""',
            '    for symbol in profile["payload"]:',
            '        robot.wait_bounded(seconds=0.001)',
            '        output = output + symbol',
            '    return output',
        ]
        original_bundle = compile_program(lines, {})
        for value in ['orbit', 'delta', 'new combination']:
            arguments = validate_arguments(declaration, {'payload': value})
            bundle = dict(original_bundle, profile=arguments)
            calls = []
            self.assertEqual(execute(bundle, {'wait_bounded': lambda **k: calls.append(k) or {}}), value)
            self.assertEqual(len(calls), len(value))
        self.assertEqual(original_bundle['profile'], {})

    def test_nested_types_bounds_and_enum(self):
        declaration = contract({'type': 'object', 'properties': {
            'items': {'type': 'array', 'items': {'type': 'integer', 'minimum': 1, 'maximum': 3},
                      'minItems': 1, 'maxItems': 2},
            'mode': {'type': 'string', 'enum': ['first', 'second']},
            'flag': {'type': 'boolean'}, 'none': {'type': 'null'},
        }, 'required': ['items']})
        value = {'items': [1, 3], 'mode': 'second', 'flag': False, 'none': None}
        self.assertEqual(validate_arguments(declaration, value), value)
        for invalid in [{'items': []}, {'items': [4]}, {'items': [True]}, {'items': [1, 2, 3]},
                        {'items': [1], 'mode': 'third'}, {'items': [1], 'extra': 1}]:
            with self.subTest(invalid=invalid), self.assertRaises(ProgramError):
                validate_arguments(declaration, invalid)

    def test_numeric_booleans_and_nonfinite_are_rejected(self):
        declaration = contract({'type': 'object', 'properties': {'value': {'type': 'number'}},
                                'required': ['value']})
        for value in [True, False, float('nan'), float('inf'), '1']:
            with self.subTest(value=value), self.assertRaises(ProgramError):
                validate_arguments(declaration, {'value': value})
        self.assertEqual(validate_arguments(declaration, {'value': 1.5}), {'value': 1.5})

    def test_boolean_is_distinct_from_integer_in_enum(self):
        declaration = contract({'type': 'object', 'properties': {
            'value': {'type': 'boolean', 'enum': [1]},
        }, 'required': ['value']})
        with self.assertRaises(ProgramError):
            validate_arguments(declaration, {'value': True})

    def test_missing_and_undeclared_arguments_are_rejected_by_default(self):
        declaration = contract({'type': 'object', 'properties': {'value': {'type': 'string'}},
                                'required': ['value']})
        for arguments in [{}, {'value': 'x', 'extra': 'y'}, [], {'value': 4}]:
            with self.subTest(arguments=arguments), self.assertRaises(ProgramError):
                validate_arguments(declaration, arguments)

    def test_explicit_additional_properties_still_require_plain_finite_json(self):
        declaration = contract({'type': 'object', 'additionalProperties': True})
        self.assertEqual(validate_arguments(declaration, {'arbitrary': [1, 'x']}), {'arbitrary': [1, 'x']})
        for value in [object(), (1, 2), float('nan')]:
            with self.subTest(value=value), self.assertRaises(ProgramError):
                validate_arguments(declaration, {'arbitrary': value})

    def test_bad_or_unsupported_schema_is_rejected(self):
        bad = [
            {'type': ['string', 'null']}, {'type': 'array'}, {'type': 'object', '$ref': 'elsewhere'},
            {'type': 'object', 'additionalProperties': {}},
            {'type': 'object', 'required': ['missing']},
            {'type': 'object', 'properties': {'x': {'type': 'string', 'minLength': True}}},
            {'type': 'object', 'properties': {'x': {'type': 'number', 'minimum': True}}},
            {'type': 'object', 'properties': {'x': {'type': 'integer', 'minimum': 3, 'maximum': 1}}},
            {'type': 'object', 'properties': {'x': {'type': 'string', 'enum': ['x', 'x']}}},
        ]
        for schema in bad:
            with self.subTest(schema=schema), self.assertRaises(ProgramError):
                validate_contract(contract(schema))

    def test_contract_shape_cycles_and_oversized_data(self):
        for change in [{'description': ''}, {'preconditions': 'not a list'}, {'permission': 'new authority'}]:
            value = contract(); value.update(change)
            with self.subTest(change=change), self.assertRaises(ProgramError):
                validate_contract(value)
        cyclic = {}; cyclic['itself'] = cyclic
        with self.assertRaises(ProgramError):
            validate_arguments(contract({'type': 'object', 'additionalProperties': True}), cyclic)
        with self.assertRaises(ProgramError):
            validate_arguments(contract({'type': 'object', 'additionalProperties': True}), {'payload': 'x' * 4097})


class LinkingTests(unittest.TestCase):
    @staticmethod
    def dependency(index):
        return {
            'candidate_id': 'candidate-' + str(index),
            'source': 'OFFSET = ' + str(index / 10) + '\n'
                      'def helper(value):\n    return value + OFFSET\n'
                      'def main(robot, profile):\n'
                      '    robot.move_tcp(xyz_m=[helper(profile["x"]), 0, 0.1], quaternion_xyzw=[0,0,0,1])\n'
                      '    return profile["label"]\n',
            'profile': {'x': 1, 'label': str(index)},
        }

    def test_three_colliding_helpers_run_in_one_program_with_original_parameters(self):
        dependencies = {'approach': self.dependency(1), 'press': self.dependency(2), 'retreat': self.dependency(3)}
        lines = ['def main(robot, profile):',
                 '    one = approach(robot, {"x": profile["x"]})',
                 '    two = press(robot, {})',
                 '    three = retreat(robot, {})',
                 '    return [one, two, three]']
        before = copy.deepcopy((lines, dependencies))
        bundle = compile_program(lines, {'x': 2}, dependencies)
        moves = []
        self.assertEqual(execute(bundle, {'move_tcp': lambda **k: moves.append(k) or {}}), ['1', '2', '3'])
        self.assertEqual([item['xyz_m'][0] for item in moves], [2.1, 1.2, 1.3])
        self.assertEqual(before, (lines, dependencies))
        self.assertEqual(bundle['entrypoint']['parameters'], ['robot', 'profile'])
        self.assertEqual(bundle['entrypoint']['dependency_aliases'], ['approach', 'press', 'retreat'])
        self.assertEqual({pin['candidate_id'] for pin in bundle['dependency_pins']}, {'candidate-1', 'candidate-2', 'candidate-3'})

    def test_namespacing_handles_same_spelled_local_parameter_and_keywords(self):
        child = {'candidate_id': 'one', 'profile': {}, 'source':
                 'value = 7\ndef helper(value):\n    return value\n'
                 'def main(robot, profile):\n    return helper(value=value)\n'}
        bundle = compile_program(['def main(robot, profile):', '    return child(robot, {})'], {}, {'child': child})
        self.assertEqual(execute(bundle), 7)

    def test_profile_defaults_are_fresh_for_each_helper_call(self):
        child = {'candidate_id': 'mutable', 'profile': {'counter': 0}, 'source':
                 'def main(robot, profile):\n    profile["counter"] = profile["counter"] + 1\n    return profile["counter"]\n'}
        bundle = compile_program(['def main(robot, profile):',
                                  '    return [child(robot, {}), child(robot, {})]'], {}, {'child': child})
        self.assertEqual(execute(bundle), [1, 1])
        self.assertEqual(child['profile'], {'counter': 0})

    def test_deterministic_linking_and_sha_pins(self):
        dependencies = {'second': self.dependency(2), 'first': self.dependency(1)}
        lines = ['def main(robot, profile):', '    return first(robot, {})']
        a = compile_program(lines, {}, dependencies)
        b = compile_program(lines, {}, dict(reversed(list(dependencies.items()))))
        self.assertEqual(a, b)
        pin = a['dependency_pins'][0]
        self.assertEqual(pin['source_sha256'], hashlib.sha256(dependencies['first']['source'].encode()).hexdigest())
        self.assertEqual(pin['profile_sha256'], hashlib.sha256(
            (json.dumps(dependencies['first']['profile'], sort_keys=True, ensure_ascii=False,
                        allow_nan=False, separators=(',', ':')) + '\n').encode()).hexdigest())

    def test_unresolved_or_colliding_aliases_and_malformed_dependencies(self):
        source = ['def main(robot, profile):', '    return None']
        for dependencies in [{'main': self.dependency(1)}, {'open': self.dependency(1)},
                             {'bad.alias': self.dependency(1)}, {4: self.dependency(1)},
                             {'a': {}}, {'a': {'source': 'bad', 'profile': {}, 'candidate_id': 'x'}},
                             {str(i): self.dependency(i) for i in range(9)}]:
            with self.subTest(dependencies=dependencies), self.assertRaises(ProgramError):
                compile_program(source, {}, dependencies)
        with self.assertRaises(ProgramError):
            compile_program(['def child(robot, profile):', '    return None', *source], {}, {'child': self.dependency(1)})
        with self.assertRaises(ProgramError):
            compile_program(['def main(robot, profile):', '    return missing(robot, {})'], {})

    def test_forbidden_source_and_profile_authority_cannot_be_linked(self):
        for source in [
            ['import os', 'def main(robot, profile):', '    return None'],
            ['def main(robot, profile):', '    return open("file")'],
            ['def main(robot, profile):', '    if False:', '        robot.surprise()', '    return None'],
        ]:
            with self.subTest(source=source), self.assertRaises(ProgramError):
                compile_program(source, {})
        with self.assertRaises(ProgramError):
            compile_program(['def main(robot, profile):', '    return None'], {'speed_percent': 1})
        child = self.dependency(1); child['profile']['disable_protection'] = True
        with self.assertRaises(ProgramError):
            compile_program(['def main(robot, profile):', '    return child(robot, {})'], {}, {'child': child})

    def test_optional_tools_require_explicit_active_whitelist(self):
        source = ['def main(robot, profile):', '    return robot.trajectory_catalog()']
        with self.assertRaises(ProgramError):
            compile_program(source, {})
        bundle = compile_program(source, {}, demo_tools=True)
        self.assertEqual(execute(bundle, {'trajectory_catalog': lambda: {'available': True}}, demo_tools=True), {'available': True})

    def test_combined_ast_budget_is_not_reset_per_dependency(self):
        source = 'def main(robot, profile):\n    value = 0\n' + '    value = value + 1\n' * 300 + '    return value\n'
        dependencies = {'item' + str(i): {'source': source, 'profile': {}, 'candidate_id': str(i)} for i in range(3)}
        with self.assertRaisesRegex(ProgramError, 'AST|budget'):
            compile_program(['def main(robot, profile):', '    return item0(robot, {})'], {}, dependencies)

    def test_invalid_lines_and_source_budget(self):
        for lines in [[], ['def main(robot, profile):\n    return None'], ['x' * 66000],
                      ['def main(robot, profile):', '    value = 0'] + ['    value = value + 1'] * 900]:
            with self.subTest(size=len(lines)), self.assertRaises(ProgramError):
                compile_program(lines, {})


if __name__ == '__main__':
    unittest.main()
