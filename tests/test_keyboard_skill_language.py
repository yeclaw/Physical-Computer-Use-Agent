"""Adversarial AST language and bounded feedback tests, with JSON-only fake tools."""

import ast
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from astra_robotics_agent.physical_keyboard import skill_language as language
from astra_robotics_agent.physical_keyboard.skill_language import (
    BudgetExceeded, Limits, SkillError, ToolError, ValidationError, run_skill)


class SkillLanguageTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.pose = [0.0, 0.0, 0.2]
        self.tools = {'observe': self.observe, 'move_tcp': self.move,
                      'wait_bounded': self.wait, 'stop': self.stop}

    def observe(self):
        self.calls.append(('observe',))
        return {'observation': {'pose': {'xyz_m': list(self.pose)}}}

    def move(self, xyz_m, quaternion_xyzw):
        self.calls.append(('move', xyz_m, quaternion_xyzw))
        self.pose = list(xyz_m)
        return {'status': 'ok', 'pose': {'xyz_m': list(xyz_m)}}

    def wait(self, seconds):
        self.calls.append(('wait', seconds))
        return {'status': 'ok'}

    def stop(self):
        self.calls.append(('stop',))
        return {'status': 'ok'}

    def run_program(self, source, *, helpers=None, profile=None, tools=None, limits=None):
        sources = ([('helpers.py', helpers)] if helpers is not None else []) + [('skill.py', source)]
        return run_skill(sources, tools=self.tools if tools is None else tools,
                         profile={} if profile is None else profile,
                         limits=Limits() if limits is None else limits)

    def test_helpers_math_arguments_and_json_return(self):
        result = self.run_program('''def main(robot, profile):
    value = distance(profile['a'], profile['b'])
    return {'distance': value, 'pi': math.pi, 'rounded': round(value, 2),
            'selected': min(3, 4), 'sum': sum([1, 2, 3])}
''', helpers='''def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
''', profile={'a': [0, 0], 'b': [3, 4]})
        self.assertEqual(result['distance'], 5)
        self.assertEqual(result['sum'], 6)
        self.assertEqual(result['pi'], math.pi)
        self.assertEqual(self.calls, [])

    def test_while_uses_actual_tool_feedback(self):
        result = self.run_program('''def main(robot, profile):
    count = 0
    actual = robot.observe()['observation']['pose']['xyz_m']
    while abs(actual[0] - profile['target']) > 0.001:
        next_pose = list(actual)
        next_pose[0] += 0.01
        robot.move_tcp(xyz_m=next_pose, quaternion_xyzw=[0, 0, 0, 1])
        actual = robot.observe()['observation']['pose']['xyz_m']
        count += 1
    return {'count': count, 'actual': actual}
''', profile={'target': 0.03})
        self.assertEqual(result['count'], 3)
        self.assertAlmostEqual(result['actual'][0], 0.03)
        self.assertEqual(len(self.calls), 7)

    def test_for_break_continue_and_loop_else(self):
        self.assertEqual(self.run_program('''def main(robot, profile):
    result = []
    for i in range(6):
        if i == 1:
            continue
        if i == 4:
            break
        result = result + [i]
    else:
        result = result + [99]
    for j in range(2):
        pass
    else:
        result = result + [j]
    return result
'''), [0, 2, 3, 1])

    def test_nested_helper_returns_and_keyword_binding(self):
        self.assertEqual(self.run_program('''def main(robot, profile):
    return helper(b=2, a=3)
''', helpers='''def helper(a, b):
    if a > b:
        return other(a - b)
    return 0
def other(value):
    return value * 2
'''), 2)

    def test_tool_error_handler_and_finally_release(self):
        attempts = []
        def rejecting_move(xyz_m, quaternion_xyzw):
            attempts.append(list(xyz_m))
            if len(attempts) == 1:
                raise ToolError('confirmed rejected')
            return {'status': 'ok'}
        result = self.run_program('''def main(robot, profile):
    message = ''
    try:
        robot.move_tcp([0, 0, 0.1], [0, 0, 0, 1])
    except ToolError as error:
        message = str(error)
    finally:
        robot.move_tcp([0, 0, 0.2], [0, 0, 0, 1])
    return {'message': message}
''', tools={**self.tools, 'move_tcp': rejecting_move})
        self.assertEqual(result, {'message': 'confirmed rejected'})
        self.assertEqual(attempts, [[0, 0, 0.1], [0, 0, 0.2]])

    def test_try_else_finally_and_return_semantics(self):
        result = self.run_program('''def helper():
    try:
        return 1
    finally:
        return 2
def main(robot, profile):
    values = []
    try:
        values = values + [helper()]
    except ToolError:
        values = [99]
    else:
        values = values + [3]
    finally:
        values = values + [4]
    return values
''')
        self.assertEqual(result, [2, 3, 4])

    def test_unhandled_error_cannot_be_suppressed_by_finally_return(self):
        with self.assertRaises(SkillError):
            self.run_program('''def main(robot, profile):
    try:
        value = 1 / 0
    finally:
        return {'success': True}
''')

    def test_transport_skill_error_latches_finally_closed(self):
        def disconnected():
            raise SkillError('transport disconnected')
        with self.assertRaisesRegex(SkillError, 'transport disconnected'):
            self.run_program('''def main(robot, profile):
    try:
        robot.observe()
    finally:
        robot.stop()
''', tools={**self.tools, 'observe': disconnected})
        self.assertEqual(self.calls, [])

    def test_unexpected_host_error_is_fatal_not_recoverable(self):
        def disconnected():
            raise RuntimeError('host failure')
        with self.assertRaisesRegex(SkillError, 'Host tool failed'):
            self.run_program('''def main(robot, profile):
    try:
        robot.observe()
    except ToolError:
        robot.stop()
    finally:
        robot.stop()
''', tools={**self.tools, 'observe': disconnected})
        self.assertEqual(self.calls, [])

    def test_all_sources_dead_branches_and_top_level_prevalidated(self):
        attacks = [
            'import os', 'class X:\n        pass', 'x = robot.__class__',
            "x = open('/tmp/forbidden')", "x = eval('1')", "x = globals()",
            "x = getattr(robot, 'stop')", 'x = [i for i in range(3)]',
            'x = lambda: 1', 'x = (1, 2)', 'x = profile.keys()',
            'x = math.factorial(100000)', 'x = made_up_function()',
            'try:\n        pass\n    except:\n        pass',
        ]
        for attack in attacks:
            source = 'def main(robot, profile):\n    robot.stop()\n    if False:\n' + '\n'.join('        ' + line for line in attack.splitlines()) + '\n    return None\n'
            with self.subTest(attack=attack), self.assertRaises(ValidationError):
                self.run_program(source)
            self.assertEqual(self.calls, [])
        with self.assertRaises(ValidationError):
            self.run_program('def main(robot, profile):\n    robot.stop()\n', helpers='import os\n')
        with self.assertRaises(ValidationError):
            self.run_program('robot.stop()\ndef main(robot, profile):\n    pass\n')
        self.assertEqual(self.calls, [])

    def test_exception_reflection_and_budget_handlers_rejected(self):
        for handler in ['except Exception:', 'except BudgetExceeded:', 'except ToolError as e:\n        x = e.args']:
            source = 'def main(robot, profile):\n    try:\n        robot.stop()\n    ' + handler + '\n        pass\n'
            with self.subTest(handler=handler), self.assertRaises(ValidationError):
                self.run_program(source)
        self.assertEqual(self.calls, [])

    def test_infinite_loops_and_recursion_stop_with_budget(self):
        programs = ['def main(robot, profile):\n    while True:\n        pass\n',
                    'def main(robot, profile):\n    return main(robot, profile)\n']
        for source in programs:
            with self.subTest(source=source), self.assertRaises(BudgetExceeded):
                self.run_program(source, limits=replace(Limits(), max_iterations=10, max_call_depth=8))

    def test_budget_cannot_use_finally_to_send_or_return(self):
        with self.assertRaises(BudgetExceeded):
            self.run_program('''def main(robot, profile):
    try:
        while True:
            pass
    finally:
        robot.stop()
        return True
''', limits=replace(Limits(), max_iterations=3))
        self.assertEqual(self.calls, [])
        with self.assertRaises(BudgetExceeded):
            self.run_program('''def main(robot, profile):
    try:
        for i in range(3):
            robot.observe()
    finally:
        robot.stop()
''', limits=replace(Limits(), max_tool_calls=2))
        self.assertEqual(self.calls, [('observe',), ('observe',)])

    def test_large_powers_repeats_and_ranges_rejected_before_allocation(self):
        expressions = ['2 ** (2 ** 30)', '[1] * (2 ** 100)', "'x' * (2 ** 100)",
                       'list(range(1000000000))', '2 ** 128']
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises(BudgetExceeded):
                self.run_program('def main(robot, profile):\n    return ' + expression + '\n')
        self.assertEqual(self.calls, [])

    def test_repeated_allocations_are_cumulatively_bounded(self):
        with self.assertRaises(BudgetExceeded):
            self.run_program('''def main(robot, profile):
    for i in range(1000):
        value = [0] * 64
    return value
''', limits=replace(Limits(), max_container_items=64))

    def test_cycles_shared_graphs_and_depth_limited(self):
        with self.assertRaises(SkillError):
            self.run_program('''def main(robot, profile):
    value = [None]
    value[0] = value
    return value
''')
        with self.assertRaises(BudgetExceeded):
            self.run_program('''def main(robot, profile):
    value = [1]
    for i in range(20):
        value = [value, value]
    return value
''')
        value = 1
        for _ in range(40):
            value = [value]
        with self.assertRaises(BudgetExceeded):
            self.run_program('def main(robot, profile):\n    return None\n', profile={'deep': value})

    def test_profile_and_namespace_are_fresh(self):
        profile = {'values': [1]}
        source = '''COUNTER = [0]
def main(robot, profile):
    COUNTER[0] += 1
    profile['values'][0] = 99
    return COUNTER[0]
'''
        self.assertEqual(self.run_program(source, profile=profile), 1)
        self.assertEqual(self.run_program(source, profile=profile), 1)
        self.assertEqual(profile, {'values': [1]})

    def test_non_json_tool_values_and_output_rejected(self):
        for result in [object(), {'x': float('nan')}, {'x': (1, 2)}, {1: 'bad key'}]:
            with self.subTest(result=result), self.assertRaises(SkillError):
                self.run_program('def main(robot, profile):\n    return robot.observe()\n',
                                 tools={**self.tools, 'observe': lambda: result})
        for expression in ['robot', 'math', 'main', 'range(3)', "float('inf')"]:
            with self.subTest(expression=expression), self.assertRaises(SkillError):
                self.run_program('def main(robot, profile):\n    return ' + expression + '\n')

    def test_invalid_tool_receipt_latches_finally_closed(self):
        with self.assertRaises(SkillError):
            self.run_program('''def main(robot, profile):
    try:
        robot.observe()
    finally:
        robot.stop()
''', tools={**self.tools, 'observe': lambda: object()})
        self.assertEqual(self.calls, [])

    def test_result_size_cap(self):
        with self.assertRaises(BudgetExceeded):
            self.run_program("def main(robot, profile):\n    return ['x' * 4096] * 9\n")

    def test_dictionary_iteration_mutation_is_structured_and_finally_runs(self):
        with self.assertRaisesRegex(SkillError, 'RuntimeError'):
            self.run_program('''def main(robot, profile):
    values = {'a': 1}
    try:
        for key in values:
            values['b'] = 2
    finally:
        robot.stop()
''')
        self.assertEqual(self.calls, [('stop',)])

    def test_tool_parameter_validation_precedes_callback(self):
        calls = ["robot.move_tcp([True, 0, 0], [0, 0, 0, 1])",
                 "robot.move_tcp([0, 0], [0, 0, 0, 1])", "robot.move_tcp([0, 0, 0], [0, 0, 0, 0])",
                 'robot.observe(extra=1)', 'robot.stop(1)', 'robot.wait_bounded(-1)',
                 'robot.wait_bounded(True)']
        for call in calls:
            with self.subTest(call=call), self.assertRaises(SkillError):
                self.run_program('def main(robot, profile):\n    ' + call + '\n')
        self.assertEqual(self.calls, [])

    def test_static_size_caps_and_duplicate_sources(self):
        with self.assertRaises(ValidationError):
            self.run_program('def main(robot, profile):\n    return 1\n', limits=replace(Limits(), max_source_bytes=8))
        with self.assertRaises(ValidationError):
            self.run_program('def main(robot, profile):\n    return 1\n', limits=replace(Limits(), max_ast_nodes=3))
        with self.assertRaises(ValidationError):
            run_skill([('same.py', 'def helper():\n    pass\n'), ('same.py', 'def main(robot, profile):\n    pass\n')],
                      tools=self.tools, profile={})
        with self.assertRaises(ValidationError):
            self.run_program('def main(robot, profile):\n    return 2 ** 200\n',
                             helpers='VALUE = ' + str(2 ** 200) + '\n')

    def test_function_defaults_decorators_and_unpacking_rejected(self):
        sources = ['def main(robot, profile, x=robot.stop()):\n    pass\n',
                   '@robot.stop()\ndef main(robot, profile):\n    pass\n',
                   'def main(robot, profile):\n    robot.move_tcp(**profile)\n',
                   'def main(robot, profile):\n    a, b = [1,2]\n']
        for source in sources:
            with self.subTest(source=source), self.assertRaises(ValidationError):
                self.run_program(source)
        self.assertEqual(self.calls, [])

    def test_python311_function_ast_without_type_params(self):
        original = ast.parse
        def old_parse(*args, **kwargs):
            tree = original(*args, **kwargs)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and hasattr(node, 'type_params'):
                    del node.type_params
            return tree
        with patch.object(language.ast, 'parse', old_parse):
            self.assertEqual(self.run_program('def main(robot, profile):\n    return 7\n'), 7)


if __name__ == '__main__':
    unittest.main()
