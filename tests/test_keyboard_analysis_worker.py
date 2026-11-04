"""Finite numerical jobs only: no hardware, network, model or knowledge writes."""
from dataclasses import asdict
import hashlib
import json
import os
import unittest

from astra_robotics_agent.physical_keyboard.runtime.analysis_worker import (
    AnalysisError, AnalysisLimits, run_analysis,
)
from astra_robotics_agent.physical_keyboard.runtime.skill_language import Limits


class AnalysisWorkerTests(unittest.TestCase):
    def rejected(self, lines, data=None, parameters=None, category=None, **kwargs):
        with self.assertRaises(AnalysisError) as caught:
            run_analysis(lines, [] if data is None else data, {} if parameters is None else parameters, **kwargs)
        if category:
            self.assertEqual(caught.exception.category, category, str(caught.exception))
        return caught.exception

    def test_reproducible_exact_candidate_result_in_separate_process(self):
        lines = ['def analyze(data, parameters):', '    return {"mean": sum(data) / len(data), "unit": parameters["unit"]}']
        first = run_analysis(lines, [.1,.2,.3], {'unit':'m'})
        second = run_analysis(lines, [.1,.2,.3], {'unit':'m'})
        self.assertEqual(first['result'], second['result'])
        self.assertEqual(first['result_sha256'], second['result_sha256'])
        self.assertEqual(first['source_sha256'], second['source_sha256'])
        self.assertEqual(first['authority'], 'candidate_only')
        self.assertNotEqual(first['environment']['worker_pid'], os.getpid())
        self.assertEqual(first['environment']['callbacks'], [])
        self.assertFalse(first['environment']['sdk_imported'])
        self.assertIsNone(Limits().max_tool_calls)

    def test_full_single_key_and_multisource_projection_fit_approved_starting_budgets(self):
        # A complete round-trip-shaped input, never a physical replay.
        key = [{'t_s':i*.03341,'xyz_m':[.4+i*.00001,-.2,.3], 'joint_rad':[i*.00001]*6} for i in range(512)]
        result = run_analysis(['def analyze(data, parameters):','    total = 0.0',
                               '    for row in data:', '        total = total + row["xyz_m"][0]',
                               '    return {"count": len(data), "sum_x": total}'], key, {})
        self.assertEqual(result['result']['count'], 512)
        projected = [[i*.03341, .4+i*.00001] for i in range(4000)]
        result = run_analysis(['def analyze(data, parameters):','    total = 0.0',
                               '    for source in data:', '        total = total + len(source)',
                               '    return total'], [projected[:2000],projected[2000:]], {})
        self.assertEqual(result['result'], 4000)
        self.assertEqual(result['budgets']['max_input_bytes'],1048576)
        self.assertEqual(result['budgets']['max_container_items'],65536)

    def test_full_input_byte_limit_is_independent_of_physical_container_default(self):
        # ~900KB finite strings, below container cap and 1MiB job input budget.
        result=run_analysis(['def analyze(data, parameters):','    return len(data)'], ['x'*30000]*30,{})
        self.assertEqual(result['result'],30)
        self.rejected(['def analyze(data, parameters):','    return 1'], ['x'*60000]*18, category='input_limit')

    def test_robot_wrapper_import_io_and_reflection_are_denied(self):
        for lines in [
            ['def analyze(data, parameters):','    return robot.observe()'],
            ['def analyze(data, parameters):','    return profile["data"]'],
            ['def main(data, parameters):','    return 1','def analyze(data, parameters):','    return 1'],
            ['import os','def analyze(data, parameters):','    return 1'],
            ['def analyze(data, parameters):','    return data.append(1)'],
        ]:
            with self.subTest(lines=lines): self.rejected(lines, category='forbidden_capability')
        for expression in ['open("/tmp/secret")','exec("1")','getattr(data, "x")','__import__("socket")']:
            self.rejected(['def analyze(data, parameters):','    return '+expression], category='invalid_source')

    def test_nonfinite_bool_budget_duplicate_and_scope_injection_do_not_run(self):
        self.rejected(['def analyze(data, parameters):','    return data'], [float('nan')], category='input_limit')
        self.rejected(['def analyze(data, parameters):','    return 1'], limits={'max_steps':True},category='invalid_budget')
        self.rejected(['def analyze(data, parameters):','    return 1'],limits={'robot':'enabled'},category='invalid_budget')

    def test_steps_iterations_container_source_result_and_process_deadlines_classified(self):
        self.rejected(['def analyze(data, parameters):','    while True:','        pass'],
                      limits={'max_steps':30},category='step_limit')
        self.rejected(['def analyze(data, parameters):','    while True:','        pass'],
                      limits={'max_iterations':3},category='iteration_limit')
        self.rejected(['def analyze(data, parameters):','    return 1'],list(range(50)),
                      limits={'max_container_items':10},category='container_limit')
        self.rejected(['def analyze(data, parameters):','    return 1'],limits={'max_source_bytes':20},category='source_limit')
        self.rejected(['def analyze(data, parameters):','    return "x" * 32769'],category='output_limit')
        self.rejected(['def analyze(data, parameters):','    return 1'], limits={'wall_time_s':.0001},category='timeout')
        self.rejected(['def analyze(data, parameters):','    return 1'], limits={'memory_bytes':1024},category='memory_limit')

    def test_fault_state_is_only_data_and_never_replaced(self):
        state={'motion_available':False,'fault':'unknown','generation':'g1','STOP':True}
        original=json.dumps(state,sort_keys=True)
        result=run_analysis(['def analyze(data, parameters):','    data["fault"] = "claimed"','    return data'],state,{})
        self.assertEqual(json.dumps(state,sort_keys=True),original)
        self.assertEqual(result['authority'],'candidate_only')


if __name__ == '__main__': unittest.main()
