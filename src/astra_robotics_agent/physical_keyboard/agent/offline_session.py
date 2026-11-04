"""Authored decision fixture, not a model or proof of autonomous learning.

It exercises the actual loop, interpreter, mock device, evidence, save, inspect
and reuse. Decisions are fixed fixtures whose feedback assertions fail loudly.
"""
import json


SHALLOW = '''def main(robot, profile):
    robot.move_tcp([0, 0, 0.16], [0, 0, 0, 1])
    robot.move_tcp([0, 0, 0.2], [0, 0, 0, 1])
    return robot.observe()
'''
REVISED = SHALLOW.replace('0.16', '0.12')


class OfflineSession:
    def __init__(self):
        self.index = 0
        self.skill_id = self.lesson_id = None

    def run_turn(self, prompt, callback, images):
        context = json.loads(prompt.split('\n')[0])
        feedback = context['feedback']
        index = self.index
        self.index += 1
        if index in (0, 2):
            decision = dict(action='run_skill', arguments=dict(label='press_attempt',
                source_lines=(SHALLOW if index == 0 else REVISED).splitlines(), profile={}),
                hypothesis='The next synthetic observation determines whether one character registered.')
        elif index == 1:
            assert feedback['execution']['status'] == 'ok'
            assert '"observed_text":""' in prompt or '"observed_text": ""' in prompt
            decision = dict(action='save_lesson', arguments=dict(id='ack_is_not_registration',
                claim='The shallow fixture movement was acknowledged but no character appeared.',
                applicability=['Synthetic mock fixture only'],
                limitations=['No physical keyboard or model learning'], evidence_ids=feedback['evidence_ids']))
        elif index == 3:
            assert '"observed_text":"a"' in prompt or '"observed_text": "a"' in prompt
            decision = dict(action='save_skill', arguments=dict(id='mock_press_release',
                source_lines=REVISED.splitlines(), profile={},
                applicability=['Mock device at declared fixture start'],
                limitations=['Handwritten fixture, not a learned hardware skill'],
                evidence_ids=feedback['evidence_ids']))
        elif index == 4:
            self.skill_id = feedback['candidate']['candidate_id']
            decision = dict(action='inspect_skill', arguments=dict(candidate_id=self.skill_id))
        elif index == 5:
            assert feedback['skill']['source'] == REVISED
            decision = dict(action='verify_skill', arguments=dict(candidate_id=self.skill_id),
                            hypothesis='Reuse the saved source; the synthetic text should become aa.')
        elif index == 6:
            assert '"observed_text":"aa"' in prompt or '"observed_text": "aa"' in prompt
            decision = dict(action='finish', result=dict(status='partial', reflection=dict(
                summary='Authored fixture exercised repair, storage and reuse.',
                skills_saved=[self.skill_id], lessons_saved=[],
                limitations=['Synthetic decisions/device; not autonomous or physical learning.'])))
        else:
            raise RuntimeError('Unexpected extra fixture turn')
        return dict(turn_id='offline-%02d' % index, final_text=json.dumps(decision),
                    usage=None, evidence_mode='synthetic')

    def close(self):
        pass


def factory(settings, *, instructions, cwd, output):
    if settings != {}:
        raise ValueError('Offline session accepts no settings')
    return OfflineSession()
