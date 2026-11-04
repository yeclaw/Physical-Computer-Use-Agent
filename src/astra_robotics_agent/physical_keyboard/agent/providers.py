"""Optional reference factory around the existing Codex app-server client.

Import and construction do not start a model. The first run_turn starts the
configured client. Other providers implement the same session/factory seam.
"""
from pathlib import Path
from astra_robotics_agent.codex import CodexSession


class CodexTaskSession:
    def __init__(self, settings, *, instructions, cwd, output, session_type=CodexSession):
        allowed = {'model', 'effort', 'service_tier', 'timeout'}
        if type(settings) is not dict or set(settings) - allowed:
            raise ValueError('Supported provider settings: model/effort/service_tier/timeout')
        self.client = session_type(cwd=Path(cwd), tools=[], instructions=instructions,
            model=settings.get('model', 'gpt-6-astra'), effort=settings.get('effort', 'xhigh'),
            service_tier=settings.get('service_tier', 'default'), disable_execution_tools=True,
            log_dir=Path(output), timeout=settings.get('timeout', 600))
        self.started = False

    def run_turn(self, prompt, callback, images):
        if not self.started:
            self.client.start()
            self.started = True
        return self.client.run_turn(prompt, callback, images=images)

    def __getattr__(self, name):
        return getattr(self.client, name)

    def close(self):
        self.client.close()


def codex_factory(settings, *, instructions, cwd, output):
    return CodexTaskSession(settings, instructions=instructions, cwd=cwd, output=output)
