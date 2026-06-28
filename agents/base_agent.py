class BaseAgent:
    def __init__(self, name):
        self.name = name

    def analyze(self, context):
        raise NotImplementedError
