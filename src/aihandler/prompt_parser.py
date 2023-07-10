from aihandler.prompt_variable import PromptVariable
from aihandler.prompt_weight_bridge import PromptWeightBridge


class PromptParser:
    """
    A class which will take a prompt, and prase it into a format that the
    AI can understand.
    """

    @classmethod
    def parse(cls, data, prompt_type, prompt, variables=None, weights=None, seed=None):
        """
        Parses a prompt into a format that the AI can understand.
        """
        # first we will run weight translation on the prompt
        prompt = PromptWeightBridge.convert(prompt)

        # next we will run variable translation on the prompt
        prompt = PromptVariable.parse(data, prompt_type, prompt, variables, weights, seed)
        prompt = PromptVariable.parse(data, prompt_type, prompt, variables, weights, seed)

        return prompt
