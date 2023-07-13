from aihandler.prompt_variable import PromptVariable
from aihandler.prompt_weight_bridge import PromptWeightBridge


class PromptParser:
    """
    A class which will take a prompt, and prase it into a format that the
    AI can understand.
    """
    @classmethod
    def do_parse(cls, prompt, variables, weights, seed):
        if not prompt:
            return ""
        # first we will run weight translation on the prompt
        parsed_prompt = PromptWeightBridge.convert(prompt)

        # next we will run variable translation on the prompt
        parsed_prompt = PromptVariable.parse(
            parsed_prompt, variables, weights, seed)
        parsed_prompt = PromptVariable.parse(
            parsed_prompt, variables, weights, seed)
        parsed_prompt = parsed_prompt.strip()

        return parsed_prompt

    @classmethod
    def parse(
        cls,
        prompt=None,
        negative_prompt=None,
        generated_prompt=None,
        generated_negative_prompt=None,
        text_weight=0,
        auto_weight=0,
        negative_text_weight=0,
        negative_auto_weight=0,
        variables=None,
        weights=None,
        seed=None
    ):
        """
        Parses a prompt into a format that the AI can understand.
        """
        prompt = cls.do_parse(prompt, variables, weights, seed)
        negative_prompt = cls.do_parse(negative_prompt, variables, weights, seed)
        generated_prompt = cls.do_parse(generated_prompt, variables, weights, seed)
        generated_negative_prompt = cls.do_parse(generated_negative_prompt, variables, weights, seed)

        if prompt != "" and text_weight > 0 and auto_weight > 0:
            prompt = f'("{prompt}", "{generated_prompt}").blend({text_weight:.2f}, {auto_weight:.2f})'
        elif text_weight == 0 or prompt == "":
            prompt = generated_prompt

        if negative_prompt != "" and negative_text_weight > 0 and negative_auto_weight > 0:
            negative_prompt = f'("{negative_prompt}", "{generated_negative_prompt}").blend({negative_text_weight:.2f}, {negative_auto_weight:.2f})'
        elif negative_text_weight == 0 or negative_prompt == "":
            negative_prompt = generated_negative_prompt

        return prompt, negative_prompt
