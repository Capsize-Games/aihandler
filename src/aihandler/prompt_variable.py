import json
import random
import re

data = json.load(open("data/prompts.json", "r"))
data["age"] = [str(n) for n in range(18, 100)]


class PromptVariable:
    """
    A class which will take a prompt, and replace all variables with random
    values.
    """
    misc_values = {
        "age": data
    }

    @classmethod
    def get_values(cls, variable_name):
        """
        Gets the values for a variable.
        :param prompt_type:
        :param variable_name:
        :return:
        """
        return cls.misc_values.get(variable_name, [])

    @classmethod
    def get_random_value(cls, prompt_type=None, variable_name="", available_variables=None):
        """
        Gets a random value for a variable.
        :param prompt_type:
        :param variable_name:
        :return:
        """
        variable_name = variable_name.lower()

        # handle special case of human_name
        if available_variables and variable_name in available_variables:
            values = available_variables.get(variable_name, [])
        else:
            values = cls.get_values(variable_name)
        if len(values) > 0:
            return random.choice(values).lower()
        print("No values found for variable: " + variable_name)
        return ""

    @classmethod
    def translate_variable(cls, prompt_type=None, variable="", available_variables=None, weights=None, seed=None):
        """
        Translates a variable into a random value.
        :param prompt_type:
        :param variable:
        :return:
        """
        if seed:
            random.seed(seed)
        # remove the $ from the variable
        variable = variable.replace("$", "")

        original_variable = None
        if variable == "gender_name":
            original_variable = "gender_name"
            variable = "gender"

        # get the random value
        if prompt_type:
            random_value = cls.get_random_value(prompt_type, variable, available_variables)
        else:
            random_value = cls.get_random_value("misc", variable, available_variables)
        if variable == "age":
            random_value = f"{random_value} years old"
        if weights and variable in weights and (original_variable is None or original_variable != "gender_name"):
            random_value = f"({random_value}){weights[variable]['weight']}"

        if original_variable == "gender_name":
            print("returning gender_name")
            return f"{random_value}_name"

        return random_value

    @classmethod
    def find_variables(cls, prompt):
        # find anything starting with a $ including $$
        pattern = r"(?<!\\)\$[a-zA-Z0-9_]+"
        matches = re.findall(pattern, prompt)
        return matches

    @classmethod
    def replace_var_with_weight(cls, match, weights=None):
        var = match.group("var")
        if var and weights:
            stripped_dollarsign = var.replace("$", "")
            if stripped_dollarsign in weights:
                val = weights[stripped_dollarsign]["value"]
                weight = weights[stripped_dollarsign]["weight"]
                if val != "":
                    val = val.lower()
                    if stripped_dollarsign == "age":
                        val = f"{val} years old"
                    value = f"({val}){weight}"
                    return value
                return match.group("var")
        return f'{match.group("var")}'

    @classmethod
    def parse(cls, prompt_type=None, prompt="", available_variables=None, weights=None, seed=None):
        """
        Finds all variables in a prompt, and replaces them with random values.
        :param prompt_type:
        :param prompt:
        :return:
        """
        variables = cls.find_variables(prompt)
        # pattern = r"(?<!\$)(?P<var>\$[a-zA-Z_0-9]+)"
        # prompt = re.sub(pattern, partial(cls.replace_var_with_weight, weights=weights), prompt)
        for variable in variables:
            translated_variable = cls.translate_variable(
                prompt_type,
                variable,
                available_variables=available_variables,
                weights=weights,
                seed=seed
            )
            # find variables of of the form $variable but not $$variable
            if variable == "$gender_name":
                prompt = prompt.replace("$$gender_name", f"${translated_variable}")
            else:
                # strip $ from variable
                variable = variable.replace("$", "")
                prompt = re.sub(r"(?<!\$)\$" + variable, translated_variable, prompt)
        return prompt
