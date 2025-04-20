from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class CustomPrompt:
    def __init__(self, template: str, pydantic_output_parser_class):
        self.template = template
        self.parser = PydanticOutputParser(pydantic_object=pydantic_output_parser_class)
        self.prompt = PromptTemplate(
            template=self.template,
            template_format='f-string',
        )
        self.prompt.template = 'Answer the USER_QUERY.\n\n{FORMAT_INSTRUCTIONS}\n\nUSER_QUERY:\n\n' + self.prompt.template
        self.prompt.partial_variables.update({"FORMAT_INSTRUCTIONS": self.parser.get_format_instructions()})

    def get_prompt(self):
        return self.prompt