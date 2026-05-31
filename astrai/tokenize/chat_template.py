from typing import Any, Dict, List, Optional

from jinja2 import Template

type MessageType = Dict[str, Any]


class ChatTemplate:
    """A chat template with Jinja2 rendering support.

    Attributes:
        name: Unique identifier for the template.
        template_str: Jinja2 template string.
        description: Optional description.
        default_variables: Optional dictionary of default variable values.
        special_tokens: Optional dictionary mapping token names to their string values.
    """

    def __init__(
        self,
        name: str = "",
        template_str: str = "",
        description: str = "",
        default_variables: Optional[Dict[str, Any]] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.template_str = template_str
        self.description = description
        self.default_variables = default_variables or {}
        self.special_tokens = special_tokens or {}
        self._compiled : Template = Template(template_str)

    @classmethod
    def from_string(
        cls,
        template_str: str,
        description: str = "",
        default_variables: Optional[Dict[str, Any]] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> "ChatTemplate":
        """Create a ChatTemplate instance directly from a template string."""
        return cls(
            name="",
            template_str=template_str,
            description=description,
            default_variables=default_variables,
            special_tokens=special_tokens,
        )

    def render(
        self,
        messages: List[MessageType],
        system_prompt: Optional[str] = None,
        **extra_variables: Any,
    ) -> str:
        """Render the template with given messages and variables.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_prompt: Optional system prompt string.
            **extra_variables: Additional variables to pass to the template.
                These override default_variables and special_tokens.

        Returns:
            Rendered prompt string.
        """
        # Merge default variables, special tokens, and extra variables
        variables = {**self.default_variables, **self.special_tokens, **extra_variables}
        variables["messages"] = messages
        if system_prompt is not None:
            variables["system_prompt"] = system_prompt

        return self._compiled.render(**variables)
