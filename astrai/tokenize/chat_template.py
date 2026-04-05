from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jinja2 import Template

# Message type for chat messages
type MessageType = Dict[str, Any]


@dataclass
class ChatTemplate:
    """A chat template with Jinja2 rendering support.

    Attributes:
        name: Unique identifier for the template.
        template_str: Jinja2 template string.
        description: Optional description.
        default_variables: Optional dictionary of default variable values
            that will be passed to the template if not overridden during rendering.
        special_tokens: Optional dictionary mapping token names to their string values.
            These tokens are automatically added to the template variables.
    """

    name: str
    template_str: str
    description: str = ""
    default_variables: Dict[str, Any] = None
    special_tokens: Dict[str, str] = None

    def __post_init__(self):
        if self.default_variables is None:
            self.default_variables = {}
        if self.special_tokens is None:
            self.special_tokens = {}

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
            name="",  # empty name for ad‑hoc templates
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

        jinja_template = Template(self.template_str)
        return jinja_template.render(**variables)
