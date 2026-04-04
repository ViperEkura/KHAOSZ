from typing import Dict, List, Optional, Tuple, Any
from jinja2 import Template
from dataclasses import dataclass
from astrai.factory import Registry

HistoryType = List[Tuple[str, str]]
MessageType = Dict[str, str]


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


# Global registry instance
_default_registry = Registry()

# Default template name
_default_template_name = "chatml"


# Convenience functions
def register_chat_template(
    name: str,
    template_str: str,
    description: str = "",
    default_variables: Optional[Dict[str, Any]] = None,
    special_tokens: Optional[Dict[str, str]] = None,
) -> ChatTemplate:
    """Register a chat template in the global registry."""
    template = ChatTemplate(
        name=name,
        template_str=template_str,
        description=description,
        default_variables=default_variables,
        special_tokens=special_tokens,
    )
    _default_registry.register(name, template, category=None, priority=0)
    return template


def set_default_chat_template(name: str) -> None:
    """Set the default chat template name globally."""
    global _default_template_name
    if not _default_registry.contains(name):
        raise KeyError(
            f"Chat template '{name}' not found. Available: {list(_default_registry.list_names())}"
        )
    _default_template_name = name


def get_default_chat_template_name() -> str:
    """Get the current default chat template name."""
    return _default_template_name


def get_chat_template(name: str) -> ChatTemplate:
    """Get a chat template from the global registry."""
    return _default_registry.get(name)


def list_chat_templates() -> List[str]:
    """List all registered chat template names."""
    return _default_registry.list_names()


def chat_template_exists(name: str) -> bool:
    """Check if a chat template exists."""
    return _default_registry.contains(name)


def build_prompt(
    query: str,
    system_prompt: Optional[str] = None,
    history: Optional[HistoryType] = None,
    template: Optional[str] = None,
    template_name: Optional[str] = None,
    **extra_variables: Any,
) -> str:
    """Build prompt using a registered chat template or a custom template string.

    This function maintains backward compatibility with the previous API.

    Args:
        query: The current user query.
        system_prompt: Optional system prompt.
        history: Optional list of (user_msg, assistant_msg) pairs.
        template: If provided, uses this exact Jinja2 template string (overrides template_name).
        template_name: Name of a registered template to use (ignored if `template` is given).
            If None, uses the globally set default template (see `set_default_chat_template`).
        **extra_variables: Additional variables to pass to the template.

    Returns:
        Rendered prompt string.

    Raises:
        KeyError: If `template_name` is not registered.
    """
    # Convert history to message format
    messages: List[MessageType] = []
    if history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": query})

    if template is not None:
        # Use the provided template string directly
        jinja_template = Template(template)
        variables = {"messages": messages, **extra_variables}
        if system_prompt is not None:
            variables["system_prompt"] = system_prompt
        return jinja_template.render(**variables)
    else:
        # Determine which template name to use
        if template_name is None:
            template_name = _default_template_name
        # Use a registered template
        chat_template = get_chat_template(template_name)
        return chat_template.render(
            messages=messages,
            system_prompt=system_prompt,
            **extra_variables,
        )


# Predefined templates
# ChatML template (original)
register_chat_template(
    name="chatml",
    template_str=(
        "{%- if system_prompt -%}\n"
        "{{ bos_token }}system\n"
        "{{ system_prompt }}{{ eos_token }}\n"
        "{%- endif -%}\n"
        "{%- for message in messages -%}\n"
        "{{ bos_token }}{{ message['role'] }}\n"
        "{{ message['content'] }}{{ eos_token }}\n"
        "{%- endfor -%}\n"
        "{{ bos_token }}assistant\n"
    ),
    description="ChatML format with configurable special tokens.",
    special_tokens={"bos_token": "<｜im▁start｜>", "eos_token": "<｜im▁end｜>"},
)

# Simplified template without special tokens (plain text)
register_chat_template(
    name="plain",
    template_str=(
        "{%- if system_prompt -%}\n"
        "System: {{ system_prompt }}\n"
        "{%- endif -%}\n"
        "{%- for message in messages -%}\n"
        "{{ message['role']|capitalize }}: {{ message['content'] }}\n"
        "{%- endfor -%}\n"
        "Assistant:"
    ),
    description="Plain text format with role labels.",
)

# Alpaca-style template
register_chat_template(
    name="alpaca",
    template_str=(
        "{%- if system_prompt -%}\n"
        "### Instruction:\n"
        "{{ system_prompt }}\n"
        "{%- endif -%}\n"
        "### Input:\n"
        "{{ messages[-1]['content'] }}\n"
        "### Response:"
    ),
    description="Alpaca instruction‑response format (single‑turn).",
    default_variables={},
)

# OpenAI chat format (approximation)
register_chat_template(
    name="openai",
    template_str=(
        "{%- if system_prompt -%}\n"
        "{{ bos_token }}system\n"
        "{{ system_prompt }}{{ eos_token }}\n"
        "{%- endif -%}\n"
        "{%- for message in messages -%}\n"
        "{{ bos_token }}{{ message['role'] }}\n"
        "{{ message['content'] }}{{ eos_token }}\n"
        "{%- endfor -%}\n"
        "{{ bos_token }}assistant\n"
    ),
    description="OpenAI‑compatible chat format with configurable special tokens.",
    special_tokens={"bos_token": "<｜im▁start｜>", "eos_token": "<｜im▁end｜>"},
)

# Llama‑2 style with [INST] tags
register_chat_template(
    name="llama2",
    template_str=(
        "{%- if system_prompt -%}\n"
        "<<SYS>>\n"
        "{{ system_prompt }}\n"
        "<</SYS>>\n"
        "{%- endif -%}\n"
        "[INST] {{ messages[-1]['content'] }} [/INST]"
    ),
    description="Llama‑2 style with [INST] tags (single‑turn).",
    default_variables={},
)


__all__ = [
    "ChatTemplate",
    "register_chat_template",
    "get_chat_template",
    "list_chat_templates",
    "chat_template_exists",
    "build_prompt",
    "set_default_chat_template",
    "get_default_chat_template_name",
    "HistoryType",
    "MessageType",
]
