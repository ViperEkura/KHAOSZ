from typing import Dict, List, Optional, Tuple
from jinja2 import Template

HistoryType = List[Tuple[str, str]]
MessageType = Dict[str, str]

# Predefined chat templates using jinja2
CHAT_TEMPLATES: Dict[str, str] = {
    "chatml": """{%- if system_prompt -%}
<｜im▁start｜>system
{{ system_prompt }}<｜im▁end｜>
{%- endif -%}
{%- for message in messages -%}
<｜im▁start｜>{{ message['role'] }}
{{ message['content'] }}<｜im▁end｜>
{%- endfor -%}
<｜im▁start｜>assistant
""",
}


def build_prompt(
    query: str,
    system_prompt: Optional[str] = None,
    history: Optional[HistoryType] = None,
    template: Optional[str] = None,
) -> str:
    """Build prompt using jinja2 template for query and history.

    Args:
        query (str): query string.
        system_prompt (Optional[str]): system prompt string.
        history (Optional[HistoryType]): history list of query and response.
        template (Optional[str]): jinja2 template string. If None, uses default chatml template.

    Returns:
        str: prompt string formatted according to the template.

    Example:
        # Use default template
        prompt = build_prompt(query="Hello", history=[...])

        # Use custom template
        custom_template = '''
        {%- for msg in messages -%}
        {{ msg['role'] }}: {{ msg['content'] }}
        {%- endfor -%}
        '''
        prompt = build_prompt(query="Hello", template=custom_template)
    """
    # Convert history to message format
    messages: List[MessageType] = []
    if history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": query})

    # Use provided template or default chatml template
    template_str = template if template is not None else CHAT_TEMPLATES["chatml"]

    # Render template
    jinja_template = Template(template_str)
    return jinja_template.render(
        messages=messages,
        system_prompt=system_prompt,
    )
