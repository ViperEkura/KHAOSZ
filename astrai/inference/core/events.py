"""Output events: the single event contract between core and frontend.

Mirrors vLLM v1's frontend/core split: the scheduler loop emits token IDs
and terminal facts only — never user callbacks, never text rendering.  The
frontend (engine + protocol adapters) consumes these events, detokenizes,
maps finish reasons and builds protocol responses.

Lifecycle: every request ends with exactly one terminal event
(``RequestFinished`` or ``RequestError``); terminal events are emitted
after all of the request's ``TokenDelta`` events.  Consumers may rely on
that ordering to release per-request state.
"""

from dataclasses import dataclass
from typing import Tuple

# Finish reasons shared by every entry point.  Protocol adapters map these
# onto their own vocabularies (OpenAI ``finish_reason``, Anthropic
# ``stop_reason``); the core never invents protocol-specific reasons.
FINISH_STOP_TOKEN = "stop_token"  # a tokenizer stop id was sampled
FINISH_LENGTH = "length"  # max_tokens / sequence cap reached
FINISH_CANCELLED = "cancelled"  # user cancelled before completion
FINISH_ABORTED = "aborted"  # engine-side termination (KV cap…)
FINISH_REJECTED = "rejected"  # request refused before running


@dataclass(frozen=True)
class TokenDelta:
    """One committed output token for a request.

    ``text`` is the incremental detokenized fragment (may be empty while a
    multi-byte character is still incomplete); it is filled by the
    frontend's OutputProcessor, not by the core.
    """

    request_id: str
    token_id: int
    sequence_no: int  # 1-based output position of this token
    text: str = ""


@dataclass(frozen=True)
class RequestFinished:
    """Terminal success event for a request."""

    request_id: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Sampled token ids, filled when the consumer asked to keep them
    # (such as rollout); streaming consumers leave it empty.
    token_ids: Tuple = ()


@dataclass(frozen=True)
class RequestError:
    """Terminal failure event for a request."""

    request_id: str
    error_code: str
    message: str
    retryable: bool = False
    finish_reason: str = FINISH_ABORTED
