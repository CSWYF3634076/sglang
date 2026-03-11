import logging
from typing import Any, Dict, List, Optional, Union

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    LogProbs,
)

logger = logging.getLogger(__name__)


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, _, token_text in token_logprobs:
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {token[2]: token[0] for token in tokens}
                )
            else:
                ret_logprobs.top_logprobs.append(None)

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs


def process_hidden_states_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[List]:
    """Process hidden states from a ret item in non-streaming response.

    Args:
        ret_item: Response item containing meta_info
        request: The original request object

    Returns:
        Processed hidden states for the last token, or None
    """
    if not request.return_hidden_states:
        return None

    hidden_states = ret_item["meta_info"].get("hidden_states", None)
    if hidden_states is not None:
        hidden_states = hidden_states[-1] if len(hidden_states) > 1 else []
    return hidden_states

def process_completion_token_ids_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[List[int]]:
    """Process completion token ids from a non-streaming response item."""
    if not getattr(request, "return_token_ids", False):
        return None

    output_ids = ret_item.get("output_ids")
    if output_ids is None:
        return None
    return list(output_ids)


def process_prompt_token_ids_from_request(
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
    prompt_index: int = 0,
) -> Optional[List[int]]:
    """Process prompt token ids from request-attached state."""
    return request.get_prompt_token_ids(prompt_index)


def process_stream_completion_token_ids(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
    completion_token_ids_so_far: Dict[int, List[int]],
    index: int,
) -> Optional[List[int]]:
    """Return newly observed completion token ids for a streaming response item.

    TokenizerManager can yield either cumulative output ids (default) or incremental
    output ids when `stream_output` is enabled. Handle both forms.
    """
    if not getattr(request, "return_token_ids", False):
        return None

    current_output_ids = ret_item.get("output_ids")
    if current_output_ids is None:
        return None

    current_output_ids = list(current_output_ids)
    previous_output_ids = completion_token_ids_so_far.get(index, [])

    if (
        len(current_output_ids) >= len(previous_output_ids)
        and current_output_ids[: len(previous_output_ids)] == previous_output_ids
    ):
        new_output_ids = current_output_ids[len(previous_output_ids) :]
        completion_token_ids_so_far[index] = current_output_ids
    else:
        new_output_ids = current_output_ids
        completion_token_ids_so_far[index] = previous_output_ids + current_output_ids

    return new_output_ids or None
