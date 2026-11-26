"""Chat-template wrapper shared by the E2 trainer.

Thinking stays off. The B_text arm is explicit text, and hidden reasoning
consumed the whole output budget without reaching an answer in the capability
screen. This is the same rule as ``experiments.e2.evaluate._wrap``.
"""

from __future__ import annotations


def wrap_user(tokenizer, prompt: str, *, think: bool = False) -> str:
    """Present ``prompt`` as a user turn when the tokenizer has a chat template.

    ``think`` defaults off and the trainer never turns it on. The flag is an
    argument so a test can see that the call site passes it.
    """
    if think:
        raise ValueError("E2 thinking is disabled; B_text must be explicit text")
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
