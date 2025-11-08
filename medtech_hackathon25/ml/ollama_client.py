"""Small helper for chatting with DeepSeek models via a local Ollama instance."""

from __future__ import annotations

from typing import Dict, Iterator, List, Set, Tuple

import ollama

DEFAULT_MODEL = "deepseek-r1:8b"
Message = Dict[str, str]
ChatHistory = List[Message]
MoodReport = Tuple[str, int]


def ensure_model_available(model: str = DEFAULT_MODEL) -> None:
    """Pull the requested model if it is not already available locally."""
    try:
        listing = ollama.list()
    except Exception as exc:  # pragma: no cover - depends on local daemon
        raise RuntimeError(
            "Unable to reach the Ollama service. Ensure `ollama serve` is running."
        ) from exc

    installed: Set[str] = set()
    models = getattr(listing, "models", [])
    for entry in models:
        if isinstance(entry, dict):
            name = entry.get("name")
        else:
            name = getattr(entry, "name", None)
        if name:
            installed.add(name)

    if model not in installed:
        ollama.pull(model)


def _build_messages(
    user_prompt: str,
    system_prompt: str | None,
    history: ChatHistory | None,
) -> ChatHistory:
    """Compose chat payload merging any prior history."""
    messages: ChatHistory = []
    if history:
        for item in history:
            role = item.get("role") if isinstance(item, dict) else None
            content = item.get("content") if isinstance(item, dict) else None
            if role and content:
                messages.append({"role": role, "content": content})
        if system_prompt and not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})
    return messages


def get_response(
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Return a blocking response string from the configured model."""
    ensure_model_available(model)
    messages = _build_messages(user_prompt, system_prompt, history=None)
    reply = ollama.chat(model=model, messages=messages)
    return reply.get("message", {}).get("content", "")


def stream_response(
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Iterator[str]:
    """Yield the model response incrementally for streaming UIs."""
    ensure_model_available(model)
    messages = _build_messages(user_prompt, system_prompt, history=None)
    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        content = chunk.get("message", {}).get("content")
        if content:
            yield content


class ChatSession:
    """Maintain chat history across multiple exchanges with an Ollama model."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        history: ChatHistory | None = None,
    ) -> None:
        ensure_model_available(model)
        self.model = model
        self.system_prompt = system_prompt
        self._history: ChatHistory = []
        if system_prompt:
            self._history.append({"role": "system", "content": system_prompt})
        if history:
            self.extend_history(history)

    @property
    def history(self) -> ChatHistory:
        return list(self._history)

    def extend_history(self, history: ChatHistory) -> None:
        for item in history:
            role = item.get("role") if isinstance(item, dict) else None
            content = item.get("content") if isinstance(item, dict) else None
            if role and content:
                self._history.append({"role": role, "content": content})

    def ask(self, prompt: str) -> str:
        messages = self.history + [{"role": "user", "content": prompt}]
        reply = ollama.chat(model=self.model, messages=messages)
        content = reply.get("message", {}).get("content", "")
        self._history.append({"role": "user", "content": prompt})
        if content:
            self._history.append({"role": "assistant", "content": content})
        return content

    def ask_with_mood(self, prompt: str) -> MoodReport:
        """Return the assistant reply along with a 1-10 mood score for the conversation."""
        response = self.ask(prompt)
        score = analyze_mood(self._history)
        return response, score

    def ask_stream(self, prompt: str) -> Iterator[str]:
        messages = self.history + [{"role": "user", "content": prompt}]
        buffer: List[str] = []
        for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
            content = chunk.get("message", {}).get("content")
            if content:
                buffer.append(content)
                yield content
        full_reply = "".join(buffer)
        self._history.append({"role": "user", "content": prompt})
        if full_reply:
            self._history.append({"role": "assistant", "content": full_reply})


def analyze_mood(history: ChatHistory) -> int:
    """Estimate overall conversation mood on a 1-10 scale using simple keyword sentiment."""
    positive_terms = {
        "good",
        "great",
        "happy",
        "excellent",
        "positive",
        "confident",
        "optimistic",
        "encouraging",
        "thank",
        "helpful",
    }
    negative_terms = {
        "bad",
        "sad",
        "angry",
        "frustrated",
        "negative",
        "concerned",
        "worried",
        "upset",
        "disappointed",
        "stress",
    }

    positive = 0
    negative = 0
    for item in history:
        content = item.get("content", "").lower()
        positive += sum(1 for word in positive_terms if word in content)
        negative += sum(1 for word in negative_terms if word in content)

    base_score = 5
    score = base_score + positive - negative
    return max(1, min(10, score))


if __name__ == "__main__":
    session = ChatSession(system_prompt="You are a concise clinical assistant.")
    question = "List two ways point-of-care testing speeds up decision making."
    answer, mood = session.ask_with_mood(question)
    print(answer)
    print(f"Mood score: {mood}/10")
