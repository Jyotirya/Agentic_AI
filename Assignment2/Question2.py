from typing import List, TypedDict, Optional
import os
from transformers import pipeline

from langgraph.graph import StateGraph, START, END
from huggingface_hub import InferenceClient

class Message(TypedDict):
    role: str
    content: str

class ChatState(TypedDict):
    messages: List[Message]
    clarified_question: Optional[str]

llm = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)


def _generate_text(prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
    out = llm(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    if isinstance(out, list) and out:
        return out[0].get("generated_text") or str(out[0])
    if isinstance(out, dict):
        return out.get("generated_text") or str(out)
    return str(out)

def question_analyzer(state: ChatState) -> ChatState:
    user_msgs = [m for m in state["messages"] if m["role"] == "user"]
    if not user_msgs:
        clarified = ""
    else:
        last_q = user_msgs[-1]["content"]
        prompt = (
            "You are a helpful assistant that rewrites and clarifies user questions. "
            "Rewrite the following question to be concise, unambiguous, and suitable for an LLM to answer.\n\n"
            f"Original question: {last_q}\n\nClarified question:"
        )
        clarified = _generate_text(prompt, max_new_tokens=128, temperature=0.0).strip()
    state["clarified_question"] = clarified
    # optionally append analyzer output to messages
    state["messages"].append({"role": "assistant", "content": f"Clarified question: {clarified}"})
    return state

def answer_generator(state: ChatState) -> ChatState:
    clarified = state.get("clarified_question") or ""
    if not clarified:
        # fallback: answer the last user message directly
        user_msgs = [m for m in state["messages"] if m["role"] == "user"]
        clarified = user_msgs[-1]["content"] if user_msgs else ""
    prompt = (
        "You are an expert assistant. Answer the question below clearly and concisely. "
        "If relevant, provide code for coding questions or a short explanation for math and factual queries.\n\n"
        f"Question: {clarified}\n\nAnswer:"
    )
    answer = _generate_text(prompt, max_new_tokens=256, temperature=0.2).strip()
    state["messages"].append({"role": "assistant", "content": answer})
    return state

graph = StateGraph(ChatState)
graph.add_node("analyzer", question_analyzer)
graph.add_node("generator", answer_generator)
graph.add_edge(START, "analyzer")
graph.add_edge("analyzer", "generator")
graph.add_edge("generator", END)
app = graph.compile()

def run_query(user_input: str) -> None:
    initial_state: ChatState = {"messages": [{"role": "user", "content": user_input}], "clarified_question": None}
    result = app.invoke(initial_state)
    assistant_msgs = [m for m in result["messages"] if m["role"].startswith("assistant")]
    clarified = assistant_msgs[-2]["content"] if len(assistant_msgs) >= 2 else "(no clarified text)"
    final = assistant_msgs[-1]["content"] if assistant_msgs else "(no answer)"
    print("User question:", user_input)
    print("Clarified question:", clarified.replace("Clarified question:", "").strip())
    print("Final answer:", final)

if __name__ == "__main__":
    queries = [
        "how to print hello world in python",
        "Find derivative x^2+3x",
        "Tell me about Albert Einstein"
    ]
    for q in queries:
        run_query(q)