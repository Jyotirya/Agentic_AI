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
    selected_agent: Optional[str]

llm = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

def _generate_text(prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
    out = llm(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    if isinstance(out, list) and out:
        return out[0].get("generated_text") or str(out[0])
    if isinstance(out, dict):
        return out.get("generated_text") or str(out)
    return str(out)

def router(state: ChatState) -> ChatState:
    user_msgs = [m for m in state["messages"] if m["role"] == "user"]
    last_msg = user_msgs[-1]["content"] if user_msgs else ""
    text = last_msg.lower()
    python_keywords = [
        "python", ".py", "pip", "django", "flask", "list", "dict", "tuple",
        "numpy", "pandas", "import", "def ", "class ", "lambda", "print", "function"
    ]
    chosen = "python" if any(k in text for k in python_keywords) else "general"
    state["selected_agent"] = chosen
    state["messages"].append({"role": "assistant", "content": f"Router: selected '{chosen}' agent."})
    return state

def python_agent(state: ChatState) -> ChatState:
    if state.get("selected_agent") != "python":
        return state
    user_msgs = [m for m in state["messages"] if m["role"] == "user"]
    question = user_msgs[-1]["content"] if user_msgs else ""
    prompt = (
        "You are a Python expert. Provide a clear, correct, and concise Python-focused "
        "answer. If code is useful, include a minimal example.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    answer = _generate_text(prompt, max_new_tokens=256, temperature=0.0).strip()
    state["messages"].append({"role": "assistant", "content": answer})
    return state

def general_agent(state: ChatState) -> ChatState:
    if state.get("selected_agent") != "general":
        return state
    user_msgs = [m for m in state["messages"] if m["role"] == "user"]
    question = user_msgs[-1]["content"] if user_msgs else ""
    prompt = (
        "You are a helpful general-knowledge assistant. Answer clearly and concisely.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    answer = _generate_text(prompt, max_new_tokens=256, temperature=0.2).strip()
    state["messages"].append({"role": "assistant", "content": answer})
    return state
def dispatch(state: ChatState) -> ChatState:
    sel = state.get("selected_agent")
    if sel == "python":
        return python_agent(state)
    else:
        return general_agent(state)

graph = StateGraph(ChatState)
graph.add_node("router", router)
graph.add_node("dispatch", dispatch)
graph.add_edge(START, "router")
graph.add_edge(START, "router")
graph.add_edge("router", "dispatch")
graph.add_edge("dispatch", END)
app = graph.compile()

def run_query(user_input: str) -> None:
    initial_state: ChatState = {"messages": [{"role": "user", "content": user_input}], "selected_agent": None}
    result = app.invoke(initial_state)
    assistant_msgs = [m for m in result["messages"] if m["role"].startswith("assistant")]
    final = assistant_msgs[-1]["content"] if assistant_msgs else "(no answer)"
    print("User question:", user_input)
    print("Final answer:", final)
    print("-" * 60)

if __name__ == "__main__":
    tests = [
        "how to print hello world in python",
        "what is the capital of France?",
        "can you provide a python function to compute factorial?",
        "who was Isaac Newton?",
        "explain the concept of class with a python example"
    ]
    for q in tests:
        run_query(q)