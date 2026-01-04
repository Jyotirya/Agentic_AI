from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from transformers import pipeline

class Message(TypedDict):
    role: str
    content: str

class ChatState(TypedDict):
    messages: List[Message]

llm = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

def llm_node(state: ChatState) -> ChatState:
    prompt = ""
    for m in state["messages"]:
        prompt += f"{m['role'].capitalize()}: {m['content']}\n"
    prompt += "Assistant:"
    out = llm(prompt, max_new_tokens=128)
    text = out[0].get("generated_text", "") if isinstance(out, list) else str(out)
    return {"messages": state["messages"] + [{"role": "assistant", "content": text}]}

graph = StateGraph(ChatState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
app = graph.compile()

def chat(app, history, user_input):
    history = history + [{"role": "user", "content": user_input}]
    result = app.invoke({"messages": history})
    return result["messages"]

history: List[Message] = []

# 1. Coding question
history = chat(app, history, "Write a Python function to print hello world.")
print("Coding Response:\n", history[-1]["content"])

# 2. Math question
history = chat(app, history, "What is the derivative of x^2 + 3x?")
print("\nMath Response:\n", history[-1]["content"])

# 3. General knowledge question
history = chat(app, history, "Who was Albert Einstein?")
print("\nGeneral Knowledge Response:\n", history[-1]["content"])