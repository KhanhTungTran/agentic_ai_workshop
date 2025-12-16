import json
import logging
import os
from typing import Literal

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLOUDCIX_API_KEY = os.getenv("CLOUDCIX_API_KEY")
client = OpenAI(api_key=CLOUDCIX_API_KEY, base_url="https://ml-openai.cloudcix.com")

MODEL = "UCCIX-Mistral-24B"
TEMP = 0.6
MAX_TOK = 2048

tools = [
    {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "Perform basic arithmetic between two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {"type": "number", "description": "First operand."},
                    "num2": {"type": "number", "description": "Second operand."},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "Arithmetic operation to perform.",
                    },
                },
                "required": ["num1", "num2", "operation"],
            },
        },
    }
]

def simple_calculator(num1: float, num2: float, operation: Literal["add", "subtract", "multiply", "divide"]) -> str:
    if operation == "add":
        result = num1 + num2
    elif operation == "subtract":
        result = num1 - num2
    elif operation == "multiply":
        result = num1 * num2
    else:
        if num2 == 0:
            return "Error: division by zero is not allowed."
        result = num1 / num2
    return f"{num1} {operation} {num2} = {result}"

def should_use_calculator(user_question: str) -> bool:
    """Ask the LLM if the question needs the calculator tool."""
    decision_messages = [
        {
            "role": "user",
            "content": (
                "Decide if the following user question needs arithmetic calculation. "
                "Reply with 'use_tool' if calculation is needed, otherwise reply with 'no_tool'.\n"
                f"{user_question}"
            ),
        },
    ]
    log.info("Requesting calculator decision from model")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=decision_messages,
        temperature=0,
        max_tokens=4,
        tool_choice="none",
    )
    decision = (resp.choices[0].message.content or "").lower()
    use_tool = "use_tool" in decision
    log.info("Decision: %s", "use_tool" if use_tool else "no_tool")
    return use_tool

demo_questions = [
    "What is 3 plus 5?",
    "Calculate 7 times 8.",
    "Divide 10 by 0.",
    "Explain what 2 - 1 equals.",
    "What is a cat",
]

for question in demo_questions:
    messages = [{"role": "user", "content": question}]
    log.info("User query: %s", question)

    use_calculator = should_use_calculator(question)
    tool_choice = {"type": "function", "function": {"name": "simple_calculator"}} if use_calculator else "none"

    first_resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
        tools=tools,
        tool_choice=tool_choice,
    )

    tool_calls = first_resp.choices[0].message.tool_calls or []
    if not tool_calls:
        log.info("Model chose not to use the calculator.")
        final_resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMP,
            max_tokens=MAX_TOK,
        )
        log.info("Answer:\n%s\n", final_resp.choices[0].message.content)
        continue

    log.info("Model requested %d tool call(s).", len(tool_calls))
    messages.append({"role": "assistant", "tool_calls": tool_calls, "content": ""})

    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        log.info("Executing calculator with args: %s", args)
        result = simple_calculator(**args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            }
        )

    final_resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=TEMP, max_tokens=MAX_TOK)
    log.info("Final Answer:\n%s\n", final_resp.choices[0].message.content)
