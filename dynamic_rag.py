"""Minimal Dynamic RAG Tutorial Example using CloudCIX tool calls.

New to agentic AI? Think of the model as an "agent" that can choose tools on its own.
Here, the model decides whether to call `retrieve_information` (vector search) before
answering, demonstrating how LLMs can plan tool use instead of being explicitly told.

Flow Overview:
1. Ask the LLM to decide if retrieval is needed for the user question.
2. If yes, set tool_choice to the retrieve_information tool; otherwise set tool_choice to none.
3. Send the user query to CloudCIX hosted LLM with the chosen tool_choice.
4. If the model requested the tool, execute retrieve_information against the CloudCIX embedding DB.
5. Feed any retrieved context back to the model for a grounded final answer.
"""

import json
import logging
import os
from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import \
    Function

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration (environment variable over hardcoded secret)
# -----------------------------------------------------------------------------
CLOUDCIX_API_KEY = os.getenv(
    "CLOUDCIX_API_KEY"
)

client = OpenAI(
    api_key=CLOUDCIX_API_KEY,
    base_url="https://ml-openai.cloudcix.com",  # CloudCIX OpenAI-compatible endpoint
)

TEMP = 0.60
MAX_TOK = 8192

model = "UCCIX-Mistral-24B"

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_information",
            "description": "Vector search in CloudCIX embedding DB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to retrieve related context snippets.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

def retrieve_information(query: str) -> str:
    """Call CloudCIX embedding DB for a small vector search.

    Returns a concatenated string of sources for easy injection as context.
    """
    import requests

    url = "https://ml.cloudcix.com/embedding_db/"

    payload = {
        "api_key": CLOUDCIX_API_KEY,
        "names": ["CloudCIX"],  # collection names
        "method": "vector_search",
        "encoder_name": "gte-large-en-v1.5",
        "query": query,
        "order_by": "euclidean_distance",
        "threshold": "25.0",
        "limit": 3,
    }

    log.info("[tool] Sending vector search request to embedding DB")
    response = requests.post(url=url, json=payload, timeout=30)
    if response.status_code != 200:
        log.warning("[tool] Non-200 response (%s): %s", response.status_code, response.text[:200])
    dct = response.json()
    log.info("[tool] Raw embedding DB response keys: %s", list(dct.keys()))

    output_lines: List[str] = []
    for item in dct.get("content", []):
        source = item[0]
        content = item[1]
        output_lines.append(f"Source {source}: {content}")
    joined = "\n\n".join(output_lines).strip()
    log.info("[tool] Compiled %d source snippet(s) for model context injection", len(output_lines))
    return joined

def should_use_retrieval(user_question: str) -> bool:
    """Ask the LLM if the question needs CloudCIX-specific retrieval."""
    decision_messages = [
        {
            "role": "user",
            "content": (
                "Decide if the following user question needs CloudCIX-specific context. "
                "Reply with 'use_tool' if retrieval is needed, otherwise reply with 'no_tool'.\n" + user_question
            ),
        },
    ]
    log.info("Requesting retrieval decision from model")
    resp = client.chat.completions.create(
        model=model,
        messages=decision_messages,
        temperature=0,
        max_tokens=4,
        tool_choice="none",
    )
    decision = (resp.choices[0].message.content or "").lower()
    use_tool = "use_tool" in decision
    log.info("Decision: %s", "use_tool" if use_tool else "no_tool")
    return use_tool

for USER_QUESTION in [
    "What is CloudCIX?",
    "What is the difference between CloudCIX and CIX?",
    "What is a cat?",
]:

    messages = [
        {"role": "user", "content": USER_QUESTION}
    ]

    log.info("User query: %s", messages[0]["content"])
    log.info("Model selected: %s | Temperature: %.2f", model, TEMP)

    log.info("Tool spec registered: %s", tools[0]["function"]["name"])

    use_retrieval = should_use_retrieval(USER_QUESTION)
    tool_choice = (
        {"type": "function", "function": {"name": "retrieve_information"}}
        if use_retrieval
        else "none"
    )

    log.info("Sending first chat completion request (tool_choice=%s)", tool_choice)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
        tools=tools,
        tool_choice=tool_choice,
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        log.info("Model chose NOT to call any tool. Proceeding without retrieval.")
    else:
        log.info("Model requested %d tool call(s).", len(tool_calls))
        for i, tc in enumerate(tool_calls, start=1):
            log.info(
                "Tool Call %d: name=%s arguments=%s", i, tc.function.name, tc.function.arguments
            )

    results = []
    for tool_call in tool_calls or []:
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        if function_name == "retrieve_information":
            parsed_args = json.loads(function_args)
            log.info("Executing tool '%s' with args: %s", function_name, parsed_args)
            result = retrieve_information(**parsed_args)
            log.info("Tool '%s' result length: %d chars", function_name, len(result))
            results.append(result)

    if tool_calls:
        messages.append({"role": "assistant", "tool_calls": tool_calls, "content": ""})
        for tool_call, result in zip(tool_calls, results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result,
                }
            )
        log.info("Injected %d retrieval result(s) back into conversation for grounding.", len(results))
    else:
        log.info("Skipping context injection step (no tool calls).")

    log.info("Requesting final grounded answer from model")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
    )

    final_answer = response.choices[0].message.content
    log.info("Final Answer:\n%s\n\n", final_answer)
