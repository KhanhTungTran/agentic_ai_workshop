"""Minimal Dynamic RAG Tutorial Example using CloudCIX tool calls.

Flow Overview:
1. Send initial user query to CloudCIX hosted LLM with a tool spec (retrieve_information).
2. Model may decide to call the tool (this is the dynamic part of RAG: model chooses retrieval).
3. We execute the tool, performing a vector search against the CloudCIX embedding DB.
4. Feed retrieved context back to the model for a grounded final answer.

Logging is added at each step to make the process transparent for tutorial purposes.
"""

import json
import logging
import os
from typing import List

from openai import OpenAI

# -----------------------------------------------------------------------------
# Logging Setup (simple, tutorial-friendly)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration (in tutorials prefer environment variable over hardcoded secret)
# -----------------------------------------------------------------------------
CLOUDCIX_API_KEY = os.getenv(
    "CLOUDCIX_API_KEY"
)

client = OpenAI(
    api_key=CLOUDCIX_API_KEY,
    base_url="https://ml-openai.cloudcix.com",  # CloudCIX OpenAI-compatible endpoint
)

TEMP = 0.10
MAX_TOK = 8192

model = "GPT-4.1" # "UCCIX-Mistral-24B"

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_information",
            "description": "Vector search in CloudCIX embedding DB for context about AI For Ireland.",
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
    import requests  # local import keeps top-level clean for tutorial

    url = "https://ml.cloudcix.com/embedding_db/"

    payload = {
        "api_key": CLOUDCIX_API_KEY,
        "names": ["AIFI"],  # collection names
        "method": "vector_search",
        "encoder_name": "test_encoder",
        "query": query,
        "order_by": "euclidean_distance",
        "threshold": "25.0",
        "limit": "3",
    }

    log.info("[tool] Sending vector search request to embedding DB")
    response = requests.post(url=url, json=payload, timeout=30)
    if response.status_code != 200:
        log.warning("[tool] Non-200 response (%s): %s", response.status_code, response.text[:200])
    dct = response.json()
    log.info("[tool] Raw embedding DB response keys: %s", list(dct.keys()))

    output_lines: List[str] = []
    for k, v in dct.items():
        output_lines.append(f"Source {k}: {v}")
    joined = "\n\n".join(output_lines).strip()
    log.info("[tool] Compiled %d source snippet(s) for model context injection", len(output_lines))
    return joined

for USER_QUESTION in [
    "What is ai for ireland?",
    "What is a cat?",
]:

    messages = [
        {"role": "user", "content": USER_QUESTION}
    ]


    log.info("User query: %s", messages[0]["content"])
    log.info("Model selected: %s | Temperature: %.2f", model, TEMP)

    log.info("Tool spec registered: %s", tools[0]["function"]["name"])

    log.info("Sending first chat completion request (tool_choice=auto)")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
        tools=tools,
        tool_choice="auto",  # let model decide if it wants retrieval
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
