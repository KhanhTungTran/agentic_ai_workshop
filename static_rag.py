"""Minimal Static RAG Tutorial Example using CloudCIX tool calls.

New to agentic AI? This example is "non-agentic": the developer always calls
`retrieve_information` (vector search) before the model answers. Use it to compare with
the dynamic version where the model decides on tool use.

Flow Overview:
1. Send initial user query to CloudCIX hosted LLM with a tool spec (retrieve_information).
2. Always call the tool.
3. We execute the tool, performing a vector search against the CloudCIX embedding DB.
4. Feed retrieved context back to the model for a grounded final answer.
"""

import json
import logging
import os
from typing import List

from openai import OpenAI

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

TEMP = 0.6
MAX_TOK = 8192

model = "UCCIX-Mistral-24B"

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_information",
            "description": "Vector search in CloudCIX embedding DB",
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
    """Call CloudCIX embedding DB for a vector search.

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


# -----------------------------------------------------------------------------
# Query without RAG (no context injection)
# -----------------------------------------------------------------------------
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

    log.info("Sending chat completion request")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
    )

    final_answer = response.choices[0].message.content
    log.info("Final Answer:\n%s\n\n", final_answer)


# -----------------------------------------------------------------------------
# Query with Static RAG (always retrieve context)
# -----------------------------------------------------------------------------
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

    retrieved_context = retrieve_information(messages[0]["content"])
    log.info("Tool 'retrieve_information' result length: %d chars", len(retrieved_context))

    messages[0]["content"] += f"\n\nContext:\n{retrieved_context}"

    log.info("Sending chat completion request")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
    )

    final_answer = response.choices[0].message.content
    log.info("Final Answer:\n%s\n\n", final_answer)
