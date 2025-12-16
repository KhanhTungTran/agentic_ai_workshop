## Agentic AI Workshop

A small set of scripts showing static vs dynamic retrieval-augmented generation (RAG) using the CloudCIX OpenAI-compatible endpoint.

### What's here
- `static_rag.py`: always retrieves context before answering.
- `dynamic_rag.py`: lets the model decide whether to call the retrieval tool.

### Run the examples
```bash
python static_rag.py
python dynamic_rag.py
```
