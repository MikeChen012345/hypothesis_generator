# agent-helper

Scientific hypothesis co-pilot built with LangGraph + LangChain. The agent focuses on helping researchers start with a single draft hypothesis, fan out multiple improved candidates, interrogate evidence gaps, and surface ranked recommendations.

## Workflow Highlights
- **Phase 1 – Hypothesis Clarification:** capture the researcher’s draft hypothesis, variables, measurement targets, and ambiguities.
- **Phase 2 – Candidate Expansion:** generate multiple hypotheses (H1…Hn) with clarified assumptions,
focusing on diverse perspectives and alternative explanations.
- **Phase 3 - Grounding Research:** retrieve relevant papers, web results, and RAG context to inform evidence gap analysis and check citation authenticity (leveraging research tools when needed).
- **Phase 4 – Evidence Gap Analysis:** walk through each candidate hypothesis, suggesting potential literature and data sources to fill gaps.
- **Phase 5 – Synthesis:** summarize all candidates side-by-side, rank them, and return a final recommendation.

## Configuration
`config.yaml` controls logging, memory, and research-specific behavior.

- Long-term (Postgres + Qdrant) storage can archive curated hypotheses for reuse.
- `Hypothesis` section introduces:
  - `candidate_count`: number of improved candidates to generate.
  - `archive_candidates`: toggle for persisting refined candidates to shared memory.
  - `show_all_candidates`: force downstream phases to present every candidate.
- `Model` section centralizes runtime parameters (model name, temperature, timeout, retry/max token limits, base URL, API key) so you can switch providers without code changes.
- `Logging` supports a dedicated `chat_history` block so interactive transcripts go to `chat_history.log` while process telemetry stays in `workflow.log`.

## Environment Variables
Create a `.env` file with at least:

```
OPENAI_API_ENDPOINT=<azure_or_openai_endpoint>
OPENAI_API_KEY=<token>
POSTGRESQL_CONNECTION_STRING=<postgres_uri>
QDRANT_URL=<qdrant_http_endpoint>
GOOGLE_SEARCH_API_KEY=<cse_key>
GOOGLE_SEARCH_ENGINE_ID=<cse_id>
```

Additional API keys (e.g., SERP providers) can be added as needed.

## Installation & Usage
1. Clone and install:
   ```bash
   git clone git@github.com:MikeChen012345/hypothesis_generator.git
   cd hypothesis_generator
   python -m venv venv
   venv\Scripts\activate   # or source venv/bin/activate
   pip install -e .
   ```
2. Ensure `config.yaml` matches your databases and desired hypothesis settings.
3. Run the workflow:
   ```bash
   python agents/workflow.py
   ```
4. Follow the interactive prompts—provide a draft hypothesis, answer clarification questions, and review every candidate snapshot presented in the final summary.
5. To compare the agentic workflow against a baseline, run:
   ```bash
   python metrics.py --model <baseline|llm|agent>
   ```
6. RAG setup (if enabled in config):
   - Ensure Qdrant is running on the specified endpoint.
   - Install `train.parquet` from "https://huggingface.co/datasets/nick007x/arxiv-papers"
   and rename it to `data/arxiv_papers.parquet`.
   - Ingest data into Qdrant:
     ```bash
     python rag.py
     ```
   - Warning: ingestion may take a long time without GPU acceleration.
   - Warning: the script will delete any existing `documents` collection in Qdrant.
7. Configurations can be adjusted in `config.yaml` as needed.