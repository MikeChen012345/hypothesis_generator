# agent-helper

Scientific hypothesis co-pilot built with LangGraph + LangChain. The agent now focuses on helping researchers start with a single draft hypothesis, fan out multiple improved candidates, interrogate evidence gaps, and surface ranked recommendations.

## Workflow Highlights
- **Phase 1 – Hypothesis Intake:** capture the researcher’s draft hypothesis, variables, measurement targets, and ambiguities.
- **Phase 2 – Candidate Expansion:** generate multiple hypotheses (H1…Hn) with clarified assumptions and grammar fixes.
- **Phase 3 – Evidence Gap Analysis:** walk through each candidate to log missing data, risks, suggested experiments, and groundedness feedback (leveraging research tools when needed).
- **Phase 4 – Synthesis:** summarize all candidates side-by-side, rank them, and outline next steps plus memory-archival notes.
- **Phase 5 – End:** close the session once the researcher has a clear plan.

## Configuration
`config.yaml` controls logging, memory, and research-specific behavior.

- Long-term (Postgres + Qdrant) storage can archive curated hypotheses for reuse.
- `Hypothesis` section introduces:
  - `candidate_count`: number of improved candidates to generate.
  - `archive_candidates`: toggle for persisting refined candidates to shared memory.
  - `show_all_candidates`: force downstream phases to present every candidate.

## Environment Variables
Create a `.env` file with at least:

```
OPENAI_API_ENDPOINT=<azure_or_openai_endpoint>
OPENAI_API_KEY=<token>
POSTGRESQL_CONNECTION_STRING=<postgres_uri>
QDRANT_URL=<qdrant_http_endpoint>
GOOGLE_SEARCH_API_KEY=<cse_key>
GOOGLE_SEARCH_ENGINE_ID=<cse_id>
SEMANTIC_SCHOLAR_API_KEY=<optional_graph_api_key>
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