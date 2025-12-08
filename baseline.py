### Baseline LLM-only hypothesis refinement workflow (multi-round with simulated user support).

from __future__ import annotations

import logging
from typing import Dict, List
import time
import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import logging_config
from inference_auth_token import get_access_token

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are a scientific hypothesis improver. You will receive a single input that contains the
researcher’s draft hypothesis. Without requesting additional clarification turns, produce one improved hypothesis that
refines variables, clarifies measurement signals, names the target population, and states anticipated directionality.

Workflow expectations:
- Interpret the provided draft carefully; if details are missing, make conservative assumptions and state them explicitly.
- Highlight concrete, testable relationships (independent vs dependent variables, controls, population segment, timeframe).
- Address obvious weaknesses (e.g., vague metrics, lack of comparator) by tightening language rather than inventing data.
- Also, include a brief rationale explaining why this candidate was chosen,
  and any references to supporting literature or data ("title|authors|year"). DO NOT fabricate any references
- Final format must be in JSON as shown below, enclosed within <output> ... </output> tags:
<output>
{
	"Hypothesis": "<best hypothesis sentence>",
	"Rationale": "<brief explanation>",
	"References": <list of citations in the format "title|authors|year">
}

Stay grounded solely in the user’s submission; do not fabricate literature or refer to external tools.
"""


def _load_model_settings() -> Dict[str, object]:
	model_cfg = logging_config.get_model_config() or {}
	return {
		"model": model_cfg.get("name", "gpt-4o-mini"),
		"temperature": float(model_cfg.get("temperature", 0.4)),
		"timeout": int(model_cfg.get("timeout", 30)),
		"max_retries": int(model_cfg.get("max_retries", 2)),
		"max_tokens": int(model_cfg.get("max_tokens", 1024)),
		"base_url": model_cfg.get("base_url"),
		"api_key": model_cfg.get("api_key"),
	}


def _build_model():
	settings = _load_model_settings()
	api_key = settings["api_key"] or get_access_token()

	return init_chat_model(
		model=settings["model"],
		model_provider="openai",
		temperature=settings["temperature"],
		timeout=settings["timeout"],
		max_retries=settings["max_retries"],
		max_tokens=settings["max_tokens"],
		base_url=settings["base_url"],
		api_key=api_key,
	)


def _message_text(message: AIMessage | HumanMessage | SystemMessage) -> str:
	content = message.content
	return content if isinstance(content, str) else str(content)


def _extract_output_block(final_raw: str) -> tuple[str, str, list[str]]:
	start_tag, end_tag = "<output>", "</output>"
	start_idx = final_raw.find(start_tag)
	end_idx = final_raw.find(end_tag, start_idx)
	if start_idx != -1 and end_idx != -1:
		final_raw = final_raw[start_idx + len(start_tag):end_idx]

	final_json = json.loads(final_raw)
	final_hypothesis = final_json.get("Hypothesis", "").strip()
	rationale = final_json.get("Rationale", "").strip()
	references = final_json.get("References", [])
	if isinstance(references, str):
		references = [ref.strip() for ref in references.split(",") if ref.strip()]
	elif not isinstance(references, list):
		references = []
		
	return final_hypothesis, rationale, references


def _gather_initial_payload() -> str:
	hypothesis = input("Enter your draft hypothesis:\n").strip()
	return hypothesis


def baseline_workflow(initial_hypothesis: str="") -> tuple[str, str, list[str], int, float]:
	"""Run a single-turn hypothesis improvement using only an LLM.
	Args:
        initial_hypothesis: An optional initial hypothesis draft to start the workflow.
    Returns:
        - The final hypothesis emitted by the synthesis phase (text inside <output> tags),
			or an empty string if the workflow could not complete.
		- The rationale provided for the final hypothesis.
		- A list of references cited in the final hypothesis.
		- The total token usage during the workflow run.
		- The total time taken for the workflow run.
    """
	now = time.time()
	hypothesis = initial_hypothesis or _gather_initial_payload()
	if not hypothesis:
		print("No hypothesis provided. Please rerun and enter a draft hypothesis to refine.")
		return "", "", [], 0, 0.0

	system_prompt = SYSTEM_PROMPT_TEMPLATE
	model = _build_model()

	initial_payload = {
		"hypothesis": hypothesis,
	}

	token_usage = 0

	def _record_usage(response, include: bool = True) -> None:
		nonlocal token_usage
		if not include or response is None:
			return
		usage = getattr(response, "usage_metadata", None) or {}
		input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
		output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
		total = 0
		if input_tokens is not None:
			total += int(input_tokens)
		if output_tokens is not None:
			total += int(output_tokens)
		if not total:
			fallback = usage.get("total_tokens") or usage.get("token_count")
			if fallback is not None:
				total = int(fallback)
		if total:
			token_usage += total

	conversation: List[HumanMessage | AIMessage | SystemMessage] = [
		SystemMessage(content=system_prompt),
		HumanMessage(content=f"Here is the current hypothesis data:\n{initial_payload}"),
	]

	assistant_message = model.invoke(conversation)
	_record_usage(assistant_message)
	assistant_text = _message_text(assistant_message)

	if "<output>" not in assistant_text:
		assistant_text = (
			assistant_text.strip()
			+ "\n\nReminder: emit the single best hypothesis using <output> tags."
		)

	final_hypothesis, rationale, citations = _extract_output_block(assistant_text)
	return final_hypothesis, rationale, citations, token_usage, time.time() - now



if __name__ == "__main__":
	initial_hypothesis = "The increase in urban green spaces leads to a measurable decrease in local air pollution levels."
	final_hypothesis, rationale, citations, token_usage, elapsed_time = \
		baseline_workflow(initial_hypothesis=initial_hypothesis)
	if final_hypothesis:
		print("\nFinal refined hypothesis:\n")
		print(final_hypothesis)
		print("\nRationale:\n")
		print(rationale)
		print("\nCitations:\n")
		for citation in citations:
			print(f"- {citation}")
	print(f"\nApproximate tokens used (excluding simulated user): {token_usage}")
	print(f"Total time taken for the workflow run: {elapsed_time:.2f} seconds")