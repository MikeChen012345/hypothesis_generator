### Documentation for memory: https://docs.langchain.com/oss/python/langgraph/add-memory#example-using-postgres-store
import os
import json
import logging
import yaml
import time
from typing import Any, Tuple, Dict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.postgres import PostgresStore 
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode
from inference_auth_token import get_access_token

import tools
import memory_manager
import logging_config # apply logging configuration when importing

logger = logging.getLogger(__name__)
agent_memory_config = logging_config.get_agent_memory_config()
hypothesis_config = logging_config.get_hypothesis_config()
model_config = logging_config.get_model_config()
graph_config = logging_config.get_graph_config()
chat_logger = logging_config.get_chat_logger()

### Custom Memory
class CustomMemory:
    """
    A memory to store variables needed during the workflow
    """
    def __init__(self):
        self.memories = {}
        logger.debug("Initialized CustomMemory")

    def get_key(self, key: str) -> Any:
        """
        Get a value by key from the memory.

        Input:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key, or None if not found.
        """
        logger.debug(f"CustomMemory {self}: Retrieving key: {key}")
        return self.memories.get(key, [])

    def set_key(self, key: str, value: Any) -> None:
        """
        Set a key-value pair in the memory.

        Input:
            key (str): The key to set.
            value (Any): The value to associate with the key.
        """
        logger.debug(f"CustomMemory {self}: Setting key: {key} to value: {value}")
        self.memories[key] = value
    
    def __repr__(self):
        return f"CustomMemory(id={id(self)}, memories={self.memories})"

    def save_json_to_memory(self, json_str: str) -> None:
        """
        Save the extracted JSON data to custom memory.

        Input:
            json_str (str): The JSON string to save.
        """
        logger.info(f"CustomMemory {self}: Saving JSON string {json_str}")
        self.set_key("json", json_str)
    
    def load_json_from_memory(self) -> str:
        """
        Load the JSON data from custom memory.

        Returns:
            str: The JSON string from custom memory.
        """
        logger.info(f"CustomMemory {self}: Loading JSON string")
        return self.get_key("json")


### Define helper functions
def extract_json_from_response(response: str) -> Tuple[str, dict]:
    """
    Extract JSON object from the model's response string.
    Assumes the JSON is enclosed within <output> and </output> tags.
    If extraction fails, returns an empty dict, along with an error message in the string.

    Input:
        response (str): The full response string from the model.
    Returns:
        Tuple[str, dict]: A tuple containing the extracted JSON string and the parsed JSON object.
            If extraction fails, returns an empty dict, along with an error message in the string.
    """
    logger.debug("Extracting JSON from response")
    start_tag = "<output>"
    end_tag = "</output>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag, start_index)
    if start_index == -1 or end_index == -1:
        logger.warning("No <output> </output> tags found in the response. Please make sure to enclose the JSON within <output> and </output> tags.")
        return ("No <output> </output> tags found in the response. Please make sure to enclose the JSON within <output> and </output> tags.", {})
    json_str = response[start_index + len(start_tag):end_index].strip()
    try: # ensure valid json
        json_obj = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("Failed to decode JSON from response:", json_str)
        return ("Failed to decode JSON from response. Please make sure the JSON is valid.", {})
    return json_str, json_obj


class AgentWorkflow:
    def __init__(self, thread_id: str, user_id: str, memory: CustomMemory|None=None, 
                 model: BaseChatModel|None=None):
        self.config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            },
            "recursion_limit": 50,
        }
        self.memory = memory or CustomMemory()
        self.hypothesis_config = hypothesis_config or {}
        self.model_config = model_config or {}
        self.graph_config = graph_config or {}
        self.config["recursion_limit"] = max(1, int(self.graph_config.get("max_iterations", 50)))
        self.candidate_target = max(1, int(self.hypothesis_config.get("candidate_count", 3)))
        self.show_all_candidates = bool(self.hypothesis_config.get("show_all_candidates", True))
        self.archive_candidates = bool(self.hypothesis_config.get("archive_candidates", True))
        model_name = self.model_config.get("name", "gpt-4o-mini")
        temperature = float(self.model_config.get("temperature", 0.5))
        timeout = int(self.model_config.get("timeout", 30))
        max_retries = int(self.model_config.get("max_retries", 3))
        max_tokens = int(self.model_config.get("max_tokens", 4096))
        base_url = self.model_config.get("base_url") or os.getenv("OPENAI_API_ENDPOINT")
        api_key = self.model_config.get("api_key") or os.getenv("OPENAI_API_KEY") or get_access_token()
        self.model = model or init_chat_model(
                                model=model_name,
                                model_provider="openai",
                                temperature=temperature,
                                timeout=timeout,
                                max_retries=max_retries,
                                max_tokens=max_tokens,
                                base_url=base_url,
                                api_key=api_key,
                            )
        self.model = self.model.bind_tools(tools.get_all_tools())
        self.config["hypothesis"] = {
            "candidate_count": self.candidate_target,
            "show_all_candidates": self.show_all_candidates,
        }
        self.config["model"] = {
            "name": model_name,
            "temperature": temperature,
            "timeout": timeout,
            "max_retries": max_retries,
            "max_tokens": max_tokens,
        }
        self.phase_loop_counts: Dict[str, int] = {}
        self.max_phase_rounds = max(1, int(self.graph_config.get("max_phase_rounds", 3)))
        self.phase_force_advance: Dict[str, bool] = {}
        self.max_tool_call_chain = max(1, int(self.graph_config.get("max_consecutive_tool_calls", 3)))
        self.tool_call_streak = 0
        self.total_token_usage = 0
        
        # Tool Calling Node
        self.tool_calling = ToolNode(tools.get_all_tools())

    ### Helper for metrics recording
    def _record_usage_from_response(self, response: BaseMessage | None, *, count: bool = True) -> None:
        """Accumulate provider-reported token usage from a model response."""
        if not count or response is None:
            return
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens")
        if input_tokens is None:
            input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = usage.get("completion_tokens")

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
            self.total_token_usage += total

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            try:
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return str(content)
        if isinstance(content, list):
            return " ".join(self._stringify_content(item) for item in content)
        return str(content)

    def _recent_tool_outputs(self, messages: list[BaseMessage], limit: int = 3) -> list[str]:
        recent: list[str] = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                text = self._stringify_content(getattr(msg, "content", "")).strip()
                if text:
                    recent.append(text)
                continue
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                continue
            break
        if not recent:
            return []
        recent.reverse()
        return recent[-limit:]

    def _trim_conversation(
        self,
        messages: list[BaseMessage],
        *,
        max_tokens: int,
        start_on: str | tuple[str, ...] | None,
        end_on: str | tuple[str, ...] | None,
    ) -> list[BaseMessage]:
        """Trim conversation history without dropping trailing tool outputs."""

        def _run_trim(start: str | tuple[str, ...] | None) -> list[BaseMessage]:
            trimmed = trim_messages(
                messages,
                strategy="last",
                token_counter=count_tokens_approximately,
                max_tokens=max_tokens,
                start_on=start,
                end_on=end_on,
            )
            return list(trimmed) if trimmed else []

        primary = _run_trim(start_on)
        if primary:
            return primary

        fallback = _run_trim(None)
        if fallback:
            return fallback

        return list(messages[-10:]) if messages else []

    ### Helpers for phase handling
    def _previous_phase_data(self) -> str:
        data = self.memory.load_json_from_memory()
        return data if isinstance(data, str) else ""

    def _reset_phase_rounds(self, phase: str) -> None:
        self.phase_loop_counts[phase] = 0
        self.phase_force_advance[phase] = False

    def _increment_phase_rounds(self, phase: str) -> int:
        current = self.phase_loop_counts.get(phase, 0) + 1
        self.phase_loop_counts[phase] = current
        return current

    def _allow_force_advance(self, phase: str) -> None:
        self.phase_force_advance[phase] = True

    def _is_force_advance(self, phase: str) -> bool:
        return self.phase_force_advance.get(phase, False)

    def _validate_phase_payload(self, payload: dict, phase: str) -> str | None:
        if phase in {"clarification_phase", "generation_phase", "exploration_phase"}:
            missing_fields = payload.get("missing_fields")
            if not isinstance(missing_fields, list):
                return "Error: Include a 'missing_fields' array (use an empty list when nothing is missing)."

        if phase in {"generation_phase", "verification_phase", "exploration_phase"}:
            candidates = payload.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                return "Error: Provide a non-empty 'candidates' array that lists each hypothesis candidate."
            missing = [str(idx) for idx, candidate in enumerate(candidates, start=1)
                       if not candidate.get("id") or not candidate.get("statement")]
            if missing:
                return f"Error: Candidate(s) {', '.join(missing)} need both 'id' and 'statement' fields."
        return None

    def _handle_phase_transition(self, state: MessagesState, content: str, phase: str) -> Command:
        json_str, json_obj = extract_json_from_response(content)
        if json_obj != {}:
            validation_error = self._validate_phase_payload(json_obj, phase)
            if validation_error:
                state["messages"].append(SystemMessage(content=validation_error))
                return Command(goto="current_phase")
            force_advance = self._is_force_advance(phase)
            self.memory.save_json_to_memory(json_str)
            self._reset_phase_rounds(phase)
            missing_fields_raw = json_obj.get("missing_fields")
            if isinstance(missing_fields_raw, list):
                missing_fields = [str(item).strip() for item in missing_fields_raw if str(item).strip()]
            else:
                missing_fields = []

            if missing_fields and not force_advance:
                state["messages"].append(
                    SystemMessage(
                        content=(
                            "The current JSON output still has missing_fields entries: "
                            + ", ".join(missing_fields)
                            + ". Ask the user for that data and append it before finalizing."
                        )
                    )
                )
                return Command(goto="user_input")

            if force_advance and missing_fields:
                logger.info("Forcing advancement from %s with outstanding fields: %s", phase, ", ".join(missing_fields))
            return Command(goto="next_phase")

        state["messages"].append(
            SystemMessage(
                content=f"""Error: Unable to extract valid JSON from your response. Please ensure your output adheres strictly to the required format.
                Specifically, {json_str}"""
            )
        )
        return Command(goto="current_phase")

    # Define nodes
    # General model call node with memory integration
    def call_model(self, state: MessagesState, config: RunnableConfig, *, 
                store: BaseStore, system_prompt: str = "") -> Dict[str, AIMessage]:
        query = str(state["messages"][-1].content)
        memory_keyword = memory_manager.retrieve_user_memory(store, agent_memory_config, config, query)  
        # memory_vector = 
        tool_notes = self._recent_tool_outputs(state["messages"])
        tool_context = ""
        if tool_notes:
            tool_context = (
                "\n\nRecent tool findings (treat these as high-confidence evidence; cite them verbatim when relevant):\n"
                + "\n---\n".join(tool_notes)
            )
        system_prompt = system_prompt + memory_keyword + tool_context # + memory_vector

        messages = self._trim_conversation(
            state["messages"],
            max_tokens=4096,
            start_on=("human", "tool", "system"),
            end_on=("human", "tool", "system"),
        )

        prompt_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)] + messages

        response = self.model.invoke(prompt_messages)
        logger.debug("Assistant response: %s", response.content)
        chat_logger.info("ASSISTANT: %s", response.content)
        self._record_usage_from_response(response)
        
        return {"messages": response}

    # User Input Node
    def user_input(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, HumanMessage]:
        user_message = input("\n")
        chat_logger.info("USER: %s", user_message)
        return {"messages": [HumanMessage(content=user_message)]}

    def simulate_user_input(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, HumanMessage]:
        """
        Use LLM to simulate user input.
        """
        system_prompt = """
        You are simulating a user interacting with a scientific hypothesis refinement assistant. 
        Based on the assistant's last message, generate a realistic user response that advances the workflow.
        Guidelines:
        - If the assistant is asking for clarifications, provide additional scientific details or context.
        - If the assistant is presenting candidate hypotheses, give feedback on their quality and suggest improvements.
        - If the assistant is summarizing findings, acknowledge the summary and propose next steps.
        Ensure the response is coherent and contextually relevant to the assistant's last message.
        You do not need to be verbose; keep responses concise, as a typical human user would.
        
        Keep in mind that you are simulating a human user, so your responses should reflect natural human behavior.
        Sometimes you can just agree with the assistant to move forward.
        Sometimes you can be a bit ambiguous and as if you are confused to simulate real user behavior.
        DO NOT output any JSON or structured data; just provide natural language responses.
        
        DO NOT write responses that are too long.
        If you have interacted for multiple rounds, you should start wrapping up the conversation
        and indicate that you are satisfied with the current results and want to proceed to the next phase.

        **DO NOT reference any literature, prior study, or research. Instead, ask the assistant for potential references.**
        """
        messages = self._trim_conversation(
            state["messages"],
            max_tokens=1024,
            start_on=("ai", "human", "tool"),
            end_on=("ai", "tool"),
        )
        response = self.model.invoke(
            [SystemMessage(content=system_prompt)] + messages
        )
        user_message = response.content
        self._record_usage_from_response(response, count=False)
        chat_logger.info("SIMULATED_USER: %s", user_message)
        return {"messages": [HumanMessage(content=user_message)]}

    # Phase 1: Clarification Phase
    def clarification_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """
        Gather the seed hypothesis details and any missing scientific context.
        """
        logger.info("Clarification Phase")
        self.memory.set_key("phase", "clarification_phase")
        self._reset_phase_rounds("clarification_phase")
        system_prompt = f"""You are a scientific hypothesis intake mentor. Begin by acknowledging the researcher’s objective, then
        guide them to articulate a precise draft hypothesis. Confirm the phenomenon, target population, independent/dependent
        variables, measurable signals, and any prior evidence. Surface ambiguities (missing variables, vague outcomes, undefined
        population) and log them as open questions so later phases can resolve them. You may at most {self.max_phase_rounds}
        rounds of clarifying exchanges before providing your best-effort JSON output. If information is still missing, populate
        the "missing_fields" array and move forward.

        Workflow:
        1. Reflect the user’s draft in your own words to show understanding.
        2. Ask for any missing scientific details (variables, measurement units, timeframe, constraints).
        3. Capture background context such as prior experiments, datasets, or operational limitations.
        4. Maintain a "missing_fields" array that lists any information you still need. At the end of every response, emit the
            JSON snapshot inside <output> ... </output>. When "missing_fields" is empty, state that the form is complete; otherwise,
            present the partial JSON (still wrapped in <output>) and clearly ask the user to supply the remaining fields.

        JSON template:
        {{
            "research_question": "Concise description of the question being asked.",
            "initial_hypothesis": "The user-provided draft hypothesis (quoted).",
            "independent_variables": "Key drivers (comma-delimited or list).",
            "dependent_variables": "Outcomes or measurements tied to the hypothesis.",
            "context_notes": "Relevant population, environment, instrumentation, prior evidence.",
            "quality_flags": ["Ambiguity or strength indicators"],
            "open_questions": ["Specific clarifications still needed in later phases."],
            "missing_fields": ["List items still outstanding (empty list when complete)"]
        }}

        Only collect information in this phase—do not generate alternative hypotheses yet.
        """
        # ensure the output adheres strictly to this JSON structure. Then save the JSON to custom memory
        return self.call_model(state, config, store=store, system_prompt=system_prompt)

    # Phase 2: Generation Phase
    def generation_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """
        Expand the seed hypothesis into multiple candidate hypotheses with clear structure.
        """
        logger.info("Generation Phase")
        self.memory.set_key("phase", "generation_phase")
        self._reset_phase_rounds("generation_phase")
        system_prompt = f"""You are a hypothesis architect. Using the prior JSON context, collaborate with the user to transform the
        single draft hypothesis into {self.candidate_target} improved candidates. Encourage iteration—ask the user how bold, conservative,
        or mechanistic each candidate should be before finalizing.

        Requirements:
        - Produce at least {self.candidate_target} total candidates with IDs H1, H2, ...
        - Each candidate must include a crisp statement, explicit independent/dependent variables, assumptions, rationale, and any grammar
          fixes needed.
        - DO NOT invent new hypotheses beyond what the user suggests; focus on refining and clarifying the original idea.
        - If the user requests changes, incorporate them into existing candidates rather than creating new ones.
        - DO NOT create candidates that are minor rewordings; each must have distinct scientific angles.
        - DO NOT cite external literature or data; that is for the verification phase.
        - DO NOT fabricate any scientific context; only use evidence from the research tools.
        - Reference research tools (web search, Semantic Scholar, ArXiv) when the user needs supporting literature.
        - Capture comparison axes (e.g., novelty, feasibility, data availability) so the next phase can analyze gaps.
        - Show all candidate statements inside the JSON so the user can review them at once.
        - Track "missing_fields" whenever information is unavailable. At the end of every response, emit the current JSON snapshot inside
            <output> ... </output> so the user can see progress. Once the list is empty, announce that generation is complete. Avoid more than
            {self.max_phase_rounds} clarification turns; if the user remains uncertain, still output your best effort with the outstanding
            items listed in "missing_fields".

        JSON template:
        {{
            "research_question": "carried from earlier phase",
            "initial_hypothesis": "carried from earlier phase",
            "candidates": [
                {{
                    "id": "H1",
                    "statement": "Improved hypothesis sentence.",
                    "independent_variables": ["variable list"],
                    "dependent_variables": ["metric list"],
                    "assumptions": ["Key assumptions"],
                    "rationale": "Why this candidate could hold true.",
                    "logistical_considerations": "Data collection or experimental overhead.",
                    "language_feedback": "Grammar/clarity tweaks applied."
                }}
            ],
            "comparison_axes": ["novelty", "evidence availability", "operational complexity"],
            "notes_for_next_phase": "What uncertainties still need investigation.",
            "missing_fields": ["What information is still needed before moving on (empty when done)"]
        }}

        Always wait for the user’s acknowledgement before emitting <output>...</output>."""
        previous_data = self._previous_phase_data()
        if previous_data:
            system_prompt = system_prompt + "\nPrevious phase data:\n" + previous_data
        return self.call_model(state, config, store=store, system_prompt=system_prompt)

    # Phase 3: Verification Phase
    def verification_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """Use an LLM critique agent to evaluate the verifiability of every candidate."""
        logger.info("Verification Phase")
        self.memory.set_key("phase", "verification_phase")
        self._reset_phase_rounds("verification_phase")

        previous_data = self._previous_phase_data()
        if not previous_data:
            warning_msg = (
                "Verifiability check could not run because no candidate data was found. "
                "Please return to the generation phase and recreate the candidates."
            )
            logger.warning(warning_msg)
            return {"messages": AIMessage(content=warning_msg)}

        try:
            payload = json.loads(previous_data)
        except json.JSONDecodeError:
            logger.exception("Failed to parse JSON from previous phase during verifiability check")
            return {
                "messages": AIMessage(
                    content=(
                        "Verifiability check failed because the prior phase JSON is invalid. "
                        "Please regenerate the candidate hypotheses."
                    )
                )
            }

        candidates = payload.get("candidates") or []
        if not candidates:
            warning_msg = (
                "Verifiability check could not run because the candidate list is empty. "
                "Please regenerate hypotheses before continuing."
            )
            logger.warning(warning_msg)
            return {"messages": AIMessage(content=warning_msg)}

        system_prompt = """You are a rigorous hypothesis critique agent. Assess each candidate for clarity, causal logic,
        measurability, comparison structure, and feasibility. Provide constructive recommendations and call out
        missing variables or confounders before the exploration phase. Always speak as an internal reviewer, not as
        the original researcher.
        You may use tools to look up relevant literature or data that can help you evaluate the hypotheses.
        Appends a field called "references" to each candidate that lists any literature or data you cite from the research tools
        following the format: "title|authors|year". DO NOT remove existing entries, paraphrase or summarize the references; just list them as-is.

        If the tool request fails, proceed with your own reasoning without tool data.
        DO NOT make tool calls if the previous tool call failed for multiple times.

        You are given the JSON emitted from the generation_phase of a hypothesis refinement workflow. For each
        candidate hypothesis, evaluate verifiability and output a refreshed JSON payload that keeps the original
        fields but appends:
        - "verifiability" (0-5 integer)
        - "strengths" (list of bullet strings)
        - "gaps" (list of specific weaknesses or missing evidence)
        - "recommendations" (list of actionable edits or data-collection suggestions)

        Rules:
        - Clearly analyze each citation given in the json and ensure that it is supported by the literature.
        - If no literature is found to support a citation, consider it as a hallucination and search for alternative literature using the research tools.
        - Preserve the research_question, initial_hypothesis, and existing candidate statements.
        - Do not invent new candidates; only annotate the existing ones.
        - DO NOT fabricate any scientific context; only use evidence and papers from the research tools.
        - Clearly cite any literature or data you reference from the research tools.
        - If information is missing, note it explicitly in "gaps" and "recommendations".
        - Return the final response as JSON enclosed inside <output> tags so downstream phases can parse it.
        """
        previous_data = json.dumps(payload, indent=2)
        system_prompt = system_prompt + "\nGeneration phase JSON:\n" + previous_data
        return self.call_model(state, config, store=store, system_prompt=system_prompt)

    # Phase 4: Exploration Phase
    def exploration_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """
        Probe each candidate hypothesis for evidence gaps, risks, and refinement steps.
        """
        logger.info("Exploration Phase")
        self.memory.set_key("phase", "exploration_phase")
        self._reset_phase_rounds("exploration_phase")
        system_prompt = f"""You are an evidence gap analyst. For each candidate hypothesis in the provided JSON:
        - Work with the user to examine assumptions, missing data, confounders, and feasibility.
        - Suggest concrete data sources (instrumentation, datasets, observational protocols) and cite any supporting literature you find via tools.
        - Highlight grammar/logic issues that still remain.
        - Encourage the user to answer clarifying questions before locking the outputs.
        - Maintain a "missing_fields" array so we know what evidence or parameters still need clarification. Do not exceed {self.max_phase_rounds}
            clarification turns; if the user cannot provide more detail, output the best-available JSON with the outstanding fields listed.
        - At the end of every response, emit the current JSON snapshot inside <output> ... </output> so users can track progress.
        - DO NOT fabricate any scientific context; only use evidence from the research tools.
        - You may use tools to look up relevant literature or data that can help you refine the hypotheses.
        - If the tool request fails, proceed with your own reasoning without tool data. DO NOT make tool calls if the previous tool call failed for multiple times.
        - If not already exists, appends a field called "references" to each candidate that lists any literature or data you cite from the research tools
        following the format: "title|authors|year". DO NOT remove existing entries, paraphrase or summarize the references; just list them as-is.
        JSON template:
        {{
            "research_question": "unchanged",
            "candidates": [
                {{
                    "id": "H1",
                    "statement": "...",
                    "evidence_gaps": ["Unverified assumption 1", "Missing control 2"],
                    "supporting_signals": ["Key prior findings or citations"],
                    "data_collection_plan": "How to gather evidence (sample size, method, tooling).",
                    "risk_notes": ["Ethical, logistical, or statistical risks"],
                    "refinement_actions": ["Rewrite outcome clause", "Clarify population"],
                    "verifiability_feedback": "Result of hypothesis_verifiability_check or manual reasoning."
                }}
            ],
            "shared_insights": "Themes that apply to multiple candidates.",
            "prioritized_questions": ["Outstanding questions to resolve before validation."],
            "missing_fields": ["Evidence or clarifications still needed (empty when complete)"]
        }}

        Only emit <output>...</output> when every candidate has been discussed with the user.
        """
        previous_data = self._previous_phase_data()
        if previous_data:
            system_prompt = system_prompt + "\nPrevious phase data:\n" + previous_data
        return self.call_model(state, config, store=store, system_prompt=system_prompt)

    # Phase 5: Synthesis Phase
    def synthesis_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """Produce the single best hypothesis for downstream metric evaluation."""
        logger.info("Synthesis Phase")
        self.memory.set_key("phase", "synthesis_phase")
        self._reset_phase_rounds("synthesis_phase")
        system_prompt = """You are a synthesis expert whose only job is to emit the single best hypothesis candidate for automated metric scoring.
        Use the prior JSON to evaluate every candidate, select the strongest one, 
        and output ONLY that hypothesis wrapped in <output> tags without additional information.

        Requirements:
        - Consider verifiability_feedback, evidence gaps, and user preferences captured in prior phases.
        - Choose the top candidate ID (e.g., H2) plus its final statement.
        - Do not restate other candidates, summaries, or next steps.
        - Also, include a brief rationale explaining why this candidate was chosen,
        and any references to supporting literature or data from prior phases ("title|authors|year").
        DO NOT fabricate any references; only include those cited in the JSON from prior phases.
        - Final format must be in JSON as shown below, enclosed within <output> ... </output> tags:
        <output>
        {
            "Hypothesis": "<best hypothesis sentence>",
            "Rationale": "<brief explanation>",
            "References": <list of citations in the format "title|authors|year">
        }
        </output>

        Stay grounded in the provided conversation only. Do not reference tools or experiment memory.
        """
        previous_data = self._previous_phase_data()
        if previous_data:
            system_prompt = system_prompt + "\nPrevious phase data:\n" + previous_data
        return self.call_model(state, config, store=store, system_prompt=system_prompt)

    # Phase 6: End Phase
    def end_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """
        Either all phases are complete, or the user has chosen to end the workflow.
        """
        logger.info("End Phase")
        self.memory.set_key("phase", "end_phase")
        self._reset_phase_rounds("end_phase")
        return {"messages": AIMessage(content="Thank you for using the assistant. If you need further help, feel free to start a new session.")}

    # Phase Control Nodes
    def current_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Command:
        """
        Execute the function corresponding to the current phase stored in custom memory.
        """
        logger.debug("Executing current phase function")
        phase = self.memory.get_key("phase")
        return Command(goto=phase)

    def next_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Command:
        """
        Proceed to the next phase based on the current phase stored in custom memory.
        """
        # Map phases to functions
        next_phase_map = {
            "start": "clarification_phase",
            "clarification_phase": "generation_phase",
            "generation_phase": "verification_phase",
            "verification_phase": "exploration_phase",
            "exploration_phase": "synthesis_phase",
            "synthesis_phase": "end_phase",
        }
        logger.debug("Executing next phase function")
        phase = self.memory.get_key("phase")
        next_phase_name = next_phase_map.get(phase)
        logger.info(f"Transitioning from phase '{phase}' to '{next_phase_name}'")
        
        # clear memory if agent memory is independent
        if agent_memory_config.get("message_history", {}).get("phase_independent", False):
            state["messages"] = []
            logger.info("Cleared state messages due to independent agent memory configuration")
        
        if next_phase_name:
            return Command(goto=next_phase_name)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def router(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Command:
        """
        Decide the next step based on the model's latest response and current phase.
        """
        logger.info("Executing router function")
        last_message = state["messages"][-1]
        raw_content = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
        content_lower = raw_content.lower()
        phase = self.memory.get_key("phase")

        # Handle tool calling signals
        if last_message.tool_calls:
            self.tool_call_streak += 1
            logger.info("Router detected tool call signal in the response (streak=%s)", self.tool_call_streak)
            if self.tool_call_streak > self.max_tool_call_chain:
                warning_msg = (
                    f"Tool call limit of {self.max_tool_call_chain} consecutive calls reached. "
                    "You are NOT allowed to make further tool calls in this phase. "
                    "Continue reasoning with the available information and summarizing your findings."
                )
                state["messages"].append(SystemMessage(content=warning_msg))
                logger.warning("Exceeded consecutive tool call limit; redirecting to current phase")
                return Command(goto="current_phase")
            return Command(goto="tool_calling")
        elif self.tool_call_streak:
            logger.debug("Resetting tool call streak from %s to 0", self.tool_call_streak)
            self.tool_call_streak = 0
        
        # Phase-specific routing logic
        if phase in ["clarification_phase", "generation_phase", "verification_phase", "exploration_phase"]:
            if "<output>" in content_lower: # Signal to move to next phase
                return self._handle_phase_transition(state, raw_content, phase)
            # elif "exit" in content or "quit" in content: # User wants to end the workflow
            #     return Command(goto="end_phase")
            # remove for potential misjudgment
            else: # Stay in current phase and get user input
                current_loops = self._increment_phase_rounds(phase)
                if current_loops >= self.max_phase_rounds:
                    state["messages"].append(
                        SystemMessage(
                            content=(
                                f"Maximum clarification rounds ({self.max_phase_rounds}) reached in {phase}. "
                                "Summarize current knowledge, list remaining items under missing_fields, and emit <output> now."
                            )
                        )
                    )
                    logger.info(f"Max rounds reached in {phase}, forcing advancement")
                    self._allow_force_advance(phase)
                    return Command(goto=phase)
                return Command(goto="user_input")
        elif phase == "synthesis_phase":
            if "<output>" in content_lower: # Signal to move to end phase
                self.memory.set_key("final_hypothesis", raw_content)
                return Command(goto="next_phase")
            else: # Error in synthesis output, stay in current phase
                state["messages"].append(
                    SystemMessage(
                        content=f"""Error: Unable to extract the output from the response. Please ensure your output adheres strictly to the required format.
                        Specifically, make sure to enclose your final summary and action plan within <output> and </output> tags.""")
                )
                logger.warning("Synthesis phase output error: missing <output> tags")
                return Command(goto="current_phase")
        else:
            raise ValueError(f"Unknown phase in router: {self.memory.get_key('phase')}")

    def run_workflow(self, initial_hypothesis: str = "", show_message: bool = True) -> tuple[str, str, list[str], int, float]:
        """Run the workflow and return the final refined hypothesis.

        Args:
            initial_hypothesis: Optional starting hypothesis text. When empty (default),
                the workflow will prompt the user for input. When provided, the workflow
                uses this value without prompting.
            show_output: Whether to print the messages to the console.

        Returns:
            - The final hypothesis emitted by the synthesis phase (text inside <output> tags),
                or an empty string if the workflow could not complete.
            - The rationale provided for the final hypothesis.
            - A list of references cited in the final hypothesis.
            - The total token usage during the workflow run.
            - The total time taken for the workflow run.
        """
        now = time.time()
        logger.info("Starting workflow run")
        self.memory.set_key("final_hypothesis", "")
        self.total_token_usage = 0

        # Initialize the graph
        logger.info("Initializing StateGraph")
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", self.call_model)

        if self.graph_config.get("use_simulated_user", False):
            builder.add_node("user_input", self.simulate_user_input)
        else:
            builder.add_node("user_input", self.user_input)

        builder.add_node("tool_calling", self.tool_calling)
        builder.add_node("clarification_phase", self.clarification_phase)
        builder.add_node("generation_phase", self.generation_phase)
        builder.add_node("verification_phase", self.verification_phase)
        builder.add_node("exploration_phase", self.exploration_phase)
        builder.add_node("synthesis_phase", self.synthesis_phase)
        builder.add_node("end_phase", self.end_phase)
        builder.add_node("current_phase", self.current_phase)
        builder.add_node("next_phase", self.next_phase)
        builder.add_node("router", self.router)

        builder.add_edge(START, "clarification_phase")
        builder.add_edge("user_input", "current_phase")
        builder.add_edge("tool_calling", "current_phase")
        builder.add_edge("clarification_phase", "router")
        builder.add_edge("generation_phase", "router")
        builder.add_edge("verification_phase", "router")
        builder.add_edge("exploration_phase", "router")
        builder.add_edge("synthesis_phase", "router")
        builder.add_edge("end_phase", END)

        self.memory.set_key("phase", "START")
        self._reset_phase_rounds("START")

        # Ensure we have a starting hypothesis message
        starting_message = (initial_hypothesis or "").strip()
        if not starting_message:
            starting_message = input("Hi! Please enter your initial hypothesis draft:\n").strip()
        if not starting_message:
            logger.error("No hypothesis provided to the workflow. Aborting run.")
            return "", "", [], 0, 0.0

        # Decide whether to open DB connections and compile the graph with them.
        def _run_stream(graph_obj, start_text: str) -> None:
            logger.info("Starting interactive workflow")
            for chunk in graph_obj.stream(
                {"messages": [{"role": "user", "content": start_text}]},
                self.config,
                stream_mode="values",
            ):
                if show_message and "messages" in chunk:
                    chunk["messages"][-1].pretty_print()
        
        # Reset the shared tool configuration before deciding which memory store to use.
        tools.configure_memory_tool(
            store=None,
            agent_memory_config=agent_memory_config,
            config=self.config,
        )

        use_long_term = agent_memory_config.get("database", {}).get("long_term", {}).get("use", False)
        use_short_term = agent_memory_config.get("database", {}).get("short_term", {}).get("use", False)
        if use_long_term or use_short_term:
            DB_URI = os.getenv("POSTGRESQL_CONNECTION_STRING")
            if DB_URI is None:
                logger.error("POSTGRESQL_CONNECTION_STRING is not set in .env file")
                return "", "", [], 0, 0.0

            # Both long_term and short_term enabled: open both resources together
            if use_long_term and use_short_term:
                logger.info("Long-term and short-term memory enabled: initializing PostgresSaver and PostgresStore")
                with PostgresSaver.from_conn_string(DB_URI) as checkpointer, PostgresStore.from_conn_string(DB_URI) as store:
                    # Create necessary tables if they don't exist. No side effects if they do.
                    store.setup()
                    checkpointer.setup()
                    tools.configure_memory_tool(
                        store=store,
                        agent_memory_config=agent_memory_config,
                        config=self.config,
                    )
                    graph = builder.compile(store=store)
                    _run_stream(graph, starting_message)

            # Only short_term enabled
            elif use_short_term:
                logger.info("Short-term memory enabled: initializing PostgresSaver")
                with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
                    checkpointer.setup()
                    graph = builder.compile(checkpointer=checkpointer)
                    _run_stream(graph, starting_message)

            # Only long_term enabled
            elif use_long_term:
                logger.info("Long-term memory enabled: initializing PostgresStore")
                with PostgresStore.from_conn_string(DB_URI) as store:
                    store.setup()
                    tools.configure_memory_tool(
                        store=store,
                        agent_memory_config=agent_memory_config,
                        config=self.config,
                    )
                    graph = builder.compile(store=store)
                    _run_stream(graph, starting_message)

        else:
            # No DB-backed memory enabled
            logger.info("No DB-backed memory enabled: compiling graph without store or checkpointer")
            graph = builder.compile()
            _run_stream(graph, starting_message)

        final_raw = self.memory.get_key("final_hypothesis")
        if isinstance(final_raw, str) and final_raw:
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

            return final_hypothesis, rationale, references, self.total_token_usage, time.time() - now
        
        logger.error("Workflow completed without producing a final hypothesis.")
        return "", "", [], 0, 0.0


if __name__ == "__main__":
    workflow = AgentWorkflow(thread_id="thread_1", user_id="user_1")
    initial_hypothesis = "The increase in urban green spaces leads to a measurable decrease in local air pollution levels."
    final_hypothesis, rationale, citations, token_usage, elapsed_time = \
        workflow.run_workflow(initial_hypothesis=initial_hypothesis, show_message=True)
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