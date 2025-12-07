### Documentation for memory: https://docs.langchain.com/oss/python/langgraph/add-memory#example-using-postgres-store
import os
import json
import logging
import yaml
from typing import Any, Tuple, Dict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

### Custom Memory
class CustomMemory:
    """
    A memory to store variables needed during the workflow
    """
    def __init__(self):
        self.memories = {}
        logging.debug("Initialized CustomMemory")

    def get_key(self, key: str) -> Any:
        """
        Get a value by key from the memory.

        Input:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key, or None if not found.
        """
        logging.debug(f"CustomMemory {self}: Retrieving key: {key}")
        return self.memories.get(key, [])

    def set_key(self, key: str, value: Any) -> None:
        """
        Set a key-value pair in the memory.

        Input:
            key (str): The key to set.
            value (Any): The value to associate with the key.
        """
        logging.debug(f"CustomMemory {self}: Setting key: {key} to value: {value}")
        self.memories[key] = value
    
    def __repr__(self):
        return f"CustomMemory(id={id(self)}, memories={self.memories})"

    def save_json_to_memory(self, json_str: str) -> None:
        """
        Save the extracted JSON data to custom memory.

        Input:
            json_str (str): The JSON string to save.
        """
        logging.info(f"CustomMemory {self}: Saving JSON string {json_str}")
        self.set_key("json", json_str)
    
    def load_json_from_memory(self) -> str:
        """
        Load the JSON data from custom memory.

        Returns:
            str: The JSON string from custom memory.
        """
        logging.info(f"CustomMemory {self}: Loading JSON string")
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
    logging.debug("Extracting JSON from response")
    start_tag = "<output>"
    end_tag = "</output>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag, start_index)
    if start_index == -1 or end_index == -1:
        logging.warning("No <output> </output> tags found in the response. Please make sure to enclose the JSON within <output> and </output> tags.")
        return ("No <output> </output> tags found in the response. Please make sure to enclose the JSON within <output> and </output> tags.", {})
    json_str = response[start_index + len(start_tag):end_index].strip()
    try: # ensure valid json
        json_obj = json.loads(json_str)
    except json.JSONDecodeError:
        logging.warning("Failed to decode JSON from response:", json_str)
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
        self.candidate_target = max(1, int(self.hypothesis_config.get("candidate_count", 3)))
        self.show_all_candidates = bool(self.hypothesis_config.get("show_all_candidates", True))
        self.archive_candidates = bool(self.hypothesis_config.get("archive_candidates", True))
        self.model = model or init_chat_model(
                                model="gpt-4o-mini",
                                temperature=0.5,
                                timeout=30,
                                max_retries=3,
                                max_tokens=4096,
                                base_url=os.getenv("OPENAI_API_ENDPOINT"),
                                openai_api_key=get_access_token(),
                            )
        self.model = self.model.bind_tools(tools.get_all_tools())
        self.config["hypothesis"] = {
            "candidate_count": self.candidate_target,
            "show_all_candidates": self.show_all_candidates,
        }
        
        # Tool Calling Node
        self.tool_calling = ToolNode(tools.get_all_tools())

    ### Define nodes
    def _previous_phase_data(self) -> str:
        data = self.memory.load_json_from_memory()
        return data if isinstance(data, str) else ""

    def _validate_phase_payload(self, payload: dict, phase: str) -> str | None:
        if phase == "clarification_phase":
            if not payload.get("initial_hypothesis"):
                return "Error: Please include an 'initial_hypothesis' field before we continue."
        if phase in {"structuring_phase", "exploration_phase"}:
            candidates = payload.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                return "Error: Provide a non-empty 'candidates' array that lists each hypothesis candidate."
            if phase == "structuring_phase" and len(candidates) < self.candidate_target:
                return f"Error: Generate at least {self.candidate_target} distinct candidates as requested."
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
            self.memory.save_json_to_memory(json_str)
            return Command(goto="next_phase")

        state["messages"].append(
            SystemMessage(
                content=f"""Error: Unable to extract valid JSON from your response. Please ensure your output adheres strictly to the required format.
                Specifically, {json_str}"""
            )
        )
        return Command(goto="current_phase")

    # General model call node with memory integration
    def call_model(self, state: MessagesState, config: RunnableConfig, *, 
                store: BaseStore, system_prompt: str = "") -> Dict[str, AIMessage]:
        query = str(state["messages"][-1].content)
        memory_keyword = memory_manager.retrieve_user_memory(store, agent_memory_config, config, query)  
        # memory_vector = 
        system_prompt = system_prompt + memory_keyword # + memory_vector

        messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=1024,
            start_on="human",
            end_on=("human", "tool"),
        )
        response = self.model.invoke(
            [{"role": "system", "content": system_prompt}] + messages
        )
        logging.debug("Assistant response: " + response.content)
        
        return {"messages": response}

    # User Input Node
    def user_input(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, HumanMessage]:
        user_message = input("\n")
        logging.info(f"User input: {user_message}")
        return {"messages": [HumanMessage(content=user_message)]}


    # Phase 1: Clarification Phase
    def clarification_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """
        Gather the seed hypothesis details and any missing scientific context.
        """
        logging.info("Clarification Phase")
        self.memory.set_key("phase", "clarification_phase")
        system_prompt = """You are a scientific hypothesis intake mentor. Begin by acknowledging the researcher’s objective, then
        guide them to articulate a precise draft hypothesis. Confirm the phenomenon, target population, independent/dependent
        variables, measurable signals, and any prior evidence. Surface ambiguities (missing variables, vague outcomes, undefined
        population) and log them as open questions so later phases can resolve them.

        Workflow:
        1. Reflect the user’s draft in your own words to show understanding.
        2. Ask for any missing scientific details (variables, measurement units, timeframe, constraints).
        3. Capture background context such as prior experiments, datasets, or operational limitations.
        4. When confident, emit <output> ... </output> containing the JSON template below. Do not advance until the user agrees the
           captured hypothesis is accurate.

        JSON template:
        {
            "research_question": "Concise description of the question being asked.",
            "initial_hypothesis": "The user-provided draft hypothesis (quoted).",
            "independent_variables": "Key drivers (comma-delimited or list).",
            "dependent_variables": "Outcomes or measurements tied to the hypothesis.",
            "context_notes": "Relevant population, environment, instrumentation, prior evidence.",
            "quality_flags": ["Ambiguity or strength indicators"],
            "open_questions": ["Specific clarifications still needed in later phases."]
        }

        Only collect information in this phase—do not generate alternative hypotheses yet.
        """
        # ensure the output adheres strictly to this JSON structure. Then save the JSON to custom memory
        return self.call_model(state, config, store=store, system_prompt=system_prompt)

    # Phase 2: Structuring Phase
    def structuring_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """
        Expand the seed hypothesis into multiple candidate hypotheses with clear structure.
        """
        logging.info("Structuring Phase")
        self.memory.set_key("phase", "structuring_phase")
        system_prompt = f"""You are a hypothesis architect. Using the prior JSON context, collaborate with the user to transform the
        single draft hypothesis into {self.candidate_target} improved candidates. Encourage iteration—ask the user how bold, conservative,
        or mechanistic each candidate should be before finalizing.

        Requirements:
        - Produce at least {self.candidate_target} total candidates with IDs H1, H2, ...
        - Each candidate must include a crisp statement, explicit independent/dependent variables, assumptions, rationale, and any grammar
          fixes needed.
        - Reference research tools (web search, Semantic Scholar) when the user needs supporting literature.
        - Capture comparison axes (e.g., novelty, feasibility, data availability) so the next phase can analyze gaps.
        - Show all candidate statements inside the JSON so the user can review them at once.

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
            "notes_for_next_phase": "What uncertainties still need investigation."
        }}

        Always wait for the user’s acknowledgement before emitting <output>...</output>."""
        previous_data = self._previous_phase_data()
        if previous_data:
            system_prompt = system_prompt + "\nPrevious phase data:\n" + previous_data
        return self.call_model(state, config, store=store, system_prompt=system_prompt)

    # Phase 3: Exploration Phase
    def exploration_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """
        Probe each candidate hypothesis for evidence gaps, risks, and refinement steps.
        """
        logging.info("Exploration Phase")
        self.memory.set_key("phase", "exploration_phase")
        system_prompt = """You are an evidence gap analyst. For each candidate hypothesis in the provided JSON:
        - Work with the user to examine assumptions, missing data, confounders, and feasibility.
        - Suggest concrete data sources (instrumentation, datasets, observational protocols) and cite any supporting literature you find via tools.
        - Highlight grammar/logic issues that still remain.
        - Encourage the user to answer clarifying questions before locking the outputs.

        JSON template:
        {
            "research_question": "unchanged",
            "candidates": [
                {
                    "id": "H1",
                    "statement": "...",
                    "evidence_gaps": ["Unverified assumption 1", "Missing control 2"],
                    "supporting_signals": ["Key prior findings or citations"],
                    "data_collection_plan": "How to gather evidence (sample size, method, tooling).",
                    "risk_notes": ["Ethical, logistical, or statistical risks"],
                    "refinement_actions": ["Rewrite outcome clause", "Clarify population"],
                    "groundedness_feedback": "Result of hypothesis_groundedness_check or manual reasoning."
                }
            ],
            "shared_insights": "Themes that apply to multiple candidates.",
            "prioritized_questions": ["Outstanding questions to resolve before validation."]
        }

        Only emit <output>...</output> when every candidate has been discussed with the user.
        """
        previous_data = self._previous_phase_data()
        if previous_data:
            system_prompt = system_prompt + "\nPrevious phase data:\n" + previous_data
        return self.call_model(state, config, store=store, system_prompt=system_prompt)

    # Phase 4: Synthesis Phase
    def synthesis_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """
        Integrate all insights into an actionable research plan covering every candidate hypothesis.
        """
        logging.info("Synthesis Phase")
        self.memory.set_key("phase", "synthesis_phase")
        system_prompt = """You are a synthesis expert wrapping up the hypothesis refinement sprint. Use the JSON input to:
        - Summarize the overall research goal and what changed from the original hypothesis.
        - Present every candidate (H1, H2, ...) with a short verdict line including strengths, key risks, and recommended next experiments.
        - Provide a ranked recommendation order (e.g., Tier 1/2) explaining the prioritization logic.
        - Outline the immediate next steps (data collection, literature dives, stakeholder reviews) and how long-term memory should archive the work.

        Output format (plain text wrapped in <output> tags):
        <output>
        Executive Summary...
        Candidate Snapshots:
        - H1: ...
        - H2: ...
        Ranking & Rationale...
        Next Steps...
        Memory Notes...
        </output>

        Mention all candidates explicitly so the user can compare them side-by-side.
        """
        previous_data = self._previous_phase_data()
        if previous_data:
            system_prompt = system_prompt + "\nPrevious phase data:\n" + previous_data
        return self.call_model(state, config, store=store, system_prompt=system_prompt)

    # Phase 5: End Phase
    def end_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, AIMessage]:
        """
        Either all phases are complete, or the user has chosen to end the workflow.
        """
        logging.info("End Phase")
        self.memory.set_key("phase", "end_phase")
        return {"messages": AIMessage(content="Thank you for using the assistant. If you need further help, feel free to start a new session.")}

    # Phase Control Nodes
    def current_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Command:
        """
        Execute the function corresponding to the current phase stored in custom memory.
        """
        logging.debug("Executing current phase function")
        phase = self.memory.get_key("phase")
        return Command(goto=phase)

    def next_phase(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Command:
        """
        Proceed to the next phase based on the current phase stored in custom memory.
        """
        # Map phases to functions
        next_phase_map = {
            "start": "clarification_phase",
            "clarification_phase": "structuring_phase",
            "structuring_phase": "exploration_phase",
            "exploration_phase": "synthesis_phase",
            "synthesis_phase": "end_phase",
        }
        logging.debug("Executing next phase function")
        phase = self.memory.get_key("phase")
        next_phase_name = next_phase_map.get(phase)
        logging.info(f"Transitioning from phase '{phase}' to '{next_phase_name}'")
        
        # clear memory if agent memory is independent
        if agent_memory_config.get("message_history", {}).get("phase_independent", False):
            state["messages"] = []
            logging.info("Cleared state messages due to independent agent memory configuration")
        
        if next_phase_name:
            return Command(goto=next_phase_name)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def router(self, state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> Command:
        """
        Decide the next step based on the model's latest response and current phase.
        """
        logging.info("Executing router function")
        last_message = state["messages"][-1]
        content = last_message.content.lower()
        phase = self.memory.get_key("phase")

        # Handle tool calling signals
        if last_message.tool_calls:
            logging.info("Router detected tool call signal in the response")
            return Command(goto="tool_calling")
        
        # Phase-specific routing logic
        if phase in ["clarification_phase", "structuring_phase", "exploration_phase"]:
            if "<output>" in content: # Signal to move to next phase
                return self._handle_phase_transition(state, content, phase)
            # elif "exit" in content or "quit" in content: # User wants to end the workflow
            #     return Command(goto="end_phase")
            # remove for potential misjudgment
            else: # Stay in current phase and get user input
                return Command(goto="user_input")
        elif phase == "synthesis_phase":
            if "<output>" in content: # Signal to move to end phase
                return Command(goto="next_phase")
            else: # Error in synthesis output, stay in current phase
                state["messages"].append(
                    SystemMessage(
                        content=f"""Error: Unable to extract the output from the response. Please ensure your output adheres strictly to the required format.
                        Specifically, make sure to enclose your final summary and action plan within <output> and </output> tags.""")
                )
                logging.warning("Synthesis phase output error: missing <output> tags")
                return Command(goto="current_phase")
        else:
            raise ValueError(f"Unknown phase in router: {self.memory.get_key('phase')}")

    def run_workflow(self) -> None:
        logging.info("Starting workflow run")

        # Initialize the graph
        logging.info("Initializing StateGraph")
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", self.call_model)
        builder.add_node("user_input", self.user_input)
        builder.add_node("tool_calling", self.tool_calling)
        builder.add_node("clarification_phase", self.clarification_phase)
        builder.add_node("structuring_phase", self.structuring_phase)
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
        builder.add_edge("structuring_phase", "router")
        builder.add_edge("exploration_phase", "router")
        builder.add_edge("synthesis_phase", "router")
        builder.add_edge("end_phase", END)

        self.memory.set_key("phase", "START")

        # Decide whether to open DB connections and compile the graph with them.
        def _run_stream(graph_obj):
            logging.info("Starting interactive workflow")
            for chunk in graph_obj.stream(
                {"messages": [{"role": "user", "content": input("Hi! What can I help you with today?\n")}]},
                self.config,
                stream_mode="values",
            ):
                if "messages" in chunk:
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
                logging.error("POSTGRESQL_CONNECTION_STRING is not set in .env file")
                return

            # Both long_term and short_term enabled: open both resources together
            if use_long_term and use_short_term:
                logging.info("Long-term and short-term memory enabled: initializing PostgresSaver and PostgresStore")
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
                    _run_stream(graph)

            # Only short_term enabled
            elif use_short_term:
                logging.info("Short-term memory enabled: initializing PostgresSaver")
                with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
                    checkpointer.setup()
                    graph = builder.compile(checkpointer=checkpointer)
                    _run_stream(graph)

            # Only long_term enabled
            elif use_long_term:
                logging.info("Long-term memory enabled: initializing PostgresStore")
                with PostgresStore.from_conn_string(DB_URI) as store:
                    store.setup()
                    tools.configure_memory_tool(
                        store=store,
                        agent_memory_config=agent_memory_config,
                        config=self.config,
                    )
                    graph = builder.compile(store=store)
                    _run_stream(graph)

        else:
            # No DB-backed memory enabled
            logging.info("No DB-backed memory enabled: compiling graph without store or checkpointer")
            graph = builder.compile()
            _run_stream(graph)


if __name__ == "__main__":
    workflow = AgentWorkflow(thread_id="thread_1", user_id="user_1")
    workflow.run_workflow()