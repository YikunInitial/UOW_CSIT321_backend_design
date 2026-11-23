from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from uuid import uuid4

from Agents.core.base_agent import BaseAgent
from Agents.core.customer import Customer, CustomerRequest
from Agents.core.event import Event
from Agents.core.event_bus import EventBus


class SemanticAgentWrapper(BaseAgent):
    """
    SemanticAgentWrapper interprets raw customer requests and converts them
    into CustomerRequest objects for the downstream CustomerAgentWrapper.

    Main responsibility:
    - receive raw / vague customer request
    - directly call an LLM
    - let the LLM extract CustomerRequest-compatible fields
    - let the LLM judge whether the request is complete
    - use Python validation as the final safety gate
    - return CustomerRequest if valid
    - return missing information and clarification question if incomplete

    It does NOT:
    - call the downstream LLMAgent
    - perform batching
    - perform optimisation
    - generate loading plans
    - assign products to vehicles or positions
    - call visualisation
    """

    ALLOWED_OBJECTIVES = {
        "minimise_vehicle_number",
        "maximise_loading_rate",
        "ensure_loading_safety",
        "minimise_runtime",
    }

    REQUIRED_CONSTRAINTS = [
        "weight_limit",
        "axle_balance",
        "centre_line_balance",
        "diagonal_balance",
        "overlap_check",
        "boundary_check",
    ]

    def __init__(
        self,
        agent_id: str,
        customer: Customer,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="semantic",
            name="Semantic Agent Wrapper",
            config=config,
        )

        self.customer = customer
        self.event_bus = event_bus
        self.config = config or {}

        # Default to your stronger local model.
        self.llm_provider = self.config.get("llm_provider", "ollama")
        self.llm_model = self.config.get("llm_model", "qwen2.5:7b")
        self.llm_base_url = self.config.get("llm_base_url", "http://127.0.0.1:11434")
        self.llm_api_key = self.config.get("llm_api_key")

        self.temperature = self.config.get("temperature", 0.0)
        self.timeout = self.config.get("timeout", 120)

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def interpret_raw_request(
        self,
        raw_request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Interpret raw customer request through LLM.

        Return examples:

        Ready:
        {
            "status": "ready",
            "customer_request": CustomerRequest(...),
            ...
        }

        Need clarification:
        {
            "status": "need_clarification",
            "customer_request": None,
            "missing_information": [...],
            "clarification_question": "...",
            ...
        }

        Failed:
        {
            "status": "failed",
            "error": "...",
            "customer_request": None
        }
        """

        self.update_status("running")
        self.set_current_task("semantic_interpretation")

        try:
            if not raw_request or not raw_request.strip():
                raise ValueError("raw_request cannot be empty.")

            context = context or {}

            self.update_belief("customer_id", self.customer.customer_id)
            self.update_belief("raw_request", raw_request)

            self.add_desire("understand_raw_customer_request")
            self.add_intention("call_llm_to_extract_customer_request")

            self._publish_event_if_possible(
                topic="semantic.request.received",
                correlation_id=None,
                payload={
                    "customer_id": self.customer.customer_id,
                    "raw_request": raw_request,
                    "context": context,
                },
            )

            prompt = self._build_prompt(
                raw_request=raw_request,
                context=context,
            )

            llm_text = self._call_llm(prompt)
            llm_json = self._parse_llm_json(llm_text)

            result = self._normalise_llm_result(
                llm_result=llm_json,
                raw_request=raw_request,
                context=context,
            )

            validation = self._validate_llm_result(result)
            result["validation"] = validation

            self.update_belief("semantic_summary", result.get("summary"))
            self.update_belief(
                "is_ready_for_customer_agent",
                validation["is_valid"],
            )
            self.update_belief(
                "missing_information",
                validation["missing_information"],
            )

            self._publish_event_if_possible(
                topic="semantic.request.interpreted",
                correlation_id=result.get("semantic_task_id"),
                payload={
                    "customer_id": self.customer.customer_id,
                    "raw_request": raw_request,
                    "llm_result": result,
                    "validation": validation,
                },
            )

            if not validation["is_valid"]:
                self.update_status("completed")
                self.clear_current_task()

                self._publish_event_if_possible(
                    topic="semantic.clarification.required",
                    correlation_id=result.get("semantic_task_id"),
                    payload={
                        "customer_id": self.customer.customer_id,
                        "raw_request": raw_request,
                        "summary": result.get("summary"),
                        "missing_information": validation["missing_information"],
                        "clarification_question": validation[
                            "clarification_question"
                        ],
                        "llm_result": result,
                    },
                )

                return {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "status": "need_clarification",
                    "message": "The request is incomplete or the LLM output is not valid enough to build a CustomerRequest.",
                    "semantic_task_id": result.get("semantic_task_id"),
                    "customer_id": self.customer.customer_id,
                    "summary": result.get("summary"),
                    "missing_information": validation["missing_information"],
                    "clarification_question": validation[
                        "clarification_question"
                    ],
                    "llm_result": result,
                    "customer_request": None,
                }

            customer_request = self._build_customer_request(result)

            self.update_status("completed")
            self.clear_current_task()

            self._publish_event_if_possible(
                topic="semantic.customer_request.ready",
                correlation_id=customer_request.request_id,
                payload={
                    "customer_id": self.customer.customer_id,
                    "request_id": customer_request.request_id,
                    "raw_request": raw_request,
                    "summary": result.get("summary"),
                    "customer_request": self._customer_request_to_dict(
                        customer_request
                    ),
                    "llm_result": result,
                },
            )

            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "ready",
                "message": "CustomerRequest generated successfully.",
                "semantic_task_id": result.get("semantic_task_id"),
                "request_id": customer_request.request_id,
                "customer_id": customer_request.customer_id,
                "destination": customer_request.destination,
                "product_count": len(customer_request.products),
                "summary": result.get("summary"),
                "missing_information": [],
                "clarification_question": None,
                "llm_result": result,
                "customer_request": customer_request,
            }

        except Exception as e:
            self.update_belief("error", str(e))
            self.update_status("failed")
            self.clear_current_task()

            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "failed",
                "message": "SemanticAgentWrapper failed to interpret raw request.",
                "error": str(e),
                "customer_request": None,
            }

    def act(self, task: Any) -> Dict[str, Any]:
        """
        Standard BaseAgent interface.

        Expected task:
        {
            "raw_request": "...",
            "context": {...}
        }

        "raw_input" is also accepted for compatibility.
        """

        if not isinstance(task, dict):
            raise TypeError("SemanticAgentWrapper.act(task) expects a dictionary.")

        raw_request = task.get("raw_request") or task.get("raw_input")
        context = task.get("context", {})

        if not raw_request:
            raise ValueError("raw_request is required.")

        return self.interpret_raw_request(
            raw_request=raw_request,
            context=context,
        )

    # ------------------------------------------------------------------
    # LLM CALLING
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """
        Directly call LLM.

        Supported:
        - ollama
        - openai_compatible

        This method does not call the downstream LLMAgent.
        """

        if self.llm_provider == "ollama":
            return self._call_ollama(prompt)

        if self.llm_provider == "openai_compatible":
            return self._call_openai_compatible(prompt)

        raise ValueError(f"Unsupported llm_provider: {self.llm_provider}")

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama native chat API:
        POST http://127.0.0.1:11434/api/chat
        """

        url = self.llm_base_url.rstrip("/") + "/api/chat"

        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Semantic Agent for a multi-agent loading system. "
                        "Return valid JSON only. Do not include markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        response_json = self._post_json(url=url, payload=payload)

        if "message" not in response_json:
            raise ValueError(f"Unexpected Ollama response format: {response_json}")

        message = response_json["message"]

        if "content" not in message:
            raise ValueError(f"Unexpected Ollama message format: {response_json}")

        return message["content"]

    def _call_openai_compatible(self, prompt: str) -> str:
        """
        Call OpenAI-compatible chat completion API:
        POST {base_url}/chat/completions

        Example:
        - http://127.0.0.1:11434/v1
        - http://127.0.0.1:11435/v1
        """

        url = self.llm_base_url.rstrip("/") + "/chat/completions"

        payload = {
            "model": self.llm_model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Semantic Agent for a multi-agent loading system. "
                        "Return valid JSON only. Do not include markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        response_json = self._post_json(url=url, payload=payload)

        try:
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected OpenAI-compatible response format: {response_json}"
            ) from e

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }

        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"

        request = urllib.request.Request(
            url=url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_text = response.read().decode("utf-8")
                return json.loads(response_text)

        except urllib.error.HTTPError as e:
            error_text = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"LLM HTTP error {e.code}: {error_text}"
            ) from e

        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to LLM endpoint: {url}. Error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # LLM OUTPUT PARSING
    # ------------------------------------------------------------------

    def _parse_llm_json(self, llm_text: str) -> Dict[str, Any]:
        """
        Parse LLM output into a JSON dictionary.
        Handles pure JSON and markdown fenced JSON blocks.
        """

        if not isinstance(llm_text, str):
            raise ValueError("LLM output must be a string.")

        text = llm_text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError(f"LLM did not return valid JSON: {llm_text}")
            result = json.loads(match.group(0))

        if not isinstance(result, dict):
            raise ValueError("LLM JSON output must be a JSON object.")

        return result

    # ------------------------------------------------------------------
    # NORMALISATION
    # ------------------------------------------------------------------

    def _normalise_llm_result(
        self,
        llm_result: Dict[str, Any],
        raw_request: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalise LLM result.

        Important:
        - Agent generates semantic_task_id and request_id itself.
        - Agent overrides customer_id itself.
        - Agent overrides metadata.raw_request itself.
        - This avoids trusting copied placeholder IDs from the LLM.
        """

        result = dict(llm_result)

        semantic_task_id = f"SEM_{uuid4().hex[:8]}"
        request_id = f"REQ_{uuid4().hex[:8]}"

        result["semantic_task_id"] = semantic_task_id
        result["request_id"] = request_id
        result["customer_id"] = self.customer.customer_id

        summary = result.get("summary")
        if self._is_placeholder_value(summary):
            summary = raw_request
        result["summary"] = summary or raw_request

        if not isinstance(result.get("missing_information"), list):
            result["missing_information"] = []

        if "is_complete" not in result or not isinstance(result["is_complete"], bool):
            result["is_complete"] = False

        if self._is_placeholder_value(result.get("clarification_question")):
            result["clarification_question"] = None

        customer_request = result.get("customer_request")

        if not isinstance(customer_request, dict):
            customer_request = {}

        customer_request["request_id"] = request_id
        customer_request["customer_id"] = self.customer.customer_id

        customer_request.setdefault("destination", None)
        customer_request.setdefault("products", [])
        customer_request.setdefault("objective", {})
        customer_request.setdefault("constraints", {})
        customer_request.setdefault("priority", "normal")
        customer_request.setdefault("due_time", None)
        customer_request.setdefault("metadata", {})

        if self._is_placeholder_value(customer_request.get("destination")):
            customer_request["destination"] = None

        if self._is_placeholder_value(customer_request.get("due_time")):
            customer_request["due_time"] = None

        if customer_request.get("priority") not in ["low", "normal", "urgent"]:
            customer_request["priority"] = "normal"

        objective = customer_request.get("objective")
        if not isinstance(objective, dict):
            objective = {}

        primary = objective.get("primary")
        secondary = objective.get("secondary")

        if self._is_placeholder_value(primary):
            primary = None

        if self._is_placeholder_value(secondary):
            secondary = None

        objective["primary"] = primary
        objective["secondary"] = secondary
        customer_request["objective"] = objective

        constraints = customer_request.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}

        # Missing constraints are defaulted to True because these are standard
        # downstream verification requirements in this project.
        for key in self.REQUIRED_CONSTRAINTS:
            value = constraints.get(key)
            if not isinstance(value, bool):
                constraints[key] = True

        customer_request["constraints"] = constraints

        metadata = customer_request.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        metadata["source"] = "semantic_agent"
        metadata["raw_request"] = raw_request
        metadata["semantic_summary"] = result["summary"]
        metadata["semantic_task_id"] = semantic_task_id
        metadata["generated_by"] = self.agent_id

        if context:
            metadata["context"] = context

        customer_request["metadata"] = metadata

        products = customer_request.get("products", [])
        customer_request["products"] = self._normalise_products(
            products=products,
            global_destination=customer_request.get("destination"),
        )

        result["customer_request"] = customer_request

        return result

    def _normalise_products(
        self,
        products: Any,
        global_destination: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not isinstance(products, list):
            return []

        normalised_products = []

        for product in products:
            if not isinstance(product, dict):
                continue

            p = dict(product)

            if self._is_placeholder_value(p.get("product_id")):
                p["product_id"] = None
            else:
                p.setdefault("product_id", None)

            if self._is_placeholder_value(p.get("destination")):
                p["destination"] = global_destination
            elif not p.get("destination"):
                p["destination"] = global_destination

            p["weight"] = self._safe_float_or_none(p.get("weight"))
            p["length"] = self._safe_float_or_none(p.get("length"))
            p["width"] = self._safe_float_or_none(p.get("width"))
            p["height"] = self._safe_float_or_none(p.get("height"))

            normalised_products.append(p)

        return normalised_products

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_llm_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Python validation is the final gate.

        Important:
        - Do not blindly trust LLM's own missing_information.
        - If concrete fields are present, they override LLM's self-judgement.
        - due_time is optional unless your business rules later make it mandatory.
        """

        missing_information = []

        customer_request = result.get("customer_request", {})

        destination = customer_request.get("destination")
        products = customer_request.get("products", [])
        objective = customer_request.get("objective", {})
        constraints = customer_request.get("constraints", {})

        # ------------------------------------------------------------
        # 1. Destination
        # ------------------------------------------------------------
        has_valid_destination = bool(destination) and not self._is_placeholder_value(destination)

        if not has_valid_destination:
            missing_information.append("destination")

        # ------------------------------------------------------------
        # 2. Products
        # ------------------------------------------------------------
        has_products = isinstance(products, list) and len(products) > 0

        if not has_products:
            missing_information.append("product list")
        else:
            for index, product in enumerate(products):
                product_id = product.get("product_id")
                product_label = product_id or f"product_{index + 1}"

                if not product_id or self._is_placeholder_value(product_id):
                    missing_information.append(f"{product_label}: product_id")

                product_destination = product.get("destination")

                if not product_destination or self._is_placeholder_value(product_destination):
                    missing_information.append(f"{product_label}: destination")

                if product.get("weight") is None:
                    missing_information.append(f"{product_label}: weight")

                if product.get("length") is None:
                    missing_information.append(f"{product_label}: length")

                if product.get("width") is None:
                    missing_information.append(f"{product_label}: width")

                if product.get("height") is None:
                    missing_information.append(f"{product_label}: height")

        # ------------------------------------------------------------
        # 3. Objective
        # ------------------------------------------------------------
        primary_objective = objective.get("primary")
        secondary_objective = objective.get("secondary")

        if not self._is_valid_objective_value(primary_objective):
            missing_information.append("valid primary objective")

        if secondary_objective is not None and not self._is_valid_objective_value(
                secondary_objective
        ):
            missing_information.append("valid secondary objective")

        # ------------------------------------------------------------
        # 4. Constraints
        # ------------------------------------------------------------
        for key in self.REQUIRED_CONSTRAINTS:
            if key not in constraints:
                missing_information.append(f"constraint: {key}")
            elif not isinstance(constraints[key], bool):
                missing_information.append(f"constraint boolean value: {key}")

        # ------------------------------------------------------------
        # 5. LLM-declared missing information
        # ------------------------------------------------------------
        # LLM may say missing_information=["products"] even when it has already
        # extracted products correctly. Therefore we only keep LLM missing items
        # if they are still truly missing after Python field validation.
        # due_time is treated as optional in this version.
        # ------------------------------------------------------------

        llm_missing = result.get("missing_information", [])

        if isinstance(llm_missing, list):
            for item in llm_missing:
                if not item or self._is_placeholder_value(item):
                    continue

                item_text = str(item).strip().lower()

                # due_time is optional for now
                if item_text in ["due_time", "due time", "delivery time", "deadline"]:
                    continue

                # Ignore product-related missing claims if products are actually valid enough
                if item_text in ["products", "product", "product list", "product details"]:
                    if has_products:
                        continue

                if item_text in ["product_dimensions", "dimensions", "product dimension"]:
                    if has_products:
                        all_dimensions_present = all(
                            p.get("weight") is not None
                            and p.get("length") is not None
                            and p.get("width") is not None
                            and p.get("height") is not None
                            for p in products
                        )
                        if all_dimensions_present:
                            continue

                if item_text in ["destination"]:
                    if has_valid_destination:
                        continue

                missing_information.append(str(item))

        missing_information = list(dict.fromkeys(missing_information))

        is_valid = len(missing_information) == 0

        clarification_question = result.get("clarification_question")

        if not is_valid:
            if not clarification_question or self._is_placeholder_value(clarification_question):
                clarification_question = (
                        "Could you please provide or correct the following information: "
                        + ", ".join(missing_information)
                        + "?"
                )
        else:
            clarification_question = None

        return {
            "is_valid": is_valid,
            "missing_information": missing_information,
            "clarification_question": clarification_question,
        }

    def _is_valid_objective_value(self, value: Any) -> bool:
        return isinstance(value, str) and value in self.ALLOWED_OBJECTIVES

    def _is_placeholder_value(self, value: Any) -> bool:
        if value is None:
            return False

        if not isinstance(value, str):
            return False

        text = value.strip().lower()

        placeholder_values = {
            "",
            "string",
            "string or null",
            "number",
            "number or null",
            "boolean",
            "true | false",
            "null",
            "none",
            "req_xxxxxxxx",
            "sem_xxxxxxxx",
            "req_<random_id>",
            "sem_<random_id>",
            "req_<same_request_id>",
            "original raw request",
            "short summary",
            "short summary of the customer request",
            "real extracted value",
            "real extracted values",
        }

        if text in placeholder_values:
            return True

        if "|" in text:
            return True

        if " or null" in text:
            return True

        if "<" in text and ">" in text:
            return True

        if text.startswith("req_") and "x" in text:
            return True

        if text.startswith("sem_") and "x" in text:
            return True

        return False

    # ------------------------------------------------------------------
    # BUILD CUSTOMER REQUEST
    # ------------------------------------------------------------------

    def _build_customer_request(self, result: Dict[str, Any]) -> CustomerRequest:
        customer_request = result["customer_request"]

        return CustomerRequest(
            request_id=customer_request["request_id"],
            customer_id=self.customer.customer_id,
            destination=customer_request["destination"],
            products=customer_request["products"],
            objective=customer_request["objective"],
            constraints=customer_request["constraints"],
            priority=customer_request.get("priority", "normal"),
            due_time=customer_request.get("due_time"),
            metadata=customer_request.get("metadata", {}),
        )

    def _customer_request_to_dict(
        self,
        request: CustomerRequest,
    ) -> Dict[str, Any]:
        return {
            "request_id": request.request_id,
            "customer_id": request.customer_id,
            "destination": request.destination,
            "products": request.products,
            "objective": request.objective,
            "constraints": request.constraints,
            "priority": request.priority,
            "due_time": request.due_time,
            "metadata": request.metadata,
        }

    # ------------------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        raw_request: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Prompt for the Semantic Agent LLM.

        The prompt avoids schema placeholders like "string or null",
        because smaller local models may copy them literally.
        """

        output_contract = {
            "semantic_task_id": None,
            "request_id": None,
            "customer_id": self.customer.customer_id,
            "summary": None,
            "is_complete": False,
            "missing_information": [],
            "clarification_question": None,
            "customer_request": {
                "request_id": None,
                "customer_id": self.customer.customer_id,
                "destination": None,
                "products": [],
                "objective": {
                    "primary": None,
                    "secondary": None,
                },
                "constraints": {
                    "weight_limit": True,
                    "axle_balance": True,
                    "centre_line_balance": True,
                    "diagonal_balance": True,
                    "overlap_check": True,
                    "boundary_check": True,
                },
                "priority": "normal",
                "due_time": None,
                "metadata": {
                    "source": "semantic_agent",
                    "raw_request": None,
                    "semantic_summary": None,
                },
            },
        }

        example_complete_output = {
            "semantic_task_id": "SEM_example",
            "request_id": "REQ_example",
            "customer_id": self.customer.customer_id,
            "summary": "The customer wants to send two steel products to Shanghai with loading safety as the primary objective.",
            "is_complete": True,
            "missing_information": [],
            "clarification_question": None,
            "customer_request": {
                "request_id": "REQ_example",
                "customer_id": self.customer.customer_id,
                "destination": "Shanghai",
                "products": [
                    {
                        "product_id": "P001",
                        "weight": 12.5,
                        "length": 2.1,
                        "width": 1.5,
                        "height": 1.5,
                        "destination": "Shanghai",
                    }
                ],
                "objective": {
                    "primary": "ensure_loading_safety",
                    "secondary": "minimise_vehicle_number",
                },
                "constraints": {
                    "weight_limit": True,
                    "axle_balance": True,
                    "centre_line_balance": True,
                    "diagonal_balance": True,
                    "overlap_check": True,
                    "boundary_check": True,
                },
                "priority": "normal",
                "due_time": None,
                "metadata": {
                    "source": "semantic_agent",
                    "raw_request": "original customer request text",
                    "semantic_summary": "short summary",
                },
            },
        }

        return f"""
You are a Semantic Agent in a multi-agent steel product loading system.

Your job:
Read the raw customer request and decide whether it contains enough information to create a CustomerRequest for the downstream Customer Agent.

The downstream Customer Agent will publish customer.request.submitted.
Batching, optimisation, verification, loading plan generation, and visualisation are handled by downstream agents.

Rules:
1. Return JSON only.
2. Do not include markdown.
3. Do not generate a loading plan.
4. Do not assign products to vehicles, wagons, slots, or coordinates.
5. Do not invent product weight, length, width, height, destination, due time, or product IDs.
6. If information is missing, use null or an empty list.
7. If the request is complete, set is_complete to true.
8. If the request is incomplete, set is_complete to false.
9. If incomplete, list missing fields in missing_information and write a clarification_question.
10. If a global destination is given, use it as the destination for products unless a product-specific destination is stated.
11. Choose objective values only from:
   - minimise_vehicle_number
   - maximise_loading_rate
   - ensure_loading_safety
   - minimise_runtime
12. Do not copy example IDs such as SEM_example or REQ_example.
13. Do not copy placeholders.
14. The values semantic_task_id and request_id may be null. The Python agent will generate final IDs.

Customer ID:
{self.customer.customer_id}

Raw customer request:
{raw_request}

Additional context:
{json.dumps(context, ensure_ascii=False)}

Required JSON keys and shape:
{json.dumps(output_contract, ensure_ascii=False, indent=2)}

Example of a complete output:
{json.dumps(example_complete_output, ensure_ascii=False, indent=2)}

Now return the JSON object for the actual raw customer request.
""".strip()

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _publish_event_if_possible(
        self,
        topic: str,
        correlation_id: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        if self.event_bus is None:
            return

        event = Event(
            topic=topic,
            source_agent_id=self.agent_id,
            correlation_id=correlation_id,
            payload=payload,
        )

        self.event_bus.publish(event)

    def _safe_float_or_none(self, value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            cleaned = value.strip()

            if cleaned == "":
                return None

            if self._is_placeholder_value(cleaned):
                return None

            cleaned = cleaned.lower()
            cleaned = cleaned.replace("tons", "")
            cleaned = cleaned.replace("ton", "")
            cleaned = cleaned.replace("kg", "")
            cleaned = cleaned.replace("t", "")
            cleaned = cleaned.replace("吨", "")
            cleaned = cleaned.strip()

            try:
                return float(cleaned)
            except ValueError:
                return None

        return None