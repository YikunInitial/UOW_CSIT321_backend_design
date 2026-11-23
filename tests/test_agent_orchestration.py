from Agents.core.agent_factory import AgentFactory
from Agents.core.agent_registry import AgentRegistry
from Agents.core.lifecycle_manager import LifecycleManager
from Agents.core.coordinator_agent import CoordinatorAgent
from Agents.core.task import AgentTask


def main():
    # --------------------------------------------------
    # 1. Create registry, factory, lifecycle manager
    # --------------------------------------------------
    registry = AgentRegistry()
    factory = AgentFactory()
    lifecycle = LifecycleManager(
        registry=registry,
        factory=factory,
    )

    # --------------------------------------------------
    # 2. Create functional agents through LifecycleManager
    # --------------------------------------------------
    lifecycle.create_agent(
        agent_type="data",
        config={
            "json_output_path": "Agents/products.json",
        },
    )

    lifecycle.create_agent(
        agent_type="rag",
        config={
            "products_json_path": "Agents/products.json",
            "retrieved_context_output_path": "Agents/retrieved_context.json",
            "knowledge_base_dir": "KnowledgeBase",
            "persist_directory": "VectorDB/chroma_db",
            "top_k": 6,
            "rebuild_vector_db": False,
            "device": "cpu",
        },
    )

    lifecycle.create_agent(
        agent_type="llm",
        config={
            "products_json_path": "Agents/products.json",
            "retrieved_context_path": "Agents/retrieved_context.json",
            "strategy_plan_output_path": "Agents/strategy_plan.json",
            "model_name": "qwen2.5:7b",
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    )

    lifecycle.create_agent(
        agent_type="optimisation",
        config={
            "products_json_path": "Agents/products.json",
            "strategy_plan_path": "Agents/strategy_plan.json",
            "loading_solution_output_path": "Agents/loading_solution.json",
            "car_length": 13000.0,
            "car_width": 3000.0,
            "max_weight_tonne": 60.0,
            "parallel_gap": 50.0,
            "bogie_tol_tons": 2.0,
            "max_weight_diff_for_pair": 1.0,
            "force_same_outer_diameter_for_pair": True,
            "seed": 42,
        },
    )

    lifecycle.create_agent(
        agent_type="visualization",
        config={
            "loading_solution_path": "Agents/loading_solution.json",
            "visualization_output_dir": "demo_outputs",
            "visualization_summary_output_path": "Agents/visualization_summary.json",
            "figure_dpi": 200,
        },
    )

    # --------------------------------------------------
    # 3. Create CoordinatorAgent
    # CoordinatorAgent needs registry, so we create it directly
    # --------------------------------------------------
    coordinator = CoordinatorAgent(
        agent_id="CoordinatorAgent_001",
        registry=registry,
    )

    registry.register(coordinator)

    # --------------------------------------------------
    # 4. Create task
    # --------------------------------------------------
    task = AgentTask(
        task_id="REQ_FULL_001",
        customer_id="Customer_001",
        task_type="loading_optimisation",
        input_file="../Agents/products_clean.xlsx",
        objective="minimise_vehicle_number",
        constraints=[
            "weight_limit",
            "boundary_check",
            "overlap_check",
            "centerline_balance",
            "axle_balance",
            "diagonal_balance",
        ],
        metadata={
            # Data Agent
            "json_output_path": "Agents/products.json",
            "products_json_path": "Agents/products.json",

            # RAG Agent
            "retrieved_context_output_path": "Agents/retrieved_context.json",
            "retrieved_context_path": "Agents/retrieved_context.json",
            "knowledge_base_dir": "KnowledgeBase",
            "persist_directory": "VectorDB/chroma_db",
            "top_k": 6,
            "rebuild_vector_db": False,

            # LLM Agent
            "strategy_plan_output_path": "Agents/strategy_plan.json",
            "strategy_plan_path": "Agents/strategy_plan.json",
            "model_name": "qwen2.5:7b",
            "temperature": 0.1,
            "num_ctx": 8192,

            # Optimisation Agent
            "loading_solution_output_path": "Agents/loading_solution.json",
            "loading_solution_path": "Agents/loading_solution.json",
            "car_length": 13000.0,
            "car_width": 3000.0,
            "max_weight_tonne": 60.0,
            "parallel_gap": 50.0,
            "bogie_tol_tons": 2.0,
            "max_weight_diff_for_pair": 1.0,
            "force_same_outer_diameter_for_pair": True,
            "seed": 42,

            # Visualization Agent
            "visualization_output_dir": "demo_outputs",
            "visualization_summary_output_path": "Agents/visualization_summary.json",
            "figure_dpi": 200,
        },
    )

    # --------------------------------------------------
    # 5. Run full workflow
    # --------------------------------------------------
    print("Before running full workflow:")
    lifecycle.print_summary()

    result = coordinator.act(task)

    print("\nFull workflow result:")
    print(result)

    print("\nTask state:")
    print(task.to_dict())

    print("\nAfter running full workflow:")
    lifecycle.print_summary()

    # --------------------------------------------------
    # 6. Optional: remove completed agents
    # --------------------------------------------------
    print("\nRemoving completed agents:")
    removed_agents = lifecycle.remove_completed_agents()

    for agent in removed_agents:
        print(f"Removed: {agent.agent_id} | type={agent.agent_type}")

    print("\nAfter cleanup:")
    lifecycle.print_summary()


if __name__ == "__main__":
    main()