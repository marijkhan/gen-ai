import asyncio
import time
from schemas import ExecutionPlan, StepResult, ExecutionResult
from tools.search import search_web
from tools.summarizer import summarize
from tools.image_generator import generate_image
from agents.generator import generate_post
from agents.editor import edit_post


def _run_step(step, topic: str, dependency_outputs: dict[int, str]) -> StepResult:
    """Execute a single plan step synchronously."""
    start = time.perf_counter()
    try:
        if step.tool == "search_web":
            # Use the step description as the search query, or fall back to topic
            query = step.description if step.description else topic
            output = search_web(query)

        elif step.tool == "summarizer":
            # Aggregate outputs from dependencies
            combined = "\n\n".join(
                dependency_outputs[dep] for dep in step.depends_on if dep in dependency_outputs
            )
            output = summarize(combined or topic)

        elif step.tool == "content_generator":
            # Gather all upstream research
            research = "\n\n".join(
                dependency_outputs[dep] for dep in step.depends_on if dep in dependency_outputs
            )
            output = generate_post(topic, research)

        elif step.tool == "content_editor":
            # Get the draft from the dependency (content_generator)
            draft = "\n\n".join(
                dependency_outputs[dep] for dep in step.depends_on if dep in dependency_outputs
            )
            output = edit_post(draft)

        elif step.tool == "image_generator":
            img_b64, err = generate_image(topic)
            if err:
                return StepResult(
                    step=step.step,
                    tool=step.tool,
                    status="error",
                    error=err,
                    duration_ms=int((time.perf_counter() - start) * 1000),
                )
            output = img_b64 or ""

        else:
            output = f"Unknown tool: {step.tool}"

        duration_ms = int((time.perf_counter() - start) * 1000)
        return StepResult(step=step.step, tool=step.tool, status="success", output=output, duration_ms=duration_ms)

    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return StepResult(step=step.step, tool=step.tool, status="error", output="", error=str(e), duration_ms=duration_ms)


async def execute_plan(plan: ExecutionPlan) -> ExecutionResult:
    """Execute a plan respecting dependencies, running independent steps in parallel."""
    completed: dict[int, StepResult] = {}
    outputs: dict[int, str] = {}  # step_number -> output string
    execution_order: list[list[int]] = []
    all_steps = {s.step: s for s in plan.steps}
    remaining = set(all_steps.keys())

    while remaining:
        # Find steps whose dependencies are all satisfied
        wave = [
            step_num for step_num in remaining
            if all(dep in completed for dep in all_steps[step_num].depends_on)
        ]

        if not wave:
            # Deadlock — dependencies can never be satisfied
            break

        execution_order.append(sorted(wave))

        # Run wave in parallel
        async def run(step_num):
            step = all_steps[step_num]
            return await asyncio.to_thread(_run_step, step, plan.topic, outputs)

        results = await asyncio.gather(*(run(s) for s in wave))

        for result in results:
            completed[result.step] = result
            outputs[result.step] = result.output
            remaining.discard(result.step)

    # Extract final post (from content_editor or content_generator)
    final_post = None
    image_base64 = None
    for step in reversed(plan.steps):
        if step.tool == "content_editor" and step.step in outputs:
            final_post = outputs[step.step]
            break
        if step.tool == "content_generator" and step.step in outputs and final_post is None:
            final_post = outputs[step.step]

    for step in plan.steps:
        if step.tool == "image_generator" and step.step in outputs:
            result = completed[step.step]
            if result.status == "success":
                image_base64 = outputs[step.step]
            break

    return ExecutionResult(
        plan=plan,
        results=list(completed.values()),
        execution_order=execution_order,
        final_post=final_post,
        image_base64=image_base64,
    )
