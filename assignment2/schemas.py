from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step: int = Field(description="Step number (unique identifier)")
    tool: str = Field(description="Tool to execute: search_web, summarizer, content_generator, content_editor, or image_generator")
    description: str = Field(description="What this step does")
    depends_on: list[int] = Field(default_factory=list, description="Step numbers that must complete before this step runs")


class ExecutionPlan(BaseModel):
    topic: str = Field(description="The original topic for the LinkedIn post")
    steps: list[PlanStep] = Field(description="Ordered list of steps to execute")


class StepResult(BaseModel):
    step: int
    tool: str
    status: str = "success"  # "success" or "error"
    output: str = ""
    duration_ms: int = 0
    error: str | None = None


class ExecutionResult(BaseModel):
    plan: ExecutionPlan
    results: list[StepResult]
    execution_order: list[list[int]] = Field(default_factory=list, description="Waves of parallel step numbers")
    final_post: str | None = None
    image_base64: str | None = None
