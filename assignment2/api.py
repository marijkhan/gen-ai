import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from schemas import ExecutionPlan, ExecutionResult
from agents.planner import create_plan
from executor import execute_plan

app = FastAPI(title="LinkedIn Content Curation Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TopicRequest(BaseModel):
    topic: str


@app.post("/plan", response_model=ExecutionPlan)
async def plan_endpoint(req: TopicRequest):
    """Generate an execution plan for a LinkedIn post on the given topic."""
    try:
        plan = create_plan(req.topic)
        return plan
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute", response_model=ExecutionResult)
async def execute_endpoint(req: TopicRequest):
    """Generate a plan, execute it, and return the full result."""
    try:
        plan = create_plan(req.topic)
        result = await execute_plan(plan)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
