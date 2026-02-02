from pydantic import BaseModel , Field
from typing import List , Annotated , TypedDict
import operator
from langgraph.graph import START , END , StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name = "openai/gpt-oss-120b")

class Task(BaseModel):
    id : int
    title : str
    brief: str = Field(... , description = "What to cover")

class Plan(BaseModel):
    blog_title :  str
    task : List[Task]

model = model.with_structured_output(Plan)

class State(TypedDict):
    topic :  str
    plan : Plan
    # for reducer agent
    sections :  Annotated[List[str] , operator.add]
    final : str


# Graph

def planner(state: State) -> dict:
    plan = model.invoke([
        SystemMessage(content=("Create a blog plan with 5-7 sections on the following topic.")),
        HumanMessage(content = f"Topic : {state['topic']}")
    ])
    return {"plan" : plan}




graph = StateGraph(State)

graph.add_node("planner" , planner)

graph.add_edge(START , "planner")
graph.add_edge("planner" , END)

app = graph.compile()

result = app.invoke({"topic" : "Self Attention"})

print(result["plan"])



