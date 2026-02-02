
from pydantic import BaseModel , Field
from typing import List , Annotated , TypedDict
import operator
from langgraph.graph import START , END , StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.types import Send
from pathlib import Path


load_dotenv()

model = ChatGroq(model_name = "openai/gpt-oss-120b")

class Task(BaseModel):
    id : int
    title : str
    brief: str = Field(... , description = "What to cover")

class Plan(BaseModel):
    blog_title :  str
    tasks : List[Task]

class State(TypedDict):
    topic :  str
    plan : Plan
    # for reducer agent
    sections :  Annotated[List[str] , operator.add]
    final : str


# Graph

def planner(state: State) -> dict:
    plan = model.with_structured_output(Plan).invoke([
        SystemMessage(content=("Create a blog plan with 5-7 sections on the following topic.")),
        HumanMessage(content = f"Topic : {state['topic']}")
    ])
    return {"plan" : plan}

def fanout(state: State):
    return [Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]})
            for task in state["plan"].tasks]

def worker(payload: dict) -> dict:

    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    blog_title = plan.blog_title

    section_md = model.invoke(
        [
            SystemMessage(content="Write one clean Markdown section."),
            HumanMessage(
                content=(
                    f"Blog: {blog_title}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Brief: {task.brief}\n\n"
                    "Return only the section content in Markdown."
                )
            ),
        ]
    ).content.strip()

    return {"sections": [section_md]}


def reducer(state: State) -> dict:
    import re

    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
    safe_title = safe_title.strip().replace(" ", "_").lower()
    filename = safe_title + ".md"

    base_dir = Path(__file__).parent
    output_path = base_dir / filename

    output_path.write_text(final_md, encoding="utf-8")
    return {"final": final_md}


graph = StateGraph(State)

graph.add_node("planner" , planner)
graph.add_node("worker" , worker)
graph.add_node("reducer" , reducer)

graph.add_edge(START , "planner")
graph.add_conditional_edges("planner" , fanout , ["worker"])
graph.add_edge("worker" , "reducer")
graph.add_edge ("reducer" , END)

app = graph.compile()

result = app.invoke({"topic" : "Self Attention" , "sections" : []})
