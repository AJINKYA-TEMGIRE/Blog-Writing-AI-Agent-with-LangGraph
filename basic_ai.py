from pydantic import BaseModel , Field
from typing import List , Annotated , TypedDict
import operator


class Task(BaseModel):
    id : int
    title : str
    brief: str = Field(... , description = "What to cover")

class Plan(BaseModel):
    blog_title :  str
    task : List[Task]


class State(TypedDict):
    topic :  str
    plan : Plan
    # for reducer agent
    sections :  Annotated[List[str] , operator.add]
    final : str


# Graph

