#!/usr/bin/env python
# coding: utf-8

from typing import Optional, Type, Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps


class MessagesWithSteps(MessagesState):
    remaining_steps: RemainingSteps


def end_or_reflect(state: MessagesWithSteps) -> Literal[END, "graph"]:
    if state["remaining_steps"] <= 0:
        print("✅ Response approved by editor after max attempts")
        return END
    if len(state["messages"]) == 0:
        return END
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "graph"
    return END

def create_reflection_graph(
        graph: CompiledStateGraph,
        reflection: CompiledStateGraph,
        state_schema: Optional[Type[Any]] = None,
        config_schema: Optional[Type[Any]] = None,
) -> StateGraph:
    _state_schema = state_schema or graph.builder.schema

    if "remaining_steps" in _state_schema.__annotations__:
        raise ValueError("Has key 'remaining_steps' in state_schema, this shadows a built in key")

    if "messages" not in _state_schema.__annotations__:
        raise ValueError("Missing required key 'messages' in state_schema")

    class StateSchema(_state_schema):
        remaining_steps: RemainingSteps

    rgraph = StateGraph(StateSchema, config_schema=config_schema)
    rgraph.add_node("graph", graph)
    rgraph.add_node("reflection", reflection)
    rgraph.add_edge(START, "graph")
    rgraph.add_edge("graph", "reflection")
    rgraph.add_conditional_edges("reflection", end_or_reflect)
    return rgraph


import requests
from io import BytesIO
from openai import OpenAI
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict
from openevals.llm import create_llm_as_judge
from dotenv import load_dotenv
from IPython.display import Markdown, display

client = OpenAI()


def create_file(client, file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        response = requests.get(file_path)
        file_content = BytesIO(response.content)
        file_name = file_path.split("/")[-1]
        file_tuple = (file_name, file_content)
        result = client.files.create(file=file_tuple, purpose="assistants")
    else:
        with open(file_path, "rb") as file_content:
            result = client.files.create(file=file_content, purpose="assistants")
    print(result.id)
    return result.id


file_id = create_file(client, "hiking_products.pdf")

vector_store = client.vector_stores.create(name="knowledge_base")
print(vector_store.id)

result = client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file_id)

result = client.vector_stores.files.list(vector_store_id=vector_store.id)
print(result)

load_dotenv()

# model = init_chat_model(model="openai:gpt-4o", model_kwargs={"use_responses_api": True})
model = init_chat_model(model="openai:gpt-4o")

openai_vector_store_ids = [vector_store.id, ]

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search_preview",
            "description": "Perform a web search and return a preview of results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search for information in uploaded files using a vector store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for in the vector store."
                    },
                    "vector_store_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of vector store IDs to search in."
                    }
                },
                "required": ["query", "vector_store_ids"]
            }
        }
    }
]

llm_with_tools = model.bind_tools(tools)

prompt = """
I'm on the marketing team for an ecommerce camping store called Contoso Outdoors.
Write me a short article that advertises the tents and sleeping bags we have in the hiking products file.
Make sure to name the specific products you include in the article. 
You should also look for the latest trends for camping in the summer in California and include those
in the article. make sure to cite the sources you used. 
You should also include trending places to camp in California and link to information about those places.
The article should use a friendly and approachable tone.
"""


# def call_model(state):
#     print('Creating your article... 📝')
#     return {"messages": llm_with_tools.invoke(state["messages"])}
def call_model(state):
    print('Creating your article... 📝')
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    updated_messages = [response]

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get('id')
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})

            # Simulate tool response (replace with actual tool execution if available)
            if tool_name == "web_search_preview":
                tool_result = {"content": "Web search results for camping trends in California..."}
            elif tool_name == "file_search":
                tool_result = {"content": "File search results from hiking_products.pdf..."}
            else:
                tool_result = {"content": f"Tool {tool_name} not implemented"}

            tool_response = {
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call_id
            }
            updated_messages.append(tool_response)

    return {"messages": updated_messages}


assistant_graph = (
    StateGraph(MessagesState).add_node(call_model).add_edge(START, "call_model").add_edge("call_model", END).compile())


class Finish(TypedDict):
    finish: bool


critique_prompt = """You are an expert judge evaluating AI responses. Your task is to critique the AI assistant's latest response in the conversation below.

Evaluate the response based on these criteria:
1. Accuracy - Is the information correct and factual?
2. Completeness - Does it fully address the user's query?
3. Clarity - Is the explanation clear and well-structured?
4. Helpfulness - Does it provide actionable and useful information?
5. Safety - Does it avoid harmful or inappropriate content?

If the response meets ALL criteria satisfactorily, set pass to True.

If you find ANY issues with the response, do NOT set pass to True. Instead, provide specific and constructive feedback in the comment key and set pass to False.

Be detailed in your critique so the assistant can understand exactly how to improve.

<response>
{outputs}
</response>"""


def judge_response(state, config):
    evaluator = create_llm_as_judge(prompt=critique_prompt, model="openai:o3-mini", feedback_key="pass", )
    print('Evaluating the article... 🧐')
    eval_result = evaluator(outputs=state["messages"][-1].content, inputs=None)
    if eval_result["score"]:
        print("✅ Response approved by editor")
        print("")
        print("Here's your article:")
        # display(Markdown(state["messages"][-1].text()))
        print(state["messages"][-1].text())
        return
    else:
        print("⚠️ Judge requested improvements")
        return {"messages": [{"role": "user", "content": eval_result["comment"]}]}


judge_graph = (
    StateGraph(MessagesState).add_node(judge_response).add_edge(START, "judge_response").add_edge("judge_response",
                                                                                                  END).compile())

reflection_app = create_reflection_graph(assistant_graph, judge_graph)
reflection_app = reflection_app.compile()

example_query = [{"role": "user", "content": f"{prompt}", }]

print("Running Content Generator and Editor with Reflection")
print("")
# result = reflection_app.invoke({"messages": example_query})
# Update the invocation to set initial remaining_steps
result = reflection_app.invoke({"messages": example_query, "remaining_steps": 3})
