import asyncio

import httpx
from ddgs import DDGS
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolCall
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from markdownify import markdownify as md

model = init_chat_model(
    model="Qwen/Qwen3-1.7B",
    model_provider="openai",
    base_url="http://localhost:8080/v1",
    api_key="none",
)


def make_search(search_query: str):
    return DDGS().text(search_query, max_results=2, timeout=10)


async def get_webpage(href: str):
    async with httpx.AsyncClient(
        timeout=5, headers={"User-Agent": "Mozilla/5.0"}
    ) as client:
        r = await client.get(href)
        r.raise_for_status()
    return r.text


async def get_extended_result(base_search):
    try:
        raw_html = await get_webpage(base_search["href"])
        raw_markdown = md(raw_html)
        summary = await model.ainvoke(
            "Summarize the markdown version of the following webpage in as few words as possible: \n"
            + raw_markdown[:2000]
        )
        return {**base_search, "full_body": raw_html, "summary": summary.content}
    except Exception as e:
        print(e)
        return {**base_search, "full_body": "", "summary": ""}


async def get_extended_search_results(base_searches):
    return await asyncio.gather(*[get_extended_result(i) for i in base_searches])


@tool
async def get_web_search_results(queries: list[str]):
    """
    Make a websearch using the provided search terms and return results

    Args:
        queries: List of search terms
    """
    search_results = [make_search(i) for i in queries]
    results_list = await asyncio.gather(
        *[get_extended_search_results(i) for i in search_results]
    )
    results_list = [
        [{k: v for k, v in j.items() if (k in ["title", "summary"])} for j in i]
        for i in results_list
    ]
    results = dict(zip(queries, results_list))
    # resutls = {}
    # results = await asyncio.gather(
    #     *[get_extended_result(i) for i in itertools.chain.from_iterable(search_results)]
    # )
    return results


tools = [
    get_web_search_results,
]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


@task
async def call_llm(messages: list[BaseMessage]):
    return await model_with_tools.ainvoke(
        [
            SystemMessage(
                "You are a helpful assistant for making simple web search and summarizing results"
            )
        ]
        + messages
    )


@task
async def call_tool(tool_call: ToolCall):
    tool = tools_by_name[tool_call["name"]]
    return await tool.ainvoke(tool_call)


@entrypoint()
async def agent(messages: list[BaseMessage]):
    model_response = await call_llm(messages=messages)
    while True:
        if not model_response.tool_calls:
            break
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = await asyncio.gather(*[fut for fut in tool_result_futures])
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = await call_llm(messages=messages)
    messages = add_messages(messages, model_response)
    return messages


# Invoke
messages = [HumanMessage(content="Make web searches on Lions and Tigers")]


async def main():
    async for chunk in agent.astream(messages, stream_mode="updates"):
        print(chunk)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
