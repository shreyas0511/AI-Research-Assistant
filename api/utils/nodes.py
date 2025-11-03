from utils.state import AgentState
from utils.prompts import *
from setup import embeddings, get_streaming_llm
import json
import feedparser
from urllib.parse import quote
import numpy as np
from langchain.docstore.document import Document
import anyio
from utils.formatting import *

# all node functions

async def planner(state: AgentState) -> AgentState:
    """ Generate a detailed execution plan for the user query """

    state["count"] += 1
    print(f"State:\n{state}\n")

    # publisher queue provided by /query
    publish = state.get("publish")
    
    # llm with token by token streaming
    llm = get_streaming_llm(publish, "planner")

    # complete message to pass to the LLM
    message = ""

    # First check if there are any previous reflections in reflection_notes
    # If yes, generate a new plan/ or additional search parameters

    if state["reflection_notes"] != "":
        print(">> GENERATING NEW PLAN...")

        await publish("planner_token", "\n### Generating New Plan\n")

        message = system_prompt_reflection + f"\nUser query:\n{state["query"]}\nOriginal plan:\n{json.dumps(state["original_plan"])}\nReflection notes:\n{state["reflection_notes"]}"
    else:
        # If no, this is the first run, run normally
        print(">> GENERATING INITIAL PLAN...")

        await publish("planner_token", "\n### Generating Initial Plan\n")

        message = system_prompt + f"\nUser query:\n{state["query"]}"

    response = await llm.ainvoke(message)
    response_json = response.content
    if response_json.startswith("```json"):
        response_json = response_json[7:-3]
    
    try:
        response_dict = json.loads(response_json)

        # Load plan JSON as python dictionary
        state["original_plan"] = response_dict

        state["plan"] = response_dict["plan"]

        print(f"PLAN GENERATED, total {len(state["plan"])} queries to be searched:")
        print(state["plan"])

        await publish("planner_token", format_plan_for_display(response_dict))

        return {
            **state,
            "original_plan": response_dict,
            "plan": response_dict["plan"]
        }
    
    except Exception as e:
        await publish("planner", "plan_parse_error", {
            "raw": response_json,
            "error": str(e),
        })
        print(response_json)
        print(e)


async def search_arxiv(state: AgentState) -> AgentState:
    """ given the current Agent State, search arxiv and return appropriate papers. """

    publish = state["publish"]

    llm = get_streaming_llm(publish, "search_arxiv")

    print("\n>> SEARCHING ARXIV ...")

    # pass the search_terms and additional_terms to the llm
    search_terms = state["plan"][0]["query"]["search_terms"]
    additional_focus = state["plan"][0]["query"]["additional_focus"]

    # make the llm generate appropriate arxiv search queries
    message = query_expansion_prompt + f"\nSearch terms:{search_terms}\nAdditional focus:{additional_focus}"

    queries = await llm.ainvoke(message)
    queries_json = queries.content
    if queries_json.startswith("```json"):
        queries_json = queries_json[7:-3]

    try:
        queries_dict = json.loads(queries_json)

        base_url = 'http://export.arxiv.org/api/query?'
        results = []
        max_results = 5

        # construct valid arxiv url for each search query returned by the LLM and get appropriate papers
        for query in queries_dict:
            # URL-encode
            search_query = quote(query["search_query"])

            url = base_url + f"search_query={search_query}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            
            print(f"url: {url}")

            # feedparser is blocking → run in thread
            feed = await anyio.to_thread.run_sync(feedparser.parse, url)

            results.append(feed.entries)

        count = 0
        for i in range(len(results)):
            for result in results[i]:
                result_dict = {
                    "title": result['title'],
                    "published": result['published'],
                    "summary": result['summary'],
                    "arxiv_link": result['link']
                }
                # not storing for now, will store later
                # "pdf_link": result['links'][2]['href'] # or just /pdf instead of /obs in the arxiv link
                count += 1
                state["results"]["arxiv"].append(result_dict)

        # finally, pop the current tool from plan (pop(0))
        state["plan"].pop(0)
        print(f">> {len(state["plan"])} SEARCHES LEFT...")

        await publish("search_arxiv_token", format_search_queries(queries_dict, count))

        return {**state}

    except Exception as e:
        await publish("search_arxiv", "error", {"raw": queries_json, "error": str(e)})
        print(queries_json)
        print(e)
        raise
    

async def router(state: AgentState) -> str:
    """ To check if any more tool calls are left and routing to the appropriate tools """

    # if plan is empty, go to the reflection step
    if len(state["plan"]) == 0:
        return "relevance"
    else:
        return "tool call"
    
async def retrieve(state: AgentState) -> AgentState:
    """ retrieve the top relevant papers from all papers retrieved from arxiv search """

    publish = state["publish"]

    print("\n>> RETRIEVING RELEVANT PAPERS...")
    docs = []

    for paper in state["results"]["arxiv"]:
        content = f"Title: {paper["title"]}\nSummary:\n{paper["summary"]}\nLink: {paper["arxiv_link"]}"
        docs.append(Document(page_content=content))

    print(f">> LOADED {len(docs)} PAPERS")

    if len(docs) == 0:
        await publish("search_arxiv_token", format_retrieval_stats(len(docs), 0, 0.0))

        return state

    # Define a query (combine user query + analysis focus)
    user_query = state["query"]

    combined_query = f"{user_query}. {' '.join(state["original_plan"]["reflection"]["analysis_focus"])}"

    # compute query embeddings, document embeddings and similarity scores
    # embeddings are blocking → run in thread
    query_emb = await anyio.to_thread.run_sync(embeddings.embed_query, combined_query)
    doc_embs = await anyio.to_thread.run_sync(
        lambda: [embeddings.embed_query(d.page_content) for d in docs]
    )

    query_emb = np.array(query_emb) / np.linalg.norm(query_emb)
    doc_embs = np.array(doc_embs)
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

    similarities = np.dot(doc_embs, query_emb)
    threshold = np.mean(similarities) + 0.005

    print("MIN, MAX, MEAN\n")
    print(np.min(similarities), np.max(similarities), np.mean(similarities), "\n")

    threshold = np.mean(similarities) + 0.005

    # Retrieve top-k relevant documents
    # Select docs above threshold
    count = 0
    for doc, score in zip(docs, similarities):
        if score >= threshold:
            count += 1
            state["relevant_docs"].append(doc.page_content)

    await publish("search_arxiv_token", format_retrieval_stats(len(docs), count, float(threshold)))

    print(f">> USING {count} / {len(docs)} PAPERS FOR REFLECTION...")

    return {
        **state
    }


async def reflection(state: AgentState) -> AgentState:
    """ reflect on the current findings """

    publish = state["publish"]

    llm = get_streaming_llm(publish, "reflection")

    # await publish("reflection", "starting_reflection", {"count": state["count"]})

    original_reflection = json.dumps(state["original_plan"]["reflection"])
    # papers_json = json.dumps(state["results"]["arxiv"])
    papers_json = '\n'.join(state["relevant_docs"])

    message = reflection_prompt + "\nplanned reflection:\n" + original_reflection + "\nTop relevant papers retrieved from arxiv search:\n"+papers_json


    if len(state["relevant_docs"]) == 0:
        message = reflection_prompt + "\nplanned reflection:\n" + original_reflection + "\nTop relevant papers retrieved from arxiv search: No relevant papers retrieved, search with different search terms and additional terms compared to the previous search parameters."
    
    response = await llm.ainvoke(message)
    response_json = response.content
    if response_json.startswith("```json"):
        response_json = response_json[7:-3]

    try:
        response_dict = json.loads(response_json)

        if not response_dict["sufficient"]:
            print(">> CURRENT PAPERS ARE NOT SUFFICIENT...")
            if state["count"] >= 3:
                # await publish("reflection", "forced_summary_route")
                # message += "Searched more than 3 times, USE WHATEVER PAPERS YOU HAVE TO GENERATE A SUMMARY"

                response_dict["notes"] += "\nSearched more than 3 times, USE WHATEVER PAPERS YOU HAVE TO GENERATE A SUMMARY"

                await publish("reflection_token", format_reflection(response_dict))

                return {
                    **state,
                    "results": {"arxiv":[]},
                    "reflection": True,
                    "reflection_notes": response_dict["notes"]
                }
        else:
            print(">> CURRENT PAPERS ARE SUFFICIENT...")

        print(f">> {response_dict["notes"]}\n")

        await publish("reflection_token", format_reflection(response_dict))

        return {
            **state,
            "results": {"arxiv":[]},
            "reflection": response_dict["sufficient"],
            "reflection_notes": response_dict["notes"]
        }
    
    except Exception as e:
        await publish("reflection", "error", {"raw": response_json, "error": str(e)})
        print(">> ERROR IN REFLECTION", response_json)
        print(e)
        raise


async def reflection_router(state: AgentState):
    if state["reflection"]:
        return "summarize"
    else:
        return "plan"
    
async def passthrough(state: AgentState) -> AgentState:
    return state


async def summarize(state: AgentState) -> AgentState:
    """ summarize the findings """

    publish = state["publish"]

    llm = get_streaming_llm(publish, "search_arxiv")

    # Summarize the findings and store them in the state["summary"]

    papers = '\n'.join(state["relevant_docs"])

    message = summarize_prompt + f"\nUser query:\n{state["query"]}\nPapers:\n{papers}"

    print(">> GENERATING SUMMARY...")

    response = await llm.ainvoke(message)
    summary = response.content
    print(">> SUMMARIZED RESULTS !!\n")

    await publish("summarize_token", format_summary(summary))

    return {
        **state,
        "summary": summary
    }