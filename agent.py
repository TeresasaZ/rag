from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

from rich.logging import RichHandler
import logging
import time
import os
from typing import Annotated
from pydantic import Field
from agent_framework.mem0 import Mem0ContextProvider
from mem0 import AsyncMemory

from app import rag_query

#logging
handler = RichHandler(show_path=False, show_time=False, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s ")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create the client
client = OpenAIChatClient(
    base_url = "https://models.github.ai/inference",
    api_key = os.environ['GITHUB_TOKEN'],
    model_id = os.getenv("MODEL_ID", "openai/gpt-4.1-mini")
)

# Memory database
# Embedder takes into stores into the vector store databse
# llm takes from 
# store memories from previous caht history
mem0_config = {
    "llm": {
        "provider": "openai",
        "config" : {
            "model": os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"),
            "api_key": os.environ["GITHUB_TOKEN"],
            "openai_base_url": "https://models.github.ai/inference"
        },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "openai/text-embedding-3-small",
            "api_key": os.environ["GITHUB_TOKEN"],
            "openai_base_url": "https://models.github.ai/inference"
        }
    },
    "vector_store" : {
        "provider": "qdrant",
        "config": {
            "collection_name": "met_agent_memories",
            "host": "localhost",
            "port": 6333,
        }
    }
 
    }
    }

# agent read over the tools and determine which tools to use
@tool #give knowlege on artworks
#pass a question to agent, agent use this tool and use rag_query to get the answer
def search_met_artworks(
    question: Annotated[str, Field(description="The question to ask about Met artworks")]
) -> str:
    """"""
    response = rag_query(question)

    return (response)

@tool
def add_artwork_to_tour_csv( #create csv file based on question
    artwork_name: Annotated[str, Field(description="The name of the artwork to add to the tour csv")],
    artwork_artist: Annotated[str, Field(description="The artist of the artwork to add to the tour csv")],
    artwork_gallery_link: Annotated[str, Field(description="The gallery link of the artwork in the MET museum")]
):
    """Add an artwork"""
    file_name = f"met_tour_{int(time.time())}.csv"
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("artwork_name, artwork_artist, artwork_gallery_link\n")

    with open(file_name, "a") as f:
        f.write(f"{artwork_name}, {artwork_artist}, {artwork_gallery_link}\n")

# agent = Agent(
#     client = client,
#     instructions = "You are a helpful assistant that answers questions about art.",
#     tools = [search_met_artworks, add_artwork_to_tour_csv]
# )

async def main():
    # response = await agent.run("Could you please prepare a tour of artworks by Italian artists in the MET museum?") #change this question

    # print(response.text)

    # customize memory based on user id
    # access memory from vector space
    user_id = input("Enter your user id: ").strip()
 
    mem0_client = await AsyncMemory.from_config(mem0_config)
    provider = Mem0ContextProvider(source_id = "mem0_memory", user_id = user_id, mem0_client=mem0_client)
 
    agent = Agent(client=client,
              instructions="You are an assistant that answers questions about artworks in the MET museum. If you don't know the answer, say you don't know, but try to use the tool to find out.",
              tools=[search_met_artworks, add_artwork_to_tour_csv],
              context_providers=[provider]
              )
    history = []
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt()):
            print("\nExiting...")
            break
 
        if not prompt:
            continue
 
        if prompt.lower() in {"exit", "quit"}:
            print("Exiting ...")
            break

        response = await agent.run(prompt, history=history)
 
        print(f"Agent: {response.text}")
 
        #add to history of session
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response. Text})
 

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())