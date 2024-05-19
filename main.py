from dotenv import load_dotenv
import os
import pandas as pd

# from llama_index.core.query_engine import PandasQueryEngine
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context

from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


population_path = os.path.join("data", "world_population.csv")
population_df = pd.read_csv(population_path)

load_dotenv()

API_KEY = os.getenv("API_KEY")

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
# population_query_engine.query("What is the population of India?")

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
]


llm = OpenAI(
    model="gpt-3.5-turbo",
)

agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit):")) != "q":
    result = agent.query(prompt)
    print(result)