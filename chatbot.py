import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import httpx
import os
from dotenv import load_dotenv
from phi.utils.pprint import pprint_run_response
from phi.agent import Agent, RunResponse
import pdb


# Load environment variables
load_dotenv()

# Disable SSL verification globally for httpx
original_init = httpx.Client.__init__

def patched_init(self, *args, **kwargs):
    kwargs['verify'] = False  # Disable SSL verification
    original_init(self, *args, **kwargs)

httpx.Client.__init__ = patched_init

# Define web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="gemma2-9b-it"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

# Define financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="gemma2-9b-it"),
    tools=[
        YFinanceTools(enable_all=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Define multi-agent
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="gemma2-9b-it"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit app interface
st.title("AI Stock Analysis Chatbot")

# User query input
query = st.text_input("Enter your query (e.g., 'Summarize analyst recommendations for NVDA'): ")

# Button to execute the query
if st.button("Analyze"):
    if query:
        with st.spinner("Processing your request..."):
            # Fetch response from the multi-agent
            response: RunResponse = multi_ai_agent.run(query)
            #pdb.set_trace()  # Execution will pause here
            if response and response.content:
                    content = response.content.strip()

                    # Display the content based on its type
                    st.write(response)
                    
            else:
                st.warning("No response received. Please try again.")

    else:
        st.warning("Please enter a query to proceed.")
