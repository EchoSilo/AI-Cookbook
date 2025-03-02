import streamlit as st
import os

# Disable LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, List, Tuple, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
import time

# Set page configuration
st.set_page_config(page_title="Jira Assistant", page_icon="🔧", layout="wide")

# Add a title
st.title("Jira Assistant")

# Add sidebar for credentials
with st.sidebar:
    st.header("Credentials")
    
    # Use session state to store API keys
    if "jira_api_token" not in st.session_state:
        st.session_state.jira_api_token = ""
    if "jira_username" not in st.session_state:
        st.session_state.jira_username = ""
    if "jira_instance_url" not in st.session_state:
        st.session_state.jira_instance_url = ""
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    
    jira_api_token = st.text_input("Jira API Token", value=st.session_state.jira_api_token, type="password")
    jira_username = st.text_input("Jira Username (Email)", value=st.session_state.jira_username)
    jira_instance_url = st.text_input("Jira Instance URL", value=st.session_state.jira_instance_url)
    openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
    
    st.session_state.jira_api_token = jira_api_token
    st.session_state.jira_username = jira_username
    st.session_state.jira_instance_url = jira_instance_url
    st.session_state.openai_api_key = openai_api_key
    
    st.checkbox("Use Jira Cloud", key="jira_cloud", value=True)
    
    model_name = st.selectbox(
        "Select OpenAI Model",
        ["gpt-4o-mini", "o1-mini", "o3-mini"],
        index=0
    )
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    if st.button("Connect"):
        if jira_api_token and jira_username and jira_instance_url and openai_api_key:
            st.session_state.connected = True
            st.success("Credentials saved!")
        else:
            st.error("Please fill in all credential fields.")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define the state schema for LangGraph
class AgentState(TypedDict):
    messages: List[Any]  # Store conversation messages
    tools: List[Any]  # Available tools

# Create the agent prompt
def create_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that can use Jira tools to help users manage their Jira instance.
        
You have access to the following tools:

{tools}

Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

# Define the agent node function with status updates
def agent_node(state: AgentState) -> dict:
    # Get available tools and prepare tool info for the prompt
    tools = state["tools"]
    tool_names = [tool.name for tool in tools]
    tool_descs = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    
    # Format the messages for the agent
    messages = state["messages"]
    
    # Execute the agent
    formatted_prompt = prompt.format_messages(
        messages=messages, 
        tools=tool_descs, 
        tool_names=", ".join(tool_names)
    )
    
    # Update status if available
    if "status" in st.session_state:
        st.session_state.status.write("🤔 **Agent thinking...**")
    
    # Get the response
    response = llm.invoke(formatted_prompt)
    
    # Show the agent's response in the status
    if "status" in st.session_state and isinstance(response, AIMessage):
        st.session_state.status.write(f"🤖 **Agent response:**\n```\n{response.content}\n```")
    
    # Return updated state with the agent's response
    return {"messages": messages + [response]}

# Define the function to route based on agent's output
def route_actions(state: AgentState) -> str:
    # Get the last message (which should be from the assistant)
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return "agent"
    
    # Check if the message contains "Final Answer:"
    if "Final Answer:" in last_message.content:
        return END
    
    # Check if the message has "Action:" and "Action Input:"
    if "Action:" in last_message.content and "Action Input:" in last_message.content:
        return "action"
    
    # Default back to the agent if no clear directive
    return "agent"

# Define function to execute tools with status updates
def execute_tools(state: AgentState) -> Dict:
    # Get the last assistant message
    last_message = state["messages"][-1]
    
    # Extract action and action input
    content = last_message.content
    action_start = content.find("Action:") + len("Action:")
    action_end = content.find("\n", action_start)
    action_input_start = content.find("Action Input:") + len("Action Input:")
    action_input_end = content.find("\n", action_input_start)
    
    if action_input_end == -1:  # If this is the last line
        action_input_end = len(content)
    
    action = content[action_start:action_end].strip()
    action_input = content[action_input_start:action_input_end].strip()
    
    # Update status if available
    if "status" in st.session_state:
        st.session_state.status.write(f"🔧 **Using tool:** `{action}`")
        st.session_state.status.write(f"📥 **Tool input:** `{action_input}`")
    
    # Find the appropriate tool
    for tool in state["tools"]:
        if tool.name == action:
            # Execute the tool
            try:
                observation = tool.invoke(action_input)
            except Exception as e:
                observation = f"Error: {str(e)}"
            break
    else:
        observation = f"Tool '{action}' not found. Available tools: {[t.name for t in state['tools']]}"
    
    # Show the observation in status
    if "status" in st.session_state:
        st.session_state.status.write(f"📤 **Observation:**\n```\n{observation}\n```")
    
    # Create a new message with the observation
    observation_message = HumanMessage(content=f"Observation: {observation}")
    
    # Return the updated state
    return {"messages": state["messages"] + [observation_message]}

# Build the graph
def build_jira_langgraph_agent():
    # Set up the agent state graph
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("action", execute_tools)
    
    # Add the edges
    workflow.add_conditional_edges("agent", route_actions, {
        "action": "action",
        END: END,
    })
    workflow.add_edge("action", "agent")
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Compile the graph
    graph = workflow.compile()
    
    return graph

# Function to run the agent with a query and show progress in status
def run_jira_agent(query: str):
    # Initialize the state
    state = {
        "messages": [HumanMessage(content=query)],
        "tools": jira_tools
    }
    
    # Set up a status container to show the agent's work
    with st.status("Agent working on your request...", expanded=True) as status:
        # Store status in session state for use in agent_node and execute_tools
        st.session_state.status = status
        
        # Run the graph
        result = jira_graph.invoke(state)
        
        # Mark as complete when done
        status.update(label="Agent finished processing", state="complete", expanded=False)
    
    # Extract the final answer from messages
    messages = result["messages"]
    for message in reversed(messages):
        if isinstance(message, AIMessage) and "Final Answer:" in message.content:
            final_answer_start = message.content.rfind("Final Answer:") + len("Final Answer:")
            return message.content[final_answer_start:].strip()
    
    # If no final answer found, return the last message but clean up the format
    if messages and isinstance(messages[-1], AIMessage):
        content = messages[-1].content
        # If it starts with "Thought:" but has no "Final Answer:", clean it up
        if "Thought:" in content and "Final Answer:" not in content:
            return content.split("Thought:")[-1].strip()
        return content
    
    return "No response generated"

# Chat input at the bottom
query = st.chat_input("Ask something about your Jira projects...")

# Check if we have credentials and a query
if query and st.session_state.get("connected", False):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Set environment variables
    os.environ["JIRA_API_TOKEN"] = st.session_state.jira_api_token
    os.environ["JIRA_USERNAME"] = st.session_state.jira_username
    os.environ["JIRA_INSTANCE_URL"] = st.session_state.jira_instance_url
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    os.environ["JIRA_CLOUD"] = str(st.session_state.jira_cloud)
    
    # Initialize components
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        streaming=True,
        max_tokens=4000
    )
    
    try:
        # Initialize Jira wrapper and toolkit
        jira = JiraAPIWrapper()
        toolkit = JiraToolkit.from_jira_api_wrapper(jira)
        jira_tools = toolkit.get_tools()
        
        # Create prompt and graph
        prompt = create_prompt()
        jira_graph = build_jira_langgraph_agent()
        
        # Run the agent and show debugging in status while getting clean response for chat
        with st.chat_message("assistant"):
            # Process the query and get a clean response
            response = run_jira_agent(query)
            
            # Display final response - clean, without the Thought/Action text
            st.markdown(response)
        
        # Add only the clean response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Please check your Jira and OpenAI credentials.")

elif query and not st.session_state.get("connected", False):
    st.error("Please enter your credentials in the sidebar and click 'Connect' first.")

# Add information about the app
with st.expander("About this app"):
    st.markdown("""
    This Streamlit app provides a conversational interface to Jira using LangChain and LangGraph.
    
    You can ask questions or give commands related to your Jira projects, and the app will use the appropriate Jira API tools to help you.
    
    Examples:
    - "Give me a list of projects"
    - "Show me the open issues in project X"
    - "Create a new issue in project Y with the title 'Bug in login form'"
    
    Made with LangChain, LangGraph, and Streamlit.
    """)