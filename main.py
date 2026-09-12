import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    user_prompt: str

@app.post("/agent/run")
async def run_agent(request: AgentRequest):
    response = await agent_workflow(request.user_prompt)
    return {"agent_response": response}


from langchain_core.tools import tool
from langchain.agents import create_agent

"""
llm_name = "openai/gpt-oss-20b:free"
base_url="https://openrouter.ai/api/v1"
api_key=os.getenv("OPENROUTER_API_KEY")
"""

# llm_name = "llama-3.1-8b-instant" # 14.4k requests per day
llm_name = "openai/gpt-oss-20b"     # 1k requests per day
base_url="https://api.groq.com/openai/v1"
api_key=os.getenv("GROQ_API_KEY")

system_prompt = (
    "You are an expert English Tutor, a native American speaker specializing in conversational English and grammar. "
    "Your primary goal is to engage the user in natural conversation while subtly correcting their mistakes. "
    "\n\n"
    "### ROLE & PERSONA ###\n"
    "- This is all baout helping the user improve their English conversation in a friendly and supportive manner. "
    "- Be patient, kind, and encouraging. Never make the user feel embarrassed about mistakes. "
    "- Use simple, clear English. Avoid overly complex jargon unless explaining it. "
    "- Do not care about the captalization of the first letter of a sentence. "
    "- Do not care about the punctuation of a sentence. "
    "- Act like a friendly conversation partner, not a rigid teacher. "
    "\n\n"
    "### INSTRUCTIONS ###\n"
    "1. **Engage First**: Always start by responding naturally to the user's question or statement to keep the conversation flowing. "
    "2. **Correct Gently**: After your response, identify any major grammar or spelling errors in the user's input. "
    "   - Do not list every single error. Focus on the most impactful ones. "
    "   - Explain *why* it is incorrect and provide the correct version. "
    "   - Use the format: 'By the way, a small tip: [Explanation of correction].' "
    "   - D0 not Use the format: 'By the way, a small tip: [Explanation of correction].' if there is no mistake. "
    "3. **Encourage**: End with a follow-up question or a prompt to keep the conversation going. "
    "\n\n"
    "### OUTPUT FORMAT ###\n"
    "- Speak in English only. "
    "- Keep responses concise but detailed enough to be helpful. "
    "- Do not use markdown headers (like # or ##) in your spoken response. "
    "- Do not mention that you are an AI. "
    "\n\n"
    "### EXAMPLE INTERACTION ###\n"
    "User: 'I go to the store yesterday and buyed apples.'\n"
    "You: 'That sounds like a great trip to the store! I hope you found some delicious apples. \n"
    "By the way, a small tip: Since this happened yesterday, we use the past tense. Instead of 'go' and 'buyed', we say 'went' and 'bought'. So, 'I went to the store yesterday and bought apples.' \n"
    "Did you buy any other snacks?' "
)


IS_DEBUG = True

def debug_log(message):
    if IS_DEBUG:
        print(f"DEBUG: {message}")
    else:
        print(message)

# --- TOOLS ---
# Define LangChain Tools
python_tools = []


# --- AGENT ENGINE ---
async def agent_workflow(user_input):
    """
    Uses LangChain to orchestrate the ReAct agent with Ollama.
    """
    debug_log("agent_workflow: Started agent_workflow")
    if not user_input.strip():
        debug_log("agent_workflow: No user input provided.")

    debug_log("agent_workflow: Setting messages with system prompt and history")    
    # 1. Prepare the History (Context)
    # Add history
    # ------------------------------------------------------------------
    # 1️⃣ Build the full message list (system prompt + chat history)
    # ------------------------------------------------------------------
    # Start with the system prompt (already defined as a string above)
    messages = []
    # Append every past user/assistant turn in order
    """
    # for entry in context:
    #     messages.append(("user", entry["user_input"]))
    #     messages.append(("assistant", entry["response"]))
    """
    # Finally, add the current user input
    messages.append(("user", user_input))

    # 2. Initialize LangChain Components    
    debug_log("agent_workflow: ChatOpenAI() for OpenRouter")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
       model=llm_name,
       temperature=0.7,
       # OpenRouter specific configuration
       openai_api_base=base_url,
       openai_api_key=api_key,
       streaming=False,
    )
    # debug_log(f"{llm.invoke('Hello, who are you?')}")

    full_agent_log = ""
        
    try:    
        # 4. Create the Agent        
        debug_log("agent_workflow: create_agent()")
        agent = create_agent(model=llm, tools=python_tools, system_prompt=system_prompt)
    except Exception as e:
        error_msg = f"agent_workflow.create_agent().Exception error: {str(e)}"
        debug_log(f"agent_workflow.create_agent.Exception")
        full_agent_log += f"\n{error_msg}\n"
        return full_agent_log
  
    # 5. Execute the Agent        
    debug_log("agent_workflow: agent.invoke()")
    try:        
        # messages_0 = [("user", user_input)]
        result = agent.invoke(input={"messages": messages}, config={"recursion_limit": 50})
        # This is the right calling format of invoke()
        #result = agent.invoke(
        #    input={"messages": [("user", "read output/generated_code.py and run it")]},
        #    config={"recursion_limit": 50}
        #)
        full_agent_log = result["messages"][-1].content    
    except Exception as e:
        error_msg = f"agent_workflow.agent.invoke().Exception: {str(e)}"
        debug_log("agent_workflow.run agent.invoke().Exception")
        full_agent_log += f"\n{error_msg}\n"
        return full_agent_log

    return full_agent_log
