import os
import bcrypt
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import UploadFile

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from model import *

import requests
import io
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import base64

memory=MemorySaver()

# Load environment variables
load_dotenv(override=True)
CLIPDROP_API_KEY = os.getenv("CLIPDROP_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq API
llm = ChatGroq(api_key=GROQ_API_KEY,
               model="llama-3.3-70b-versatile",
               temperature=0.2,
               max_retries=2)

#initialize FASTAPI
app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_user(password:str,username:str):
    db= SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user.hashed_password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"user_id":user.id,"username":user.username}


#function to store chat message
def store_text_image_record(user_id: int, raw_text: str, draft_text: str,raw_image:str|None,
                         text_image: str|None, enh_image:str|None):
    db = SessionLocal()
    try:
        if not raw_text.strip():  # Ensure message is not empty
            print("Warning: Attempted to store an empty message.")
            return  # Exit function without storing
        
        # Create and add new chat entry
        chat_entry = TextToImageRecord(user_id=user_id, raw_text=raw_text, draft_text=draft_text,
                                       raw_image=raw_image, text_image=text_image, enh_image=enh_image)
        db.add(chat_entry)
        db.commit()
        db.refresh(chat_entry)  # Ensure the object is fully committed

          
    except Exception as e:
        db.rollback()  # Rollback transaction if there's an error
        print(f"Error storing advice message: {e}")
    
    finally:
        db.close()  # Ensure session is always closed.
        
        
#Function to create new user
@app.post("/register")
def register_user(request:CreateUser):
    db = SessionLocal()
    hash_password = bcrypt.hashpw(request.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    db_user = db.query(User).filter(User.username== request.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = User(username=request.username, hashed_password=hash_password, image=request.image,
                    name=request.name)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    db.close()
    
    return {"message": "User registered successfully"}

#function to authenticate user
@app.post("/login")
def login_user(form_data:OAuth2PasswordRequestForm =Depends()):
    #get user from database
    db = SessionLocal()
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not bcrypt.checkpw(form_data.password.encode("utf-8"), user.hashed_password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token_payload = {
        "username": user.username,
        "name": user.name,
        "user_id": user.id
        #"exp": datetime.datetime() + datetime.timedelta(hours=1)  # Token expires in 1 hour
    }
    #jwt can be use to manage data validity her
    
    return {"response": token_payload}

#Retrieve previous conversation
def retrieve_text_to_image(user_id: int):
    db = SessionLocal()
    try:
        # Check if the user has previous advice
        messages = db.query(TextToImageRecord).filter(
            TextToImageRecord.user_id == user_id
        ).order_by(TextToImageRecord.id.asc()).all()

        #Return empty response if no messages found
        if not messages:
            return {"response": ""}

        # Format and return history
        return {
            "user_id": user_id,
            "history": [{"id": msg.id, "raw_text": msg.raw_text, "text_image":msg.text_image,
                         "enh_image":msg.enh_image} for msg in messages]
        }
    
    finally:
        db.close()  # Ensure the session is closed
        
#Retrieve previous chat history for particular stock
@app.post("/history")
def get_advice_history(request:TextToImageHistory):
    #fetch history with username
    user= get_user(request.password, request.username)
    history = retrieve_text_to_image(user_id=user["user_id"])
    return history

#Graph building starts  from here
#
#
#Prompt contructions LLM
INPUT_PROMPT="""You are an interior and exterior decorator experts. from the user input text provided\
    - enhance and develop it into a well detail descriptions of the user intents with dimensions.
    - General user intents are pictorial imagination, put into text to describe how their interior or exterior\
        looks, or how it should look like. 
    -----    
        """

SPATIAL_PROMPT="""You are an interior and exterior decorator experts with artistic skills to extract spatial details for sketch drawing.\
    Extracts spatial details from information provided in the content below:
    ------
    content:{contents}"""
    
COMPOSITION_PROMPT="""You are a professional artist generate composition details for production:
    - Shapes and proportions
    - Orientation, balance, and harmony of visual elements
    - Contrast and tonal values
    - Rhythm and gamut
    - Perspective
    - Symmetry
    - Stylization
    - Visual focus
    
    """
IMAGE_PROMPT="""You are a Professional artistic editor generate composition details for background replacement:
    - Shapes and proportions
    - Orientation, balance, and harmony of visual elements
    - Perspective
    - Symmetry
    - Stylization 
 """
    
# Define the system state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    image:  Optional[str] = None
    task:  Optional[str] = None
    refined:  Optional[str] = None
    draft:  Optional[str] = None
    user_id:  Optional[str] = None
    gen_image: Optional[str] = None
    
# Entry point functions "starter"
def handle_user_input(state: AgentState):
    # Ensure messages exist
    if not state["messages"]:  # If empty, return early
        return {"response": "No input provided."}

    # Extract text content from HumanMessage objects
    if isinstance(state["messages"], list):
        user_message = "\n".join(
            msg.content if isinstance(msg, HumanMessage) else str(msg) for msg in state["messages"]
        )
    else:
        user_message = str(state["messages"])

    messages = [
        SystemMessage(content=INPUT_PROMPT),
        HumanMessage(content=user_message)  # ✅ Now it's a string
    ]

    # Check if an image is provided
    if state.get("image"):
        messages.insert(0, SystemMessage(content=IMAGE_PROMPT))  # Add IMAGE_PROMPT first
        response = llm.invoke(messages)
        return {"messages": messages, "task": response.content, "image": state["image"]}
    else:
        response = llm.invoke(messages)
        return {"messages": messages, "task": response.content}


"""def handle_user_input(state:AgentState):
    #If message is empty END process
    messages =[
        SystemMessage(content=INPUT_PROMPT),
        HumanMessage(content=state["messages"])
    ]
    if state["messages"] ==[]:
        response=llm.invoke(messages)
        return{'response':response.content}
    else:
        if state.get("image", None) != None: # Check for image
            messages = [
                SystemMessage(content=IMAGE_PROMPT),
                HumanMessage(content=state['messages'][-1])
            ]
            response=llm.invoke(messages)
            return {'messages':[messages], 'task':[response.content], 'image':state['image']}
        else:
            response= llm.invoke(messages)
            return {'messages':[messages], 'task':[response.content]}
"""
#Extract spatial data information from user text "process"
def proccess_spatial_data(state:AgentState):
    process_task=[
    SystemMessage(
        content=SPATIAL_PROMPT.format(contents=state['task'])
    )
    ]
    _task=llm.invoke(process_task)
    return {'messages':state['messages'], 'refined':_task.content}

#Process composition for text to image "composition"
def text_composition_process(state:AgentState):
    composition=[
        SystemMessage(content=COMPOSITION_PROMPT),
        HumanMessage(content=state['refined'])
    ]
    art_composite=llm.invoke(composition)
    return {'messages':state['messages'], 'draft':art_composite.content}

#image generating function "generation"
def generate_image_model(state:AgentState):
    #generate image png
    if state.get("image",None):
        result=image_to_reimagine_model({'draft':state["draft"], 'image':state["image"]})
        state["gen_image"] = result["content"] if "content" in result else None
    else:
        result= text_to_image_model({'draft':state["draft"]})
        state["gen_image"] = result["content"] if "content" in result else None
        
    return state
    
        
#conditional edge function
def should_process(state: AgentState):
    if state['messages'] is None or state['messages'] == []:
        return END
    return 'process'

def text_to_image_model(state:AgentState):
    url='https://clipdrop-api.co/text-to-image/v1'
    files= {'prompt':(None, state['draft'], 'text/plain')}
    headers= {'x-api-key': CLIPDROP_API_KEY}
    response= requests.post(url,files=files,headers=headers)
    if response.ok:
        image_bytes = response.content
        return {"content": image_bytes, "media_type": "image/png"}

    return {
        "content": response.text,
        "media_type": "text/plain",
        "status_code": response.status_code
    }   
            

def image_to_reimagine_model(state: AgentState):
    url = 'https://clipdrop-api.co/replace-background/v1'
    headers = {'x-api-key': CLIPDROP_API_KEY, 'accept': 'image/png'}
    data = {'prompt': state["draft"]}
    # Ensure state["image"] is an UploadFile object
    if isinstance(state["image"], UploadFile):
        files = {
            'image_file': (state["image"].filename, state["image"].file, state["image"].content_type)
        }
    else:
        raise ValueError("Invalid image input. Expected an UploadFile object.")
    response = requests.post(url, files=files, data=data, headers=headers)
    if response.ok:
        return {"content": response.content, "media_type": "image/png"}
    return {
        "content": response.text,
        "media_type": "text/plain",
        "status_code": response.status_code
    }

         
  
#Graph Building start here        
builder=StateGraph(AgentState)

builder.add_node("starter",handle_user_input)
builder.add_node("process", proccess_spatial_data)
builder.add_node("composition", text_composition_process)
builder.add_node("generate",generate_image_model)

builder.set_entry_point("starter")

builder.add_conditional_edges(
    "starter",
    should_process,
    {END:END, "process":"process"}
)
builder.add_edge("starter","process")
builder.add_edge("process","composition")
builder.add_edge("composition","generate")
builder.add_edge("generate",END)

user_graph = builder.compile(checkpointer=memory)
anon_graph = builder.compile()


@app.post("/api")
async def start_generation_user(request:UserState):
    #start the graph process
    response=user_graph.invoke(
        {"messages":request["messages"], "image":request["image"]},
        stream_mode="values", 
        config = {"configurable": {"thread_id":request["user_id"]}}
        )
    #Extracting content safely
    #input text= response.get("messages", "No content available")[:1]
    #media_type = response.get("media_type", "application/json")  # Default media type
    #draft = response.get("draft", None)
    #input image= response.get("image",None)
    try:
        image_bytes = response.get("gen_image", None)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = plt.imread(io.BytesIO(image_array))

        # Save image to buffer
        img_bytes = io.BytesIO()
        plt.imsave(img_bytes, image, format="png")
        plt.close()
        img_bytes.seek(0)

        # Return the image as FastAPI response
        return Response(content=img_bytes.getvalue(), media_type="image/png")
    except AttributeError as e:
        return{"Error":e,
                "Response":response.get("content",None),
                "status_code":response.get("status_code",None)
                }
    
    

@app.post("/anonymous")
async def start_generation_anon(request:AnonymousState):
    #start the graph process
    response=anon_graph.invoke(
        {"messages":request["messages"], "image":request["image"]},
        stream_mode="values", 
        )
    #Extracting content safely
    #input text= response.get("messages", "No content available")[:1]
    #media_type = response.get("media_type", "application/json")  # Default media type
    #draft = response.get("draft", None)
    #input image= response.get("image",None)
    try:
        image_bytes = response.get("gen_image", None)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = plt.imread(io.BytesIO(image_array))

        # Save image to buffer
        img_bytes = io.BytesIO()
        plt.imsave(img_bytes, image, format="png")
        plt.close()
        img_bytes.seek(0)

        # Encode the image in Base64
        encoded = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        # Build the Data URL
        data_url = f"data:image/png;base64,{encoded}"
        
        # Return the Data URL in a JSON response
        return JSONResponse(content={"image": data_url})
    
    except AttributeError as e:
        return{"Error":e,
                "Response":response.get("content",None),
                "status_code":response.get("status_code",None)
                }
    
    
    
    
    
    
