import os
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine,URL, Column, Integer, String, Text, ForeignKey, Date
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError
from typing_extensions import TypedDict

#Database setup
#DATABASE_URL=URL.create("sqlite",host="localhost",database="chat_history.db")
DATABASE_URL="sqlite:///chathistory.db"
engine = create_engine(DATABASE_URL,connect_args={"check_same_thread": False})
SessionLocal=sessionmaker(bind=engine, autoflush=False,autocommit=False)

#SQLAlchemy ORM model
Base= declarative_base()


class User(Base):
    __tablename__ = "user"
    
    id = Column(Integer, primary_key=True, index=True)
    username= Column(String,unique=True, index=True)
    name= Column(String)
    image= Column(String)
    hashed_password= Column(String)


class TextToImageRecord(Base):
    __tablename__ = "text_to_image" 
    
    id = Column(Integer,primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    raw_text= Column(String)
    draft_text= Column(String)
    raw_image= Column(String)
    text_image= Column(String)
    enh_image= Column(String)
    
        
    user= relationship("User")    
    

#Create Tables
Base.metadata.create_all(bind=engine)

#OAuth2 for authentication
oauth2_scheme =OAuth2PasswordBearer(tokenUrl="Login")

#pydantic models
class CreateUser(BaseModel):
    username: str
    password: str
    name: str
    image: str| None
    
class TextToImageHistory(BaseModel):
    password: str | None
    username: str |None
    user_id: str | None
    
class AnonymousState(TypedDict):
    messages: str |None
    image: str |None
    
class UserState(TypedDict):
    password: str | None
    username: str |None
    messages: str |None
    image: str |None