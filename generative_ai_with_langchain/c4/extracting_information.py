from pydantic import BaseModel
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

"""
=== models ===
"""


class Experience(BaseModel):
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]


class Study(Experience):
    degree: Optional[str]
    university: Optional[str]
    country: Optional[str]
    grade: Optional[str]


class WorkExperience(Experience):
    company: str
    job_title: str


class Resume(BaseModel):
    first_name: str
    last_name: str
    linkedin_url: Optional[str]
    email_address: Optional[str]
    nationality: Optional[str]
    skill: Optional[str]
    study: Optional[Study]
    work_experience: Optional[WorkExperience]
    hobby: Optional[str]


"""
=== Chains ===
"""
pdf_file_path = "<pdf_file_path>"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()

llm = ChatOpenAI()
chain = llm.with_structured_output(schema=Resume)
chain.invoke(docs)



