from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    file_name: str
    query: str

class SummarizeRequest(BaseModel):
    file_name: str
    section_prompt: str
    image_filenames: Optional[List[str]] = Field(default=None)