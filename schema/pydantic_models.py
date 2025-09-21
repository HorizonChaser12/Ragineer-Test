from pydantic import BaseModel
from typing import Optional, Literal, List, Dict


# Pydantic models for API requests
class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 10
    temperature: Optional[float] = None
    excel_file_path: Optional[str] = None


class InitializeRequest(BaseModel):

    excel_file_path: str
    temperature: Optional[float] = 0.7
    concise_prompt: Optional[bool] = False
    use_sentence_transformers: Optional[bool] = True
    use_reranker: Optional[bool] = True


class RebuildIndexRequest(BaseModel):

    force_rebuild: Optional[bool] = False


class FileData(BaseModel):
    name: str
    content: str
    size: int
    type: str
    is_binary: Optional[bool] = False


class DirectoryFilesRequest(BaseModel):
    directory_name: str
    files: List[FileData]