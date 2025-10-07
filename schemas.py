
from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime


# --- Template Variable ---
class TemplateVariableOut(BaseModel):
    id: str
    key: str
    label: Optional[str] = None
    description: Optional[str] = None
    example: Optional[str] = None
    required: bool
    type: Optional[str] = "string" # Add type for validation (e.g., "string", "date", "number")

    class Config:
        from_attributes = True   # ✅ Pydantic v2


# --- Template ---
class TemplateOut(BaseModel):
    id: str
    title: str
    fileDescription: Optional[str] = None
    jurisdiction: Optional[str] = None
    docType: Optional[str] = None
    similarityTags: List[str] = []
    bodyMd: str
    createdAt: datetime
    updatedAt: datetime
    variables: List[TemplateVariableOut] = []

    class Config:
        from_attributes = True   # ✅ Pydantic v2


# --- Document ---
class DocumentOut(BaseModel):
    id: str
    status: str
    insights: Optional[Any] = None
    fullText: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime
    documentType: Optional[str] = None

    class Config:
        from_attributes = True   # ✅ Pydantic v2