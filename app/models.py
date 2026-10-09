"""Pydantic models for validated LLM responses and internal payloads."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Entity(BaseModel):
    text: str
    type: str = Field(min_length=2)


class AnalysisResult(BaseModel):
    summary: str
    sentiment: str
    keywords: List[str]
    entities: List[Entity]
    important_information: List[str]

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("summary must not be empty")
        return v.strip()

    @field_validator("sentiment")
    @classmethod
    def sentiment_valid(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in {"positive", "negative", "neutral", "mixed"}:
            raise ValueError("sentiment must be one of positive/negative/neutral/mixed")
        return s


class QAResult(BaseModel):
    answer: str

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("answer must not be empty")
        return v.strip()
