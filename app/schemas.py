from pydantic import BaseModel, field_validator
from typing import Optional,Literal


class Message(BaseModel):
    """A single message in the conversation history.
    request body shows: 
    {"role": "user"|"assistant", "content": "..."}
    """
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat.
    
    "The API is stateless. Every POST /chat call carries the full conversation history."
    """
    messages: list[Message]


class Recommendation(BaseModel):
    """A single assessment recommendation.
    
    Response schema:
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
    """
    name: str
    url: str
    test_type: str

    @field_validator("test_type")
    @classmethod
    def validate_test_type(cls, v):
        """Each code must be one of the valid SHL test type codes.
        Source: SHL product catalog page — filter sidebar lists these categories.
        """
        valid_codes = {"A", "B", "C", "D", "E", "K", "P", "S"}
        codes = [c.strip() for c in v.split(",")]
        for code in codes:
            if code not in valid_codes:
                raise ValueError(
                    f"Invalid test_type code '{code}'. "
                    f"Valid codes: {valid_codes}"
                )
        return v


class ChatResponse(BaseModel):
    """Response body for POST /chat.
    
    Response schema:
    - reply: agent's text response
    - recommendations: EMPTY when gathering context or refusing,
      array of 1-10 items when agent has committed to a shortlist
    - end_of_conversation: true only when agent considers task complete
    """
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

    @field_validator("recommendations")
    @classmethod
    def validate_recommendation_count(cls, v):
        """It is an array of 1 to 10 items when the agent has committed to a shortlist.
        We allow 0 (empty) for clarification/refusal turns.
        """
        if len(v) > 10:
            raise ValueError(f"Maximum 10 recommendations allowed, got {len(v)}")
        return v
