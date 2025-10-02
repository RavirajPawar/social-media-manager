"""
Pydantic models for data validation and type safety.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class UserInput(BaseModel):
    """Model for user input data."""

    raw_content: str = Field(..., description="The raw content or topic from the user")
    platform: str = Field(..., description="Target social media platform")
    feedback: Optional[str] = Field(None, description="User feedback for refinement")


class GeneratedContent(BaseModel):
    """Model for generated social media content."""

    platform: str = Field(..., description="Target platform")
    caption: str = Field(..., description="Generated caption/content")
    hashtags: List[str] = Field(..., description="SEO-optimized hashtags")
    is_satisfied: bool = Field(
        False, description="Whether user is satisfied with the content"
    )


class ConversationState(BaseModel):
    """Model for tracking conversation state."""

    current_platform: str = Field(..., description="Currently selected platform")
    user_content: str = Field(..., description="User's original content")
    generated_content: Optional[GeneratedContent] = Field(
        None, description="Latest generated content"
    )
    refinement_count: int = Field(0, description="Number of refinement iterations")
