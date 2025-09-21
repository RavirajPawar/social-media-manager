from pydantic import BaseModel, Field
from typing import List, Literal


class GenerationRequest(BaseModel):
    core_message: str
    platform: Literal["twitter", "linkedin", "instagram"] = "twitter"


class SocialMediaPost(BaseModel):
    post_text: str = Field(description="The full text of the social media post.")
    hashtags: List[str] = Field(description="A list of relevant hashtags for the post.")


class GenerationResponse(BaseModel):
    platform: str
    post_text: str
    hashtags: List[str]
