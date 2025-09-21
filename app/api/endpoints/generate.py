from fastapi import APIRouter, HTTPException
from app.services.post_generation import generate_post
from app.models.schemas import GenerationRequest, GenerationResponse

router = APIRouter()


@router.post("/generate", response_model=GenerationResponse)
async def generate_social_media_post(request: GenerationRequest):
    """
    Generate a social media post for the specified platform.
    """
    try:
        # Call the service to generate the post
        result = generate_post(request.core_message, request.platform)

        # Return the response
        return GenerationResponse(
            platform=request.platform,
            post_text=result.post_text,
            hashtags=result.hashtags,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate post: {str(e)}"
        )
