from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.models.schemas import SocialMediaPost


# Initialize the LLM client
llm = ChatOpenAI(
    model=settings.openai_model_name,
    openai_api_base=settings.openai_api_base,
    openai_api_key=settings.openai_api_key,
    temperature=0.7,
)

# Create the parser
parser = PydanticOutputParser(pydantic_object=SocialMediaPost)


def get_platform_prompt(platform: str) -> str:
    """Return the appropriate prompt for each platform"""
    prompts = {
        "twitter": """
        You are an expert social media manager for a tech company.
        Write an engaging Twitter post about: {core_message}
        
        Remember to:
        - Keep it under 280 characters
        - Use a casual and engaging tone
        - Include relevant hashtags
        - Add a call to action
        
        {format_instructions}
        """,
        "linkedin": """
        You are an expert B2B marketing professional.
        Write a professional LinkedIn post about: {core_message}
        
        Remember to:
        - Keep it professional but engaging
        - Focus on business value
        - Include relevant hashtags
        - Add a call to action
        
        {format_instructions}
        """,
        "instagram": """
        You are an expert Instagram content creator.
        Write an engaging Instagram caption about: {core_message}
        
        Remember to:
        - Use an authentic and relatable tone
        - Include relevant hashtags
        - Add emojis where appropriate
        - Encourage engagement
        
        {format_instructions}
        """,
    }
    return prompts.get(platform, prompts["twitter"])


def generate_post(core_message: str, platform: str = "twitter") -> SocialMediaPost:
    """Generate a social media post for the specified platform"""
    try:
        # Get the appropriate prompt
        prompt_template = get_platform_prompt(platform)

        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create the chain
        chain = prompt | llm | parser

        # Generate the post
        result = chain.invoke(
            {
                "core_message": core_message,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        return result
    except Exception as e:
        # In a real application, you'd want more specific error handling
        raise Exception(f"Failed to generate post: {str(e)}")
