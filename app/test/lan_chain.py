from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List
import json
import re


# 1. Define our structured output schema
class SocialMediaPost(BaseModel):
    post_text: str = Field(description="The content of the social media post")
    hashtags: List[str] = Field(description="Relevant hashtags for the post")


# 2. Set up the LLM connection to your local Ollama
llm = ChatOpenAI(
    model="gemma3:270m",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="sk-dummy-key",
    temperature=0.7,
)

# 3. Create a more explicit prompt template
prompt_template = """
You are a social media content creator. Create an engaging post about: {topic}

Please provide your response in EXACTLY this JSON format:
{{
  "post_text": "Your post content here",
  "hashtags": ["hashtag1", "hashtag2", "hashtag3"]
}}

Requirements:
- Create a compelling message about the topic
- Include 3-5 relevant hashtags as an array
- Return ONLY the JSON object, no additional text
"""

# 4. Set up the output parser
parser = PydanticOutputParser(pydantic_object=SocialMediaPost)

# 5. Create the final prompt
prompt = ChatPromptTemplate.from_template(prompt_template)

# 6. Create the chain
chain = prompt | llm


# 7. Helper function to extract JSON from response
def extract_json_from_text(text):
    """Try to extract JSON from potentially messy response"""
    try:
        # Look for JSON pattern in the text
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        return None
    except:
        return None


# 8. Test with a simple topic
if __name__ == "__main__":
    try:
        print("Testing connection to local LLM...")

        # Test the chain
        topic = "The benefits of meditation for software developers"
        response = chain.invoke({"topic": topic})

        print(f"Raw response: {response.content}")

        # Try to parse the response
        try:
            # First try direct parsing
            result = parser.parse(response.content)
        except:
            # If that fails, try to extract JSON from the response
            json_data = extract_json_from_text(response.content)
            if json_data:
                result = SocialMediaPost(**json_data)
            else:
                # Fallback: manual parsing
                print("Failed to parse JSON, using fallback parsing")
                lines = response.content.split("\n")
                post_text = lines[0] if lines else "No content generated"
                hashtags = []

                # Try to find hashtags in the text
                hashtag_matches = re.findall(r"#\w+", response.content)
                hashtags = [tag.replace("#", "") for tag in hashtag_matches][:5]

                if not hashtags:
                    hashtags = ["meditation", "developers", "wellbeing"]

                result = SocialMediaPost(post_text=post_text, hashtags=hashtags)

        print("\n✅ Success! Generated content:")
        print(f"Post: {result.post_text}")
        print(f"Hashtags: {result.hashtags}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is running: docker-compose up -d llm-engine")
        print("2. Check if model is loaded: curl http://localhost:11434/api/tags")
        print("3. Verify the API base URL includes /v1")
