"""
Instagram-specific content generator.
Generates captions and SEO-optimized hashtags for Instagram.
"""

from langchain.prompts import ChatPromptTemplate
import re
from core.llm_client import LLMClient
from models.schemas import GeneratedContent


class InstagramGenerator:
    """Generator for Instagram-specific content."""

    def __init__(self):
        self.llm = LLMClient.create_llm()
        self._hashtag_cache = {}  # Simple cache for hashtag suggestions

    def generate_caption(
        self, user_content: str, feedback: str = None
    ) -> GeneratedContent:
        """
        Generate an Instagram caption with SEO-optimized hashtags.

        Args:
            user_content: The raw content/topic from user
            feedback: Optional feedback for refinement

        Returns:
            GeneratedContent: Structured content with caption and hashtags
        """
        prompt_template = self._build_prompt(feedback)
        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = prompt | self.llm
        response = chain.invoke({"user_content": user_content})

        caption, hashtags = self._parse_response(response.content)

        return GeneratedContent(
            platform="instagram",
            caption=caption,
            hashtags=hashtags,
            is_satisfied=False,  # User needs to confirm satisfaction
        )

    def _build_prompt(self, feedback: str = None) -> str:
        """Build the appropriate prompt based on context."""
        base_prompt = """
        Create an engaging Instagram caption for: {user_content}
        
        Requirements:
        - Write compelling copy that tells a story
        - Include 5-8 highly relevant, SEO-optimized hashtags
        - Use 2-3 appropriate emojis
        - Encourage engagement (ask questions, prompt comments)
        - Keep it authentic and relatable
        - 125-150 characters ideal
        
        Return format:
        [Caption text here]
        [Blank line]  
        #[hashtag1] #[hashtag2] #[hashtag3]...
        """

        if feedback:
            return f"""
            Refine this Instagram caption based on user feedback:
            
            FEEDBACK: {feedback}
            
            {base_prompt}
            """

        return base_prompt

    def _parse_response(self, response: str) -> tuple[str, list[str]]:
        """Parse LLM response into caption and hashtags."""
        lines = response.strip().split("\n")

        # Extract caption (lines before the first empty line or hashtag)
        caption_lines = []
        hashtag_lines = []

        found_break = False
        for line in lines:
            line = line.strip()
            if not line:
                found_break = True
                continue

            if line.startswith("#") or found_break:
                hashtag_lines.append(line)
            else:
                caption_lines.append(line)

        caption = " ".join(caption_lines)
        hashtags = self._extract_hashtags(" ".join(hashtag_lines))

        return caption, hashtags

    def _extract_hashtags(self, text: str) -> list[str]:
        """Extract hashtags from text."""
        hashtags = re.findall(r"#(\w+)", text)
        return hashtags[:8]  # Limit to 8 hashtags
