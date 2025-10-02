"""
Twitter-specific content generator with character limits and engagement optimization.
"""

from langchain.prompts import ChatPromptTemplate
import re
from core.llm_client import LLMClient
from models.schemas import GeneratedContent


class TwitterGenerator:
    """Generator for Twitter-specific content with engagement optimization."""

    def __init__(self):
        self.llm = LLMClient.create_llm()

    def generate_post(
        self, user_content: str, feedback: str = None
    ) -> GeneratedContent:
        """
        Generate a Twitter post with engagement hooks and trending hashtags.

        Args:
            user_content: The raw content/topic from user
            feedback: Optional feedback for refinement

        Returns:
            GeneratedContent: Structured Twitter content
        """
        prompt_template = self._build_prompt(feedback)
        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = prompt | self.llm
        response = chain.invoke({"user_content": user_content})

        content, hashtags = self._parse_response(response.content)

        # Ensure character limit compliance
        content = self._ensure_character_limit(content)

        return GeneratedContent(
            platform="twitter", caption=content, hashtags=hashtags, is_satisfied=False
        )

    def _build_prompt(self, feedback: str = None) -> str:
        """Build Twitter-specific prompt."""
        base_prompt = """
        Create an engaging Twitter post about: {user_content}
        
        Twitter Requirements:
        - Maximum 280 characters (CRITICAL)
        - Start with a strong hook to stop scrolling
        - Use 2-3 trending/relevant hashtags
        - Include engagement elements (questions, polls, CTAs)
        - Conversational and concise tone
        - Optimize for retweets and engagement
        
        Return format:
        [Tweet content here]
        
        #[hashtag1] #[hashtag2] #[hashtag3]
        """

        if feedback:
            return f"""
            Refine this Twitter post based on user feedback:
            
            FEEDBACK: {feedback}
            
            {base_prompt}
            """

        return base_prompt

    def _parse_response(self, response: str) -> tuple[str, list[str]]:
        """Parse LLM response into content and hashtags."""
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        content_lines = []
        hashtag_lines = []

        for line in lines:
            if line.startswith("#"):
                hashtag_lines.append(line)
            else:
                content_lines.append(line)

        content = " ".join(content_lines)
        hashtags = self._extract_hashtags(" ".join(hashtag_lines))

        return content, hashtags

    def _extract_hashtags(self, text: str) -> list[str]:
        """Extract hashtags from text."""
        hashtags = re.findall(r"#(\w+)", text)
        return hashtags[:3]  # Twitter typically uses 2-3 hashtags

    def _ensure_character_limit(self, content: str) -> str:
        """Ensure content doesn't exceed Twitter's character limit."""
        if len(content) > 280:
            # Simple truncation - in production, you might want smarter shortening
            return content[:277] + "..."
        return content
