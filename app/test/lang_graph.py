"""
Social Media Content Generation Graph using LangGraph.

This module defines a stateful graph that generates platform-specific
social media content using a local LLM via Ollama.
"""

import re
from typing import Dict, List, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from core.config import settings


class GraphState(TypedDict):
    """
    State definition for the social media content generation graph.

    Attributes:
        topic: The main topic for the social media post.
        platform: The target social media platform.
        content: The generated post content.
        hashtags: List of generated hashtags for the post.
        current_step: Tracks the current execution step in the graph.
    """

    topic: str
    platform: str
    content: str
    hashtags: List[str]
    current_step: str


class SocialMediaPost(BaseModel):
    """
    Structured output model for social media posts.

    Attributes:
        content: The main content of the social media post.
        hashtags: Relevant hashtags for the post.
    """

    content: str = Field(description="The main content of the social media post")
    hashtags: List[str] = Field(description="Relevant hashtags for the post")


def create_llm() -> ChatOpenAI:
    """
    Create and configure the LLM instance using environment variables.

    Returns:
        ChatOpenAI: Configured LLM instance for content generation.

    Raises:
        ValueError: If required environment variables are not set.
    """
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key
    model_name = settings.openai_model_name

    if not all([api_base, api_key, model_name]):
        raise ValueError(
            "Missing required environment variables. "
            "Please set OPENAI_API_BASE, OPENAI_API_KEY, and OPENAI_MODEL_NAME."
        )

    return ChatOpenAI(
        model=model_name,
        openai_api_base=api_base,
        openai_api_key=api_key,
        temperature=0.7,
        max_retries=2,
        timeout=30,
    )


def extract_hashtags(text: str, max_hashtags: int = 8) -> List[str]:
    """
    Extract hashtags from text content.

    Args:
        text: The text content to extract hashtags from.
        max_hashtags: Maximum number of hashtags to return.

    Returns:
        List[str]: List of extracted hashtags (without the # symbol).
    """
    hashtags = re.findall(r"#(\w+)", text)
    return hashtags[:max_hashtags]


def platform_router(state: GraphState) -> Dict[str, str]:
    """
    Route to the appropriate platform-specific content generator.

    Args:
        state: The current graph state containing topic and platform.

    Returns:
        Dict[str, str]: Updated state with next step to execute.
    """
    platform = state["platform"].lower()

    platform_mapping = {
        "twitter": "twitter_generator",
        "x": "twitter_generator",
        "linkedin": "linkedin_generator",
        "professional": "linkedin_generator",
        "instagram": "instagram_generator",
        "ig": "instagram_generator",
    }

    next_step = platform_mapping.get(platform, "general_generator")
    return {"current_step": next_step}


def twitter_generator(state: GraphState, llm: ChatOpenAI) -> Dict:
    """
    Generate Twitter-specific content.

    Args:
        state: The current graph state.
        llm: The LLM instance for content generation.

    Returns:
        Dict: Updated state with generated content and hashtags.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Create a Twitter post about: {topic}
        
        Requirements:
        - Maximum 280 characters
        - Engaging and concise
        - Include 3-5 relevant hashtags
        - Casual, conversational tone
        
        Return only the post content with hashtags.
    """
    )

    chain = prompt | llm
    response = chain.invoke({"topic": state["topic"]})

    return {
        "content": response.content,
        "hashtags": extract_hashtags(response.content),
        "current_step": "completed",
    }


def linkedin_generator(state: GraphState, llm: ChatOpenAI) -> Dict:
    """
    Generate LinkedIn-specific content.

    Args:
        state: The current graph state.
        llm: The LLM instance for content generation.

    Returns:
        Dict: Updated state with generated content and hashtags.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Create a LinkedIn post about: {topic}
        
        Requirements:
        - Professional tone
        - Focus on business value and insights
        - Include 2-3 relevant hashtags
        - 2-3 paragraphs maximum
        
        Return only the post content with hashtags.
    """
    )

    chain = prompt | llm
    response = chain.invoke({"topic": state["topic"]})

    return {
        "content": response.content,
        "hashtags": extract_hashtags(response.content),
        "current_step": "completed",
    }


def instagram_generator(state: GraphState, llm: ChatOpenAI) -> Dict:
    """
    Generate Instagram-specific content.

    Args:
        state: The current graph state.
        llm: The LLM instance for content generation.

    Returns:
        Dict: Updated state with generated content and hashtags.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Create an Instagram caption about: {topic}
        
        Requirements:
        - Conversational and engaging tone
        - Include 5-8 relevant hashtags
        - Add emojis where appropriate
        - Encourage engagement in comments
        
        Return only the caption with hashtags.
    """
    )

    chain = prompt | llm
    response = chain.invoke({"topic": state["topic"]})

    return {
        "content": response.content,
        "hashtags": extract_hashtags(response.content),
        "current_step": "completed",
    }


def general_generator(state: GraphState, llm: ChatOpenAI) -> Dict:
    """
    Generate general social media content (fallback).

    Args:
        state: The current graph state.
        llm: The LLM instance for content generation.

    Returns:
        Dict: Updated state with generated content and hashtags.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Create a social media post about: {topic}
        Make it engaging and include relevant hashtags.
        
        Return only the post content with hashtags.
    """
    )

    chain = prompt | llm
    response = chain.invoke({"topic": state["topic"]})

    return {
        "content": response.content,
        "hashtags": extract_hashtags(response.content),
        "current_step": "completed",
    }


def create_social_media_graph(llm: ChatOpenAI) -> StateGraph:
    """
    Create and configure the social media content generation graph.

    Args:
        llm: The LLM instance for content generation.

    Returns:
        StateGraph: Compiled social media content generation graph.
    """
    # Initialize the graph with state schema
    workflow = StateGraph(GraphState)

    # Add nodes to the graph
    workflow.add_node("platform_router", platform_router)
    workflow.add_node("twitter_generator", lambda state: twitter_generator(state, llm))
    workflow.add_node(
        "linkedin_generator", lambda state: linkedin_generator(state, llm)
    )
    workflow.add_node(
        "instagram_generator", lambda state: instagram_generator(state, llm)
    )
    workflow.add_node("general_generator", lambda state: general_generator(state, llm))

    # Set entry point
    workflow.set_entry_point("platform_router")

    # Add conditional edges from router
    workflow.add_conditional_edges(
        "platform_router",
        lambda state: state["current_step"],
        {
            "twitter_generator": "twitter_generator",
            "linkedin_generator": "linkedin_generator",
            "instagram_generator": "instagram_generator",
            "general_generator": "general_generator",
        },
    )

    # Connect all generators to END
    for generator_node in [
        "twitter_generator",
        "linkedin_generator",
        "instagram_generator",
        "general_generator",
    ]:
        workflow.add_edge(generator_node, END)

    return workflow.compile()


def main():
    """
    Main execution function to demonstrate the social media graph.
    """
    try:
        # Initialize LLM and graph
        llm = create_llm()
        social_media_graph = create_social_media_graph(llm)

        # Test cases
        test_cases = [
            {"topic": "AI in healthcare", "platform": "twitter"},
            {"topic": "Remote work best practices", "platform": "linkedin"},
            {"topic": "Morning coffee routine", "platform": "instagram"},
            {"topic": "Sustainability tips", "platform": "facebook"},
        ]

        print("🚀 Social Media Content Generation Graph")
        print("=" * 50)

        for i, test_case in enumerate(test_cases, 1):
            print(
                f"\n📋 Test Case {i}: {test_case['topic']} for {test_case['platform']}"
            )
            print("-" * 40)

            # Initialize state
            initial_state = GraphState(
                topic=test_case["topic"],
                platform=test_case["platform"],
                content="",
                hashtags=[],
                current_step="start",
            )

            # Execute graph
            result = social_media_graph.invoke(initial_state)

            # Display results
            print(f"📝 Content: {result['content']}")
            print(f"🏷️  Hashtags: {result['hashtags']}")
            print(f"🔧 Platform: {test_case['platform']}")

    except ValueError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Execution error: {e}")


if __name__ == "__main__":
    main()
