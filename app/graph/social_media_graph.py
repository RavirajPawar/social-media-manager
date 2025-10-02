"""
Social Media Content Generation Graph using LangGraph.
Handles multi-platform content creation with state management and user refinement.
"""

from typing import Literal, Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from core.llm_client import LLMClient
from platforms.instagram_generator import InstagramGenerator
from platforms.twitter_generator import TwitterGenerator


class GraphState(TypedDict):
    """
    State management for social media content generation workflow.

    Attributes:
        user_topic: Original content topic from user
        current_platform: Platform currently being processed
        target_platforms: List of platforms user wants to target
        generated_content: Content for current platform
        hashtags: SEO-optimized hashtags
        user_feedback: User's refinement feedback
        is_satisfied: Whether user is satisfied with current content
        processed_platforms: List of completed platforms
        conversation_history: Track all interactions
        next_step: Determines workflow progression
    """

    user_topic: str
    current_platform: Literal["instagram", "twitter", None]
    target_platforms: List[Literal["instagram", "twitter"]]
    generated_content: str
    hashtags: List[str]
    user_feedback: Optional[str]
    is_satisfied: bool
    processed_platforms: List[str]
    conversation_history: Annotated[List[str], add_messages]
    next_step: Literal[
        "generate_content", "get_feedback", "refine_content", "next_platform", "end"
    ]


class SocialMediaGraph:
    """
    LangGraph-based social media content generator with state management.

    Features:
    - Multi-platform content generation
    - Iterative refinement with user feedback
    - State persistence across platform transitions
    - Conditional workflow routing
    """

    def __init__(self):
        self.llm = LLMClient.create_llm()
        self.instagram_gen = InstagramGenerator()
        self.twitter_gen = TwitterGenerator()
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create and compile the social media content generation graph."""
        workflow = StateGraph(GraphState)

        # Add nodes to the graph
        workflow.add_node("platform_selector", self._select_platform)
        workflow.add_node("content_generator", self._generate_content)
        workflow.add_node("user_feedback", self._get_user_feedback)
        workflow.add_node("content_refiner", self._refine_content)
        workflow.add_node("platform_advancer", self._advance_to_next_platform)

        # Define workflow structure
        workflow.set_entry_point("platform_selector")

        # Main flow: Platform selection → Content generation → User feedback
        workflow.add_edge("platform_selector", "content_generator")
        workflow.add_edge("content_generator", "user_feedback")

        # Conditional edges from user feedback
        workflow.add_conditional_edges(
            "user_feedback",
            self._should_refine_content,
            {
                "refine": "content_refiner",
                "next_platform": "platform_advancer",
                "end": END,
            },
        )

        # Refinement loop back to feedback
        workflow.add_edge("content_refiner", "user_feedback")

        # Platform advancement loop
        workflow.add_conditional_edges(
            "platform_advancer",
            self._has_more_platforms,
            {"continue": "content_generator", "end": END},
        )

        return workflow.compile()

    def _select_platform(self, state: GraphState) -> dict:
        """
        Select the first platform to process from target platforms.

        Args:
            state: Current graph state

        Returns:
            Updated state with current platform set
        """
        if not state["target_platforms"]:
            raise ValueError("No target platforms specified")

        # Get first platform that hasn't been processed
        available_platforms = [
            p
            for p in state["target_platforms"]
            if p not in state["processed_platforms"]
        ]

        if not available_platforms:
            return {"next_step": "end"}

        current_platform = available_platforms[0]

        return {
            "current_platform": current_platform,
            "next_step": "generate_content",
            "conversation_history": state["conversation_history"]
            + [f"Starting content generation for {current_platform}"],
        }

    def _generate_content(self, state: GraphState) -> dict:
        """
        Generate platform-specific content based on user topic.

        Args:
            state: Current graph state

        Returns:
            Updated state with generated content
        """
        platform = state["current_platform"]
        topic = state["user_topic"]
        feedback = state.get("user_feedback")

        # Select appropriate generator based on platform
        if platform == "instagram":
            content_result = self.instagram_gen.generate_caption(topic, feedback)
        elif platform == "twitter":
            content_result = self.twitter_gen.generate_post(topic, feedback)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        conversation_entry = (
            f"Generated {platform} content: {content_result.caption}\n"
            f"Hashtags: {content_result.hashtags}"
        )

        return {
            "generated_content": content_result.caption,
            "hashtags": content_result.hashtags,
            "is_satisfied": False,  # User needs to confirm
            "user_feedback": None,  # Clear previous feedback
            "next_step": "get_feedback",
            "conversation_history": state["conversation_history"]
            + [conversation_entry],
        }

    def _get_user_feedback(self, state: GraphState) -> dict:
        """
        Get user feedback on generated content using interrupt for real user input.

        Args:
            state: Current graph state

        Returns:
            Updated state with user feedback and satisfaction
        """
        platform = state["current_platform"]
        content = state["generated_content"]
        hashtags = state["hashtags"]

        # Display current content to user
        print(f"\n📱 {platform.upper()} CONTENT:")
        print(f"{content}")
        print(f"🏷️  HASHTAGS: {', '.join([f'#{tag}' for tag in hashtags])}")

        # Use LangGraph interrupt to get real user input
        try:
            feedback = interrupt(
                "💬 Are you satisfied with this content? "
                "(Type 'yes', 'no' with feedback, or 'next' for another platform): "
            )

            if feedback.lower() in ["yes", "y", ""]:
                return {
                    "is_satisfied": True,
                    "user_feedback": None,
                    "next_step": "next_platform",
                }
            elif feedback.lower() in ["next", "skip"]:
                return {
                    "is_satisfied": False,
                    "user_feedback": None,
                    "next_step": "next_platform",
                }
            else:
                return {
                    "is_satisfied": False,
                    "user_feedback": feedback,
                    "next_step": "refine",
                }

        except Exception as e:
            # Fallback if interrupt not available
            print(f"⚠️  Using simulated feedback: {e}")
            return {
                "is_satisfied": True,  # Auto-approve for demo
                "user_feedback": None,
                "next_step": "next_platform",
            }

    def _refine_content(self, state: GraphState) -> dict:
        """
        Refine content based on user feedback.

        Args:
            state: Current graph state

        Returns:
            Updated state for content regeneration
        """
        return {
            "next_step": "generate_content",
            "conversation_history": state["conversation_history"]
            + [f"Refining content based on: {state['user_feedback']}"],
        }

    def _advance_to_next_platform(self, state: GraphState) -> dict:
        """
        Move to the next platform or end workflow.

        Args:
            state: Current graph state

        Returns:
            Updated state with next platform or end signal
        """
        current_platform = state["current_platform"]
        processed_platforms = state["processed_platforms"] + [current_platform]

        # Check if more platforms need processing
        remaining_platforms = [
            p for p in state["target_platforms"] if p not in processed_platforms
        ]

        if remaining_platforms:
            next_platform = remaining_platforms[0]
            return {
                "current_platform": next_platform,
                "processed_platforms": processed_platforms,
                "next_step": "generate_content",
                "conversation_history": state["conversation_history"]
                + [f"Moving to {next_platform}"],
            }
        else:
            return {"processed_platforms": processed_platforms, "next_step": "end"}

    def _should_refine_content(
        self, state: GraphState
    ) -> Literal["refine", "next_platform", "end"]:
        """Determine next step based on user feedback."""
        if state["next_step"] == "refine":
            return "refine"
        elif state["next_step"] == "next_platform":
            return "next_platform"
        else:
            return "end"

    def _has_more_platforms(self, state: GraphState) -> Literal["continue", "end"]:
        """Check if more platforms need processing."""
        remaining = [
            p
            for p in state["target_platforms"]
            if p not in state["processed_platforms"]
        ]
        return "continue" if remaining else "end"

    def generate_content_workflow(self, user_topic: str, platforms: List[str]) -> dict:
        """
        Execute the complete content generation workflow.

        Args:
            user_topic: The main content topic
            platforms: List of platforms to generate content for

        Returns:
            Final state with all generated content
        """
        initial_state = GraphState(
            user_topic=user_topic,
            current_platform=None,
            target_platforms=platforms,
            generated_content="",
            hashtags=[],
            user_feedback=None,
            is_satisfied=False,
            processed_platforms=[],
            conversation_history=[f"Starting workflow for: {user_topic}"],
            next_step="generate_content",
        )

        return self.graph.invoke(initial_state)
