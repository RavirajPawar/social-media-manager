"""
Main application using LangGraph for stateful social media content generation.
"""

from graph.social_media_graph import SocialMediaGraph


def main():
    """Run the LangGraph-powered social media content generator."""
    print("\n" + "=" * 60)
    print("🚀 LANGGRAPH SOCIAL MEDIA CONTENT GENERATOR")
    print("=" * 60)
    print("\nI'll help you create engaging content across multiple platforms!")
    print("We'll work on one platform at a time with your feedback.")

    try:
        # Get user input
        user_topic = input(
            "\n💭 What would you like to create content about?\n> "
        ).strip()

        if not user_topic:
            print("❌ Please provide a topic to continue.")
            return

        # Platform selection
        print("\n📱 Available platforms: Instagram, Twitter")
        platform_input = (
            input("Which platforms? (e.g., 'instagram twitter' or 'all'): ")
            .strip()
            .lower()
        )

        if platform_input == "all":
            target_platforms = ["instagram", "twitter"]
        else:
            target_platforms = [
                p for p in platform_input.split() if p in ["instagram", "twitter"]
            ]

        if not target_platforms:
            print("❌ No valid platforms selected. Using Instagram & Twitter.")
            target_platforms = ["instagram", "twitter"]

        print(f"\n🎯 Generating content for: {', '.join(target_platforms)}")
        print(
            "💡 You'll be able to refine content for each platform before moving to the next."
        )
        print("-" * 50)

        # Initialize and run LangGraph workflow
        graph_processor = SocialMediaGraph()
        final_state = graph_processor.generate_content_workflow(
            user_topic, target_platforms
        )

        # Display final results
        print("\n🎉 CONTENT GENERATION COMPLETE!")
        print("=" * 50)
        print(f"📊 Processed {len(final_state['processed_platforms'])} platforms:")
        for platform in final_state["processed_platforms"]:
            print(f"   ✅ {platform.capitalize()}")

        print(
            f"\n💾 Conversation history: {len(final_state['conversation_history'])} entries"
        )

    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Thank you for using the generator!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please try again or check your configuration.")


if __name__ == "__main__":
    main()
