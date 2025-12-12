#!/usr/bin/env python3
"""
Main entry point for Medical-ACE system

This demonstrates how to use the Medical Assistant Agent
to analyze medical images using multimodal AI.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents import medical_assistant


def main():
    """Synchronous main function for simple queries."""
    print("🏥 Medical-ACE: Medical Visual Question Answering System")
    print("=" * 60)
    
    # Use the pre-configured agent
    print("\n📦 Loading Medical Assistant Agent...")
    agent = medical_assistant
    print("✅ Agent loaded!\n")
    
    # Example usage
    print("=" * 60)
    print("Example Query:")
    print("=" * 60)
    
    query = """
    I have a pathology image at 'image.png'. 
    Can you analyze it and tell me what type of tissue or organ is shown?
    """
    
    print(f"Query: {query.strip()}\n")
    print("Processing...\n")
    
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        response = result["messages"][-1].content
        
        print("\n" + "=" * 60)
        print("Agent Response:")
        print("=" * 60)
        print(response)
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. OPENAI_API_KEY is set in your .env file")
        print("2. Required models are accessible")
        print("3. Image files exist in the project directory")


async def async_main():
    """Async main function for multiple concurrent queries."""
    print("🏥 Medical-ACE: Medical Visual Question Answering System (Async)")
    print("=" * 60)
    
    # Use the pre-configured agent
    print("\n📦 Loading Medical Assistant Agent...")
    agent = medical_assistant
    print("✅ Agent loaded!\n")
    
    # Multiple test queries
    queries = [
        {
            "query": "Analyze the image at 'image.png'. What type of tissue is shown?",
            "description": "First pathology image analysis"
        },
        {
            "query": "Look at 'image copy.png'. Describe the cellular features visible in this medical image.",
            "description": "Second pathology image analysis"
        }
    ]
    
    # Process all queries
    for i, item in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {item['description']}")
        print(f"{'='*60}")
        print(f"Question: {item['query']}\n")
        
        try:
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": item['query']}]
            })
            response = result["messages"][-1].content
            print(f"\n✅ Response:\n{response}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def interactive_mode():
    """Interactive mode for continuous querying."""
    print("🏥 Medical-ACE: Interactive Mode")
    print("=" * 60)
    print("\nLoading agent...")
    
    agent = medical_assistant
    
    print("✅ Ready! Type your questions (or 'quit' to exit)\n")
    print("Example: Analyze image.png and tell me what organ this is from")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🔬 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n⏳ Processing...\n")
            result = agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            response = result["messages"][-1].content
            print(f"🤖 Agent: {response}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Medical-ACE: AI-powered medical image analysis"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "interactive"],
        default="single",
        help="Execution mode: single (one query), batch (multiple queries), or interactive"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "single":
            main()
        elif args.mode == "batch":
            asyncio.run(async_main())
        elif args.mode == "interactive":
            interactive_mode()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)

