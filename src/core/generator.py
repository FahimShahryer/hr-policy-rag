"""
LLM Generator with Simple Conversation Memory
Uses Gemini 2.0 Flash with buffer window memory (last 5 Q&A pairs)
"""

from typing import List, Dict, Any, Optional
import google.generativeai as genai
from config.settings import settings


class ConversationMemory:
    """
    Simple buffer window memory for conversation history

    Stores last N question-answer pairs per session
    No persistence, no vector storage - just simple in-memory buffer
    """

    def __init__(self, max_pairs: int = 5):
        """
        Initialize memory

        Args:
            max_pairs: Maximum Q&A pairs to keep (default: 5)
        """
        self.max_pairs = max_pairs
        self.max_messages = max_pairs * 2  # Each pair = 2 messages (Q + A)
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        """
        Add user question to history

        Args:
            message: User question
        """
        self.history.append({
            "role": "user",
            "content": message
        })
        self._trim_history()

    def add_assistant_message(self, message: str):
        """
        Add assistant response to history

        Args:
            message: Assistant response
        """
        self.history.append({
            "role": "assistant",
            "content": message
        })
        self._trim_history()

    def _trim_history(self):
        """
        Keep only last N messages (N = max_pairs * 2)
        """
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get full conversation history

        Returns:
            List of messages with role and content
        """
        return self.history.copy()

    def get_history_text(self) -> str:
        """
        Format history as readable text for prompt

        Returns:
            str: Formatted conversation history
        """
        if not self.history:
            return "No previous conversation."

        formatted = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def clear(self):
        """
        Clear all conversation history
        """
        self.history = []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics

        Returns:
            Dict with memory stats
        """
        return {
            "max_pairs": self.max_pairs,
            "max_messages": self.max_messages,
            "current_messages": len(self.history),
            "current_pairs": len(self.history) // 2
        }


class LLMGenerator:
    """
    Production-grade LLM generator with Gemini 2.0 Flash

    Features:
    - Gemini 2.0 Flash (fast, cost-effective)
    - Simple conversation memory (last 5 Q&A pairs)
    - Comprehensive error handling
    - Configurable parameters
    - Safety settings
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        memory_max_pairs: Optional[int] = None
    ):
        """
        Initialize LLM generator

        Args:
            model_name: Gemini model name (default from settings)
            temperature: Sampling temperature (default from settings)
            max_output_tokens: Max tokens in response (default from settings)
            top_p: Nucleus sampling parameter (default from settings)
            top_k: Top-k sampling parameter (default from settings)
            memory_max_pairs: Max conversation pairs to remember (default from settings)
        """
        # Configuration
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.max_output_tokens = max_output_tokens or settings.LLM_MAX_OUTPUT_TOKENS
        self.top_p = top_p or settings.LLM_TOP_P
        self.top_k = top_k or settings.LLM_TOP_K

        # Memory
        max_pairs = memory_max_pairs or settings.MEMORY_MAX_PAIRS
        self.memory = ConversationMemory(max_pairs=max_pairs)

        # Model (lazy loaded)
        self._model = None
        self._is_connected = False

    def connect(self) -> bool:
        """
        Connect to Gemini API and initialize model

        Returns:
            bool: True if successful

        Raises:
            Exception: If connection fails
        """
        try:
            # Check API key
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")

            # Configure Gemini
            genai.configure(api_key=settings.GEMINI_API_KEY)

            # Initialize model with generation config
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
            }

            # Safety settings (moderate - not too restrictive)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            print(f"[OK] Connected to Gemini: {self.model_name}")
            print(f"[OK] Temperature: {self.temperature}")
            print(f"[OK] Memory: Last {self.memory.max_pairs} Q&A pairs")

            self._is_connected = True
            return True

        except Exception as e:
            print(f"[ERROR] Failed to connect generator: {str(e)}")
            raise

    def generate(
        self,
        question: str,
        context: str,
        use_memory: bool = True
    ) -> str:
        """
        Generate response to user question

        Args:
            question: User question
            context: Retrieved context from documents
            use_memory: Whether to include conversation history (default: True)

        Returns:
            str: Generated response

        Raises:
            RuntimeError: If generator not connected
            Exception: If generation fails
        """
        if not self._is_connected:
            raise RuntimeError("Generator not connected. Call connect() first.")

        try:
            # Get conversation history
            history_text = ""
            if use_memory:
                history_text = self.memory.get_history_text()

            # Build prompt using template
            prompt = settings.SYSTEM_PROMPT.format(
                context=context,
                history=history_text,
                question=question
            )

            # Generate response
            response = self._model.generate_content(prompt)

            # Check if response was blocked or empty
            try:
                answer = response.text.strip()

                if not answer:
                    # Response is empty
                    if hasattr(response, 'prompt_feedback'):
                        return f"I apologize, but I cannot answer this question due to safety filters. Reason: {response.prompt_feedback}"
                    return "I apologize, but I cannot generate a response for this question."

            except (ValueError, AttributeError) as e:
                # response.text raised an exception (safety block or no valid parts)
                print(f"[WARN] Response blocked or invalid: {str(e)}")

                # Check finish reason
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason

                    if finish_reason == 2:  # SAFETY
                        return "I apologize, but I cannot answer this question as it may have been flagged by safety filters. Please try rephrasing your question."
                    elif finish_reason == 3:  # RECITATION
                        return "I apologize, but I cannot provide this response due to content policy restrictions."
                    else:
                        return f"I apologize, but I cannot generate a response (finish_reason: {finish_reason})."

                # Generic fallback
                return "I apologize, but I cannot generate a response for this question. Please try rephrasing."

            # Update memory
            if use_memory:
                self.memory.add_user_message(question)
                self.memory.add_assistant_message(answer)

            return answer

        except Exception as e:
            print(f"[ERROR] Generation failed: {str(e)}")
            # Return user-friendly error
            return settings.ERROR_RESPONSE

    def generate_with_fallback(
        self,
        question: str,
        context: str,
        use_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response with error handling and fallback

        Args:
            question: User question
            context: Retrieved context
            use_memory: Whether to include conversation history

        Returns:
            Dict with 'response', 'success', and optional 'error'
        """
        try:
            # Handle empty context
            if not context or context == "No relevant information found.":
                response = settings.NO_CONTEXT_RESPONSE
                return {
                    "response": response,
                    "success": True,
                    "warning": "No relevant context found"
                }

            # Generate response
            response = self.generate(question, context, use_memory)

            return {
                "response": response,
                "success": True
            }

        except Exception as e:
            print(f"[ERROR] Generation with fallback failed: {str(e)}")
            return {
                "response": settings.ERROR_RESPONSE,
                "success": False,
                "error": str(e)
            }

    def clear_memory(self):
        """
        Clear conversation history
        """
        self.memory.clear()
        print("[INFO] Conversation memory cleared")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics

        Returns:
            Dict with memory stats
        """
        return self.memory.get_stats()

    def health_check(self) -> Dict[str, Any]:
        """
        Check generator health status

        Returns:
            Dict with health status information
        """
        status = {
            "connected": self._is_connected,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "memory_stats": self.get_memory_stats()
        }

        return status

    def close(self):
        """
        Cleanup resources
        """
        self.memory.clear()
        self._model = None
        self._is_connected = False
        print("[INFO] Generator closed")


# ========== Testing ==========

def test_generator():
    """
    Test generator functionality
    """
    print("=" * 80)
    print("  TESTING LLM GENERATOR")
    print("=" * 80)
    print()

    try:
        # Initialize and connect
        generator = LLMGenerator()
        print("[1/4] Connecting to generator...")
        generator.connect()
        print()

        # Test context
        test_context = """[Document 1]
Section: 5 - Vacation and Leave
Pages 15-16
Content: All full-time employees are entitled to 15 days of paid vacation per year. Vacation days accrue at a rate of 1.25 days per month. Employees must request vacation at least 2 weeks in advance.

[Document 2]
Section: 5 - Vacation and Leave
Pages 16-17
Content: Unused vacation days can be carried over to the next year, up to a maximum of 5 days. Vacation days cannot be cashed out."""

        # Test question 1
        test_question_1 = "How many vacation days do I get per year?"
        print(f"[2/4] Testing Question 1: '{test_question_1}'")
        response_1 = generator.generate(test_question_1, test_context)
        print(f"\nResponse 1:\n{response_1}\n")

        # Test question 2 (with memory)
        test_question_2 = "Can I carry them over to next year?"
        print(f"[3/4] Testing Question 2 (with memory): '{test_question_2}'")
        response_2 = generator.generate(test_question_2, test_context)
        print(f"\nResponse 2:\n{response_2}\n")

        # Memory stats
        print("[4/4] Memory Stats:")
        stats = generator.get_memory_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nConversation History:")
        print(generator.memory.get_history_text())

        # Health check
        print("\n" + "=" * 80)
        print("HEALTH CHECK:")
        print("=" * 80)
        health = generator.health_check()
        for key, value in health.items():
            print(f"  {key}: {value}")

        print("\n[SUCCESS] Generator test completed!")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_generator()
