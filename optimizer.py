
import openai
import json
import os
import re

class LLMPromptOptimizer:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set it as an environment variable or pass it to the constructor.")
        openai.api_key = api_key
        self.model = model

    def _call_openai_api(self, messages, temperature=0.7, max_tokens=500):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message["content"]
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return None

    def optimize_prompt(self, original_prompt, target_audience="general", optimization_goals=None):
        if optimization_goals is None:
            optimization_goals = [
                "clarity",
                "conciseness",
                "effectiveness for target audience",
                "avoiding ambiguity"
            ]

        system_message = (
            "You are an expert prompt engineer. Your task is to optimize user-provided prompts "
            f"for a {target_audience} audience, focusing on the following goals: {', '.join(optimization_goals)}. "
            "Provide only the optimized prompt, without any additional commentary or explanation."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Optimize the following prompt: \"" + original_prompt + "\""}
        ]

        optimized_prompt = self._call_openai_api(messages)
        return optimized_prompt

    def analyze_prompt_effectiveness(self, prompt, expected_output_characteristics):
        system_message = (
            "You are an AI assistant specialized in evaluating prompt effectiveness. "
            "Analyze the given prompt and determine how well it is likely to achieve the desired output characteristics. "
            "Provide a score (1-10) and a brief explanation."
        )

        user_message = (
            f"Prompt: \"" + prompt + "\"\n"
            f"Expected Output Characteristics: {expected_output_characteristics}\n"
            "Evaluate its effectiveness. Output format: SCORE: [score]/10\nEXPLANATION: [explanation]"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        analysis = self._call_openai_api(messages)
        return analysis

    def generate_variations(self, prompt, num_variations=3):
        system_message = (
            "You are a creative AI assistant. Generate several alternative phrasings or structures "
            f"for the given prompt, aiming for {num_variations} distinct variations. "
            "Each variation should be presented on a new line, prefixed with a number (e.g., '1. Variation 1')."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate variations for the prompt: \"" + prompt + "\""}
        ]

        variations_raw = self._call_openai_api(messages)
        if variations_raw:
            return [line.strip() for line in variations_raw.split('\n') if line.strip() and re.match(r'^\d+\.', line)]
        return []

if __name__ == '__main__':
    # Ensure you have OPENAI_API_KEY set in your environment variables
    # For testing, you can uncomment the line below and replace with your key
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    try:
        optimizer = LLMPromptOptimizer()

        # Example 1: Optimize a prompt
        original_p = "Tell me about AI."
        optimized_p = optimizer.optimize_prompt(original_p, target_audience="technical", optimization_goals=["depth", "technical accuracy"])
        print(f"\nOriginal Prompt: {original_p}")
        print(f"Optimized Prompt: {optimized_p}")

        # Example 2: Analyze prompt effectiveness
        analysis_result = optimizer.analyze_prompt_effectiveness(
            optimized_p or original_p,
            "A comprehensive, technically accurate overview of AI, including its subfields and challenges."
        )
        print(f"\nPrompt Effectiveness Analysis:\n{analysis_result}")

        # Example 3: Generate variations
        variations = optimizer.generate_variations(original_p)
        print(f"\nVariations for \"{original_p}\":")
        for i, var in enumerate(variations):
            print(f"  {i+1}. {var}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Commit timestamp: 2023-10-26 00:00:00 - 295
