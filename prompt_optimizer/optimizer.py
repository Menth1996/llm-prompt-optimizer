import random
import time
from typing import List, Callable

class Optimizer:
    """
    Evolutionary algorithm for prompt optimization.
    """
    def __init__(self, initial_prompts: List[str], evaluator: Callable, llm_client: str, model: str, population_size: int = 10):
        self.population = initial_prompts
        self.evaluator = evaluator
        self.llm_client = llm_client
        self.model = model
        self.population_size = population_size
        
        # Ensure initial population is up to size
        while len(self.population) < self.population_size:
            self.population.append(random.choice(initial_prompts))

    def _mutate(self, prompt: str) -> str:
        """
        Simulates mutating a prompt using an LLM.
        In a real scenario, this would call an LLM to rephrase or improve the prompt.
        """
        mutations = [
            " Please be concise.",
            " Think step-by-step.",
            " Act as an expert.",
            " Ensure high accuracy."
        ]
        return prompt + random.choice(mutations)

    def _crossover(self, prompt1: str, prompt2: str) -> str:
        """
        Simulates crossing over two prompts.
        """
        words1 = prompt1.split()
        words2 = prompt2.split()
        mid1 = len(words1) // 2
        mid2 = len(words2) // 2
        return " ".join(words1[:mid1] + words2[mid2:])

    def run(self, generations: int = 5) -> str:
        """
        Runs the evolutionary optimization loop.
        """
        print(f"Starting optimization for {generations} generations...")
        
        for gen in range(generations):
            print(f"--- Generation {gen + 1} ---")
            
            # Evaluate current population
            scores = []
            for prompt in self.population:
                score = self.evaluator(prompt)
                scores.append((prompt, score))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Best score this generation: {scores[0][1]:.4f} | Prompt: '{scores[0][0]}'")
            
            # Select top 50%
            survivors = [p[0] for p in scores[:self.population_size // 2]]
            
            # Create next generation
            next_gen = list(survivors)
            
            # Mutate and crossover to fill the rest
            while len(next_gen) < self.population_size:
                if random.random() < 0.5:
                    # Mutate
                    parent = random.choice(survivors)
                    next_gen.append(self._mutate(parent))
                else:
                    # Crossover
                    p1, p2 = random.sample(survivors, 2)
                    next_gen.append(self._crossover(p1, p2))
                    
            self.population = next_gen
            time.sleep(0.1) # Simulate processing time
            
        # Final evaluation
        final_scores = [(p, self.evaluator(p)) for p in self.population]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nOptimization complete!")
        return final_scores[0][0]

# Dummy evaluator for testing
def dummy_evaluator(prompt: str) -> float:
    # Reward longer prompts and specific keywords for demonstration
    score = len(prompt) * 0.01
    if "expert" in prompt.lower(): score += 0.5
    if "step-by-step" in prompt.lower(): score += 0.8
    return min(score, 1.0) # Max score 1.0

if __name__ == "__main__":
    opt = Optimizer(
        initial_prompts=["Translate this text:", "Summarize the following:"],
        evaluator=dummy_evaluator,
        llm_client="mock",
        model="mock-v1"
    )
    best = opt.run(generations=3)
    print(f"\nFinal Best Prompt: {best}")
