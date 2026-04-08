# LLM Prompt Optimizer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-API-green)
![License](https://img.shields.io/badge/license-Apache_2.0-blue)

An automated framework for discovering the optimal prompt for a given task using evolutionary algorithms and LLM-as-a-Judge evaluation.

## Features
- Genetic algorithm for prompt mutation and crossover
- Integration with OpenAI, Anthropic, and local LLMs
- Automated evaluation metrics (accuracy, relevance, toxicity)
- Checkpointing and experiment tracking

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from prompt_optimizer import Optimizer
from prompt_optimizer.evaluators import ExactMatchEvaluator

optimizer = Optimizer(
    initial_prompts=["Translate this to French:", "Provide a French translation:"],
    evaluator=ExactMatchEvaluator(dataset="translation_data.json"),
    llm_client="openai",
    model="gpt-4"
)
best_prompt = optimizer.run(generations=10)
print(f"Best Prompt: {best_prompt}")
```
