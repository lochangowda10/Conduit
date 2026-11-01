"""Synthetic, *labelled* LLM workload for benchmarking.

Why synthetic: real chat logs are private, and we need ground truth. Every
prompt carries a label ``"intent:entity"``; two prompts with the same label
are paraphrases (a cache hit between them is correct), two prompts with
different labels are not (a cache hit between them is a *false hit*, i.e.
the user got someone else's answer).

The workload is designed to be adversarial for a semantic cache:
* paraphrase families ("What's the capital of X" / "Tell me X's capital"),
* near-miss pairs that share almost every word but differ in one entity
  ("Explain recursion simply" vs "Explain hash tables simply",
  "12 times 7" vs "12 times 8"),
* a long tail of one-off prompts that should never hit,
* Zipf-distributed popularity, because real traffic is skewed: a few
  questions are asked constantly, most are asked once.

Syllabus link: Experiment 6 - ``generate_workload`` is a generator, so a
million-request workload costs the same memory as a ten-request one.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterator

INTENTS: dict[str, dict] = {
    "capital": {
        "templates": [
            "What is the capital of {e}?",
            "What's the capital city of {e}?",
            "Tell me the capital of {e}.",
            "Which city is the capital of {e}?",
        ],
        "entities": ["France", "Spain", "Japan", "Brazil", "Kenya", "Canada",
                     "Italy", "Egypt", "India", "Germany", "Peru", "Norway"],
    },
    "explain": {
        "templates": [
            "Explain {e} in simple terms.",
            "Can you explain {e} simply?",
            "Give me a simple explanation of {e}.",
            "Explain {e} like I'm a beginner.",
        ],
        "entities": ["gradient descent", "backpropagation", "overfitting", "the attention mechanism",
                     "recursion", "hash tables", "binary search", "dynamic programming",
                     "garbage collection", "database indexing", "TCP handshakes", "public-key cryptography"],
    },
    "python": {
        "templates": [
            "Write a Python function to {e}.",
            "How do I {e} in Python?",
            "Python code to {e}, please.",
            "Show me how to {e} using Python.",
        ],
        "entities": ["reverse a linked list", "check if a string is a palindrome", "merge two sorted lists",
                     "find duplicates in a list", "flatten a nested list", "count word frequencies",
                     "read a CSV file", "validate an email address", "reverse a string", "sort a dictionary by value"],
    },
    "translate": {
        "templates": [
            'Translate "{e}" to French.',
            'How do you say "{e}" in French?',
            'What is "{e}" in French?',
        ],
        "entities": ["good morning", "thank you very much", "where is the station", "I am hungry",
                     "see you tomorrow", "how much does this cost"],
    },
    "arithmetic": {
        "templates": [
            "What is {a} times {b}?",
            "Calculate {a} * {b}.",
            "What do you get when you multiply {a} by {b}?",
        ],
        "entities": [(12, 7), (12, 8), (25, 4), (13, 17), (99, 3), (45, 6), (14, 7)],
    },
    "history": {
        "templates": [
            "Summarize the history of {e} in three sentences.",
            "Give me a three-sentence summary of the history of {e}.",
            "In three sentences, what is the history of {e}?",
        ],
        "entities": ["the internet", "Python", "the printing press", "the Roman Empire", "jazz music", "the Olympic Games"],
    },
    "design": {  # hard prompts -> should route to the large tier
        "templates": [
            "Design a {e} and compare the trade-offs of at least two approaches for scalability, consistency and fault tolerance. Analyze the failure modes step by step.",
            "How would you architect a {e}? Evaluate the trade-offs, analyze failure modes and the complexity of each component step by step, and justify the design for scale.",
            "Walk me through the system design of a {e}: compare alternative architectures, analyze bottlenecks and trade-offs, and explain how it scales in a distributed setting.",
        ],
        "entities": ["distributed rate limiter", "URL shortener", "real-time chat system",
                     "news feed ranking service", "distributed job scheduler", "payment ledger"],
    },
    "debug": {  # medium/large: contains code
        "templates": [
            "Debug this Python code and explain why it fails: ```{e}```",
            "Why does this code not work? Analyze it and suggest a fix: ```{e}```",
        ],
        "entities": [
            "def avg(xs): return sum(xs) / len(xs)\nprint(avg([]))",
            "items = [1, 2, 3]\nfor i in range(len(items) + 1): print(items[i])",
            "def add(x, acc=[]): acc.append(x); return acc",
            "import threading\ncount = 0\ndef inc():\n    global count\n    for _ in range(10**6): count += 1",
        ],
    },
}

# Ground truth for the routing-quality proxy: which tier each kind of task really needs.
REQUIRED_TIER = {"design": "large", "debug": "medium"}  # everything else: "small"

_ADJ = ["quiet", "neon", "rusty", "brave", "tiny", "cosmic", "lazy", "golden", "frozen", "electric",
        "hidden", "wild", "silver", "humble", "rapid", "ancient"]
_NOUN = ["teapot", "lighthouse", "robot", "garden", "violin", "glacier", "lantern", "bicycle",
         "notebook", "comet", "harbor", "orchard", "satellite", "bakery", "compass", "volcano"]
_TAIL_TEMPLATES = [
    "Write a two-line poem about a {a} {n}.",
    "Suggest a name for a startup that sells {a} {n}s.",
    "Describe a {a} {n} in one vivid sentence.",
]


# Real users wrap the same question in different filler, so exact string
# repeats are rarer than "same question" repeats.
_PREFIXES = ["", "", "", "Hey, ", "Quick question: ", "Please help: ", "Hi! ", "I need help. "]
_SUFFIXES = ["", "", "", " Thanks!", " Keep it short.", " Thank you.", " Be concise."]


def _surface_noise(prompt: str, rng: random.Random) -> str:
    prompt = rng.choice(_PREFIXES) + prompt + rng.choice(_SUFFIXES)
    if rng.random() < 0.1:
        prompt = prompt.lower()  # people often don't capitalise
    return prompt


@dataclass(frozen=True)
class WorkItem:
    prompt: str
    label: str      # ground truth: equal labels <=> semantically equivalent
    intent: str


def _families() -> list[tuple[str, list[str], str]]:
    """Every (intent, entity) pair, with all of its paraphrase variants."""
    fams = []
    for intent, spec in INTENTS.items():
        for ent in spec["entities"]:
            if intent == "arithmetic":
                a, b = ent
                variants = [t.format(a=a, b=b) for t in spec["templates"]]
                label = f"{intent}:{a}x{b}"
            else:
                variants = [t.format(e=ent) for t in spec["templates"]]
                label = f"{intent}:{ent}"
            fams.append((intent, variants, label))
    return fams


def generate_workload(n: int, seed: int = 7, zipf_s: float = 1.1, tail_fraction: float = 0.2,
                      noise: bool = True) -> Iterator[WorkItem]:
    """Yield ``n`` labelled prompts.

    * ``tail_fraction`` of requests are unique one-offs (never repeat).
    * The rest pick a family with Zipf popularity, then a random paraphrase,
      then (if ``noise``) random greeting/sign-off filler and casing.
    """
    rng = random.Random(seed)
    families = _families()
    rng.shuffle(families)  # which family is "most popular" is random but reproducible
    weights = [1.0 / (rank ** zipf_s) for rank in range(1, len(families) + 1)]
    tail = [(t, a, nn) for t in _TAIL_TEMPLATES for a in _ADJ for nn in _NOUN]
    rng.shuffle(tail)
    tail_iter = itertools.cycle(tail)
    for i in range(n):
        if rng.random() < tail_fraction:
            t, a, nn = next(tail_iter)
            yield WorkItem(t.format(a=a, n=nn), f"tail:{i}", "tail")
        else:
            intent, variants, label = rng.choices(families, weights=weights, k=1)[0]
            prompt = rng.choice(variants)
            yield WorkItem(_surface_noise(prompt, rng) if noise else prompt, label, intent)


def labelled_pairs() -> Iterator[tuple[str, str, bool]]:
    """Every pair of prompts within each intent, with whether they are equivalent.

    Used to measure an embedder's raw separability, independent of cache order.
    """
    by_intent: dict[str, list[tuple[str, str]]] = {}
    for intent, variants, label in _families():
        by_intent.setdefault(intent, []).extend((v, label) for v in variants)
    for items in by_intent.values():
        for (p1, l1), (p2, l2) in itertools.combinations(items, 2):
            yield p1, p2, l1 == l2


if __name__ == "__main__":
    for item in itertools.islice(generate_workload(15), 15):
        print(f"{item.label:<40} {item.prompt[:70]}")
