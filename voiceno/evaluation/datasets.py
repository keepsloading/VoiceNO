"""Diverse open-vocabulary evaluation benchmarks for VoiceNO.

Covers 6 distinct linguistic categories:
1. Short utterances
2. Normal natural-language sentences
3. Longer complex sentences
4. Conversational natural language
5. Unusual / non-templated phrasing
6. Viseme-ambiguous vocabulary pairs
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalBenchmarkItem:
    """A benchmark sentence with category and contextual ground truth."""
    id: str
    category: str
    prompt: str
    expected_viseme_ambiguities: List[str] = field(default_factory=list)
    task_context: Optional[Dict[str, str]] = None


# Open-vocabulary benchmark suite
OPEN_VOCABULARY_BENCHMARK: List[EvalBenchmarkItem] = [
    # 1. Short Utterances
    EvalBenchmarkItem(
        id="short_01",
        category="short_utterance",
        prompt="OPEN THIS",
        expected_viseme_ambiguities=["HOPE THIS", "OPEN THIS"],
        task_context={"active_app": "browser", "action": "open"},
    ),
    EvalBenchmarkItem(
        id="short_02",
        category="short_utterance",
        prompt="EXPLAIN THIS",
        expected_viseme_ambiguities=["EXPLAIN THIS", "EXPAND THIS"],
        task_context={"active_app": "code_editor", "action": "explain"},
    ),
    EvalBenchmarkItem(
        id="short_03",
        category="short_utterance",
        prompt="SEARCH THIS",
        expected_viseme_ambiguities=["SEARCH THIS", "SUCH THIS"],
        task_context={"active_app": "browser", "action": "search"},
    ),
    EvalBenchmarkItem(
        id="short_04",
        category="short_utterance",
        prompt="REWRITE THIS",
        expected_viseme_ambiguities=["REWRITE THIS", "READ THIS"],
        task_context={"active_app": "document_viewer", "action": "rewrite"},
    ),
    EvalBenchmarkItem(
        id="short_05",
        category="short_utterance",
        prompt="MAKE THIS SHORTER",
        expected_viseme_ambiguities=["TAKE THIS SHORTER", "MAKE THIS SHORTER"],
        task_context={"active_app": "document_viewer", "domain_keywords": ["shorter", "concise"]},
    ),

    # 2. Normal Natural-Language Sentences
    EvalBenchmarkItem(
        id="normal_01",
        category="normal_sentence",
        prompt="CAN YOU SUMMARIZE THIS ARTICLE FOR ME",
        expected_viseme_ambiguities=["SUMMARIZE THIS ARTICLE"],
        task_context={"active_app": "browser", "selected_text": "article text about technology"},
    ),
    EvalBenchmarkItem(
        id="normal_02",
        category="normal_sentence",
        prompt="PLEASE TRANSLATE THIS TO SPANISH",
        expected_viseme_ambiguities=["TRANSLATE THIS"],
        task_context={"active_app": "document_viewer", "domain_keywords": ["spanish", "translate"]},
    ),
    EvalBenchmarkItem(
        id="normal_03",
        category="normal_sentence",
        prompt="OPEN MY CALENDAR FOR TOMORROW",
        expected_viseme_ambiguities=["OPEN MY CALENDAR"],
        task_context={"active_app": "calendar", "action": "calendar"},
    ),

    # 3. Longer Complex Sentences
    EvalBenchmarkItem(
        id="long_01",
        category="longer_sentence",
        prompt="EXPLAIN WHY THIS PARAGRAPH CONTRADICTS THE PREVIOUS SECTION",
        expected_viseme_ambiguities=["CONTRADICTS", "PREVIOUS SECTION"],
        task_context={"active_app": "document_viewer", "selected_text": "paragraph contradicts the conclusion"},
    ),
    EvalBenchmarkItem(
        id="long_02",
        category="longer_sentence",
        prompt="REWRITE THIS TO SOUND MORE PROFESSIONAL AND CONCISE",
        expected_viseme_ambiguities=["PROFESSIONAL AND CONCISE"],
        task_context={"active_app": "document_viewer", "domain_keywords": ["professional", "rewrite"]},
    ),
    EvalBenchmarkItem(
        id="long_03",
        category="longer_sentence",
        prompt="SEARCH FOR THE CHEAPEST OPTION AVAILABLE ONLINE",
        expected_viseme_ambiguities=["CHEAPEST OPTION", "SHEEPEST OPTION"],
        task_context={"active_app": "browser", "domain_keywords": ["cheapest", "option", "price"]},
    ),

    # 4. Conversational Natural Language
    EvalBenchmarkItem(
        id="conv_01",
        category="conversational",
        prompt="I DONT REALLY UNDERSTAND WHAT THIS MEANS",
        expected_viseme_ambiguities=["DONT UNDERSTAND", "WHAT THIS MEANS"],
        task_context={"active_app": "general", "domain_keywords": ["understand", "means"]},
    ),
    EvalBenchmarkItem(
        id="conv_02",
        category="conversational",
        prompt="WHAT DO YOU THINK ABOUT THIS OPTION",
        expected_viseme_ambiguities=["THINK ABOUT", "THIS OPTION"],
        task_context={"active_app": "browser", "domain_keywords": ["option"]},
    ),

    # 5. Unusual / Non-Templated Phrasing
    EvalBenchmarkItem(
        id="unusual_01",
        category="unusual_phrasing",
        prompt="WHY DOES THIS CODE FAIL ON EVERY ATTEMPT",
        expected_viseme_ambiguities=["CODE FAIL", "ATTEMPT"],
        task_context={"active_app": "code_editor", "selected_text": "TypeError: null pointer in function attempt"},
    ),
    EvalBenchmarkItem(
        id="unusual_02",
        category="unusual_phrasing",
        prompt="WHERE IS THE NEAREST COFFEE SHOP FROM HERE",
        expected_viseme_ambiguities=["NEAREST COFFEE", "SHOP"],
        task_context={"active_app": "browser", "domain_keywords": ["coffee", "map", "nearest"]},
    ),

    # 6. Viseme-Ambiguous Vocabulary Pairs
    EvalBenchmarkItem(
        id="ambig_01",
        category="viseme_ambiguous",
        prompt="I WANT TO GO TO THE MALL",
        expected_viseme_ambiguities=["MALL", "BALL", "HALL", "FALL"],
        task_context={"active_app": "browser", "domain_keywords": ["shopping", "mall", "stores"]},
    ),
    EvalBenchmarkItem(
        id="ambig_02",
        category="viseme_ambiguous",
        prompt="I AM DRINKING WATER",
        expected_viseme_ambiguities=["DRINKING WATER", "GET WATER", "TAKING WATER"],
        task_context={"active_app": "general", "domain_keywords": ["drinking", "water"]},
    ),
    EvalBenchmarkItem(
        id="ambig_03",
        category="viseme_ambiguous",
        prompt="PLEASE PACK MY BOX CAREFULLY",
        expected_viseme_ambiguities=["PACK", "BACK", "BLACK"],
        task_context={"active_app": "general", "domain_keywords": ["pack", "box"]},
    ),
]


def get_open_vocab_benchmark() -> List[EvalBenchmarkItem]:
    """Returns complete open-vocabulary benchmark suite."""
    return list(OPEN_VOCABULARY_BENCHMARK)
