"""Phonetically diverse calibration and evaluation sentence prompts for VoiceNO.

Specialized for Private Silent Speech Interaction with AI Agents.
Separates real-world AI command calibration prompts from held-out evaluation prompts.
"""

from typing import List, Optional

# Core AI Agent Command Calibration Corpus
# Spans canonical action verbs (summarize, explain, rewrite, convert, search, open, fix),
# targets (code, document, error, image, pdf, calendar), and modifiers.
AI_AGENT_CALIBRATION_PROMPTS: List[str] = [
    # 1. Document & Text Transformation
    "SUMMARIZE THIS DOCUMENT",
    "EXPLAIN WHAT THIS PARAGRAPH MEANS",
    "REWRITE THIS TO SOUND MORE PROFESSIONAL",
    "MAKE THIS SHORTER AND MORE CONCISE",
    "TRANSLATE THIS TEXT TO SPANISH",
    "EXTRACT THE KEY TAKEAWAYS FROM THIS ARTICLE",
    "PROOFREAD THIS FOR GRAMMAR AND SPELLING",
    "FORMAT THIS LIST AS BULLET POINTS",

    # 2. File & Format Conversion
    "CONVERT THIS IMAGE TO PDF",
    "EXTRACT THE TABLES FROM THIS FILE",
    "SAVE THIS AS A MARKDOWN FILE",
    "TRANSFORM THIS DATA INTO JSON",
    "EXPORT THIS NOTEBOOK TO HTML",

    # 3. Code & Technical Assistance
    "WHY DOES THIS CODE FAIL",
    "EXPLAIN THIS ERROR MESSAGE",
    "FIX THE BUG IN THIS FUNCTION",
    "REFACTOR THIS FUNCTION FOR PERFORMANCE",
    "WRITE UNIT TESTS FOR THIS CODE",
    "OPTIMIZE THIS SQL QUERY",
    "EXPLAIN HOW THIS ALGORITHM WORKS",

    # 4. Search, Navigation & Productivity
    "SEARCH FOR THE CHEAPEST OPTION",
    "OPEN MY CALENDAR",
    "FIND RECENT EMAILS ABOUT THE PROJECT",
    "DRAFT A POLITE REPLY TO THIS MESSAGE",
    "CREATE A TASK FOR TOMORROW MORNING",
    "LOOK UP THE DOCUMENTATION FOR THIS API",

    # 5. Inquiry & Reasoning
    "WHAT DOES THIS SYMBOL MEAN",
    "I DONT UNDERSTAND THIS PART",
    "HOW DO I SOLVE THIS ISSUE",
    "COMPARE THESE TWO APPROACHES",
    "GIVE ME THREE ALTERNATIVES FOR THIS",
]

# Held-Out Evaluation Prompts (Disjoint set for measuring generalizability to unseen AI commands)
AI_AGENT_EVALUATION_PROMPTS: List[str] = [
    "REWRITE THIS EMAIL TO BE MORE CASUAL",
    "EXPLAIN THIS CHART AND ITS TRENDS",
    "CONVERT THIS SPREADSHEET TO CSV",
    "DEBUG THIS NULL POINTER EXCEPTION",
    "SEARCH FOR LOCAL RESTAURANTS NEARBY",
    "SCHEDULE A MEETING FOR NEXT MONDAY",
    "SIMPLIFY THIS EXPLANATION FOR BEGINNERS",
    "WHAT ARE THE PROS AND CONS OF THIS OPTION",
    "GENERATE A PYTHON SCRIPT TO AUTOMATE THIS",
    "CLOSE ALL INACTIVE BROWSER TABS",
]

# Legacy pangram prompts retained for research comparison
PANGRAM_CALIBRATION_PROMPTS: List[str] = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS",
    "HOW VEXINGLY QUICK DAFT ZEBRAS JUMP",
    "SPHERES OF GLOSSY BLACK GLASS ARE VERY SMOOTH",
    "BRIGHT SUNLIGHT ILLUMINATES THE OCEAN SURFACE",
]

# Default to AI Agent prompts
CALIBRATION_PROMPTS = AI_AGENT_CALIBRATION_PROMPTS
EVALUATION_PROMPTS = AI_AGENT_EVALUATION_PROMPTS


def get_calibration_prompts(num_prompts: int = 10, category: Optional[str] = None) -> List[str]:
    """Returns a subset of AI agent calibration prompts."""
    return CALIBRATION_PROMPTS[:min(num_prompts, len(CALIBRATION_PROMPTS))]


def get_evaluation_prompts(num_prompts: int = 5) -> List[str]:
    """Returns held-out AI agent evaluation prompts."""
    return EVALUATION_PROMPTS[:min(num_prompts, len(EVALUATION_PROMPTS))]
