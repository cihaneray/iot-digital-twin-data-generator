# Copilot Instructions

## 0. This project uses uv for python packages.

## 1. Interaction Protocol: Think Before Coding
LLMs often pick an interpretation silently. This protocol forces explicit reasoning to avoid wasted cycles.

* **State Assumptions Explicitly:** If any part of the request is ambiguous, list your assumptions before writing code. If uncertain, ask rather than guess.
* **Present Multiple Interpretations:** If a prompt can be read in two ways, present both. Do not pick one silently.
* **Push Back:** If a requested approach is unnecessarily complex, suggest a simpler alternative.
* **Stop When Confused:** If the logic doesn't add up, name exactly what is unclear and wait for clarification.

## 2. Implementation Philosophy
* **The Senior Smell Test:** Before submitting, ask: *"Would a senior engineer say this is overcomplicated?"* If yes, simplify.
* **Be Surgical:** Modify only the specific lines or blocks necessary to satisfy the request. 
* **Density Over Volume:** If 200 lines of code can be functionally achieved in 50 lines, rewrite it for brevity. 
* **Minimum Viable Code:** Provide the absolute minimum code that solves the problem. No speculative features, no "just in case" logic, and no "gold-plating."
* **Zero Speculation:** Do not add features beyond what was explicitly asked for.
* **Pragmatic Error Handling:** Do not write error handling for "impossible" scenarios or edge cases that weren't requested.

## 3. Code Modification Rules
When editing existing files, adhere to these strict constraints:

* **Touch Only What You Must:** Clean up your own mess, but do not go on a cleaning spree of the surrounding environment.
* **Preserve Context:** * Do **not** "improve" adjacent code, comments, or formatting. 
    * Do **not** refactor things that aren't broken.
* **Match Existing Style:** Mimic the project’s current indentation, naming conventions, and architectural patterns—even if you personally prefer a different style.
* **Dead Code Policy:** If you spot unrelated dead code, mention it in your response, but **do not delete it** unless specifically instructed.

## 4. Constraint Summary
* **Don't assume** it if it's not requested.
* **Don't hide** confusion.
* **Surface tradeoffs** immediately.