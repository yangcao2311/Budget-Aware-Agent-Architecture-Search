"""Prompt registry. prompt_id -> {family -> template}.

Templates receive: task (problem statement), solution (current best, may be
empty), feedback (verifier feedback, may be empty).
Answer format contracts: code inside one ```python fence; math final answer
inside \\boxed{}.
"""

CODE_FORMAT = ("Write a self-contained Python function solving the task. "
               "Return ONLY one ```python code block with the complete solution.")
MATH_FORMAT = ("End your response with the final answer in \\boxed{...}.")

PROMPTS = {
    "solve_direct": {
        "code": "{task}\n\n" + CODE_FORMAT,
        "math": "{task}\n\n" + MATH_FORMAT,
    },
    "solve_cot": {
        "code": ("{task}\n\nFirst reason step by step about edge cases and the "
                 "algorithm, then " + CODE_FORMAT),
        "math": ("{task}\n\nReason step by step, checking each step. " + MATH_FORMAT),
    },
    "self_check": {
        "code": ("Task:\n{task}\n\nCandidate solution:\n{solution}\n\n"
                 "Carefully review the candidate for bugs and edge cases. If it is "
                 "correct, return it unchanged; otherwise fix it. " + CODE_FORMAT),
        "math": ("Problem:\n{task}\n\nCandidate solution:\n{solution}\n\n"
                 "Verify each step. If correct, restate the answer; otherwise "
                 "redo the solution. " + MATH_FORMAT),
    },
    "refine_from_feedback": {
        "code": ("Task:\n{task}\n\nPrevious attempt:\n{solution}\n\n"
                 "Automatic test feedback:\n{feedback}\n\n"
                 "Fix the solution so all tests pass. " + CODE_FORMAT),
        "math": ("Problem:\n{task}\n\nPrevious attempt:\n{solution}\n\n"
                 "Feedback:\n{feedback}\n\n"
                 "Rework the solution carefully. " + MATH_FORMAT),
    },
    "decompose": {
        "code": ("{task}\n\nBreak this task into 2-4 subproblems (input handling, "
                 "core algorithm, edge cases). For each, sketch the approach in "
                 "2-3 lines. Do not write the final solution yet."),
        "math": ("{task}\n\nBreak this problem into 2-4 subproblems and solve "
                 "each briefly. Do not state the final answer yet."),
    },
    "aggregate_sub": {
        "code": ("Task:\n{task}\n\nYour subproblem analysis:\n{solution}\n\n"
                 "Now write the complete final solution. " + CODE_FORMAT),
        "math": ("Problem:\n{task}\n\nYour subproblem work:\n{solution}\n\n"
                 "Combine the pieces into a complete solution. " + MATH_FORMAT),
    },
    "check_math": {
        # Gold-free internal verifier for math: independent re-derivation.
        # The runner compares the checker's boxed answer with the candidate's.
        "math": ("Solve this problem completely independently, step by step. "
                 "Do not assume any previous attempt is correct.\n\n"
                 "Problem:\n{task}\n\n" + MATH_FORMAT),
    },
}


def render(prompt_id: str, family: str, task: str, solution: str = "", feedback: str = "") -> str:
    # Literal replace, not str.format: templates and task text contain LaTeX
    # braces (\boxed{...}) that format() would misparse.
    out = PROMPTS[prompt_id][family]
    for k, v in (("{task}", task), ("{solution}", solution), ("{feedback}", feedback)):
        out = out.replace(k, v)
    return out
