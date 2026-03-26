from __future__ import annotations


SPEC_DOCSTRING = """
Single-machine total tardiness scheduling.

Input:
- processing_times: list[int]
- due_dates: list[int]

Output:
- schedule: list[int], permutation of all job indices [0..n-1]

Hard constraints:
- Assign each job exactly once.
- Do not mutate processing_times or due_dates.
- Do not create/delete jobs.
"""


def assignment_template(processing_times: list[int], due_dates: list[int]) -> list[int]:
    """Simple EDD baseline template for LLM mutations."""
    return sorted(range(len(processing_times)), key=lambda i: (due_dates[i], processing_times[i]))
