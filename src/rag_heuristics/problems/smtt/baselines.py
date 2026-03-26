from __future__ import annotations


def edd(processing_times: list[int], due_dates: list[int]) -> list[int]:
    return sorted(range(len(processing_times)), key=lambda i: (due_dates[i], processing_times[i]))


def spt(processing_times: list[int], due_dates: list[int]) -> list[int]:
    del due_dates
    return sorted(range(len(processing_times)), key=lambda i: processing_times[i])


def mdd(processing_times: list[int], due_dates: list[int]) -> list[int]:
    unscheduled = set(range(len(processing_times)))
    schedule: list[int] = []
    current_time = 0
    while unscheduled:
        best = min(unscheduled, key=lambda j: max(due_dates[j], current_time + processing_times[j]))
        schedule.append(best)
        unscheduled.remove(best)
        current_time += processing_times[best]
    return schedule


def mddc_like(processing_times: list[int], due_dates: list[int]) -> list[int]:
    unscheduled = set(range(len(processing_times)))
    schedule: list[int] = []
    current_time = 0.0
    while unscheduled:
        jobs = list(unscheduled)
        p_u = [processing_times[j] for j in jobs]
        d_u = [due_dates[j] for j in jobs]
        p_max = max(p_u)
        p_avg = sum(p_u) / len(p_u)
        scores: dict[int, float] = {}
        for idx, j in enumerate(jobs):
            mu = max(1.1 * p_u[idx] + current_time, d_u[idx])
            rho = min(p_u[idx] / (current_time + p_max), 1.0)
            theta = (rho**2) / (1.0 + rho**2)
            sigma = p_u[idx] / (current_time + p_avg)
            scores[j] = float(mu * (1 + theta) + sigma)
        best = min(scores, key=scores.get)
        schedule.append(best)
        unscheduled.remove(best)
        current_time += processing_times[best]
    return schedule
