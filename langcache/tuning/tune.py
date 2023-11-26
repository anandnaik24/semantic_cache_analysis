from typing import List
from langcache.statistics.simple import SimpleStatistics


def tune(stats_list: List[SimpleStatistics], policy: str, present_threshold, append_flag, sensitivity_fn, sensitivity_fp):
    if policy == "precision":
        return tune_precision(stats_list)
    elif policy == "recall":
        return tune_recall(stats_list)
    elif policy == "balance":
        return tune_balance(stats_list)
    elif policy == "dynamic":
        return tune_dynamic(append_flag, present_threshold, stats_list, sensitivity_fn, sensitivity_fp)
    else:
        raise NotImplementedError(f"{policy} is not supported")


def tune_precision(stats_list: List[SimpleStatistics]):
    distance_threshold = 2
    for stats in stats_list:
        if stats.fp > 0:
            distance_threshold = min(distance_threshold, stats.distance - 0.001)
    return distance_threshold


def tune_recall(stats_list: List[SimpleStatistics]):
    distance_threshold = 0x0
    for stats in stats_list:
        if stats.fn > 0:
            distance_threshold = max(distance_threshold, stats.distance + 0.001)
    return distance_threshold


def tune_balance(stats_list: List[SimpleStatistics]):
    # Simply take an avarage of two extreme for now.
    return (tune_precision(stats_list) + tune_recall(stats_list)) / 2


def tune_dynamic(append_flag, present_threshold, stats_list: List[SimpleStatistics], sensitivity_fn, sensitivity_fp):

    new_threshold = present_threshold

    if append_flag:
        stats = stats_list[-1]
        print("New stats added")
    else:
        return new_threshold

    if stats.fn > 0:
        new_threshold = present_threshold + (stats.distance - present_threshold) * sensitivity_fn
    if stats.fp > 0:
        new_threshold = present_threshold - (present_threshold - stats.distance) * sensitivity_fp
    return new_threshold
