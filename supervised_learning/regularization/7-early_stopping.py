#!/usr/bin/env python3
"""early-stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """early-stopping"""
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0
    if count < patience:
        return False, count
    return True, count
