# Author: imyhxy
# File: general.py
# Date: 12/5/23


def calculate_split_ratios(ratios: str):
    split_ratios = [float(x) for x in ratios.split(":")]
    for i in range(1, len(split_ratios)):
        split_ratios[i] = split_ratios[i - 1] + split_ratios[i]
    return [x / split_ratios[-1] for x in split_ratios]
