#!/usr/bin/env python3
"""
Test fuzzy similarity scoring between the two building names.
"""

from fuzzywuzzy import fuzz

def test_building_name_similarity():
    name1 = "Edelweiss Club"
    name2 = "Victoria Edelweiss Club"
    
    ratio_score = fuzz.ratio(name1, name2)
    partial_ratio = fuzz.partial_ratio(name1, name2)
    token_sort_ratio = fuzz.token_sort_ratio(name1, name2)
    token_set_ratio = fuzz.token_set_ratio(name1, name2)
    
    print(f"Building name similarity test:")
    print(f"  '{name1}' vs '{name2}'")
    print(f"  ratio: {ratio_score}")
    print(f"  partial_ratio: {partial_ratio}")
    print(f"  token_sort_ratio: {token_sort_ratio}")
    print(f"  token_set_ratio: {token_set_ratio}")
    print(f"  threshold: 85")
    print(f"  would match: {ratio_score >= 85}")

if __name__ == "__main__":
    test_building_name_similarity()