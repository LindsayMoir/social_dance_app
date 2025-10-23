"""
Centralized fuzzy string matching utility.

This module provides FuzzyMatcher class for consistent fuzzy string matching
across the application, eliminating duplication and standardizing thresholds.
"""
from rapidfuzz import fuzz
from typing import Optional, List, Tuple
import logging


class FuzzyMatcher:
    """Centralized fuzzy string matching utility."""

    # Standard thresholds for different use cases
    THRESHOLDS = {
        'exact': 95,
        'high': 90,
        'moderate': 80,
        'loose': 70,
    }

    @staticmethod
    def compare(str1: str, str2: str,
                threshold: int = 80,
                algorithm: str = 'token_set') -> bool:
        """
        Compare two strings with fuzzy matching.

        Args:
            str1: First string to compare
            str2: Second string to compare
            threshold: Score threshold (0-100), default 80
            algorithm: 'token_set' (default), 'partial', or 'ratio'

        Returns:
            bool: True if similarity score >= threshold, False otherwise

        Examples:
            >>> FuzzyMatcher.compare("The Duke Saloon", "Duke Saloon", threshold=75)
            True
            >>> FuzzyMatcher.compare("Victoria", "Vancouver", threshold=90)
            False
        """
        if not str1 or not str2:
            return False

        s1 = str1.lower().strip()
        s2 = str2.lower().strip()

        if algorithm == 'token_set':
            score = fuzz.token_set_ratio(s1, s2)
        elif algorithm == 'partial':
            score = fuzz.partial_ratio(s1, s2)
        else:  # ratio (default)
            score = fuzz.ratio(s1, s2)

        return score >= threshold

    @staticmethod
    def find_best(needle: str,
                  haystack: List[Tuple[int, str]],
                  threshold: int = 80,
                  algorithm: str = 'token_set') -> Optional[Tuple[int, str, int]]:
        """
        Find best matching item from a list of candidates.

        Args:
            needle: String to search for
            haystack: List of (id, string) tuples to search in
            threshold: Minimum score required (0-100)
            algorithm: 'token_set' (default), 'partial', or 'ratio'

        Returns:
            tuple: (id, matched_string, score) or None if no match found

        Example:
            >>> candidates = [(123, "The Duke Saloon"), (618, "The Duke Saloon, Victoria")]
            >>> FuzzyMatcher.find_best("Duke Saloon", candidates, threshold=75)
            (123, 'The Duke Saloon', 95)
        """
        if not needle or not haystack:
            return None

        best_match = None
        best_score = threshold
        needle_lower = needle.lower().strip()

        for item_id, item_str in haystack:
            if not item_str:
                continue

            item_lower = item_str.lower().strip()

            if algorithm == 'token_set':
                score = fuzz.token_set_ratio(needle_lower, item_lower)
            elif algorithm == 'partial':
                score = fuzz.partial_ratio(needle_lower, item_lower)
            else:
                score = fuzz.ratio(needle_lower, item_lower)

            if score > best_score:
                best_score = score
                best_match = (item_id, item_str, score)

        return best_match

    @staticmethod
    def get_score(str1: str, str2: str,
                  algorithm: str = 'token_set') -> int:
        """
        Get raw similarity score (0-100) for two strings.
        Useful for flexible threshold decisions.

        Args:
            str1: First string
            str2: Second string
            algorithm: 'token_set' (default), 'partial', or 'ratio'

        Returns:
            int: Similarity score 0-100
        """
        if not str1 or not str2:
            return 0

        s1 = str1.lower().strip()
        s2 = str2.lower().strip()

        if algorithm == 'token_set':
            return fuzz.token_set_ratio(s1, s2)
        elif algorithm == 'partial':
            return fuzz.partial_ratio(s1, s2)
        else:
            return fuzz.ratio(s1, s2)

    @staticmethod
    def fuzzy_match_by_score(str1: str, str2: str) -> int:
        """
        Get the best (highest) score from multiple algorithms.
        More robust than single algorithm.

        Args:
            str1: First string
            str2: Second string

        Returns:
            int: Best score (max of ratio, partial_ratio, token_set_ratio)
        """
        if not str1 or not str2:
            return 0

        scores = [
            fuzz.ratio(str1.lower(), str2.lower()),
            fuzz.partial_ratio(str1.lower(), str2.lower()),
            fuzz.token_set_ratio(str1.lower(), str2.lower()),
        ]
        return max(scores)
