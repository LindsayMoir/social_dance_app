"""Tests for FuzzyMatcher utility."""
import pytest
from src.utils.fuzzy_utils import FuzzyMatcher


class TestFuzzyMatcher:
    """Test suite for FuzzyMatcher class."""

    def test_exact_match(self):
        """Test exact string match returns True."""
        assert FuzzyMatcher.compare("The Duke Saloon", "The Duke Saloon", threshold=95)

    def test_high_similarity(self):
        """Test high similarity match returns True."""
        assert FuzzyMatcher.compare("The Duke Saloon", "Duke Saloon", threshold=80)

    def test_no_match_high_threshold(self):
        """Test dissimilar strings with high threshold returns False."""
        assert not FuzzyMatcher.compare("Victoria", "Vancouver", threshold=90)

    def test_find_best_match(self):
        """Test finding best match from candidates."""
        candidates = [(123, "The Duke Saloon"), (618, "The Duke")]
        result = FuzzyMatcher.find_best("Duke Saloon", candidates, threshold=75)
        assert result is not None
        assert result[0] == 123  # Should match first one (higher score)

    def test_get_score(self):
        """Test getting raw similarity score."""
        score = FuzzyMatcher.get_score("The Duke Saloon", "Duke Saloon")
        assert score > 80

    def test_empty_strings(self):
        """Test that empty strings return False."""
        assert not FuzzyMatcher.compare("", "test", threshold=80)
        assert not FuzzyMatcher.compare("test", "", threshold=80)

    def test_find_best_empty_haystack(self):
        """Test find_best with empty candidate list."""
        result = FuzzyMatcher.find_best("test", [], threshold=75)
        assert result is None

    def test_none_strings(self):
        """Test that None strings return False."""
        assert not FuzzyMatcher.compare(None, "test", threshold=80)
        assert not FuzzyMatcher.compare("test", None, threshold=80)

    def test_fuzzy_match_by_score(self):
        """Test fuzzy_match_by_score returns best score."""
        score = FuzzyMatcher.fuzzy_match_by_score("Victoria", "Victoria")
        assert score == 100

    def test_case_insensitive(self):
        """Test that matching is case insensitive."""
        assert FuzzyMatcher.compare("THE DUKE SALOON", "the duke saloon", threshold=95)
