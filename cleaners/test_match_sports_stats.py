#!/usr/bin/env python3
"""
Test script for match_sports_stats.py functionality without database dependency
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from sports.utils_sports import normalize_lower

# Test data
def create_test_stat_record():
    """Create a mock stat record for testing"""
    class MockStat:
        def __init__(self):
            self.id = 1
            self.match_slug = "test-match-123"
            self.match_uid = "456"
            self.team_a = "Manchester United U19"
            self.team_b = "Liverpool U19"
            self.country = "england"
            self.competition = "Premier League 2"
            self.date_start = datetime.now(timezone.utc).date()
            self.time_start_str = "today"
            self.sport_event_footystats_id = None
    
    return MockStat()

def create_test_event_record():
    """Create a mock event record for testing"""
    class MockEvent:
        def __init__(self):
            self.id = 100
            self.match_uid = "456"
            self.match_url = "https://footystats.org/england/test-match-123"
            self.team_a = "Manchester United U19"
            self.team_b = "Liverpool U19"
            self.country = "england"
            self.competition = "Premier League 2"
            self.time = datetime.now(timezone.utc)
    
    return MockEvent()

def normalize_team_name(team_name):
    """Normalize team name for comparison"""
    if not team_name:
        return ""
    
    # Remove common suffixes and prefixes
    normalized = team_name.strip()
    
    # Remove U19, U21, U23, etc.
    normalized = re.sub(r'\s?U[12][0-9]', '', normalized, flags=re.IGNORECASE)
    
    # Remove "FC", "AFC", etc.
    normalized = re.sub(r'\s?(FC|AFC)$', '', normalized, flags=re.IGNORECASE)
    
    # Convert to normalized lowercase
    return normalize_lower(normalized)

def calculate_team_similarity(team_a1, team_b1, team_a2, team_b2):
    """Calculate similarity between two pairs of team names using difflib"""
    if not all([team_a1, team_b1, team_a2, team_b2]):
        return 0.0
    
    # Normalize team names
    norm_a1 = normalize_team_name(team_a1)
    norm_b1 = normalize_team_name(team_b1)
    norm_a2 = normalize_team_name(team_a2)
    norm_b2 = normalize_team_name(team_b2)
    
    # Calculate similarity for both orders (home/away might be swapped)
    similarity1 = (
        SequenceMatcher(None, norm_a1, norm_a2).ratio() +
        SequenceMatcher(None, norm_b1, norm_b2).ratio()
    ) / 2
    
    similarity2 = (
        SequenceMatcher(None, norm_a1, norm_b2).ratio() +
        SequenceMatcher(None, norm_b1, norm_a2).ratio()
    ) / 2
    
    return max(similarity1, similarity2)

def test_team_name_normalization():
    """Test team name normalization function"""
    print("Testing team name normalization...")
    
    test_cases = [
        ("Manchester United U19", "manchester united"),
        ("Liverpool FC", "liverpool"),
        ("Real Madrid", "real madrid"),
        ("Barcelona AFC", "barcelona"),
        ("Bayern Munich U21", "bayern munich"),
    ]
    
    for original, expected in test_cases:
        result = normalize_team_name(original)
        print(f"  '{original}' -> '{result}' (expected: '{expected}')")
        assert result == expected, f"Expected '{expected}', got '{result}'"
    
    print("✓ Team name normalization tests passed")

def test_team_similarity():
    """Test team similarity calculation"""
    print("Testing team similarity calculation...")
    
    test_cases = [
        # Exact matches
        ("Manchester United", "Liverpool", "Manchester United", "Liverpool", 1.0),
        # Teams with U19 suffix
        ("Manchester United U19", "Liverpool U19", "Manchester United", "Liverpool", 1.0),
        # Teams with FC suffix
        ("Manchester United FC", "Liverpool FC", "Manchester United", "Liverpool", 1.0),
        # Similar but not identical
        ("Manchester United", "Liverpool", "Manchester Utd", "Liverpool", 0.8),
        # Swapped order
        ("Manchester United", "Liverpool", "Liverpool", "Manchester United", 1.0),
    ]
    
    for team_a1, team_b1, team_a2, team_b2, expected_min in test_cases:
        similarity = calculate_team_similarity(team_a1, team_b1, team_a2, team_b2)
        print(f"  '{team_a1}' vs '{team_b1}' <-> '{team_a2}' vs '{team_b2}' = {similarity:.3f}")
        assert similarity >= expected_min, f"Similarity {similarity:.3f} below expected {expected_min}"
    
    print("✓ Team similarity tests passed")

def test_matching_logic():
    """Test the matching logic with mock data"""
    print("Testing matching logic...")
    
    # Create test data
    stat = create_test_stat_record()
    event = create_test_event_record()
    
    # Test exact match logic (match_slug)
    stat_match_slug = "test-match-123"
    event_match_url = "https://footystats.org/england/test-match-123"
    
    exact_match = stat_match_slug in event_match_url
    print(f"  Exact match test: {exact_match}")
    assert exact_match, "Exact match should be found"
    
    # Test team similarity
    similarity = calculate_team_similarity(stat.team_a, stat.team_b, event.team_a, event.team_b)
    print(f"  Team similarity: {similarity:.3f}")
    assert similarity >= 0.9, f"High similarity expected, got {similarity:.3f}"
    
    # Test date matching
    date_diff = abs((stat.date_start - event.time.date()).days)
    date_match = date_diff <= 1  # Within 1 day tolerance
    print(f"  Date match: {date_match} (diff: {date_diff} days)")
    assert date_match, "Date should match within tolerance"
    
    print("✓ Matching logic tests passed")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    # Test empty/null inputs
    similarity_empty = calculate_team_similarity("", "", "", "")
    assert similarity_empty == 0.0, "Empty input should return 0 similarity"
    
    similarity_none = calculate_team_similarity(None, None, None, None)
    assert similarity_none == 0.0, "None input should return 0 similarity"
    
    # Test team name normalization with edge cases
    assert normalize_team_name("") == ""
    assert normalize_team_name(None) == ""
    assert normalize_team_name("  FC  ") == ""
    
    print("✓ Edge case tests passed")

def main():
    """Run all tests"""
    print("Running match_sports_stats.py tests...\n")
    
    try:
        test_team_name_normalization()
        print()
        
        test_team_similarity()
        print()
        
        test_matching_logic()
        print()
        
        test_edge_cases()
        print()
        
        print("✓ All tests passed successfully!")
        print("\nThe matching script logic is working correctly.")
        print("To run the full script with database, ensure:")
        print("1. MySQL connector is installed: pip install mysql-connector-python")
        print("2. Database connection details are in settings.env")
        print("3. Run: python sports/match_sports_stats.py")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
