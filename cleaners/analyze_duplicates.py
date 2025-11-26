#!/usr/bin/env python3

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import textdistance
from datetime import timezone, datetime

# Load environment
env_file = 'settings.env'
dotenv_path = Path(env_file)
load_dotenv(dotenv_path=dotenv_path)

# Database connection
db_database = os.environ['MYSQL_DATABASE']
db_user = os.environ['MYSQL_USER']
db_password = os.environ['MYSQL_PASSWORD']
db_ip_address = os.environ['MYSQL_IP_ADDRESS']
db_port = os.environ['MYSQL_PORT']

MYSQL_CONNECTOR_STRING = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_ip_address}:{db_port}/{db_database}?charset=utf8mb4&collation=utf8mb4_general_ci'
engine = create_engine(MYSQL_CONNECTOR_STRING, echo=False, pool_pre_ping=True, pool_recycle=300)





from utils_sports import *

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import *
from utils import *
from utils_db import *
from utils_ai import *

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'log/sports/update_sports_results_{datetime.now(timezone.utc).strftime("%Y%m%d")}.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# %%
def find_sport_event_mapping_duplicates(session):
    """Find duplicates where different footystats_id share same bookmaker_id"""
    duplicates = session.query(
        SportEventMapping.sport_event_bookmaker_id,
        func.count(func.distinct(SportEventMapping.sport_event_footystats_id)).label('footystats_count'),
        func.count(SportEventMapping.id).label('total_mappings')
    ).filter(
        SportEventMapping.status.not_in(['wrong_match', 'deleted_duplicate']),
        SportEventMapping.sport_event_bookmaker_id.isnot(None),
        SportEventMapping.sport_event_footystats_id.isnot(None)
    ).group_by(
        SportEventMapping.sport_event_bookmaker_id
    ).having(
        func.count(func.distinct(SportEventMapping.sport_event_footystats_id)) > 1
    ).all()
    
    return duplicates

def calculate_similarity(bookmaker_event, footystats_event):
    """Calculate similarity between bookmaker and footystats events"""
    if not bookmaker_event or not footystats_event:
        return 0.0, 0.0, 0.0, False, None
    
    bkm_team_a = normalize_lower(bookmaker_event.team_a or "")
    bkm_team_b = normalize_lower(bookmaker_event.team_b or "")
    fs_team_a = normalize_lower(footystats_event.team_a or "")
    fs_team_b = normalize_lower(footystats_event.team_b or "")
    
    # Team name similarities
    team_a_sim = textdistance.jaro_winkler.normalized_similarity(bkm_team_a, fs_team_a)
    team_b_sim = textdistance.jaro_winkler.normalized_similarity(bkm_team_b, fs_team_b)
    
    def clean_team_name(team_name):
        team_name = team_name.strip().replace("-", " ")
        if team_name.endswith(" u19") or team_name.endswith(" u20") or team_name.endswith(" u21") or team_name.endswith(" u22") or team_name.endswith(" u23") or team_name.endswith(" u24"):
            team_name = team_name[:-4]
        if team_name.endswith(" w"):
            team_name = team_name[:-2]
        if team_name.endswith(" (women)"):
            team_name = team_name[:-7]
        if team_name.endswith(" fc"):
            team_name = team_name[:-3]
        if team_name.startswith("fc ") or team_name.startswith("cd ") or team_name.startswith("ac ") or team_name.startswith("sc "):
            team_name = team_name[3:]
        return team_name
    
    def has_common_word(name1, name2, min_length=4):
        """Check if both team names have a common word with minimum length"""
        words1 = set(word for word in name1.split() if len(word) >= min_length)
        words2 = set(word for word in name2.split() if len(word) >= min_length)
        return bool(words1.intersection(words2))
    
    if team_a_sim < 0.7 and bkm_team_a and fs_team_a:
        bkm_team_a = clean_team_name(bkm_team_a)
        fs_team_a = clean_team_name(fs_team_a)
        team_a_sim = textdistance.jaro_winkler.normalized_similarity(bkm_team_a, fs_team_a)
        # if team_a_sim > 0.7:
        #     print(f"✅ Team A ({bkm_team_a}) - FootyStats team A ({fs_team_a}) | similarity improved to {team_a_sim}")
        if team_a_sim < 0.7 and (bkm_team_a in fs_team_a or fs_team_a in bkm_team_a):
            # print(f"✅ Team A ({bkm_team_a}) matches FootyStats team A ({fs_team_a})")
            team_a_sim = 0.8
        if team_a_sim < 0.7 and has_common_word(bkm_team_a, fs_team_a):
            # print(f"✅ Team A ({bkm_team_a}) - FootyStats team A ({fs_team_a}) | common word match")
            team_a_sim = 0.8
    if team_b_sim < 0.7 and bkm_team_b and fs_team_b:
        bkm_team_b = clean_team_name(bkm_team_b)
        fs_team_b = clean_team_name(fs_team_b)
        team_b_sim = textdistance.jaro_winkler.normalized_similarity(bkm_team_b, fs_team_b)
        # if team_b_sim > 0.7:
        #     print(f"✅ Team B ({bkm_team_b}) - FootyStats team B ({fs_team_b}) | similarity improved to {team_b_sim}")
        if team_b_sim < 0.7 and (bkm_team_b in fs_team_b or fs_team_b in bkm_team_b):
            # print(f"✅ Team B ({bkm_team_b}) matches FootyStats team B ({fs_team_b})")
            team_b_sim = 0.8
        if team_b_sim < 0.7 and has_common_word(bkm_team_b, fs_team_b):
            # print(f"✅ Team B ({bkm_team_b}) - FootyStats team B ({fs_team_b}) | common word match")
            team_b_sim = 0.8
    
    
    # Overall similarity (average of team similarities)
    overall_sim = (team_a_sim + team_b_sim) / 2
    
    # Time comparison
    exact_start_time = False
    time_diff_minutes = None
    
    if bookmaker_event.time and footystats_event.time:
        # Convert both times to UTC for comparison
        bm_time = bookmaker_event.time.replace(tzinfo=timezone.utc) if bookmaker_event.time.tzinfo is None else bookmaker_event.time
        fs_time = footystats_event.time.replace(tzinfo=timezone.utc) if footystats_event.time.tzinfo is None else footystats_event.time
        
        exact_start_time = (bm_time == fs_time)
        time_diff = abs((bm_time - fs_time).total_seconds())
        time_diff_minutes = int(time_diff / 60)
    
    return team_a_sim, team_b_sim, overall_sim, exact_start_time, time_diff_minutes

def get_duplicate_details_with_similarity(session, bookmaker_id):
    """Get detailed information for a specific duplicate bookmaker_id with similarity analysis"""
    mappings = session.query(SportEventMapping).filter(
        SportEventMapping.status.not_in(['wrong_match', 'deleted_duplicate']),
        SportEventMapping.sport_event_bookmaker_id == bookmaker_id
    ).all()
    
    bookmaker_event = session.query(SportEventBookmaker).filter(
        SportEventBookmaker.id == bookmaker_id
    ).first()
    
    footystats_ids = [m.sport_event_footystats_id for m in mappings if m.sport_event_footystats_id]
    footystats_events = session.query(SportEventFootystats).filter(
        SportEventFootystats.id.in_(footystats_ids)
    ).all()
    
    # Calculate similarities for each FootyStats event
    similarities = []
    for fs_event in footystats_events:
        team_a_sim, team_b_sim, overall_sim, exact_time, time_diff = calculate_similarity(bookmaker_event, fs_event)
        similarities.append({
            'footystats_event': fs_event,
            'team_a_similarity': team_a_sim,
            'team_b_similarity': team_b_sim,
            'overall_similarity': overall_sim,
            'exact_start_time': exact_time,
            'time_diff_minutes': time_diff
        })
    
    # Sort by overall similarity (highest first)
    similarities.sort(key=lambda x: x['overall_similarity'], reverse=True)
    
    return {
        'bookmaker_event': bookmaker_event,
        'footystats_events': footystats_events,
        'mappings': mappings,
        'similarities': similarities
    }

def get_duplicate_details(session, bookmaker_id):
    """Get detailed information for a specific duplicate bookmaker_id"""
    mappings = session.query(SportEventMapping).filter(
        SportEventMapping.sport_event_bookmaker_id == bookmaker_id
    ).all()
    
    bookmaker_event = session.query(SportEventBookmaker).filter(
        SportEventBookmaker.id == bookmaker_id
    ).first()
    
    footystats_ids = [m.sport_event_footystats_id for m in mappings if m.sport_event_footystats_id]
    footystats_events = session.query(SportEventFootystats).filter(
        SportEventFootystats.id.in_(footystats_ids)
    ).all()
    
    return {
        'bookmaker_event': bookmaker_event,
        'footystats_events': footystats_events,
        'mappings': mappings
    }

def update_mappings_status(session, mapping_ids_str):
    """Update multiple mappings status to wrong_match and nullify bookmaker_id"""
    try:
        # Parse comma-separated mapping IDs
        mapping_ids = [int(id_str.strip()) for id_str in mapping_ids_str.split(',') if id_str.strip()]
        
        if not mapping_ids:
            print("No valid mapping IDs provided.")
            return
        
        print(f"Updating {len(mapping_ids)} mappings: {mapping_ids}")
        
        # Get mappings from database
        mappings = session.query(SportEventMapping).filter(
            SportEventMapping.status.not_in(['wrong_match', 'deleted_duplicate']),
            SportEventMapping.id.in_(mapping_ids)
        ).all()
        
        if not mappings:
            print("No mappings found with the provided IDs.")
            return
        
        updated_count = 0
        for mapping in mappings:
            print(f"Updating Mapping #{mapping.id}:")
            print(f"  Current status: {mapping.status}")
            # print(f"  Current bookmaker_id: {mapping.sport_event_bookmaker_id}")
            
            mapping.status = 'wrong_match'
            # mapping.sport_event_bookmaker_id = None
            mapping.date_updated = datetime.now(timezone.utc)
            updated_count += 1
            
            print(f"  ✅ Updated to status='wrong_match'")
        
        session.commit()
        print(f"\n🎉 Successfully updated {updated_count} mappings!")
        
    except ValueError as e:
        print(f"Error parsing mapping IDs: {e}")
    except Exception as e:
        print(f"Error updating mappings: {e}")
        session.rollback()

def verify_mapping_constraints(session, similarity_threshold=0.8, threshold_range=None, severe_only=False):
    """Verify constraints like matching start times between FootyStats and bookmaker events"""
    print("🔍 Verifying mapping constraints...")
    if severe_only:
        print("🚨 SEVERE MODE: Showing only events with MULTIPLE violations")
    if threshold_range:
        print(f"Using similarity threshold range: {threshold_range[0]:.1f} - {threshold_range[1]:.1f}")
    else:
        print(f"Using similarity threshold: {similarity_threshold}")
    print("="*80)
    
    # Get all active mappings with both FootyStats and bookmaker events
    mappings = session.query(SportEventMapping).filter(
        SportEventMapping.status.not_in(['wrong_match', 'deleted_duplicate']),
        SportEventMapping.sport_event_bookmaker_id.isnot(None),
        SportEventMapping.sport_event_footystats_id.isnot(None)
    ).all()
    
    print(f"Checking {len(mappings)} active mappings for constraint violations...")
    print()
    
    time_mismatch_count = 0
    similarity_violation_count = 0
    multiple_issues_found = []
    issues_found = []
    
    for mapping in mappings:
        # Get bookmaker event
        bookmaker_event = session.query(SportEventBookmaker).filter(
            SportEventBookmaker.id == mapping.sport_event_bookmaker_id
        ).first()
        
        # Get FootyStats event
        footystats_event = session.query(SportEventFootystats).filter(
            SportEventFootystats.id == mapping.sport_event_footystats_id
        ).first()
        
        if not bookmaker_event or not footystats_event:
            continue
        
        violations = []
        
        # Check team name similarity
        team_a_sim, team_b_sim, overall_sim, _, _ = calculate_similarity(bookmaker_event, footystats_event)
        
        # Check if similarity falls within the specified range or below threshold
        similarity_violation = False
        if threshold_range:
            # Range mode: flag if similarity is within the specified range
            if threshold_range[0] <= overall_sim <= threshold_range[1]:
                if (team_a_sim >= 0.8 or team_b_sim >= 0.8) and overall_sim > 0.65:
                    pass
                elif (team_a_sim >= 0.7 and team_b_sim >= 0.7) and overall_sim >= 0.7:
                    pass
                else:
                    similarity_violation = True
                    similarity_violation_count += 1
                    violations.append(f"❌ Similarity in range {threshold_range[0]:.1f}-{threshold_range[1]:.1f}: {overall_sim:.3f} (Team A: {team_a_sim:.3f}, Team B: {team_b_sim:.3f})")
        else:
            # Threshold mode: flag if similarity is below threshold
            if overall_sim < similarity_threshold:
                if (team_a_sim >= 0.8 or team_b_sim >= 0.8) and overall_sim > 0.65:
                    pass
                elif (team_a_sim >= 0.7 and team_b_sim >= 0.7) and overall_sim >= 0.7:
                    pass
                else:
                    similarity_violation = True
                    similarity_violation_count += 1
                    violations.append(f"❌ Low team similarity: {overall_sim:.3f} (Team A: {team_a_sim:.3f}, Team B: {team_b_sim:.3f})")

        # Check time difference
        time_diff = None
        if bookmaker_event.time and footystats_event.time:
            time_diff = abs((bookmaker_event.time - footystats_event.time).total_seconds() / 60)  # in minutes
            
            if (overall_sim < similarity_threshold or (threshold_range and overall_sim < threshold_range[1])) and time_diff > 60:  # More than 1 hour difference
                time_mismatch_count += 1
                hours = int(time_diff // 60)
                minutes = int(time_diff % 60)
                if hours > 0:
                    violations.append(f"❌ Time difference: {hours}h {minutes}m")
                else:
                    violations.append(f"❌ Time difference: {minutes}m")
        
        # If any violations found, record and display them
        if violations:
            is_severe = len(violations) > 1
            
            issue = {
                'mapping_id': mapping.id,
                'time_diff_minutes': time_diff,
                'overall_similarity': overall_sim,
                'team_a_similarity': team_a_sim,
                'team_b_similarity': team_b_sim,
                'bookmaker_event': bookmaker_event,
                'footystats_event': footystats_event,
                'violations': violations,
                'is_severe': is_severe
            }
            issues_found.append(issue)

            if is_severe:
                multiple_issues_found.append(mapping.id)
            
            # Only display if not in severe_only mode, or if it's a severe case
            if not severe_only or is_severe:
                print(f"{'🚨' if is_severe else '⚠️'}  CONSTRAINT VIOLATION{'S' if is_severe else ''} - Mapping #{mapping.id}")
                print(f"    Bookmaker: {bookmaker_event.team_a} vs {bookmaker_event.team_b} \t Time: {bookmaker_event.time} ({bookmaker_event.competition})")
                print(f"    FootyStats: {footystats_event.team_a} vs {footystats_event.team_b} \t Time: {footystats_event.time} ({footystats_event.competition})")
                for violation in violations:
                    print(f"    {violation}")
                print()
    
    # Summary
    print("="*80)
    print("📊 CONSTRAINT VERIFICATION SUMMARY")
    print(f"Total mappings checked: {len(mappings)}")
    
    if severe_only:
        severe_issues = [issue for issue in issues_found if issue['is_severe']]
        print(f"Severe violations found: {len(severe_issues)} (showing only severe cases)")
        displayed_issues = severe_issues
    else:
        print(f"Total violations found: {len(issues_found)} | ({len(multiple_issues_found)} severe)")
        print(f" - Time mismatches ({time_mismatch_count})")
        print(f" - Low similarity ({similarity_violation_count})")
        displayed_issues = issues_found
    
    if displayed_issues:
        print(f"\n💡 To invalidate mappings, you can run:")
        mapping_ids = [str(issue['mapping_id']) for issue in displayed_issues]
        print(f"   python analyze_duplicates.py invalidate {','.join(mapping_ids)}")
        if not severe_only and multiple_issues_found:
            print(f"\n💡 To invalidate mappings with multiple issues, you can run:")
            mapping_ids = [str(mapping_id) for mapping_id in multiple_issues_found]
            print(f"   python analyze_duplicates.py invalidate {','.join(mapping_ids)}")
    else:
        if severe_only:
            print("✅ No severe violations found!")
        else:
            print("✅ All mappings passed constraint verification!")
    
    return issues_found

if __name__ == "__main__":
    import sys
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Default similarity threshold
    similarity_threshold = 0.9
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "invalidate" and len(sys.argv) > 2:
            # Invalidate mode: python analyze_duplicates.py invalidate "123,456,789"
            mapping_ids_str = sys.argv[2]
            update_mappings_status(session, mapping_ids_str)
            session.close()
            exit()
        elif sys.argv[1] == "verify":
            # Verify constraints mode: python analyze_duplicates.py verify [threshold|range] [severe]
            verify_threshold = 0.8  # default
            threshold_range = None
            severe_only = False
            
            # Check for 'severe' argument
            if len(sys.argv) > 2 and sys.argv[-1] == "severe":
                severe_only = True
                # Remove 'severe' from args for threshold parsing
                args_without_severe = sys.argv[:-1]
            else:
                args_without_severe = sys.argv
            
            if len(args_without_severe) > 2:
                threshold_arg = args_without_severe[2]
                
                # Check if it's a range (contains dash)
                if '-' in threshold_arg:
                    try:
                        range_parts = threshold_arg.split('-')
                        if len(range_parts) == 2:
                            min_thresh = float(range_parts[0])
                            max_thresh = float(range_parts[1])
                            
                            if 0.0 <= min_thresh <= 1.0 and 0.0 <= max_thresh <= 1.0 and min_thresh <= max_thresh:
                                threshold_range = (min_thresh, max_thresh)
                            else:
                                print("Range values must be between 0.0 and 1.0, and min <= max. Using default threshold: 0.8")
                        else:
                            print("Invalid range format. Use format: 0.5-0.7. Using default threshold: 0.8")
                    except ValueError:
                        print(f"Invalid range: {threshold_arg}. Using default threshold: 0.8")
                else:
                    # Single threshold value
                    try:
                        verify_threshold = float(threshold_arg)
                        if not (0.0 <= verify_threshold <= 1.0):
                            print("Similarity threshold must be between 0.0 and 1.0. Using default: 0.8")
                            verify_threshold = 0.8
                    except ValueError:
                        print(f"Invalid threshold: {threshold_arg}. Using default: 0.8")
            
            verify_mapping_constraints(session, verify_threshold, threshold_range, severe_only)
            session.close()
            exit()
        elif sys.argv[1] == "help":
            print("Usage:")
            print("  python analyze_duplicates.py                      # Run duplicate analysis (threshold=0.9)")
            print("  python analyze_duplicates.py 0.8                  # Run with custom similarity threshold")
            print("  python analyze_duplicates.py verify               # Verify mapping constraints (similarity=0.8)")
            print("  python analyze_duplicates.py verify 0.5           # Verify with custom similarity threshold")
            print("  python analyze_duplicates.py verify 0.5-0.7       # Verify mappings with similarity in range")
            print("  python analyze_duplicates.py verify 0-0.5         # Find mappings with low similarity (0-50%)")
            print("  python analyze_duplicates.py verify severe        # Show only events with multiple violations")
            print("  python analyze_duplicates.py verify 0.5 severe    # Show severe cases with custom threshold")
            print("  python analyze_duplicates.py verify 0.5-0.7 severe # Show severe cases in range")
            print("  python analyze_duplicates.py invalidate 123,456    # Invalidate mapping IDs (set to wrong_match)")
            print("  python analyze_duplicates.py help                 # Show this help")
            session.close()
            exit()
        else:
            # Try to parse as similarity threshold
            try:
                threshold = float(sys.argv[1])
                if 0.0 <= threshold <= 1.0:
                    similarity_threshold = threshold
                    print(f"Using similarity threshold: {similarity_threshold}")
                else:
                    print("Similarity threshold must be between 0.0 and 1.0. Using default: 0.9")
            except ValueError:
                print(f"Invalid argument: {sys.argv[1]}. Using default threshold: 0.9")
    
    # Default behavior: run duplicate analysis
    duplicates = find_sport_event_mapping_duplicates(session)
    
    print(f"Found {len(duplicates)} bookmaker events with duplicate FootyStats mappings:")
    
    for dup in duplicates:
        print(f"\n{'='*80}")
        print(f"Bookmaker ID: {dup.sport_event_bookmaker_id}")
        print(f"FootyStats Count: {dup.footystats_count}, Total Mappings: {dup.total_mappings}")
        
        details = get_duplicate_details_with_similarity(session, dup.sport_event_bookmaker_id)
        
        if details['bookmaker_event']:
            bm = details['bookmaker_event']
            print(f"Bookmaker Event: {bm.team_a} vs {bm.team_b} ({bm.competition}) - {bm.time}")
        
        print("\nFootyStats Events (sorted by similarity):")
        
        # Check for duplicate FootyStats events (same match_url) and identify newest
        footystats_by_url = {}
        for sim_data in details['similarities']:
            fs_event = sim_data['footystats_event']
            if fs_event.match_url:
                if fs_event.match_url not in footystats_by_url:
                    footystats_by_url[fs_event.match_url] = []
                footystats_by_url[fs_event.match_url].append(sim_data)
        
        # Find duplicate URLs and mark older ones for removal
        duplicate_footystats_to_remove = set()
        for url, events in footystats_by_url.items():
            if len(events) > 1:
                # Sort by date_scraped (newest first)
                events.sort(key=lambda x: x['footystats_event'].date_scraped or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
                # Mark all but the newest for removal
                for event_data in events[1:]:
                    duplicate_footystats_to_remove.add(event_data['footystats_event'].id)
                    print(f"     📋 DUPLICATE FOOTYSTATS URL: {url}")
                    print(f"         Keeping newest: {events[0]['footystats_event'].id} (scraped: {events[0]['footystats_event'].date_scraped})")
                    print(f"         Removing older: {event_data['footystats_event'].id} (scraped: {event_data['footystats_event'].date_scraped})")
        
        # Check if we have a high-confidence match (exact time + similarity >= threshold)
        high_confidence_match = None
        if details['similarities']:
            best_match = details['similarities'][0]
            if best_match['exact_start_time'] and best_match['overall_similarity'] >= similarity_threshold:
                high_confidence_match = best_match['footystats_event'].id
        
        for i, sim_data in enumerate(details['similarities'], 1):
            fs = sim_data['footystats_event']
            team_a_sim = sim_data['team_a_similarity']
            team_b_sim = sim_data['team_b_similarity']
            overall_sim = sim_data['overall_similarity']
            exact_time = sim_data['exact_start_time']
            time_diff = sim_data['time_diff_minutes']
            
            # Find the mapping for this FootyStats event to get mapping ID
            mapping_id = None
            for m in details['mappings']:
                if m.sport_event_footystats_id == fs.id:
                    mapping_id = m.id
                    break
            
            print(f"  {i}. {fs.team_a} vs {fs.team_b} ({fs.competition}) - {fs.time} \t [Mapping #{mapping_id} | FootyStats #{fs.id}]")
            print(f"     Similarity: Team A: {team_a_sim:.3f}, Team B: {team_b_sim:.3f}, Overall: {overall_sim:.3f}")
            
            # Time comparison and action
            time_matches = False
            if exact_time:
                print(f"     Time: ✓ EXACT MATCH")
                time_matches = True
            elif time_diff is not None:
                if time_diff == 0:
                    print(f"     Time: ✓ EXACT MATCH")
                    time_matches = True
                elif time_diff <= 60:
                    print(f"     Time: ⚠ {time_diff} minutes difference")
                else:
                    hours = time_diff // 60
                    minutes = time_diff % 60
                    if hours > 0:
                        print(f"     Time: ✗ {hours}h {minutes}m difference")
                    else:
                        print(f"     Time: ✗ {minutes}m difference")
            else:
                print(f"     Time: ? Unable to compare")
            
            # Get the mapping (already found above)
            mapping = None
            for m in details['mappings']:
                if m.sport_event_footystats_id == fs.id:
                    mapping = m
                    break
            
            # Update status logic
            should_mark_wrong = False
            reason = ""
            
            # Case 1: Different start times (existing logic)
            if not time_matches and time_diff is not None:
                should_mark_wrong = True
                reason = "time mismatch"
            
            # Case 2: High-confidence match exists and this is not it
            elif high_confidence_match and fs.id != high_confidence_match:
                should_mark_wrong = True
                reason = "better match found"
            
            # Case 3: Duplicate FootyStats event (older one)
            elif fs.id in duplicate_footystats_to_remove:
                should_mark_wrong = True
                reason = "duplicate footystats (older)"
            
            if should_mark_wrong and mapping:# and mapping.status != 'wrong_match':
                if fs.id in duplicate_footystats_to_remove:
                    mapping.status = 'deleted_duplicate'
                    status_text = 'deleted_duplicate'
                else:
                    mapping.status = 'wrong_match'
                    status_text = 'wrong_match'
                # mapping.sport_event_bookmaker_id = None
                mapping.date_updated = datetime.now(timezone.utc)
                print(f"     🔄 STATUS UPDATED: Mapping #{mapping.id} set to '{status_text}' ({reason})")
                session.commit()
            
            # Highlight the best match
            if i == 1:
                if overall_sim >= 0.8:
                    print(f"     *** BEST MATCH (High Confidence) ***")
                elif overall_sim >= 0.6:
                    print(f"     *** BEST MATCH (Medium Confidence) ***")
                else:
                    print(f"     *** BEST MATCH (Low Confidence) ***")
    
    session.close()
