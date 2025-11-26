#!/usr/bin/env python3
"""
Script to match scraped sports stats with footystats events and mappings
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging
from difflib import SequenceMatcher
import re

# Environment setup
from dotenv import load_dotenv
env_file = 'settings.env'
dotenv_path = Path(env_file)
load_dotenv(dotenv_path=dotenv_path)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database configuration
db_database = os.environ['MYSQL_DATABASE']
db_user = os.environ['MYSQL_USER']
db_password = os.environ['MYSQL_PASSWORD']
db_ip_address = os.environ['MYSQL_IP_ADDRESS']
db_port = os.environ['MYSQL_PORT']
SERVER_MODE = os.environ['ENV']
MYSQL_CONNECTOR_STRING = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_ip_address}:{db_port}/{db_database}?charset=utf8mb4&collation=utf8mb4_general_ci'
engine = create_engine(MYSQL_CONNECTOR_STRING, echo=False, pool_pre_ping=True, pool_recycle=300, pool_size=20, max_overflow=0)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import *
from utils_sports import *

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'log/sports/match_sports_stats_{datetime.now(timezone.utc).strftime("%Y%m%d")}.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SIMILARITY_THRESHOLD_HIGH = 0.90
SIMILARITY_THRESHOLD_MEDIUM = 0.80
SIMILARITY_THRESHOLD_LOW = 0.70
DATE_TOLERANCE_DAYS = 1


def normalize_team_name(team_name):
    """Normalize team name for comparison"""
    if not team_name:
        return ""
    
    # Remove common suffixes and prefixes
    normalized = team_name.strip()
    
    # Remove U19, U21, U23, etc.
    normalized = re.sub(r'\sU[12][0-9]', '', normalized, flags=re.IGNORECASE)
    
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


def exact_match_stats_to_events(session, stats_records):
    """Try exact matching by match_slug and date"""
    matches = []
    
    for stat in stats_records:
        if not stat.match_slug or not stat.date_start:
            continue
            
        # Look for exact match by match_slug and date (±1 day)
        potential_events = session.query(SportEventFootystats).filter(
            SportEventFootystats.match_slug == stat.match_slug,
            SportEventFootystats.time >= datetime.combine(stat.date_start, datetime.min.time()) - timedelta(days=DATE_TOLERANCE_DAYS),
            SportEventFootystats.time <= datetime.combine(stat.date_start, datetime.min.time()) + timedelta(days=DATE_TOLERANCE_DAYS + 1)
        ).all()
        
        if potential_events:
            # If multiple events found, pick the closest date
            if len(potential_events) > 1:
                logger.warning(f"Multiple events found for {stat.match_slug} ({stat.date_start}): {len(potential_events)}")
            best_event = min(potential_events, key=lambda e: abs((e.time.date() - stat.date_start).days))
            matches.append({
                'stat': stat,
                'event': best_event,
                'match_type': 'exact',
                'confidence': 1.0
            })
            logger.debug(f"Exact match found: stat_id={stat.id} -> event_id={best_event.id} ({stat.match_slug})")
    
    return matches


def strong_match_stats_to_events(session, stats_records):
    """Try strong matching by team names, date, and country"""
    matches = []
    
    for stat in stats_records:
        if not stat.team_a or not stat.team_b or not stat.date_start:
            continue
            
        # Look for events within date range
        date_start_min = datetime.combine(stat.date_start, datetime.min.time()) - timedelta(days=DATE_TOLERANCE_DAYS)
        date_start_max = datetime.combine(stat.date_start, datetime.min.time()) + timedelta(days=DATE_TOLERANCE_DAYS + 1)
        
        potential_events = session.query(SportEventFootystats).filter(
            SportEventFootystats.time >= date_start_min,
            SportEventFootystats.time <= date_start_max
        ).all()
        
        best_match = None
        best_similarity = 0
        
        for event in potential_events:
            if not event.team_a or not event.team_b:
                continue
                
            similarity = calculate_team_similarity(
                stat.team_a, stat.team_b,
                event.team_a, event.team_b
            )
            
            # Additional country matching if available
            country_match = True
            if stat.country and event.country:
                country_match = normalize_lower(stat.country) == normalize_lower(event.country)
            
            if similarity >= SIMILARITY_THRESHOLD_HIGH and country_match and similarity > best_similarity:
                best_match = event
                best_similarity = similarity
        
        if best_match:
            matches.append({
                'stat': stat,
                'event': best_match,
                'match_type': 'strong',
                'confidence': best_similarity
            })
            logger.debug(f"Strong match found: stat_id={stat.id} -> event_id={best_match.id} (similarity={best_similarity:.3f})")
    
    return matches


def weak_match_stats_to_events(session, stats_records):
    """Try weak matching by team names and date only"""
    matches = []
    
    for stat in stats_records:
        if not stat.team_a or not stat.team_b or not stat.date_start:
            continue
            
        # Look for events within date range
        date_start_min = datetime.combine(stat.date_start, datetime.min.time()) - timedelta(days=DATE_TOLERANCE_DAYS)
        date_start_max = datetime.combine(stat.date_start, datetime.min.time()) + timedelta(days=DATE_TOLERANCE_DAYS + 1)
        
        potential_events = session.query(SportEventFootystats).filter(
            SportEventFootystats.time >= date_start_min,
            SportEventFootystats.time <= date_start_max
        ).all()
        
        best_match = None
        best_similarity = 0
        
        for event in potential_events:
            if not event.team_a or not event.team_b:
                continue
                
            similarity = calculate_team_similarity(
                stat.team_a, stat.team_b,
                event.team_a, event.team_b
            )
            
            if similarity >= SIMILARITY_THRESHOLD_MEDIUM and similarity > best_similarity:
                best_match = event
                best_similarity = similarity
        
        if best_match:
            matches.append({
                'stat': stat,
                'event': best_match,
                'match_type': 'weak',
                'confidence': best_similarity
            })
            logger.info(f"Weak match found: stat_id={stat.id} -> event_id={best_match.id} (similarity={best_similarity:.3f})")
    
    return matches


def create_footystats_event_from_stat(session, stat):
    """Create a new SportEventFootystats from a stat record"""
    try:
        # Only create events for stats with match_slug
        if not stat.match_slug:
            logger.warning(f"Cannot create event without match_slug: stat_id={stat.id}")
            return None
            
        # Check if event already exists
        existing_event = session.query(SportEventFootystats).filter(
            SportEventFootystats.match_slug == stat.match_slug,
            SportEventFootystats.time >= datetime.combine(stat.date_start, datetime.min.time()) - timedelta(days=DATE_TOLERANCE_DAYS),
            SportEventFootystats.time <= datetime.combine(stat.date_start, datetime.min.time()) + timedelta(days=DATE_TOLERANCE_DAYS + 1)
        ).first()
        
        if existing_event:
            logger.info(f"Event already exists for match_slug: {stat.match_slug}")
            return existing_event
        
        # Create new event
        new_event = SportEventFootystats(
            match_uid=stat.match_uid,
            match_slug=stat.match_slug,
            time=datetime.combine(stat.date_start, datetime.min.time()).replace(tzinfo=timezone.utc),
            time_str=stat.time_start_str,
            country=stat.country,
            competition=stat.competition,
            team_a=stat.team_a,
            team_b=stat.team_b,
            scrape_script='match_sports_stats.py'
        )
        
        session.add(new_event)
        session.commit()
        logger.info(f"Created new event: {new_event.id} for stat_id={stat.id} ({stat.match_slug})")
        return new_event
        
    except Exception as e:
        logger.error(f"Error creating event from stat {stat.id}: {e}")
        session.rollback()
        return None


def update_stat_with_event_mapping(session, stat, event, confidence, match_type):
    """Update stat record with event mapping"""
    try:
        stat.sport_event_footystats_id = event.id
        stat.date_updated = datetime.now(timezone.utc)
        
        # Add confidence score to a comment field or log it
        logger.info(f"Updated stat_id={stat.id} with event_id={event.id} (type={match_type}, confidence={confidence:.3f})")
        
        session.commit()
        return True
        
    except Exception as e:
        logger.error(f"Error updating stat {stat.id} with event mapping: {e}")
        session.rollback()
        return False


def match_sports_stats(session, create_missing=True, dry_run=False):
    """Main function to match sports stats with footystats events"""
    logger.info(f"Starting sports stats matching process (dry_run={dry_run})")
    
    # Get unmatched stats records
    unmatched_stats = session.query(SportPotentialFootystats).filter(
        SportPotentialFootystats.sport_event_footystats_id.is_(None)
    ).all()
    
    logger.info(f"Found {len(unmatched_stats)} unmatched stats records")
    
    if not unmatched_stats:
        logger.info("No unmatched stats records found")
        return
    
    total_matched = 0
    total_created = 0
    matched_stat_ids = set()  # Track matched IDs in memory
    
    # Try exact matching first
    logger.info("Trying exact matching...")
    exact_matches = exact_match_stats_to_events(session, unmatched_stats)
    for match in exact_matches:
        if match['stat'].id not in matched_stat_ids:
            if dry_run:
                logger.info(f"[DRY RUN] Would update stat_id={match['stat'].id} with event_id={match['event'].id} (type={match['match_type']}, confidence={match['confidence']:.3f})")
                total_matched += 1
                matched_stat_ids.add(match['stat'].id)
            else:
                if update_stat_with_event_mapping(session, match['stat'], match['event'], match['confidence'], match['match_type']):
                    total_matched += 1
                    matched_stat_ids.add(match['stat'].id)
    
    # Get remaining unmatched stats
    remaining_stats = [s for s in unmatched_stats if s.id not in matched_stat_ids]

    logger.info(f"Found {len(remaining_stats)} remaining stats records on total {len(unmatched_stats)}")
    
    # Try strong matching
    if remaining_stats:
        logger.info("Trying strong matching...")
        strong_matches = strong_match_stats_to_events(session, remaining_stats)
        for match in strong_matches:
            if match['stat'].id not in matched_stat_ids:
                if dry_run:
                    logger.info(f"[DRY RUN] Would update stat_id={match['stat'].id} with event_id={match['event'].id} (type={match['match_type']}, confidence={match['confidence']:.3f})")
                    total_matched += 1
                    matched_stat_ids.add(match['stat'].id)
                else:
                    if update_stat_with_event_mapping(session, match['stat'], match['event'], match['confidence'], match['match_type']):
                        total_matched += 1
                        matched_stat_ids.add(match['stat'].id)
    
    # Get remaining unmatched stats
    remaining_stats = [s for s in unmatched_stats if s.id not in matched_stat_ids]
    
    logger.info(f"Found {len(remaining_stats)} remaining stats records on total {len(unmatched_stats)}")
    
    # Try weak matching
    if remaining_stats:
        logger.info("Trying weak matching...")
        weak_matches = weak_match_stats_to_events(session, remaining_stats)
        for match in weak_matches:
            if match['stat'].id not in matched_stat_ids:
                if dry_run:
                    logger.info(f"[DRY RUN] Would update stat_id={match['stat'].id} with event_id={match['event'].id} (type={match['match_type']}, confidence={match['confidence']:.3f})")
                    total_matched += 1
                    matched_stat_ids.add(match['stat'].id)
                else:
                    if update_stat_with_event_mapping(session, match['stat'], match['event'], match['confidence'], match['match_type']):
                        total_matched += 1
                        matched_stat_ids.add(match['stat'].id)
    
    # Create missing events for stats with match_slug
    # TODO: if no teams found, scrape the page.
    
    # if create_missing and not dry_run:
    #     remaining_stats = [s for s in unmatched_stats if s.id not in matched_stat_ids]
    #     if remaining_stats:
    #         logger.info(f"Creating events for {len(remaining_stats)} remaining stats with match_slug...")
    #         for stat in remaining_stats:
    #             if stat.match_slug:  # Only create if we have match_slug
    #                 new_event = create_footystats_event_from_stat(session, stat)
    #                 if new_event:
    #                     if update_stat_with_event_mapping(session, stat, new_event, 1.0, 'created'):
    #                         total_created += 1
    #                         matched_stat_ids.add(stat.id)
    
    # Final report
    remaining_stats = [s for s in unmatched_stats if s.id not in matched_stat_ids]
    logger.info(f"Matching completed: {total_matched} matched, {total_created} created, {len(remaining_stats)} still unmatched")
    
    if remaining_stats:
        logger.info("Sample of unmatched stats:")
        for stat in remaining_stats[:5]:  # Show first 5
            logger.info(f"  - stat_id={stat.id}, match_slug={stat.match_slug}, teams={stat.team_a} vs {stat.team_b}, date={stat.date_start}")


def main():
    """Main execution function"""
    try:
        # Create database session
        Session = sessionmaker(bind=engine, autocommit=False, autoflush=True)
        session = Session()
        
        # Check command line arguments
        create_missing = True  # Default: create missing events
        dry_run = False  # Default: actually update database
        
        if len(sys.argv) > 1:
            if '--no-create' in sys.argv:
                create_missing = False
            if '--dry-run' in sys.argv:
                dry_run = True
        
        logger.info(f"Running match_sports_stats with create_missing={create_missing}, dry_run={dry_run}")
        
        # Run matching process
        match_sports_stats(session, create_missing=create_missing, dry_run=dry_run)
        
        logger.info("Script execution completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        sys.exit(1)
    finally:
        if 'session' in locals():
            session.close()


if __name__ == "__main__":
    main()
