#!/usr/bin/env python3
"""
Database utilities for sports matches performance interface
Functions to query matched events with footystats and bookmaker data
"""

import streamlit as st
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, and_, or_, func, extract, desc, text
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sports.models import (
    SportEventMapping, SportEventFootystats, 
    SportEventBookmaker, SportPotentialFootystats
)

# Database connection
@st.cache_resource
def get_db_engine():
    """Create database engine"""
    db_database = os.environ['MYSQL_DATABASE']
    db_user = os.environ['MYSQL_USER']
    db_password = os.environ['MYSQL_PASSWORD']
    db_ip_address = os.environ['MYSQL_IP_ADDRESS']
    db_port = os.environ['MYSQL_PORT']
    MYSQL_CONNECTOR_STRING = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_ip_address}:{db_port}/{db_database}?charset=utf8mb4&collation=utf8mb4_general_ci'
    return create_engine(MYSQL_CONNECTOR_STRING, echo=False, pool_pre_ping=True, pool_recycle=300)

def get_session():
    """Get database session"""
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    return Session()

@st.cache_data
def get_matched_events(filters=None):
    """
    Get all matched events with their footystats and bookmaker data
    Returns comprehensive dataset for analysis
    """
    session = get_session()
    try:
        # Base query joining all tables through SportEventMapping
        query = session.query(
            SportEventMapping.id.label('mapping_id'),
            SportEventMapping.start_time,
            SportEventMapping.country,
            SportEventMapping.competition,
            SportEventMapping.team_a,
            SportEventMapping.team_b,
            
            # Footystats data
            SportEventFootystats.id.label('footystats_id'),
            SportEventFootystats.avg_potential,
            SportEventFootystats.btts_potential,
            SportEventFootystats.o15_potential,
            SportEventFootystats.o25_potential,
            SportEventFootystats.o05_potential,
            SportEventFootystats.o35_potential,
            SportEventFootystats.o05HT_potential,
            SportEventFootystats.btts_fhg_potential,
            SportEventFootystats.btts_2hg_potential,
            SportEventFootystats.offsides_potential,
            SportEventFootystats.ppg_a,
            SportEventFootystats.ppg_b,
            SportEventFootystats.team_a_xg_prematch,
            SportEventFootystats.team_b_xg_prematch,
            SportEventFootystats.odds_btts_yes,
            SportEventFootystats.odds_ft_1,
            SportEventFootystats.odds_ft_x,
            SportEventFootystats.odds_ft_2,
            SportEventFootystats.odds_ft_over05,
            SportEventFootystats.odds_ft_over15,
            SportEventFootystats.odds_ft_over25,
            SportEventFootystats.odds_ft_over35,
            SportEventFootystats.score_ft,
            
            # Bookmaker data
            SportEventBookmaker.bookmaker,
            SportEventBookmaker.time.label('bookmaker_time'),
            SportEventBookmaker.odds_ft_1.label('bookmaker_odds_1'),
            SportEventBookmaker.odds_ft_x.label('bookmaker_odds_x'),
            SportEventBookmaker.odds_ft_2.label('bookmaker_odds_2'),
            SportEventBookmaker.odds_btts_yes.label('bookmaker_odds_btts'),
            SportEventBookmaker.score_ft.label('bookmaker_score_ft'),
            SportEventBookmaker.score_fh.label('bookmaker_score_fh'),
            SportEventBookmaker.score_2h.label('bookmaker_score_2h'),
            SportEventBookmaker.status.label('match_status')
        ).join(
            SportEventFootystats,
            SportEventMapping.sport_event_footystats_id == SportEventFootystats.id
        ).join(
            SportEventBookmaker,
            SportEventMapping.sport_event_bookmaker_id == SportEventBookmaker.id
        )
        
        # Apply filters if provided
        if filters:
            if filters.get('country'):
                query = query.filter(SportEventMapping.country == filters['country'])
            
            if filters.get('competition'):
                query = query.filter(SportEventMapping.competition == filters['competition'])
            
            if filters.get('date_from'):
                query = query.filter(SportEventMapping.start_time >= filters['date_from'])
            
            if filters.get('date_to'):
                query = query.filter(SportEventMapping.start_time <= filters['date_to'])
            
            # Range filters for betting simulation
            if filters.get('ppg_a_min') is not None and filters.get('ppg_a_min') > 0:
                query = query.filter(SportEventFootystats.ppg_a >= filters['ppg_a_min'])
            
            if filters.get('ppg_a_max') is not None and filters.get('ppg_a_max') < 3:
                query = query.filter(SportEventFootystats.ppg_a <= filters['ppg_a_max'])
            
            if filters.get('ppg_b_min') is not None and filters.get('ppg_b_min') > 0:
                query = query.filter(SportEventFootystats.ppg_b >= filters['ppg_b_min'])
            
            if filters.get('ppg_b_max') is not None and filters.get('ppg_b_max') < 3:
                query = query.filter(SportEventFootystats.ppg_b <= filters['ppg_b_max'])
            
            if filters.get('team_a_xg_min') is not None and filters.get('team_a_xg_min') > 0:
                query = query.filter(SportEventFootystats.team_a_xg_prematch >= filters['team_a_xg_min'])
            
            if filters.get('team_a_xg_max') is not None and filters.get('team_a_xg_max') < 5:
                query = query.filter(SportEventFootystats.team_a_xg_prematch <= filters['team_a_xg_max'])
            
            if filters.get('team_b_xg_min') is not None and filters.get('team_b_xg_min') > 0:
                query = query.filter(SportEventFootystats.team_b_xg_prematch >= filters['team_b_xg_min'])
            
            if filters.get('team_b_xg_max') is not None and filters.get('team_b_xg_max') < 5:
                query = query.filter(SportEventFootystats.team_b_xg_prematch <= filters['team_b_xg_max'])
            
            # Odds filters (footystats odds for BTTS and 1X2 markets)
            if filters.get('footystats_odds_btts_min') is not None and filters.get('footystats_odds_btts_min') > 1.01:
                query = query.filter(or_(
                    SportEventFootystats.odds_btts_yes >= filters['footystats_odds_btts_min'],
                    SportEventFootystats.odds_btts_yes.is_(None)
                ))
            
            if filters.get('footystats_odds_btts_max') is not None and filters.get('footystats_odds_btts_max') < 20:
                query = query.filter(or_(
                    SportEventFootystats.odds_btts_yes <= filters['footystats_odds_btts_max'],
                    SportEventFootystats.odds_btts_yes.is_(None)
                ))
            
            if filters.get('footystats_odds_1_min') is not None and filters.get('footystats_odds_1_min') > 1.01:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_1 >= filters['footystats_odds_1_min'],
                    SportEventFootystats.odds_ft_1.is_(None)
                ))
            
            if filters.get('footystats_odds_1_max') is not None and filters.get('footystats_odds_1_max') < 20:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_1 <= filters['footystats_odds_1_max'],
                    SportEventFootystats.odds_ft_1.is_(None)
                ))
            
            if filters.get('footystats_odds_x_min') is not None and filters.get('footystats_odds_x_min') > 1.01:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_x >= filters['footystats_odds_x_min'],
                    SportEventFootystats.odds_ft_x.is_(None)
                ))
            
            if filters.get('footystats_odds_x_max') is not None and filters.get('footystats_odds_x_max') < 20:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_x <= filters['footystats_odds_x_max'],
                    SportEventFootystats.odds_ft_x.is_(None)
                ))
            
            if filters.get('footystats_odds_2_min') is not None and filters.get('footystats_odds_2_min') > 1.01:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_2 >= filters['footystats_odds_2_min'],
                    SportEventFootystats.odds_ft_2.is_(None)
                ))
            
            if filters.get('footystats_odds_2_max') is not None and filters.get('footystats_odds_2_max') < 20:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_2 <= filters['footystats_odds_2_max'],
                    SportEventFootystats.odds_ft_2.is_(None)
                ))
            
            # Odds filters (footystats odds for Over/Under markets)
            if filters.get('footystats_odds_over15_min') is not None:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_over15 >= filters['footystats_odds_over15_min'],
                    SportEventFootystats.odds_ft_over15.is_(None)
                ))
            
            if filters.get('footystats_odds_over15_max') is not None:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_over15 <= filters['footystats_odds_over15_max'],
                    SportEventFootystats.odds_ft_over15.is_(None)
                ))
            
            if filters.get('footystats_odds_over25_min') is not None:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_over25 >= filters['footystats_odds_over25_min'],
                    SportEventFootystats.odds_ft_over25.is_(None)
                ))
            
            if filters.get('footystats_odds_over25_max') is not None:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_over25 <= filters['footystats_odds_over25_max'],
                    SportEventFootystats.odds_ft_over25.is_(None)
                ))
            
            if filters.get('footystats_odds_over35_min') is not None:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_over35 >= filters['footystats_odds_over35_min'],
                    SportEventFootystats.odds_ft_over35.is_(None)
                ))
            
            if filters.get('footystats_odds_over35_max') is not None:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_over35 <= filters['footystats_odds_over35_max'],
                    SportEventFootystats.odds_ft_over35.is_(None)
                ))
            
            if filters.get('footystats_odds_over05_min') is not None:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_over05 >= filters['footystats_odds_over05_min'],
                    SportEventFootystats.odds_ft_over05.is_(None)
                ))
            
            if filters.get('footystats_odds_over05_max') is not None:
                query = query.filter(or_(
                    SportEventFootystats.odds_ft_over05 <= filters['footystats_odds_over05_max'],
                    SportEventFootystats.odds_ft_over05.is_(None)
                ))
            
            if filters.get('avg_potential_min') is not None and filters.get('avg_potential_min') > 0:
                query = query.filter(SportEventFootystats.avg_potential >= filters['avg_potential_min'])
            
            if filters.get('avg_potential_max') is not None:
                query = query.filter(SportEventFootystats.avg_potential <= filters['avg_potential_max'])
            
            if filters.get('btts_potential_min') is not None and filters.get('btts_potential_min') > 0:
                query = query.filter(SportEventFootystats.btts_potential >= filters['btts_potential_min'])
            
            if filters.get('btts_potential_max') is not None and filters.get('btts_potential_max') < 1:
                query = query.filter(SportEventFootystats.btts_potential <= filters['btts_potential_max'])
            
            if filters.get('btts_fhg_min') is not None and filters.get('btts_fhg_min') > 0:
                query = query.filter(SportEventFootystats.btts_fhg_potential >= filters['btts_fhg_min'])
            
            if filters.get('btts_fhg_max') is not None and filters.get('btts_fhg_max') < 1:
                query = query.filter(SportEventFootystats.btts_fhg_potential <= filters['btts_fhg_max'])
            
            if filters.get('btts_2hg_min') is not None and filters.get('btts_2hg_min') > 0:
                query = query.filter(SportEventFootystats.btts_2hg_potential >= filters['btts_2hg_min'])
            
            if filters.get('btts_2hg_max') is not None and filters.get('btts_2hg_max') < 1:
                query = query.filter(SportEventFootystats.btts_2hg_potential <= filters['btts_2hg_max'])
            
            if filters.get('o15_potential_min') is not None and filters.get('o15_potential_min') > 0:
                query = query.filter(SportEventFootystats.o15_potential >= filters['o15_potential_min'])
            
            if filters.get('o15_potential_max') is not None and filters.get('o15_potential_max') < 1:
                query = query.filter(SportEventFootystats.o15_potential <= filters['o15_potential_max'])
            
            if filters.get('o25_potential_min') is not None and filters.get('o25_potential_min') > 0:
                query = query.filter(SportEventFootystats.o25_potential >= filters['o25_potential_min'])
            
            if filters.get('o25_potential_max') is not None and filters.get('o25_potential_max') < 1:
                query = query.filter(SportEventFootystats.o25_potential <= filters['o25_potential_max'])
            
            if filters.get('o35_potential_min') is not None and filters.get('o35_potential_min') > 0:
                query = query.filter(SportEventFootystats.o35_potential >= filters['o35_potential_min'])
            
            if filters.get('o35_potential_max') is not None and filters.get('o35_potential_max') < 1:
                query = query.filter(SportEventFootystats.o35_potential <= filters['o35_potential_max'])
            
            if filters.get('o05ht_potential_min') is not None and filters.get('o05ht_potential_min') > 0:
                query = query.filter(SportEventFootystats.o05HT_potential >= filters['o05ht_potential_min'])
            
            if filters.get('o05ht_potential_max') is not None and filters.get('o05ht_potential_max') < 1:
                query = query.filter(SportEventFootystats.o05HT_potential <= filters['o05ht_potential_max'])
        
        # Order by start time ascending for consistent simulation tracking
        query = query.order_by(SportEventMapping.start_time)
        
        # Execute and return as DataFrame
        results = query.all()
        
        # Handle empty results
        if not results:
            return pd.DataFrame()
        
        # Convert SQLAlchemy Row objects to DataFrame using _mapping
        df = pd.DataFrame([dict(row._mapping) for row in results])
        
        # Add calculated difference columns for filtering and analysis
        if not df.empty:
            df['ppg_diff'] = df['ppg_a'] - df['ppg_b']
            df['xg_diff'] = df['team_a_xg_prematch'] - df['team_b_xg_prematch']
        
        # Apply post-query filters for difference values
        if filters and not df.empty:
            # PPG Difference filters
            if filters.get('ppg_diff_min') is not None:
                df = df[df['ppg_diff'] >= filters['ppg_diff_min']]
            
            if filters.get('ppg_diff_max') is not None:
                df = df[df['ppg_diff'] <= filters['ppg_diff_max']]
            
            # XG Difference filters
            if filters.get('xg_diff_min') is not None:
                df = df[df['xg_diff'] >= filters['xg_diff_min']]
            
            if filters.get('xg_diff_max') is not None:
                df = df[df['xg_diff'] <= filters['xg_diff_max']]
        
        return df
        
    finally:
        session.close()

def get_potential_stats(event_ids=None):
    """
    Get detailed potential stats for specific events
    Includes team-specific potentials from SportPotentialFootystats
    """
    session = get_session()
    try:
        query = session.query(SportPotentialFootystats)
        
        if event_ids:
            query = query.filter(SportPotentialFootystats.sport_event_footystats_id.in_(event_ids))
        
        results = query.all()
        df = pd.DataFrame([{
            'id': row.id,
            'sport_event_footystats_id': row.sport_event_footystats_id,
            'stat': row.stat,
            'match_uid': row.match_uid,
            'match_slug': row.match_slug,
            'time_start': row.time_start,
            'date_start': row.date_start,
            'country': row.country,
            'competition': row.competition,
            'team_a': row.team_a,
            'team_b': row.team_b,
            'team_potential': row.team_potential,
            'average': row.average,
            'matches_count': row.matches_count,
            'probability': row.probability,
            'odd': row.odd,
            'probabilities': row.probabilities,
            'odds': row.odds
        } for row in results])
        
        return df
        
    finally:
        session.close()

def get_filter_options():
    """
    Get available filter options from the database
    Returns lists of countries, competitions, etc.
    """
    session = get_session()
    try:
        # Get unique countries
        countries_query = session.query(SportEventMapping.country).filter(
            SportEventMapping.country.isnot(None)
        ).distinct().all()
        countries = sorted([c[0] for c in countries_query if c[0]])
        
        # Get unique competitions with country context
        competitions_query = session.query(
            SportEventMapping.competition, 
            SportEventMapping.country
        ).filter(
            SportEventMapping.competition.isnot(None),
            SportEventMapping.country.isnot(None)
        ).distinct().all()
        
        # Create competition-country pairs and deduplicate
        competition_pairs = set()
        for comp, country in competitions_query:
            if comp and country:
                competition_pairs.add(f"{comp} ({country})")
        competitions = sorted(list(competition_pairs))
        
        # Get unique bookmakers
        bookmakers_query = session.query(SportEventBookmaker.bookmaker).filter(
            SportEventBookmaker.bookmaker.isnot(None)
        ).distinct().all()
        bookmakers = sorted([b[0] for b in bookmakers_query if b[0]])
        
        # Get unique stat types from potentials
        stats_query = session.query(SportPotentialFootystats.stat).filter(
            SportPotentialFootystats.stat.isnot(None)
        ).distinct().all()
        stat_types = sorted([s[0] for s in stats_query if s[0]])
        
        # Get date range
        date_range_query = session.query(
            func.min(SportEventMapping.start_time).label('min_date'),
            func.max(SportEventMapping.start_time).label('max_date')
        ).filter(SportEventMapping.start_time.isnot(None)).first()
        
        return {
            'countries': countries,
            'competitions': competitions,
            'bookmakers': bookmakers,
            'stat_types': stat_types,
            'min_date': date_range_query.min_date,
            'max_date': date_range_query.max_date
        }
        
    finally:
        session.close()

def calculate_potential_accuracy(df, potential_column, result_parser=None):
    """
    Calculate accuracy of potentials vs actual results
    """
    if df.empty or potential_column not in df.columns:
        return None
    
    # Filter out rows with missing data
    valid_df = df[df[potential_column].notna()].copy()
    
    if valid_df.empty:
        return None
    
    # Parse results if parser provided
    if result_parser:
        valid_df = valid_df.copy()
        valid_df['parsed_result'] = valid_df['bookmaker_score_ft'].apply(result_parser)
    
    # Calculate accuracy metrics
    accuracy_metrics = {
        'total_matches': len(valid_df),
        'matches_with_potential_data': len(valid_df[valid_df[potential_column].notna()]),
        'avg_potential': valid_df[potential_column].mean(),
        'max_potential': valid_df[potential_column].max(),
        'min_potential': valid_df[potential_column].min(),
        'potential_std': valid_df[potential_column].std()
    }
    
    return accuracy_metrics

def parse_score(score_str):
    """Parse score string into home and away goals"""
    if pd.isna(score_str) or score_str == '':
        return 0, 0
    
    try:
        # Handle various score formats: "1-2", "1:2", "1 - 2", "1 : 2"
        score_str = str(score_str).strip()
        if ':' in score_str:
            parts = score_str.split(':')
        elif '-' in score_str:
            parts = score_str.split('-')
        else:
            return 0, 0
        
        if len(parts) == 2:
            home = int(parts[0].strip())
            away = int(parts[1].strip())
            return home, away
        else:
            return 0, 0
    except (ValueError, IndexError):
        return 0, 0

def parse_all_scores(df):
    """Parse all score columns (fh, 2h, ft) into structured data"""
    if df.empty:
        return df
    
    # Parse full-time scores
    ft_scores = df['bookmaker_score_ft'].apply(lambda x: pd.Series(parse_score(x)))
    ft_scores.columns = ['home_ft', 'away_ft']
    
    # Parse half-time scores
    if 'bookmaker_score_fh' in df.columns:
        fh_scores = df['bookmaker_score_fh'].apply(lambda x: pd.Series(parse_score(x)))
        fh_scores.columns = ['home_fh', 'away_fh']
    else:
        fh_scores = pd.DataFrame({'home_fh': 0, 'away_fh': 0}, index=df.index)
    
    # Parse second half scores
    if 'bookmaker_score_2h' in df.columns:
        sh_scores = df['bookmaker_score_2h'].apply(lambda x: pd.Series(parse_score(x)))
        sh_scores.columns = ['home_2h', 'away_2h']
    else:
        sh_scores = pd.DataFrame({'home_2h': 0, 'away_2h': 0}, index=df.index)
    
    # Calculate derived metrics
    result_df = pd.concat([df, ft_scores, fh_scores, sh_scores], axis=1)
    
    # Total goals
    result_df['total_ft'] = result_df['home_ft'] + result_df['away_ft']
    result_df['total_fh'] = result_df['home_fh'] + result_df['away_fh']
    result_df['total_2h'] = result_df['home_2h'] + result_df['away_2h']
    
    # Goal timing analysis
    result_df['goals_1h_half'] = result_df['total_fh'] / result_df['total_ft'].where(result_df['total_ft'] > 0, 1)
    result_df['goals_2h_half'] = result_df['total_2h'] / result_df['total_ft'].where(result_df['total_ft'] > 0, 1)
    
    # Comeback analysis
    result_df['ht_home_winning'] = result_df['home_fh'] > result_df['away_fh']
    result_df['ht_away_winning'] = result_df['away_fh'] > result_df['home_fh']
    result_df['ht_draw'] = result_df['home_fh'] == result_df['away_fh']
    
    result_df['ft_home_winning'] = result_df['home_ft'] > result_df['away_ft']
    result_df['ft_away_winning'] = result_df['away_ft'] > result_df['home_ft']
    result_df['ft_draw'] = result_df['home_ft'] == result_df['away_ft']
    
    # Comeback detection
    result_df['home_comeback'] = (~result_df['ht_home_winning']) & result_df['ft_home_winning']
    result_df['away_comeback'] = (~result_df['ht_away_winning']) & result_df['ft_away_winning']
    result_df['any_comeback'] = result_df['home_comeback'] | result_df['away_comeback']
    
    # HT/FT double result
    result_df['ht_ft_result'] = (
        result_df.apply(lambda row: get_ht_ft_result(row), axis=1)
    )
    
    return result_df

def get_ht_ft_result(row):
    """Get HT/FT double result (H/H, H/D, H/A, D/H, D/D, D/A, A/H, A/D, A/A)"""
    if pd.isna(row['home_fh']) or pd.isna(row['away_fh']) or pd.isna(row['home_ft']) or pd.isna(row['away_ft']):
        return 'Unknown'
    
    def get_result(home, away):
        if home > away:
            return 'H'
        elif home < away:
            return 'A'
        else:
            return 'D'
    
    ht_result = get_result(row['home_fh'], row['away_fh'])
    ft_result = get_result(row['home_ft'], row['away_ft'])
    
    return f"{ht_result}/{ft_result}"

def analyze_potential_performance(df, potential_thresholds=None):
    """
    Analyze performance at different potential thresholds
    """
    if df.empty:
        return None
    
    if potential_thresholds is None:
        potential_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    for threshold in potential_thresholds:
        # Filter events above threshold
        threshold_df = df[df['avg_potential'] >= threshold].copy()
        
        if threshold_df.empty:
            continue
        
        # Parse scores and calculate success metrics
        threshold_df[['home_goals', 'away_goals', 'total_goals']] = threshold_df['bookmaker_score_ft'].apply(
            lambda x: pd.Series(parse_score(x))
        )
        
        # Calculate various success metrics
        total_events = len(threshold_df)
        events_with_scores = len(threshold_df[threshold_df['total_goals'].notna()])
        
        if events_with_scores > 0:
            avg_total_goals = threshold_df['total_goals'].mean()
            over_2_5_rate = (threshold_df['total_goals'] > 2.5).mean()
            btts_rate = ((threshold_df['home_goals'] > 0) & (threshold_df['away_goals'] > 0)).mean()
        else:
            avg_total_goals = 0
            over_2_5_rate = 0
            btts_rate = 0
        
        results.append({
            'threshold': threshold,
            'total_events': total_events,
            'events_with_scores': events_with_scores,
            'avg_potential': threshold_df['avg_potential'].mean(),
            'avg_total_goals': avg_total_goals,
            'over_2_5_rate': over_2_5_rate,
            'btts_rate': btts_rate
        })
    
    return pd.DataFrame(results)

def export_to_csv(df, filename=None):
    """
    Export DataFrame to CSV file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sports_matches_export_{timestamp}.csv"
    
    df.to_csv(filename, index=False)
    return filename
