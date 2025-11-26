#!/usr/bin/env python3
# %%
import pandas as pd
from sqlalchemy.engine import result
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import random
import json
import asyncio

import os
from pathlib import Path

from dotenv import load_dotenv
env_file = 'settings.env'
dotenv_path = Path(env_file)
load_dotenv(dotenv_path=dotenv_path)

from sqlalchemy import create_engine, and_, exc, or_, extract, func
from sqlalchemy.orm import sessionmaker

db_database = os.environ['MYSQL_DATABASE']
db_user = os.environ['MYSQL_USER']
db_password = os.environ['MYSQL_PASSWORD']
db_ip_address = os.environ['MYSQL_IP_ADDRESS']
db_port = os.environ['MYSQL_PORT']
SERVER_MODE = os.environ['ENV']
MYSQL_CONNECTOR_STRING = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_ip_address}:{db_port}/{db_database}?charset=utf8mb4&collation=utf8mb4_general_ci'
engine = create_engine(MYSQL_CONNECTOR_STRING, echo=False, pool_pre_ping=True, pool_recycle=300, pool_size=20, max_overflow=0)





import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import *
from utils import *
# from utils_db import *
from utils_ai import *
from utils_sports import *

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'log/update_sports_results_{datetime.now(timezone.utc).strftime("%Y%m%d")}.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)



# %% Proxy Providers
SPORTS_SCRAPING_RESULTS_PROXY_PROVIDER = os.environ.get('SPORTS_SCRAPING_RESULTS_PROXY_PROVIDER', None)




# %% 22bet Scraping Functions
def scrape_22bet_results(champId, dateFrom, dateTo):
    url = f"https://22bet.com/service-api/result/web/api/v3/games?champId={champId}&dateFrom={dateFrom}&dateTo={dateTo}&lng=en"
    
    try:
        proxy_dict = {}
        proxy = fetch_proxy(provider=SPORTS_SCRAPING_RESULTS_PROXY_PROVIDER)
        if proxy:
            proxy_dict = {'http': proxy, 'https': proxy}
        # print(proxy_dict)
        
        headers = {
            "User-Agent": get_random_user_agent(),
            'Cache-Control': 'no-cache', 'Pragma': 'no-cache',
        }
        
        response = requests.get(url, headers=headers, proxies=proxy_dict, timeout=10)
        if response.status_code in [403, 429, 202, 101]:
            logger.warning(f"Warning! WAF blocked request. Status: " + str(response.status_code))
            time.sleep(1)
            response = requests.get(url, headers=headers, proxies=proxy_dict, timeout=10)
            response.raise_for_status()
        response.raise_for_status()

        response_json = response.json()
        # Cleaning data and returning necessary information
        if response_json and 'items' in response_json:
            return response_json['items']
        return None
    except requests.RequestException as e:
        logger.error(f"Error scraping 22bet results: {e}")
        return None

# TODO: Alternative function to scrape 22bet results by search
# Example: https://22bet.com/service-api/result/web/api/v1/search?country=187&dateFrom=1761346800&dateTo=1761433200&gr=151&lng=en&ref=151&text=Slovan+Liberec+U19
# def scrape_22bet_results_by_search(query, dateFrom, dateTo):
    # TODO: avoid competition name="Alternative Matches"
def scrape_22bet_results_by_search(q, dateFrom, dateTo):
    url = f"https://22bet.com/service-api/result/web/api/v1/search?country=&dateFrom={dateFrom}&dateTo={dateTo}&gr=151&lng=en&ref=151&text={q}"
    
    try:
        proxy_dict = {}
        proxy = fetch_proxy(provider=SPORTS_SCRAPING_RESULTS_PROXY_PROVIDER)
        if proxy:
            proxy_dict = {'http': proxy, 'https': proxy}
        # print(proxy_dict)
        
        headers = {
            "User-Agent": get_random_user_agent(),
            'Cache-Control': 'no-cache', 'Pragma': 'no-cache',
        }
        
        response = requests.get(url, headers=headers, proxies=proxy_dict, timeout=10)
        if response.status_code in [403, 429, 202, 101]:
            logger.warning(f"Warning! WAF blocked request. Status: " + str(response.status_code))
            time.sleep(1)
            response = requests.get(url, headers=headers, proxies=proxy_dict, timeout=10)
            response.raise_for_status()
        response.raise_for_status()

        response_json = response.json()
        # Cleaning data and returning necessary information
        if response_json and 'items' in response_json:
            return response_json['items']
        return None
    except requests.RequestException as e:
        logger.error(f"Error scraping 22bet results: {e}")
        return None




def update_22bet_results(session, df, bookmaker, league_22bet_results):
    """
    Update 22bet results for a specific league
    
    Args:
        session: Database session
        df: DataFrame with events to update
        bookmaker: Bookmaker name (should be '22bet')
        league_22bet_results: List of results from scrape_22bet_results
    """
    if league_22bet_results:
        for result in league_22bet_results:
            r_id = result.get('id',None)
            r_champId = result.get('champId',None)
            r_start_time = result.get('dateStart',None) # in epoch
            r_start_date = datetime.fromtimestamp(r_start_time).date()
            r_team_a = result.get('opp1',None)
            r_team_b = result.get('opp2',None)
            # logger.debug(f"Looking for {r_champId}/{r_id} {r_start_date} {r_team_a} - {r_team_b}")

            if (df.bkm_game_id.eq(r_id) & df.bkm_league_id.eq(r_champId)).any():
                df_row = df[(df.bkm_game_id.eq(r_id) & df.bkm_league_id.eq(r_champId))].iloc[0]
                logger.info(f"Found by id: id={df_row.id} bkm_id={df_row.bkm_id} bkm_game_id={df_row.bkm_game_id} bkm_league_id={df_row.bkm_league_id} | {r_start_date} ({r_team_a} - {r_team_b})")
            
            # TODO: find match in df by team_a and team_b and same start_date (maybe bookmaker id changed)
            elif (df.start_date.eq(r_start_date) & df.team_a.eq(r_team_a) & df.team_b.eq(r_team_b)).any():
                df_row = df[(df.start_date.eq(r_start_date) & df.team_a.eq(r_team_a) & df.team_b.eq(r_team_b))].iloc[0]
                logger.info(f"Found by date/teams: id={df_row.id} bkm_id={df_row.bkm_id} bkm_game_id={df_row.bkm_game_id} bkm_league_id={df_row.bkm_league_id} | {r_start_date} ({r_team_a} - {r_team_b})")
            else:
                print(f"Match {r_champId}/{r_id} not found in df")
                df_row = pd.Series([], dtype=object)
            
            if not df_row.empty:
                # print(f"Found id={df_row.id} bkm_id={df_row.bkm_id} bkm_game_id={df_row.bkm_game_id} bkm_league_id={df_row.bkm_league_id}")
                sport_event_mapping = session.query(SportEventMapping).filter(SportEventMapping.id == int(df_row.id)).first()
                if sport_event_mapping:
                    sport_event_bookmaker = session.query(SportEventBookmaker).filter(SportEventBookmaker.id == int(df_row.bkm_id)).first()
                    logger.info(f"Found match: [Mapping #{sport_event_mapping.id}] | {bookmaker} | {df_row.start_date} | {sport_event_bookmaker.league_id}/{sport_event_bookmaker.game_id} ({sport_event_bookmaker.team_a} - {sport_event_bookmaker.team_b}) score={result.get('score',None)}")
                    if r_id != sport_event_bookmaker.game_id:
                        # logger.warning(f"Match {r_champId}/{r_id} not found in df, updating game_id")
                        sport_event_bookmaker.game_id = r_id
                    
                    
                    score_ok = True
                    score_structure = parse_score(result.get('score',None))
                    if score_structure:
                        sport_event_bookmaker.score = score_structure.get('error',None) if score_structure.get('error',None) else result.get('score',None) # score="3:1(2:0,1:1)"  score_ft=3:1 score_fh=2:0 score_ht=1:1
                        sport_event_bookmaker.score_ft = score_structure.get('ft',None)
                        sport_event_bookmaker.score_fh = score_structure.get('fh',None)
                        sport_event_bookmaker.score_2h = score_structure.get('sh',None)
                        if score_structure.get('error',None):
                            score_ok = False
                            logger.warning(f"Error parsing score in match {result.get('champId',None)}/{result.get('id',None)}: {score_structure.get('error',None)}")
                    
                    if 'subGame' in result:
                        logger.debug(f"Match {result.get('champId',None)}/{result.get('id',None)} has subGame={result.get('subGame')}")
                        sport_event_bookmaker.subgames = result.get('subGame',None)
                        # TODO: if subGame has in the list one with title 'Corners', make the same structure of split_score
                        for subgame in result.get('subGame',None):
                            if subgame.get('title',None) == 'Corners':
                                corners_structure = parse_score(subgame.get('score','')) if subgame.get('score','') else None
                                if corners_structure:
                                    sport_event_bookmaker.corners_ft = corners_structure.get('ft',None)
                                    sport_event_bookmaker.corners_fh = corners_structure.get('fh',None)
                                    sport_event_bookmaker.corners_2h = corners_structure.get('sh',None)
                                    
                    
                    if not score_ok:
                        sport_event_bookmaker.status = 'score_error'
                        sport_event_mapping.status = 'score_error'
                    elif 'status' in result:
                        logger.debug(f"Match {result.get('champId',None)}/{result.get('id',None)} has status={result.get('status')}")
                        sport_event_bookmaker.status = result.get('status',None)
                        sport_event_mapping.status = result.get('status',None)
                    else:
                        sport_event_bookmaker.status = 'finished'
                        sport_event_mapping.status = 'finished'
                    
                    logger.info(f"Match updated: {result.get('champId',None)}/{result.get('id',None)} , status={sport_event_mapping.status}")

                    sport_event_bookmaker.date_updated = datetime.now(timezone.utc)
                    sport_event_mapping.date_updated = datetime.now(timezone.utc)
                    # break
                
                session.commit()
    else:
        logger.warning(f"No results found for {bookmaker}")






# %% 22bet Database Functions
def update_22bet_event(session, id=None, event={}, sport='football'):
    """Update 22bet event in database"""
    try:
        event_dict = {
            'league_id': event['league_id'],
            'game_id': event['game_id'],
            'match_url': f"/live/{sport}/{event['league_id']}/{event['game_id']}",
            'time': event['start_date'],
            'country': event['country'],
            'competition': event['league_name'],
            'team_a': event['team_a'],
            'team_b': event['team_b'],
        }
        event_model_id = None
        # Check if same match_uid exists to ignore saving
        existing_event = session.query(SportEventBookmaker).filter(SportEventBookmaker.bookmaker == bookmaker, SportEventBookmaker.league_id == event['league_id'], SportEventBookmaker.game_id == event['game_id']).first()
        if existing_event:
            # Check if exists, but have different time
            if session.query(SportEventBookmaker).filter(SportEventBookmaker.bookmaker == bookmaker, SportEventBookmaker.league_id == event['league_id'], SportEventBookmaker.game_id == event['game_id'], SportEventBookmaker.time != event_dict['time']).first():
                logger.info(f"Event {event['league_id']}/{event['game_id']} ({event['team_a']} - {event['team_b']}) already exists with different time, updating..")
                event_dict['date_updated'] = datetime.now(timezone.utc)
                session.query(SportEventBookmaker).filter(SportEventBookmaker.bookmaker == bookmaker, SportEventBookmaker.league_id == event['league_id'], SportEventBookmaker.game_id == event['game_id']).update(event_dict)
            else:
                logger.info(f"Event {event['league_id']}/{event['game_id']} ({event['team_a']} - {event['team_b']}) already exists")
            event_model_id = existing_event.id
        else:
            # Create a new instance of the model
            # print(event_dict)
            event_dict['date_scraped'] = datetime.now(timezone.utc)
            event_model = SportEventBookmaker(**event_dict)
            logger.info(f"Event {event['league_id']}/{event['game_id']} {event['country']} {event['team_a']} - {event['team_b']} saved to database")
            # Add the instance to the session
            session.add(event_model)
            event_model_id = event_model.id
            
        # Commit the session to save changes
        session.commit()
        if event_model_id:
            return event_model_id
        
    except Exception as e:
        logger.error(f"Error saving 22bet event: {e.__class__.__name__}: {e}")
        session.rollback()



# %% Main Scraping Functions
def scrape_and_save_sports_results(session):
    """Main function to scrape and save sports data from various sources"""
    
    # try:
    if True:
        print("Starting sports data scraping process")
        
        # Scraping logic :
        # 1. Get matched events from DB (sport_event_mapping status=matched)
        # 2. 
        # Save the data to the database

        # Get upcoming unmatched FootyStats events from DB, and scrape 22bet livefeed for future matching
        matched_events = get_matched_sports_events(session)
        if not matched_events:
            logger.info("No matched events to update")
            return
        
        dtypes = {
            'id': int,
            'sport_event_footystats_id': int,
            'sport_event_bookmaker_id': int,
            'sport_event_id': int,
            'bkm_id': int,
            'bkm_bookmaker': str,
            'bkm_league_id': int,
            'bkm_game_id': int,
            'team_a': str,
            'team_b': str,
            'status': str,
            'start_time': 'datetime64[ns]',
            'start_date': 'datetime64[ns]',
            'date_added': 'datetime64[ns]',
            'date_updated': 'datetime64[ns]',
        }
        df = pd.DataFrame(columns=list(dtypes.keys()))
        df = df.astype(dtypes)
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        
        for event in matched_events:
            if event.start_time and event.start_time.replace(tzinfo=None) < (pd.Timestamp.now(tz='UTC').tz_convert(None) - pd.Timedelta(hours=48)):
                continue
            df.loc[len(df)] = {
                'id': int(event.id),
                'sport_event_footystats_id': event.sport_event_footystats_id,
                'sport_event_bookmaker_id': event.sport_event_bookmaker_id,
                'bkm_id': int(event.sport_event_bookmaker.id),
                'bkm_bookmaker': event.sport_event_bookmaker.bookmaker,
                'bkm_league_id': int(event.sport_event_bookmaker.league_id),
                'bkm_league_name': event.sport_event_bookmaker.competition,
                'bkm_game_id': int(event.sport_event_bookmaker.game_id),
                'bkm_country': event.sport_event_bookmaker.country,
                'team_a': event.sport_event_bookmaker.team_a,
                'team_b': event.sport_event_bookmaker.team_b,
                'status': event.status,
                'start_time': event.start_time,
                'start_date': event.start_time.date(),
                'date_added': event.date_added,
                'date_updated': event.date_updated
            }

        # Only keep df events that have start_time 2 hours ago
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        # df = df[(df['start_time'] >= (pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=2)))]
        # df = df[(df['start_time'] >= (pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=6)))]

        print(f"Number of events to update: {len(df)}")

        grouped_events_by_league = df.sort_values('start_time', ascending=True).groupby(['bkm_bookmaker','bkm_league_id','start_date']).agg({'bkm_game_id':'count', 'start_date':'first', 'start_time':'first'})

        for i, row  in grouped_events_by_league.iterrows():
            print(i[0], i[1], row.start_time)
            bookmaker = i[0]
            if bookmaker == '22bet':
                scrape_league_id = i[1]
                scrape_dateFrom = int((row.start_time.replace(minute=0, microsecond=0) - timedelta(hours=2)).timestamp())
                scrape_dateTo = scrape_dateFrom + (2 * 24 * 60 * 60) # +2 days
                scrape_url_params = f"champId={scrape_league_id}&dateFrom={scrape_dateFrom}&dateTo={scrape_dateTo}"

                # Show bulletpoint info for each event in this league
                league_events = df[(df['bkm_bookmaker'] == bookmaker) & (df['bkm_league_id'] == scrape_league_id) & (df['start_date'] == row.start_date)].sort_values('start_time')
                logger.info(f"Scraping {bookmaker} for {row.bkm_game_id} results) : {scrape_url_params}")
                print(f"  Events to search for results ({len(league_events)} events):")
                for _, event in league_events.iterrows():
                    print(f"    • Mapping #{event['id']} | game_id={event['bkm_game_id']} \t {event['team_a']} vs {event['team_b']} - {event['start_time'].strftime('%Y-%m-%d %H:%M')}")
                print()
                
                league_22bet_results = scrape_22bet_results(champId=scrape_league_id, dateFrom=scrape_dateFrom, dateTo=scrape_dateTo)
                # print(league_22bet_results)
                if league_22bet_results:
                    update_22bet_results(session, df, bookmaker, league_22bet_results)
                else:
                    logger.info(f"No results found for {bookmaker} {scrape_league_id} {scrape_dateFrom} {scrape_dateTo}")
                
            session.commit()
            time.sleep(3)
            # break
        
    # except Exception as e:
    #     logger.error(f"Error in sports data scraping: {e}")
    #     raise




# %% Main Execution
if __name__ == "__main__":
    # try:
    if True:
        # Create a session
        Session = sessionmaker(bind=engine, autocommit=False, autoflush=True)
        session = Session()
        
        # Check for command line arguments to determine scraping mode
        scraping_mode = "all"  # default mode
        if len(sys.argv) > 1:
            scraping_mode = sys.argv[1]
        
        logger.debug(f"Running update_sports_results in '{scraping_mode}' mode")
        
        # Main scraping function
        scrape_and_save_sports_results(session)
        
        logger.debug("Script execution completed successfully")
        
    # except Exception as e:
    #     logger.error(f"Fatal error in main execution: {e}")
    #     sys.exit(1)
    # finally:
    #     if 'session' in locals():
    #         session.close()
