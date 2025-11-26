#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import logging

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

# Packages specific to this script
from bs4 import BeautifulSoup
import pytz
import textdistance

# Load utils_sports
from utils_sports import *


# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import *
from utils import *
from utils_db import *
from utils_ai import *


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'log/scrape_sports_stats_{datetime.now(timezone.utc).strftime("%Y%m%d")}.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# %% Configuration
# Proxy Providers
SPORTS_SCRAPING_FOOTYSTATS_STATS_PROXY_PROVIDER = os.environ.get('SPORTS_SCRAPING_FOOTYSTATS_STATS_PROXY_PROVIDER', None)


# %% Scraping Functions
# Example: https://footystats.org/stats/btts-stats
def scrape_footystats_page(page_path):
    """Scrape FootyStats stats page"""

    proxy_dict = {}
    proxy = fetch_proxy(provider=SPORTS_SCRAPING_FOOTYSTATS_STATS_PROXY_PROVIDER)
    if proxy:
        proxy_dict = {'http': proxy, 'https': proxy}
    print(proxy_dict)

    url = f"https://footystats.org/{page_path}"

    headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:141.0) Gecko/20100101 Firefox/141.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'Origin': 'https://footystats.org',
    'Connection': 'keep-alive',
    'Referer': url,
    'Cookie': 'tz=Africa/Lagos; darkmode=off;',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'Priority': 'u=0',
    'TE': 'trailers'
    }

    payload = {}
    
    # Make the API request
    print(page_path)
    if page_path.startswith("ajax_wdw.php"):
        # get date from query string then remove it
        url_day = page_path.split("=")[1]
        if url_day == "today":
            scrape_date = int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        elif url_day == "tomorrow":
            scrape_date = int((datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        url = url.replace(f"?day={url_day}", "")
        print(url, url_day, scrape_date)
        payload = {'data1': scrape_date, 'data2': '', 'data3': 'x'}
    
    if payload:
        response = requests.post(url, headers=headers, proxies=proxy_dict, timeout=10, data=payload)
    else:
        response = requests.get(url, headers=headers, proxies=proxy_dict, timeout=10)
    if response.status_code in [403, 429, 202, 101]:
        logger.warning(f"Warning! WAF blocked request. Status: " + str(response.status_code))
        time.sleep(1)
        if payload:
            response = requests.post(url, headers=headers, proxies=proxy_dict, timeout=10, data=payload)
        else:
            response = requests.get(url, headers=headers, proxies=proxy_dict, timeout=10)
        response.raise_for_status()
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    return soup
 
def get_stats_from_table(config, soup=None):
    columns = []
    stats_raw = []
    stats = []
    if config is None or 'stat_name' not in config or 'css_element' not in config:
        logger.warning("Warning! No config provided or invalid config")
        return None
    if soup is None:
        logger.warning(f"Warning! No soup provided for {config['stat_name']}")
        return None
    
    logger.info(f"Scraping stats table: {config}")

    stats_table = soup.select(config['css_element'])
    if stats_table and len(stats_table) > 0:
        stats_table = stats_table[0]
    if stats_table:
        for ele in stats_table.find('thead').find_all('th'):
            for div in ele.find_all('div'):
                div.decompose()

        columns = [ele.text.strip() for ele in stats_table.find('thead').find_all('th')]
        columns[0] = '#'
        columns = [re.sub(r'Over (\d+(?:\.\d+)?)\+? %?', 'Probability', ele) if ele else ele for ele in columns]
        columns = [re.sub(r'(\d+(?:\.\d+)?)\+? Corners %?', 'Probability', ele) if ele else ele for ele in columns]
        columns = [re.sub(r'(\d+(?:\.\d+)?\+?|Draw|Corners) Odds', 'Odds', ele) if ' Odds' in ele else ele for ele in columns]
        columns = [ele\
            .replace('Next Match', 'Date') \
            .replace('BTTS %', 'Probability') \
            .replace('AVG Draw %', 'Probability') \
            .replace('AVG Corners', 'Average') \
            .replace('Goals / Match', 'Average') \
            .replace('Corners / Match', 'Average') \
            for ele in columns if ele]
        columns = [ele\
            .replace('BTTS', 'Matches Count') \
            .replace('SBH', 'Matches Count') \
            .replace('%', 'Probability') \
            for ele in columns if ele]

        rows = stats_table.find('tbody').find_all('tr')
        stats_raw = []
        for row in rows:
            cols = [ele for ele in row.find_all('td')]
            stats_raw.append(dict(zip(columns, cols)))
        
        logger.info(f"Columns: {columns}")
        logger.info(f"Scraped {len(stats_raw)} stats")

        for row in stats_raw:
            stat_single = {
                'stat': config['stat_name'],
                'match_uid': None,
                'match_slug': None,
                'time_start_str': None,
                'time_start': None,
                'date_start': None,
                'country': None,
                'competition': None,
                'team_a': None,
                'team_b': None,
                'team_potential': None,
                'average': None,
                'matches_count': None,
                'probability': None,
                'odd': None,
                'probabilities': None,
                'odds': None,
            }
            stat_single['time_start_str'] = row['Date'].find('a').text.strip() if 'Date' in columns else None
            if stat_single['time_start_str'] and stat_single['time_start_str'].lower() in ['today', 'tomorrow', 'in 2 days']: #, 'in 3 days']:
                # Convert time_start_str to date_start
                stat_single['date_start'] = datetime.now(timezone.utc).date()
                if stat_single['time_start_str'].lower() == 'tomorrow':
                    stat_single['date_start'] += timedelta(days=1)
                if stat_single['time_start_str'].lower() == 'in 2 days':
                    stat_single['date_start'] += timedelta(days=2)
                if stat_single['time_start_str'].lower() == 'in 3 days':
                    stat_single['date_start'] += timedelta(days=3)
                # stat_single['date_start'] = stat_single['date_start'].strftime('%Y-%m-%d')
            else:
                continue

            match_url = row['Date'].find('a').get('href')
            match_url_re = re.search(r'\/([a-z0-9-]+)\/([a-z0-9-]+)#?(\d+)?$', match_url)
            if match_url_re:
                stat_single['country'] = match_url_re.group(1)
                stat_single['match_slug'] = match_url_re.group(2)
                stat_single['match_uid'] = match_url_re.group(3)
            # if Match column exists: teams names in this format (Inter Milan U19 vs Kairat U19)
            match_teams = row['Match'].text.strip() if 'Match' in columns else None
            if match_teams:
                stat_single['team_a'] = match_teams.split('vs')[0].strip()
                stat_single['team_b'] = match_teams.split('vs')[1].strip()
            # if Team column exists: team name in this format (Inter Milan U19)
            stat_single['team_potential'] = row['Team'].text.strip() if 'Team' in columns else None
            stat_single['probability'] = float(row['Probability'].text.strip().replace('%', '')) / 100 if 'Probability' in columns else None
            if stat_single['probability'] and config.get('probability_threshold', None) and stat_single['probability'] < config['probability_threshold']:
                continue

            stat_single['average'] = float(row['Average'].text.strip().replace('/ match', '')) if 'Average' in columns else None
            if stat_single['average'] and config.get('average_threshold', None) and stat_single['average'] < config['average_threshold']:
                continue

            stat_single['matches_count'] = None
            matches_count_str = re.sub(r'[^\w\s]','/', row['Matches Count'].text.strip()) if 'Matches Count' in columns else None
            if matches_count_str:
                matches_count = re.search(r'\d+\s\/(\d+) matches', matches_count_str)
                if matches_count:
                    stat_single['matches_count'] = int(matches_count.group(1))
            if stat_single['matches_count'] and config.get('matches_count_threshold', None) and stat_single['matches_count'] < config['matches_count_threshold']:
                continue
            
            if config['stat_name'] == "wdw":
                odds = {}
                odd_order = {0: "1", 1: "X", 2: "2"}
                odd_count = 0
                for li in row['Odds'].find_all('li'):
                    text = li.text.strip()
                    if text:
                        odds[odd_order[odd_count]] = float(text)
                    odd_count += 1
                stat_single['odds'] = odds
            else:
                stat_single['odd'] = float(row['Odds'].text.strip()) if 'Odds' in columns else None
            
            stats.append(stat_single)
    return stats



def scrape_footystats_stats(session):
    """
    Scrape FootyStats stats pages into structured data
    
    Returns:
        list: List of events or None if error
    """
    try:
    # if True:
        stats_config = {
            # Odds & Probability for BTTS
            "stats/btts-stats": [
                {
                    "stat_name": "btts-upcoming-matches",
                    "probability_threshold": 0.7,
                    "css_element": ".section:has(#goUpcomingMatches) table",
                },
                {
                    "stat_name": "btts-top-teams",
                    "probability_threshold": 0.8,
                    "css_element": ".section:has(#goBTTS) table",
                }
            ],
            # Odds & Probability for Draws
            'stats/draws': [
                {
                    "stat_name": "draws-upcoming-matches",
                    "probability_threshold": 0.5,
                    "css_element": ".section:has(#goUpcomingMatches) table",
                },
                {
                    "stat_name": "draws-top-teams",
                    "probability_threshold": 0.6,
                    "css_element": ".section:has(#goTeams) table",
                }
            ],
            # Odds & Probability for Over 9.5+ corners
            'stats/corner-stats': [
                {
                    "stat_name": "corners-top-teams", 
                    "probability_threshold": 0.5,
                    "average_threshold": 12,
                    "css_element": ".section:has(#cornerStats) table",
                },
                {
                    "stat_name": "corners-upcoming-matches", 
                    "probability_threshold": 0.6,
                    "average_threshold": None,
                    "css_element": ".section:has(#goUpcomingMatches) table",
                },
            ],
            # Odds & Probability for Over 9.5+ corners
            'ajax_corners.php?type=corners-o95': [
                {
                    "stat_name": "corners-o95",
                    "probability_threshold": 0.7,
                    "css_element": ".section table",
                }
            ],
            # Odds & Probability for Over 10.5+ corners
            'ajax_corners.php?type=corners-o105': [
                {
                    "stat_name": "corners-o105",
                    "probability_threshold": 0.6,
                    "css_element": ".section table",
                }
            ],
            # Odds & Probability for Over 11.5+ corners
            'ajax_corners.php?type=corners-o115': [
                {
                    "stat_name": "corners-o115",
                    "probability_threshold": 0.6,
                    "css_element": ".section table",
                }
            ],
            # Odds & Probability for Over 12.5+ corners
            'ajax_corners.php?type=corners-o125': [
                {
                    "stat_name": "corners-o125",
                    "probability_threshold": 0.6,
                    "css_element": ".section table",
                }
            ],

            # # Odds & Probability for Win-Draw-Win
            # 'ajax_wdw.php?day=today': [
            #     {
            #         "stat_name": "wdw",
            #         "css_element": "table",
            #     }
            # ],
            # 'ajax_wdw.php?day=tomorrow': [
            #     {
            #         "stat_name": "wdw",
            #         "css_element": "table",
            #     }
            # ],
            
        }
        for stat_page_path, stat_config in stats_config.items():
            soup = scrape_footystats_page(stat_page_path)
            # print(soup)
            for stat_table_config in stat_config:
                logger.info(f"Scraping {stat_table_config['stat_name']} stats table")
                stats_data = get_stats_from_table(config=stat_table_config, soup=soup)
                # print(stats)

                # break   # FOR TEST PURPOSE
                
                stat_name = stat_table_config['stat_name']
                for stat_data in stats_data:
                    # print(stat_data)
                    if not stat_data['match_slug']:
                        logger.warning(f"No match slug found for stat data: {stat_data}")
                        continue
                    # Save or update each stat record in SportPotentialFootystats
                    existing_stat = session.query(SportPotentialFootystats).filter(
                        SportPotentialFootystats.stat == stat_name,
                        SportPotentialFootystats.match_slug == stat_data['match_slug'],
                        SportPotentialFootystats.date_start >= stat_data['date_start'] - timedelta(days=1),
                        SportPotentialFootystats.date_start <= stat_data['date_start'] + timedelta(days=1),
                    ).first()
                    if existing_stat:
                        for key, value in stat_data.items():
                            setattr(existing_stat, key, value)
                        existing_stat.date_updated = datetime.now(timezone.utc)
                        logger.info(f"Updated existing stat: {stat_data['match_slug']}")
                    else:
                        # Check if match already exists in SportEventFootystats
                        match_event = session.query(SportEventFootystats).filter(
                            SportEventFootystats.match_slug == stat_data['match_slug'],
                            SportEventFootystats.time >= stat_data['date_start'] - timedelta(days=1),
                            SportEventFootystats.time <= stat_data['date_start'] + timedelta(days=3),
                        ).first()
                        if match_event:
                            logger.info(f"Match already exists in SportEventFootystats: {stat_data['match_slug']}")
                            match_event_id = match_event.id
                        else:
                            logger.info(f"Creating new match in SportEventFootystats: {stat_data['match_slug']}")
                            new_match_event = SportEventFootystats(
                                match_uid=stat_data['match_uid'],
                                match_slug=stat_data['match_slug'],
                                time=stat_data['date_start'],
                                country=stat_data['country'],
                                competition=stat_data['competition'],
                                team_a=stat_data['team_a'],
                                team_b=stat_data['team_b'],
                                scrape_script='scrape_sports_stats',
                            )
                            session.add(new_match_event)
                            session.commit()
                            match_event_id = new_match_event.id

                        # Create a new row
                        new_stat = SportPotentialFootystats(
                            stat=stat_name,
                            sport_event_footystats_id=match_event_id,
                            match_uid=stat_data['match_uid'],
                            match_slug=stat_data['match_slug'],
                            time_start_str=stat_data['time_start_str'],
                            time_start=stat_data['time_start'],
                            date_start=stat_data['date_start'],
                            country=stat_data['country'],
                            competition=stat_data['competition'],
                            team_a=stat_data['team_a'],
                            team_b=stat_data['team_b'],
                            team_potential=stat_data['team_potential'],
                            average=stat_data['average'],
                            matches_count=stat_data['matches_count'],
                            probability=stat_data['probability'],
                            odd=stat_data['odd'],
                            date_updated=datetime.now(timezone.utc)
                        )
                        session.add(new_stat)
                        logger.info(f"Added new stat: {stat_data['match_slug']}")
                    session.commit()
                logger.info(f"Scraping completed for {stat_name}")
            print()
            time.sleep(10)

        return stats_data
    except Exception as e:
        logger.error(f"Error scraping FootyStats: {e}")
        return None


# %% Database Functions


# %% Main Processing Function
def scrape_and_save_sports_stats(session):
    """
    Main function to scrape and save sports statistics data
    
    Args:
        session: Database session
    """
    try:
    # if True:
        logger.debug("Starting sports statistics scraping process")
        
        # 1. Scrape data from various sources
        stats = scrape_footystats_stats(session)

        # 2. Process and clean the data
        # 3. Save to database
        
        logger.debug("Sports statistics scraping completed")
        
    except Exception as e:
        logger.error(f"Error in sports statistics scraping: {e}")
        raise


if __name__ == "__main__":
    try:
    # if True:
        # Create a session
        Session = sessionmaker(bind=engine, autocommit=False, autoflush=True)
        session = Session()
        
        # Check for command line arguments
        scraping_mode = "all"  # default mode
        if len(sys.argv) > 1:
            scraping_mode = sys.argv[1]
        
        logger.info(f"Running scraper in '{scraping_mode}' mode")
        
        # Main scraping function
        scrape_and_save_sports_stats(session)
        
        logger.info("Script execution completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        sys.exit(1)
    finally:
        if 'session' in locals():
            session.close()