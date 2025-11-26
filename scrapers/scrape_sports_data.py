#!/usr/bin/env python3
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


# Packages specific to this script
from bs4 import BeautifulSoup
import pytz
import textdistance

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import *
from utils import *
# from utils_db import *
from utils_ai import *
from utils_sports import *

# Logging setup
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'log/scrape_sports_data_{datetime.now(timezone.utc).strftime("%Y%m%d")}.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# %% Proxy Providers
SPORTS_SCRAPING_BOOKMAKER_PROXY_PROVIDER = os.environ.get('SPORTS_SCRAPING_BOOKMAKER_PROXY_PROVIDER', None)
SPORTS_SCRAPING_STARTS_IN = int(os.environ.get('SPORTS_SCRAPING_STARTS_IN', 15))


# %% FootyStats Scraping Functions
def scrape_footystats(sport='football', starts_in=15):
    try:
    # if True:
        # Define the URL for the API endpoint

        website_tz = "Africa/Lagos"

        url = "https://footystats.org/ajax_matchSearch_neo.php"
        start_date = int((datetime.now(pytz.timezone(website_tz)).replace(tzinfo=pytz.timezone(website_tz)) - timedelta(hours=1)).timestamp())
        end_date = int((datetime.now(pytz.timezone(website_tz)).replace(tzinfo=pytz.timezone(website_tz)) + timedelta(hours=1)).timestamp())
        # print(f"start_date={start_date} , end_date={end_date}")
        
        payload = f"filters%5Bsearch_type%5D=default&filters%5Bstart_date%5D={start_date}&filters%5Bend_date%5D={end_date}&filters%5Bpre_match_home_ppg%5D%5B%5D=0&filters%5Bpre_match_home_ppg%5D%5B%5D=3&filters%5Bpre_match_away_ppg%5D%5B%5D=0&filters%5Bpre_match_away_ppg%5D%5B%5D=3&filters%5Bppg_type%5D=home_away&filters%5Bavg_potential%5D%5B%5D=0&filters%5Bavg_potential%5D%5B%5D=100&filters%5Border%5D=date_unix&filters%5Border_way%5D=ASC&filters%5Border_label%5D=KO+Time&filters%5Bmatches_completed_minimum%5D%5B%5D=3&filters%5Bmatches_completed_minimum%5D%5B%5D=100&filters%5Bbtts_potential%5D%5B%5D=0&filters%5Bbtts_potential%5D%5B%5D=100&filters%5Bbtts_fhg_potential%5D%5B%5D=0&filters%5Bbtts_fhg_potential%5D%5B%5D=100&filters%5Bbtts_2hg_potential%5D%5B%5D=0&filters%5Bbtts_2hg_potential%5D%5B%5D=100&filters%5Bo05_potential%5D%5B%5D=0&filters%5Bo05_potential%5D%5B%5D=100&filters%5Bo15_potential%5D%5B%5D=0&filters%5Bo15_potential%5D%5B%5D=100&filters%5Bo25_potential%5D%5B%5D=0&filters%5Bo25_potential%5D%5B%5D=100&filters%5Bo35_potential%5D%5B%5D=0&filters%5Bo35_potential%5D%5B%5D=100&filters%5Bo45_potential%5D%5B%5D=0&filters%5Bo45_potential%5D%5B%5D=100&filters%5Bu05_potential%5D%5B%5D=0&filters%5Bu05_potential%5D%5B%5D=100&filters%5Bu15_potential%5D%5B%5D=0&filters%5Bu15_potential%5D%5B%5D=100&filters%5Bu25_potential%5D%5B%5D=0&filters%5Bu25_potential%5D%5B%5D=100&filters%5Bu35_potential%5D%5B%5D=0&filters%5Bu35_potential%5D%5B%5D=100&filters%5Bu45_potential%5D%5B%5D=0&filters%5Bu45_potential%5D%5B%5D=100&filters%5Bcorners_potential%5D%5B%5D=0&filters%5Bcorners_potential%5D%5B%5D=30&filters%5Bcorners_o85_potential%5D%5B%5D=0&filters%5Bcorners_o85_potential%5D%5B%5D=100&filters%5Bcorners_o95_potential%5D%5B%5D=0&filters%5Bcorners_o95_potential%5D%5B%5D=100&filters%5Bcorners_o105_potential%5D%5B%5D=0&filters%5Bcorners_o105_potential%5D%5B%5D=100&filters%5Bcards_potential%5D%5B%5D=-2&filters%5Bcards_potential%5D%5B%5D=15&filters%5Bo05HT_potential%5D%5B%5D=0&filters%5Bo05HT_potential%5D%5B%5D=100&filters%5Bo15HT_potential%5D%5B%5D=0&filters%5Bo15HT_potential%5D%5B%5D=100&filters%5Bo05_2H_potential%5D%5B%5D=0&filters%5Bo05_2H_potential%5D%5B%5D=100&filters%5Bo15_2H_potential%5D%5B%5D=0&filters%5Bo15_2H_potential%5D%5B%5D=100&filters%5Boffsides_potential%5D%5B%5D=0&filters%5Boffsides_potential%5D%5B%5D=20&filters%5Bteam_a_xg_prematch%5D%5B%5D=0.1&filters%5Bteam_a_xg_prematch%5D%5B%5D=5&filters%5Bteam_b_xg_prematch%5D%5B%5D=0.1&filters%5Bteam_b_xg_prematch%5D%5B%5D=5&filters%5Bodds_ft_1%5D%5B%5D=0&filters%5Bodds_ft_1%5D%5B%5D=100&filters%5Bodds_ft_x%5D%5B%5D=0&filters%5Bodds_ft_x%5D%5B%5D=100&filters%5Bodds_ft_2%5D%5B%5D=0&filters%5Bodds_ft_2%5D%5B%5D=100&filters%5Bodds_ft_over25%5D%5B%5D=0&filters%5Bodds_ft_over25%5D%5B%5D=100&filters%5Bodds_ft_over35%5D%5B%5D=0&filters%5Bodds_ft_over35%5D%5B%5D=100&filters%5Bodds_ft_over45%5D%5B%5D=0&filters%5Bodds_ft_over45%5D%5B%5D=100&filters%5Bodds_ft_over15%5D%5B%5D=0&filters%5Bodds_ft_over15%5D%5B%5D=100&filters%5Bodds_ft_over05%5D%5B%5D=0&filters%5Bodds_ft_over05%5D%5B%5D=100&filters%5Bodds_ft_under05%5D%5B%5D=0&filters%5Bodds_ft_under05%5D%5B%5D=100&filters%5Bodds_ft_under15%5D%5B%5D=0&filters%5Bodds_ft_under15%5D%5B%5D=100&filters%5Bodds_ft_under25%5D%5B%5D=0&filters%5Bodds_ft_under25%5D%5B%5D=100&filters%5Bodds_ft_under35%5D%5B%5D=0&filters%5Bodds_ft_under35%5D%5B%5D=100&filters%5Bodds_ft_under45%5D%5B%5D=0&filters%5Bodds_ft_under45%5D%5B%5D=100&filters%5Bodds_btts_yes%5D%5B%5D=0&filters%5Bodds_btts_yes%5D%5B%5D=100&filters%5Bodds_btts_no%5D%5B%5D=0&filters%5Bodds_btts_no%5D%5B%5D=100&filters%5Bodds_corners_over_85%5D%5B%5D=0&filters%5Bodds_corners_over_85%5D%5B%5D=100&filters%5Bodds_corners_over_95%5D%5B%5D=0&filters%5Bodds_corners_over_95%5D%5B%5D=100&filters%5Bodds_corners_over_105%5D%5B%5D=0&filters%5Bodds_corners_over_105%5D%5B%5D=100&filters%5Bodds_corners_over_115%5D%5B%5D=0&filters%5Bodds_corners_over_115%5D%5B%5D=100&filters%5Bdefaults%5D%5Bavg_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bavg_potential%5D%5B1%5D=10&filters%5Bdefaults%5D%5Bbtts_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bbtts_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bbtts_fhg_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bbtts_fhg_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bbtts_2hg_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bbtts_2hg_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bo05_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bo05_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bo15_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bo15_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bo25_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bo25_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bo35_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bo35_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bo45_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bo45_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bu05_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bu05_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bu15_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bu15_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bu25_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bu25_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bu35_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bu35_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bu45_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bu45_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bcorners_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bcorners_potential%5D%5B1%5D=30&filters%5Bdefaults%5D%5Bcorners_o85_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bcorners_o85_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bcorners_o95_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bcorners_o95_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bcorners_o105_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bcorners_o105_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bcards_potential%5D%5B0%5D=-2&filters%5Bdefaults%5D%5Bcards_potential%5D%5B1%5D=15&filters%5Bdefaults%5D%5Bo05HT_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bo05HT_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bo15HT_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bo15HT_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bo05_2H_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bo05_2H_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bo15_2H_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bo15_2H_potential%5D%5B1%5D=100&filters%5Bdefaults%5D%5Boffsides_potential%5D%5B0%5D=0&filters%5Bdefaults%5D%5Boffsides_potential%5D%5B1%5D=20&filters%5Bdefaults%5D%5Bteam_a_xg_prematch%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bteam_a_xg_prematch%5D%5B1%5D=5&filters%5Bdefaults%5D%5Bteam_b_xg_prematch%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bteam_b_xg_prematch%5D%5B1%5D=5&filters%5Bdefaults%5D%5Bodds_ft_1%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_1%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_x%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_x%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_2%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_2%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_over25%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_over25%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_over35%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_over35%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_over45%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_over45%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_over15%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_over15%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_over05%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_over05%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_under05%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_under05%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_under15%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_under15%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_under25%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_under25%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_under35%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_under35%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_ft_under45%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_ft_under45%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_btts_yes%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_btts_yes%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_btts_no%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_btts_no%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_corners_over_85%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_corners_over_85%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_corners_over_95%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_corners_over_95%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_corners_over_105%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_corners_over_105%5D%5B1%5D=100&filters%5Bdefaults%5D%5Bodds_corners_over_115%5D%5B0%5D=0&filters%5Bdefaults%5D%5Bodds_corners_over_115%5D%5B1%5D=100&filters%5BsavedGames%5D=0&filters%5BgameCount%5D=0&filters%5Binitial%5D=1&filters%5Bprevious%5D=0&filters%5Bshow_esports%5D=0&filters%5BloadMore%5D=0&filters%5Bcols%5D%5B%5D=date_unix&filters%5Bcols%5D%5B%5D=avg_potential&filters%5Bcols%5D%5B%5D=btts_potential&filters%5Bcols%5D%5B%5D=o15_potential&filters%5Bcols%5D%5B%5D=o25_potential&filters%5Bcols%5D%5B%5D=corners_potential&filters%5Bcols%5D%5B%5D=cards_potential&filters%5Bcols%5D%5B%5D=o05_potential&filters%5Bcols%5D%5B%5D=o35_potential&filters%5Bcols%5D%5B%5D=o05HT_potential&filters%5Bcols%5D%5B%5D=team_a_xg_prematch&filters%5Bcols%5D%5B%5D=team_b_xg_prematch&filters%5Bcols%5D%5B%5D=btts_fhg_potential&filters%5Bcols%5D%5B%5D=btts_2hg_potential&filters%5Bcols%5D%5B%5D=odds_ft_1&filters%5Bcols%5D%5B%5D=odds_ft_x&filters%5Bcols%5D%5B%5D=odds_ft_2&filters%5Bcols%5D%5B%5D=odds_btts_yes&filters%5Bcols%5D%5B%5D=odds_ft_over05&filters%5Bcols%5D%5B%5D=odds_ft_over15&filters%5Bcols%5D%5B%5D=odds_ft_over25&filters%5Bcols%5D%5B%5D=odds_ft_over35&filters%5Bcols%5D%5B%5D=odds_ft_under15&filters%5Bcols%5D%5B%5D=odds_ft_under25&filters%5Bcols%5D%5B%5D=odds_ft_under35&filters%5Bcols%5D%5B%5D=offsides_potential&filters%5Bcols%5D%5B%5D=corners_o85_potential&filters%5Bcols%5D%5B%5D=corners_o95_potential&filters%5Bcols%5D%5B%5D=corners_o105_potential"
        headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:141.0) Gecko/20100101 Firefox/141.0',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://footystats.org',
        'Connection': 'keep-alive',
        'Referer': 'https://footystats.org/matches',
        'Cookie': 'tz=Africa/Lagos; darkmode=off;',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Priority': 'u=0',
        'TE': 'trailers'
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, data=payload, proxies=None, timeout=10)
        if response.status_code in [403, 429, 202, 101]:
            logger.warning(f"Warning! WAF blocked request. Status: " + str(response.status_code))
            time.sleep(1)
            response = requests.post(url, headers=headers, data=payload, proxies=None, timeout=10)
            response.raise_for_status()
        response.raise_for_status()

        # Parse the JSON response
        # print(response.content)
        response_json = response.json()

        events = []
        
        if 'table' in response_json:
            table = response_json['table']
            if not "<div" in table:
                return Exception("No HTML content in table")
            table_html = BeautifulSoup(table, features="html.parser")
            # print(table_html.prettify())
            table_events = table_html.find_all('div', {'class': 'team'})
            for table_event in table_events:
                if 'first-row' in table_event['class']:
                    continue
                # Structured data
                info_div = table_event.find('div', {'class': 'col', 'data-stat': 'info'})
                match_preview_a = info_div.find('a', {'class': 'overlay showMatchPreview'})
                match_preview_uid = match_preview_a['data-uid']
                match_preview_url = match_preview_a['data-url']
                match_slug = match_preview_url.split('/')[-1]

                team_item_div = table_event.find('div', {'class': 'team-item info match incomplete'})
                meta_div = team_item_div.find('div', {'class': 'meta'})
                time_div = meta_div.find('time', {'class': 'desktop-only'})
                time_str = time_div.text.strip()
                time_py = convert_ordinal_date_time(time_str)
                time_py = (time_py - timedelta(hours=0)).replace(tzinfo=timezone.utc)
                time_starts_in = int((time_py - datetime.now(timezone.utc)).total_seconds() // 60) if time_py else 0


                comp_country = match_preview_url.split('/')[1] if match_preview_url else ""
                comp_m_span = meta_div.find('span', {'class': 'comp m'})
                comp_m_str = comp_m_span.text.strip()
                comp_d_span = meta_div.find('span', {'class': 'comp d'})
                comp_d_str = comp_d_span.text.strip()

                teams_div = table_event.find('div', {'class': 'teams'})
                if teams_div:
                    inner_teams = teams_div.find_all('div', {'class': 'inner-team'})
                    teams = []
                    for inner_team in inner_teams:
                        team_link_a = inner_team.find('a', {'class': 'team-link'})
                        team_name = team_link_a.text.strip()
                        teams.append(team_name)
                    ppg_div = table_event.find('div', {'class': 'ppg'})
                    if ppg_div:
                        form_boxes = ppg_div.find_all('div', {'class': 'form-box'})
                        ppg = []
                        for form_box in form_boxes:
                            ppg_value = form_box.text.strip()
                            ppg.append(ppg_value)
                else:
                    teams = []
                    ppg = []
                
                col_divs = table_event.find_all('div', {'class': 'col'})
                data_stat_elements = {}
                for col_div in col_divs:
                    if col_div['data-stat'] and (col_div['data-stat'].startswith('cards_') or col_div['data-stat'].startswith('corners_')):
                        continue
                    if col_div.find('a', {'class': 'premium-link'}):
                        data_stat_elements[col_div['data-stat']] = 'premium'
                    else:
                        stat_div = col_div.find('div', {'class': 'stat'})
                        if stat_div:
                            data_stat_elements[col_div['data-stat']] = stat_div.text.strip()
                
                event = {
                    'match_uid': match_preview_uid,
                    'match_slug': match_slug,
                    'time_str': time_str,
                    'time': time_py,
                    'starts_in': time_starts_in,
                    'country': comp_country,
                    'competition': comp_d_str,
                    'competition_m': comp_m_str,
                    'teams': teams,
                    'ppg': ppg,
                    'stats': data_stat_elements,

                }
                if time_starts_in < 0 or time_starts_in > starts_in:
                    continue
                if 'israel' in comp_country:
                    # Free Palestine
                    continue
                # print(event)
                # print(f"Match #{event['match_uid']} {teams} starts_in: {starts_in}")
                events.append(event)

                # print(table_event)
        return events
    except requests.RequestException as e:
        logger.error(f"Error scraping FootyStats: {e}")
        return None
    except Exception as e:
        logger.error(f"Error scraping FootyStats: {e}")
        return None

# %% FootyStats Database Functions
def save_footystats_events(events_footystats, session):
    """Save FootyStats events to database"""
    try:
        saved = 0
        for event in events_footystats:

            event_dict = {
                'match_uid': event['match_uid'],
                'match_slug': event['match_slug'],
                'time': event['time'].strftime('%Y-%m-%d %H:%M:%S'),
                'time_str': event['time_str'],
                'country': event['country'],
                'competition': event['competition'],
                'team_a': event['teams'][0],
                'team_b': event['teams'][1],
                'ppg_a': float(event['ppg'][0]) if event['ppg'][0] != 'N/A' else None,
                'ppg_b': float(event['ppg'][1]) if event['ppg'][1] != 'N/A' else None,
                'team_a_xg_prematch': float(event['stats']['team_a_xg_prematch']) if event['stats']['team_a_xg_prematch'] != 'N/A' else None,
                'team_b_xg_prematch': float(event['stats']['team_b_xg_prematch']) if event['stats']['team_b_xg_prematch'] != 'N/A' else None,
                'avg_potential': float(event['stats'].get('avg_potential', "")) if event['stats'].get('avg_potential', "") != 'N/A' else None,
                'offsides_potential': float(event['stats'].get('offsides_potential', "")) if event['stats'].get('offsides_potential', "") != 'N/A' else None,
                'btts_potential': float(event['stats'].get('btts_potential', "").strip('%'))/100 if event['stats'].get('btts_potential', "").strip('%') != 'N/A' else None,
                'o15_potential': float(event['stats'].get('o15_potential', "").strip('%'))/100 if event['stats'].get('o15_potential', "").strip('%') != 'N/A' else None,
                'o25_potential': float(event['stats'].get('o25_potential', "").strip('%'))/100 if event['stats'].get('o25_potential', "").strip('%') != 'N/A' else None,
                'o05_potential': float(event['stats'].get('o05_potential', "").strip('%'))/100 if event['stats'].get('o05_potential', "").strip('%') != 'N/A' else None,
                'o35_potential': float(event['stats'].get('o35_potential', "").strip('%'))/100 if event['stats'].get('o35_potential', "").strip('%') != 'N/A' else None,
                'o05HT_potential': float(event['stats'].get('o05HT_potential', "").strip('%'))/100 if event['stats'].get('o05HT_potential', "").strip('%') != 'N/A' else None,
                'btts_fhg_potential': float(event['stats'].get('btts_fhg_potential', "").strip('%'))/100 if event['stats'].get('btts_fhg_potential', "").strip('%') != 'N/A' else None,
                'btts_2hg_potential': float(event['stats'].get('btts_2hg_potential', "").strip('%'))/100 if event['stats'].get('btts_2hg_potential', "").strip('%') != 'N/A' else None,
                'odds_ft_1': float(event['stats'].get('odds_ft_1', "")) if event['stats'].get('odds_ft_1', "") != 'N/A' else None,
                'odds_ft_x': float(event['stats'].get('odds_ft_x', "")) if event['stats'].get('odds_ft_x', "") != 'N/A' else None,
                'odds_ft_2': float(event['stats'].get('odds_ft_2', "")) if event['stats'].get('odds_ft_2', "") != 'N/A' else None,
                'odds_btts_yes': float(event['stats'].get('odds_btts_yes', "")) if event['stats'].get('odds_btts_yes', "") != 'N/A' else None,
                'odds_ft_over05': float(event['stats'].get('odds_ft_over05', "")) if event['stats'].get('odds_ft_over05', "") != 'N/A' else None,
                'odds_ft_over15': float(event['stats'].get('odds_ft_over15', "")) if event['stats'].get('odds_ft_over15', "") != 'N/A' else None,
                'odds_ft_over25': float(event['stats'].get('odds_ft_over25', "")) if event['stats'].get('odds_ft_over25', "") != 'N/A' else None,
                'odds_ft_over35': float(event['stats'].get('odds_ft_over35', "")) if event['stats'].get('odds_ft_over35', "") != 'N/A' else None,
                'odds_ft_under15': float(event['stats'].get('odds_ft_under15', "")) if event['stats'].get('odds_ft_under15', "") != 'N/A' else None,
                'odds_ft_under25': float(event['stats'].get('odds_ft_under25', "")) if event['stats'].get('odds_ft_under25', "") != 'N/A' else None,
                'odds_ft_under35': float(event['stats'].get('odds_ft_under35', "")) if event['stats'].get('odds_ft_under35', "") != 'N/A' else None,
            }

            # Check if same match_uid exists to ignore saving (date +-1 day)
            find_event_by_match_uid = session.query(SportEventFootystats).filter(
                SportEventFootystats.match_uid == event['match_uid'],
                SportEventFootystats.time >= event['time'] - timedelta(days=1),
                SportEventFootystats.time <= event['time'] + timedelta(days=1)
            ).first()
            if find_event_by_match_uid:
                # Check if exists, but have different time
                if find_event_by_match_uid.time.strftime('%Y-%m-%d %H:%M:%S') != event_dict['time']:
                    logger.info(f"FootyStats Event {event['match_uid']} ({event['teams']}) already exists with different time ({find_event_by_match_uid.time.strftime('%Y-%m-%d %H:%M:%S')} != {event_dict['time']}), updating..")
                    event_dict['date_updated'] = datetime.now(timezone.utc)
                    session.query(SportEventFootystats).filter(SportEventFootystats.id == find_event_by_match_uid.id).update(event_dict)
                else:
                    logger.debug(f"FootyStats Event {event['match_uid']} ({event['teams']}) already exists")
                    continue
            else:
                # Check if same match_slug and time exists to ignore saving
                find_event_by_match_slug = session.query(SportEventFootystats).filter(
                    or_(SportEventFootystats.match_uid == None, SportEventFootystats.match_uid != event['match_uid']),
                    SportEventFootystats.match_slug == event['match_slug'],
                    SportEventFootystats.time >= event['time'] - timedelta(days=1),
                    SportEventFootystats.time <= event['time'] + timedelta(days=1)
                ).first()
                if find_event_by_match_slug:
                    logger.debug(f"Event {event['match_uid']} ({event['teams']}) already exists")
                    event_dict['date_updated'] = datetime.now(timezone.utc)
                    logger.info(f"Event: different match_uid={event['match_uid']} match_slug={event['match_slug']} time={event['time']} | {event['country']} {event['teams']} .. updated to database")
                    session.query(SportEventFootystats).filter(SportEventFootystats.id == find_event_by_match_slug.id).update(event_dict)
                else:
                    # Create a new instance of the model
                    # print(event_dict)
                    event_dict['scrape_script'] = 'scrape_sports_data'
                    event_dict['date_scraped'] = datetime.now(timezone.utc)
                    event_model = SportEventFootystats(**event_dict)
                    logger.info(f"Event {event['match_slug']} #{event['match_uid']} {event['country']} ({event['teams']}) saved to database")
                    # Add the instance to the session
                    session.add(event_model)
                    saved += 1
            
        # Commit the session to save changes
        session.commit()
        if saved > 0:
            logger.info(f"Saved {saved} FootyStats events to database")
        
    except Exception as e:
        logger.error(f"Error saving FootyStats events: {e}")
        session.rollback()


# %% 22bet Scraping Functions
def scrape_22bet_livefeed(sport=''):
    sports_id = None
    if sport == 'football':
        sports_id = 1
    else:
        return None
    url = f"https://22bet.com/LiveFeed/Get1x2_VZip?sports={sports_id}&count=100&lng=en&mode=4"
    
    try:
        proxy_dict = {}
        proxy = fetch_proxy(provider=SPORTS_SCRAPING_BOOKMAKER_PROXY_PROVIDER)
        if proxy:
            proxy_dict = {'http': proxy, 'https': proxy}
        print(proxy_dict)
        
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
        data = []
        for event in response_json['Value']:
            event = {
                'country': event.get('CN', None),
                'league_name': event.get('L', None),
                'league_id': event.get('LI', None),
                'game_id': event.get('I', None),
                'team_a': event.get('O1', '').strip(),
                'team_b': event.get('O2', '').strip(),
                'extra_name': event.get('DI', None),
                'start_date': event.get('S', None), # in format: 1756393200
                'sport': event.get('SE', None)
            }
            event['match_url'] = f"/live/{sport}/{event['league_id']}/{event['game_id']}"
            if event.get('start_date'):
                event['start_date'] = datetime.fromtimestamp(event['start_date'], tz=timezone.utc).replace(second=0, microsecond=0)
            # if not event['country'] in ['World', 'Europe', 'Asia', 'Africa', 'America', 'South America', 'North America', 'Oceania', 'England', 'France', 'Germany', 'Italy', 'Spain', 'Portugal', 'Netherlands', 'Belgium', 'United States']:
            #     logger.debug(f"Event ignored: {event['country']} \t {event['league_name']} \t {event['team_a']} - {event['team_b']}")
            #     continue
            if event['country'] in ['Israel']:
                # Free Palestine
                continue
            if event['extra_name'] or ("Short Football" in event['league_name'] or "2x2" in event['league_name'] or "3x3" in event['league_name'] or "4x4" in event['league_name'] or "5x5" in event['league_name'] or "6x6" in event['league_name'] or "7x7" in event['league_name'] or "daily league" in event['league_name'] or "Student League" in event['league_name'] or "Indoor" in event['league_name'] or "Subsoccer" in event['league_name'] or "Team vs Player" in event['league_name'] or "Special bets" in event['league_name']):
                logger.debug(f"Event ignored: {event['country']} \t {event['league_name']} \t {event['team_a']} - {event['team_b']} \t extra_name: {event['extra_name']}")
                continue
            if event['team_b'] == "" or "special bets" in event['team_a'].lower():
                continue
            data.append(event)
        return data
    except requests.RequestException as e:
        logger.error(f"Error scraping 22bet: {e}")
        return None

def scrape_22bet_linefeed(sport='', starts_in=15):
    sports_id = None
    if sport == 'football':
        sports_id = 1
    else:
        return None
    url = f"https://22bet.com/LineFeed/Get1x2_VZip?sports={sports_id}&count=50&lng=en&tf={starts_in}&tz=0&mode=4&getEmpty=true"
    
    try:
        proxy_dict = {}
        proxy = fetch_proxy(provider=SPORTS_SCRAPING_BOOKMAKER_PROXY_PROVIDER)
        if proxy:
            proxy_dict = {'http': proxy, 'https': proxy}
        print(proxy_dict)
        
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
        # print(response_json)
        # Cleaning data and returning necessary information
        data = []
        for event in response_json['Value']:
            event = {
                'country': event.get('CN', None),
                'league_name': event.get('L', None),
                'league_id': event.get('LI', None),
                'game_id': event.get('I', None),
                'team_a': event.get('O1', '').strip(),
                'team_b': event.get('O2', '').strip(),
                'extra_name': event.get('DI', None),
                'start_date': event.get('S', None), # in format: 1756393200
                'location': event.get('MIO', None).get('Loc', None) if 'MIO' in event and 'Loc' in event.get('MIO') else None,
                'round': event.get('MIO', None).get('TSt', None) if 'MIO' in event and 'TSt' in event.get('MIO') else None,
                'sport': event.get('SE', None)
            }
            event['match_url'] = f"/line/{sport}/{event['league_id']}/{event['game_id']}"
            if event.get('start_date'):
                event['start_date'] = datetime.fromtimestamp(event['start_date'], tz=timezone.utc).replace(second=0, microsecond=0)
            # if not event['country'] in ['World', 'Europe', 'Asia', 'Africa', 'America', 'South America', 'North America', 'Oceania', 'England', 'France', 'Germany', 'Italy', 'Spain', 'Portugal', 'Netherlands', 'Belgium', 'United States']:
            #     logger.debug(f"Event ignored: {event['country']} \t {event['league_name']} \t {event['team_a']} - {event['team_b']}")
            #     continue
            if event['country'] in ['Israel']:
                # Free Palestine
                continue
            if event['extra_name'] or ("Short Football" in event['league_name'] or "2x2" in event['league_name'] or "3x3" in event['league_name'] or "4x4" in event['league_name'] or "5x5" in event['league_name'] or "6x6" in event['league_name'] or "7x7" in event['league_name'] or "daily league" in event['league_name'] or "Student League" in event['league_name'] or "Indoor" in event['league_name'] or "Subsoccer" in event['league_name'] or "Team vs Player" in event['league_name'] or "Special bets" in event['league_name']):
                logger.debug(f"Event ignored: {event['country']} \t {event['league_name']} \t {event['team_a']} - {event['team_b']} \t extra_name: {event['extra_name']}")
                continue
            if event['team_b'] == "" or "special bets" in event['team_a'].lower():
                continue
            data.append(event)
        return data
    except requests.RequestException as e:
        logger.error(f"Error scraping 22bet: {e}")
        return None

# %% 22bet Database Functions
def save_22bet_event(session, event={}, sport='football'):
    """Save 22bet event to database"""
    bookmaker = '22bet'
    try:
        event_dict = {
            'bookmaker': bookmaker,
            'league_id': event['league_id'],
            'game_id': event['game_id'],
            'match_url': event['match_url'],
            'time': event['start_date'],
            'country': event['country'],
            'competition': event['league_name'],
            'team_a': event['team_a'],
            'team_b': event['team_b'],
            'location': event['location'],
            'round': event['round'],
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


# %% Event Mapping Functions
def create_event_mapping(session, event_footystats, event_22bet_id, step=""):
    try:
        # Check if same match_uid exists to ignore saving
        existing_event_mapping = session.query(SportEventMapping).filter(SportEventMapping.sport_event_footystats_id == event_footystats.id, SportEventMapping.sport_event_bookmaker_id == event_22bet_id).first()
        if existing_event_mapping:
            logger.debug(f"Event Mapping FS#{event_footystats.id} ({event_footystats.team_a} - {event_footystats.team_b}) with {event_22bet_id} already exists")
            return existing_event_mapping.id
        # Create a new SportEventMapping
        event_mapping = SportEventMapping(
            sport_event_footystats_id=event_footystats.id,
            sport_event_bookmaker_id=event_22bet_id,
            start_time=event_footystats.time,
            status='matched',
            step=step,
            country=event_footystats.country,
            competition=event_footystats.competition,
            team_a=event_footystats.team_a,
            team_b=event_footystats.team_b,
            date_added=datetime.now(timezone.utc),
        )
        session.add(event_mapping)
        session.commit()
        return event_mapping.id
    except Exception as e:
        logger.error(f"Error creating event mapping: {e.__class__.__name__}: {e}")
        session.rollback()





def match_footystats_events(unmatched_events_footystats, events_22bet, session):
    logger.info(f"Matching ({len(unmatched_events_footystats)}) FootyStats events with ({len(events_22bet)}) 22bet events")
    try:
    # if True:
        matched_recently = get_matched_sports_events(session=session, last_x_hours=1)
        if matched_recently:
            logger.info(f"Found ({len(matched_recently)}) matched events recently...")

        matched_events = []
        matched_targets = []
        # Organize events_22bet by country
        events_22bet_by_country = {}
        for event_22bet in events_22bet:
            already_matched = False
            for e in matched_recently:
                # print(f"Event in matched_recently: {e.sport_event_bookmaker.league_id}/{e.sport_event_bookmaker.game_id} ({e.sport_event_bookmaker.team_a} - {e.sport_event_bookmaker.team_b})")
                if int(event_22bet['league_id']) == int(e.sport_event_bookmaker.league_id) and int(event_22bet['game_id']) == int(e.sport_event_bookmaker.game_id):
                    logger.debug(f"Event {event_22bet['league_id']}/{event_22bet['game_id']} ({event_22bet['team_a']} - {event_22bet['team_b']}) already matched")
                    already_matched = True
                    break
            if not already_matched:
                events_22bet_by_country.setdefault(normalize_lower(event_22bet['country']).replace(" ", "-").replace("republic-of-", ""), []).append(event_22bet)
        for country in events_22bet_by_country:
            logger.debug(f"{country}: {len(events_22bet_by_country[country])}")
            for event_22bet in events_22bet_by_country[country]:
                logger.debug(f"\t ({event_22bet['league_id']}/{event_22bet['game_id']}) {event_22bet['league_name']} <{event_22bet['team_a']} - {event_22bet['team_b']}> @ {event_22bet['start_date']}")
        
        # Step 1: Match events by country / team names similarity
        for id, event_footystats in unmatched_events_footystats.items():
            # print(f"Looking for match for {event.match_uid} country=<{event.country}> competition=<{event.competition}> teams=<{event.team_a} - {event.team_b}>")
            if normalize_lower(event_footystats.country) in events_22bet_by_country:
                for event_22bet in events_22bet_by_country[normalize_lower(event_footystats.country)]:
                    if event_22bet['game_id'] in matched_targets:
                        continue
                    exact_start_time = (event_footystats.time.replace(tzinfo=timezone.utc) == event_22bet['start_date'].replace(tzinfo=timezone.utc))
                    # print(f"Exact start time: {exact_start_time} ({event_footystats.time} | {event_22bet['start_date']})")
                    
                    similarity_team_a = textdistance.jaro_winkler.normalized_similarity(normalize_lower(event_footystats.team_a), normalize_lower(event_22bet['team_a']))
                    similarity_team_b = textdistance.jaro_winkler.normalized_similarity(normalize_lower(event_footystats.team_b), normalize_lower(event_22bet['team_b']))
                    # print(f"Similarity: team_a ({event.team_a} | {event_22bet['team_a']}): {similarity_team_a} | team_b ({event.team_b} | {event_22bet['team_b']}): {similarity_team_b}")

                    if similarity_team_a >= 0.75 and similarity_team_b >= 0.75:
                        # event.sport_event_id = event_22bet['sport_event_id']
                        if not exact_start_time:
                            logger.info(f"Step_1a | Match found with different start time for {event_footystats.match_uid} country=<{event_footystats.country}> competition=<{event_footystats.competition}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> sport_event_id=<{event_22bet['game_id']}>")
                            continue
                        else:
                            logger.info(f"Step_1a | Match found for {event_footystats.match_uid} country=<{event_footystats.country}> competition=<{event_footystats.competition}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> sport_event_id=<{event_22bet['game_id']}>")
                            matched_events.append(event_footystats.match_uid)
                            matched_targets.append(event_22bet['game_id'])
                            event_22bet_id = save_22bet_event(session=session, event=event_22bet, sport="football")
                            if event_22bet_id:
                                # Create a new SportEventMapping
                                logger.info(f"Step_1a | Creating event mapping for {event_footystats.match_uid} country=<{event_footystats.country}> competition=<{event_footystats.competition}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> event_22bet_id=<{event_22bet_id}>")
                                event_mapping_id = create_event_mapping(session=session, event_footystats=event_footystats, event_22bet_id=event_22bet_id, step="1a")
                                if event_mapping_id:
                                    logger.info(f"Step_1a | Event mapping created for {event_footystats.match_uid} country=<{event_footystats.country}> competition=<{event_footystats.competition}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> event_22bet_id=<{event_22bet_id}> event_mapping_id=<{event_mapping_id}>")
                            session.commit()
                            
            if not event_footystats.match_uid in matched_events:
                for event_22bet in events_22bet:
                    if event_22bet['game_id'] in matched_targets:
                        continue
                    exact_start_time = (event_footystats.time.replace(tzinfo=timezone.utc) == event_22bet['start_date'].replace(tzinfo=timezone.utc))
                    # print(f"Exact start time: {exact_start_time} ({event_footystats.time} | {event_22bet['start_date']})")
                    
                    similarity_team_a = textdistance.jaro_winkler.normalized_similarity(normalize_lower(event_footystats.team_a), normalize_lower(event_22bet['team_a']))
                    similarity_team_b = textdistance.jaro_winkler.normalized_similarity(normalize_lower(event_footystats.team_b), normalize_lower(event_22bet['team_b']))
                    # print(f"Similarity: team_a ({event.team_a} | {event_22bet['team_a']}): {similarity_team_a} | team_b ({event.team_b} | {event_22bet['team_b']}): {similarity_team_b}")

                    if similarity_team_a >= 0.8 and similarity_team_b >= 0.8:
                        # event.sport_event_id = event_22bet['sport_event_id']
                        if not exact_start_time:
                            logger.info(f"Step_1b | Match found with different start time for {event_footystats.match_uid} country=<{event_footystats.country}> competition=<{event_footystats.competition}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> sport_event_id=<{event_22bet['game_id']}>")
                            continue
                        else:
                            logger.info(f"Step_1b | Match found for {event_footystats.match_uid} country=<{event_footystats.country}> competition=<{event_footystats.competition}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> sport_event_id=<{event_22bet['game_id']}>")
                            matched_events.append(event_footystats.match_uid)
                            matched_targets.append(event_22bet['game_id'])
                            event_22bet_id = save_22bet_event(session=session, event=event_22bet, sport="football")
                            if event_22bet_id:
                                # Create a new SportEventMapping
                                logger.info(f"Step_1b | Creating event mapping for {event_footystats.match_uid} country=<{event_footystats.country}> competition=<{event_footystats.competition}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> event_22bet_id=<{event_22bet_id}>")
                                event_mapping_id = create_event_mapping(session=session, event_footystats=event_footystats, event_22bet_id=event_22bet_id, step="1b")
                            session.commit()
        
        # Step 2: Ask AI to match events
        unmatched_events_footystats = {int(event.id): event for id, event in unmatched_events_footystats.items() if event.match_uid not in matched_events}
        unmatched_events_footystats_json = [{
            'my_event_id': int(event.id),
            'country': event.country if event.country else "",
            'competition': normalize_latin_characters(event.competition) if event.competition else "",
            'team_a': normalize_latin_characters(event.team_a),
            'team_b': normalize_latin_characters(event.team_b),
            'start_date': event.time.strftime('%Y-%m-%d %H:%M'),
        } for id, event in unmatched_events_footystats.items()]
        
        unmatched_events_22bet = {int(event['game_id']): event for event in events_22bet if int(event['game_id']) not in matched_targets}
        unmatched_events_22bet_json = [{
            'bookmaker_event_id': int(id),
            'country': event['country'],
            'competition': normalize_latin_characters(event['league_name']),
            'team_a': normalize_latin_characters(event['team_a']),
            'team_b': normalize_latin_characters(event['team_b']),
            'start_date': event['start_date'].strftime('%Y-%m-%d %H:%M'),
        } for id, event in unmatched_events_22bet.items()]

        if len(unmatched_events_footystats) == 0:
            logger.info("Step_2 | All FootyStats events were matched")
        if len(unmatched_events_footystats) > 0 and len(unmatched_events_22bet) > 0:
            logger.info(f"Step_2 | AI: Matching ({len(unmatched_events_footystats)}) FootyStats events with ({len(unmatched_events_22bet)}) 22bet events")
            # Ask AI to match events
            ai_prompt = generate_ai_prompt_sports_events(my_events=unmatched_events_footystats_json, scraped_events=unmatched_events_22bet_json)
            # print(f"AI Prompt (Sports Events Matching):\n{ai_prompt}")
            ai_response = asyncio.run(ai_openrouter(prompt=ai_prompt))
            # ai_response = [{'my_event_id': 8277547, 'bookmaker_event_id': 661834066, 'country': 'iceland', 'competition': 'Urvalsdeild', 'team_a': 'Fram', 'team_b': 'Stjarnan', 'teams_names_similarity': '79%', 'start_date': '2025-10-20 19:15'}] # DEMO
            if ai_response:
                if 'error' in ai_response:
                    logger.error(f"AI Response Error: {ai_response['error']}")
                else:
                    for event_ai in ai_response:
                        logger.info(f"AI Response: Event {event_ai}")
                        if 'my_event_id' in event_ai and 'bookmaker_event_id' in event_ai and str(event_ai['my_event_id']).isdigit() and str(event_ai['bookmaker_event_id']).isdigit() and int(event_ai['my_event_id']) in unmatched_events_footystats.keys() and int(event_ai['bookmaker_event_id']) in unmatched_events_22bet.keys():
                            event_footystats = unmatched_events_footystats[int(event_ai['my_event_id'])]
                            event_22bet = unmatched_events_22bet[int(event_ai['bookmaker_event_id'])]
                            logger.info(f"Step_2 | Match found for {event_ai['my_event_id']} country=<{event_ai['country']}> competition=<{event_ai['competition']}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> event_22bet_id=<{event_ai['bookmaker_event_id']}> teams=<{event_22bet['team_a']} - {event_22bet['team_b']}>")
                            if 'teams_names_similarity' in event_ai and percent_to_float(event_ai['teams_names_similarity']) < 0.6: # teams_names_similarity < "60%"
                                logger.info(f"Step_2 | Teams names similarity too low for {event_ai['my_event_id']} country=<{event_ai['country']}> competition=<{event_ai['competition']}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> event_22bet_id=<{event_ai['bookmaker_event_id']}> teams=<{event_22bet['team_a']} - {event_22bet['team_b']}>")
                                continue
                            if event_footystats.time.strftime('%Y-%m-%d %H:%M') != event_22bet['start_date'].strftime('%Y-%m-%d %H:%M'):
                                if 'teams_names_similarity' in event_ai and percent_to_float(event_ai['teams_names_similarity']) < 0.8: # teams_names_similarity < "80%"
                                    logger.info(f"Step_2 | Start date too different for {event_ai['my_event_id']} country=<{event_ai['country']}> competition=<{event_ai['competition']}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> @ {event_footystats.time.strftime('%Y-%m-%d %H:%M')} | {event_22bet['start_date'].strftime('%Y-%m-%d %H:%M')} event_22bet_id=<{event_ai['bookmaker_event_id']}> teams=<{event_22bet['team_a']} - {event_22bet['team_b']}>")
                                    continue
                            matched_events.append(event_ai['my_event_id'])
                            matched_targets.append(event_ai['bookmaker_event_id'])
                            event_22bet_id = save_22bet_event(session=session, event=event_22bet, sport="football")
                            if event_22bet_id:
                                # Create a new SportEventMapping
                                logger.info(f"Step_2 | Creating event mapping for {event_footystats.id} country=<{event_footystats.country}> competition=<{event_footystats.competition}> teams=<{event_footystats.team_a} - {event_footystats.team_b}> event_22bet_id=<{event_22bet_id}>")
                                event_mapping_id = create_event_mapping(session=session, event_footystats=event_footystats, event_22bet_id=event_22bet_id, step="2")
                            session.commit()
                        else:
                            logger.info(f"AI Response: Could not match with any event {event_ai.get('my_event_id', '')}")
    except Exception as e:
        logger.error(f"Error matching FootyStats events: {e}", exc_info=True)
        session.rollback()

def scrape_and_save_sports_data(session):
    """Main function to scrape and save sports data from various sources"""
    
    try:
    # if True:
        print("Starting sports data scraping process")
        
        # Add your scraping logic here
        # This is where you would call various scraping functions
        # and save the data to the database

        # Example structure:
        # 1. Scrape data from source 1
        # 2. Process and clean the data
        # 3. Save to database
        # 4. Repeat for other sources

        starts_in = SPORTS_SCRAPING_STARTS_IN

        # Get upcoming FootyStats events and save (and stop)
        events_footystats = scrape_footystats(sport='football', starts_in=starts_in)
        if not events_footystats:
            logger.info("No new FootyStats events found")
        else:
            save_footystats_events(events_footystats, session=session)
        
        # exit()
        
        # Get upcoming unmatched FootyStats events from DB, and scrape 22bet livefeed for future matching
        unmatched_events_footystats = get_unmatched_footystats_events(session, starts_in=starts_in)
        # print(unmatched_events_footystats)
        if not unmatched_events_footystats:
            logger.debug("No unmatched FootyStats events")
        else:
            events_22bet = scrape_22bet_linefeed(sport='football', starts_in=starts_in)
            # print(events_22bet)
            if not events_22bet:
                logger.warning("No 22bet events found")
            else:
                unmatched_events_footystats = {event.match_uid: event for event in unmatched_events_footystats}
                match_footystats_events(unmatched_events_footystats, events_22bet, session=session)
        
    except Exception as e:
        logger.error(f"Error in sports data scraping: {e}")
        raise


if __name__ == "__main__":
    try:
    # if True:
        # Create a session
        Session = sessionmaker(bind=engine, autocommit=False, autoflush=True)
        session = Session()
        
        # Check for command line arguments to determine scraping mode
        scraping_mode = "all"  # default mode
        if len(sys.argv) > 1:
            scraping_mode = sys.argv[1]
        
        logger.debug(f"Running scraper in '{scraping_mode}' mode")
        
        # Main scraping function
        scrape_and_save_sports_data(session)
        
        logger.debug("Script execution completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        sys.exit(1)
    finally:
        if 'session' in locals():
            session.close()
