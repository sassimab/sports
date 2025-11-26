import re
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import unicodedata
import json

from models import *

import logging
logger = logging.getLogger(__name__)


# %% Date Functions
def convert_ordinal_date_time(date_string: str) -> datetime:
    """
    Converts a date-time string (e.g., '11th, 22:30') to a Python datetime 
    object, determining the current/next month based on the current UTC time.
    The resulting datetime object will be OFFSET-AWARE (UTC).
    """
    # 1. Establish the reference time in UTC (Offset-Aware)
    now_utc_aware = datetime.now(timezone.utc)
    
    # We strip the timezone info to get a Naive object for string parsing (strptime cannot parse time zones)
    now_naive = now_utc_aware.replace(tzinfo=None) 
    
    # 2. Split and Clean the Input
    try:
        day_part = date_string.split(',')[0].strip()
        time_part = date_string.split(',')[1].strip()
    except IndexError:
        raise ValueError("Input format must be 'Day_Ordinal, HH:MM' (e.g., '11th, 22:30')")

    match = re.search(r'(\d+)', day_part)
    if not match:
        raise ValueError(f"Could not find a day number in '{day_part}'")
        
    target_day = int(match.group(1))

    # 3. Initial Target Date Calculation (Naive)
    
    # Combine UTC year, UTC month, target day, and target time into a naive string
    temp_date_string = f"{now_naive.year}-{now_naive.month}-{target_day} {time_part}"
    date_format = "%Y-%m-%d %H:%M"
    
    try:
        # Parse the string into an Offset-Naive datetime object
        target_datetime_naive = datetime.strptime(temp_date_string, date_format)
    except ValueError:
        # Handles cases where target_day is invalid for the current month (e.g., 31st in a 30-day month)
        target_datetime_naive = now_naive.replace(day=1) + relativedelta(months=1)
        target_datetime_naive = target_datetime_naive.replace(day=target_day)

    # 4. Make the Target Date Offset-Aware (UTC) for comparison
    target_datetime_aware = target_datetime_naive.replace(tzinfo=timezone.utc)

    # 5. Core Logic: Check if the calculated target_datetime is in the past (relative to UTC)
    # Comparison is now safe: Offset-Aware vs. Offset-Aware
    if target_datetime_aware < now_utc_aware:
        # If the date/time is in the past, move to the next month's occurrence.
        
        # Start at the 1st of the current month, jump forward one month
        # Use the Naive target object for calculation to avoid recursive awareness issues
        target_datetime_naive = target_datetime_naive.replace(day=1) + relativedelta(months=1)
        
        # Now set the correct target day (will automatically handle overflow)
        try:
            target_datetime_naive = target_datetime_naive.replace(day=target_day)
        except ValueError:
             target_datetime_naive = target_datetime_naive + relativedelta(days=target_day - target_datetime_naive.day)
             
        # Re-make the final object Offset-Aware (UTC)
        target_datetime_aware = target_datetime_naive.replace(tzinfo=timezone.utc)
    
    return target_datetime_aware

# %% String Functions
def normalize_latin_characters(text):
    """
    Normalizes Latin characters by removing diacritics and converting
    special characters to their basic ASCII equivalents.
    """
    normalized_text = unicodedata.normalize('NFKD', text)
    return normalized_text.encode('ascii', 'ignore').decode('utf-8')

def normalize_lower(string):
    return normalize_latin_characters(string).lower()


def percent_to_float(percentage_string):
    """
    Converts a percentage string (e.g., "99.5%") to its decimal float equivalent.
    """
    # Remove the '%' sign and any leading/trailing whitespace
    stripped_string = percentage_string.strip().rstrip('%')
    # Convert to float and divide by 100
    try:
        decimal_value = float(stripped_string) / 100
        return decimal_value
    except ValueError:
        print(f"Error: Could not convert '{stripped_string}' to a number.")
        return None


# %% Score Functions
def parse_score(score_string):
    SCORE_REGEX = re.compile(r"^(?P<ft>\d+:\d+)(?:\s*\((?P<fh>\d+:\d+)(?:\s*,\s*(?P<sh>\d+:\d+))?(?:\s*,\s*(?P<et>\d+:\d+))?(?:\s*,\s*(?P<pen>\d+:\d+))?\))?$")
    match = SCORE_REGEX.match(score_string)
    if match:
        ft = match.group("ft")
        fh = match.group("fh")
        sh = match.group("sh")
        if not fh or not sh:
            return {"ft": ft, "fh": None, "sh": None, "et": None, "pen": None, "error": None}
        else:
            fh_match = re.match(r"(\d+):(\d+)", fh)
            sh_match = re.match(r"(\d+):(\d+)", sh)
            if fh_match and sh_match:
                ft = f"{int(fh_match.group(1)) + int(sh_match.group(1))}:{int(fh_match.group(2)) + int(sh_match.group(2))}"
                # print(ft)
            return {"ft": ft, "fh": fh, "sh": sh, "et": match.group("et"), "pen": match.group("pen"), "error": None}
    return {"ft": None, "fh": None, "sh": None, "et": None, "pen": None, "error": f"Invalid format"}


# %% AI Functions
# Generate AI prompt for matching sports events
def generate_ai_prompt_sports_events(my_events=[], scraped_events=[]):
    return f'# Task:\r\n\
Match my events with scraped bookmaker events to ensure correct event mappings.\r\n\
# Instructions:\r\n\
 - Read my events and scraped events.\r\n\
 - Match my events with scraped events by checking and validating all these criterias: start_date (exact match), country (identical), competition name (identical), teams names (identical using text similarity).\r\n\
 - Return matched events with event mapping information.\r\n\
# Output:\r\n\
Return matched events in json format with my_event_id, bookmaker_event_id, country, competition, team_a, team_b, start_date, teams_names_similarity ; else return error with brief REASON.\r\n\
Example: `{json.dumps([{"my_event_id":"123","bookmaker_event_id":"456","country":"World","competition":"World Cup","team_a":"Team A","team_b":"Team B","teams_names_similarity":"50%","start_date":"2025-08-28 17:00"}])}` or `{json.dumps({"error": "REASON"})}`\r\n\r\n\
# Input:\r\n\
## My Events:\r\n\
{json.dumps(my_events)}\r\n\
## Bookmaker Events:\r\n\
{json.dumps(scraped_events)}'



# %% Database Functions

def get_unmatched_footystats_events(session, starts_in=15):
    try:
        # Return SportEventFootystats that doesn't have SportEventMapping
        events = session.query(SportEventFootystats).join(
            SportEventMapping, SportEventFootystats.id == SportEventMapping.sport_event_footystats_id,
            isouter=True
        ).filter(
            SportEventMapping.id.is_(None),
            SportEventFootystats.match_uid.isnot(None),
            SportEventFootystats.team_a.isnot(None),
            SportEventFootystats.team_b.isnot(None),
            SportEventFootystats.time > datetime.now(timezone.utc) - timedelta(minutes=starts_in),
            SportEventFootystats.time < datetime.now(timezone.utc) + timedelta(minutes=starts_in)
        ).all()
        return events
    except Exception as e:
        logger.error(f"Error getting unmatched FootyStats events: {e}")
        return None

def get_matched_sports_events(session, last_x_hours=None):
    """Get matched FootyStats events from database"""
    try:
        matched_events = session.query(SportEventMapping).filter(
            SportEventMapping.sport_event_footystats_id.isnot(None),
            SportEventMapping.sport_event_bookmaker_id.isnot(None),
            SportEventMapping.status == 'matched',
        )
        if last_x_hours:
            matched_events = matched_events.filter(SportEventMapping.start_time >= datetime.now(timezone.utc) - timedelta(hours=last_x_hours))
        matched_events = matched_events.all()
        return matched_events
    except Exception as e:
        logger.error(f"Error getting matched FootyStats events: {e}")
        return None

