import requests
import random

# %%
import os
from pathlib import Path

from dotenv import load_dotenv
env_file = 'settings.env'
dotenv_path = Path(env_file)
load_dotenv(dotenv_path=dotenv_path)


from bs4 import BeautifulSoup
import markdown
import re
import json
import math
import time
import logging
import string


####################################
#        NOTIFICATIONS
####################################

async def send_notification(api_token="", user_key="", message="", title="", priority="0", retry=0, expire=0, sound=None, html=0):
    # return False # TEST
    
    url = "https://api.pushover.net/1/messages.json"

    payload = {
        'token': api_token,
        'user': user_key,
        'message': message,
    }
    if title:
        payload['title'] = title

    if priority:
        payload['priority'] = priority
    if retry:
        payload['retry'] = retry
    if expire:
        payload['expire'] = expire

    if sound:
        payload['sound'] = sound

    if html in [True, 1]:
        payload['html'] = 1

    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()  # Raises an exception for HTTP errors

        # Check for success
        if response.json().get('status') == 1:
            print("Notification sent successfully!")
        else:
            print("Failed to send notification:", response.json().get('error'))

    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)
        raise



# %%
# SCRAPING FUNCTIONS
def get_random_user_agent():
    return random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:111.0) Gecko/20100101 Firefox/111.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:110.0) Gecko/20100101 Firefox/110.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13.0; rv:110.0) Gecko/20100101 Firefox/110.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    ])


def fetch_proxy(provider=""):
    if provider == "tailscale":
        PROXY_TAILSCALE = os.environ['PROXY_TAILSCALE']
        return f"{PROXY_TAILSCALE}"
    if provider == "dataimpulse":
        PROXY_DATAIMPULSE = os.environ['PROXY_DATAIMPULSE']
        return f"{PROXY_DATAIMPULSE}"
    if provider == "dataimpulse_mobile":
        PROXY_DATAIMPULSE_MOBILE = os.environ['PROXY_DATAIMPULSE_MOBILE']
        return f"{PROXY_DATAIMPULSE_MOBILE}"
    if provider == "dataimpulse_residential":
        PROXY_DATAIMPULSE_RESIDENTIAL = os.environ['PROXY_DATAIMPULSE_RESIDENTIAL']
        return f"{PROXY_DATAIMPULSE_RESIDENTIAL}"
    if provider == "geonode":
        PROXY_GEONODE = os.environ['PROXY_GEONODE']
        return f"{PROXY_GEONODE}"
    if provider == "geonode_2":
        PROXY_GEONODE_2 = os.environ['PROXY_GEONODE_2']
        return f"{PROXY_GEONODE_2}"
    if provider == "brightdata":
        PROXY_BRIGHTDATA = os.environ['PROXY_BRIGHTDATA']
        PROXY_BRIGHTDATA = re.sub(r'xxx', generate_random_string(8), PROXY_BRIGHTDATA)
        return f"{PROXY_BRIGHTDATA}"
    return None

def fetch_proxy_dict(provider=""):
    if provider == "dataimpulse_mobile":
        PROXY_SERVER = os.environ['PROXY_DATAIMPULSE_MOBILE_SERVER'] or ""
        PROXY_USERNAME = os.environ['PROXY_DATAIMPULSE_MOBILE_USERNAME'] or ""
        PROXY_PASSWORD = os.environ['PROXY_DATAIMPULSE_MOBILE_PASSWORD'] or ""
    elif provider == "dataimpulse_residential":
        PROXY_SERVER = os.environ['PROXY_DATAIMPULSE_RESIDENTIAL_SERVER'] or ""
        PROXY_USERNAME = os.environ['PROXY_DATAIMPULSE_RESIDENTIAL_USERNAME'] or ""
        PROXY_PASSWORD = os.environ['PROXY_DATAIMPULSE_RESIDENTIAL_PASSWORD'] or ""
    elif provider == "geonode":
        PROXY_SERVER = os.environ['PROXY_GEONODE_SERVER'] or ""
        PROXY_USERNAME = os.environ['PROXY_GEONODE_USERNAME'] or ""
        PROXY_PASSWORD = os.environ['PROXY_GEONODE_PASSWORD'] or ""
    else:
        return None
    return {
        'server': PROXY_SERVER,
        'username': PROXY_USERNAME,
        'password': PROXY_PASSWORD
    }



# %%
# TEXT PROCESSING FUNCTIONS
def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text)).text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)    
    return text

def md_to_text(md):
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()

def clean_json_md_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string


# %%
####################################
#        POLYMARKET
####################################
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, TradeParams, MarketOrderArgs
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.exceptions import PolyApiException

async def pm_limit_order(token_id="", side=None, size=1, price=0.01, logger_custom=None):
    log = logger_custom if logger_custom else logging.getLogger(__name__)
    log.debug(f"[pm_limit_order] token_id={token_id} side={side} size={size} price={price}")
    POLYMARKET_HOST: str = "https://clob.polymarket.com"
    POLYMARKET_KEY: str = os.environ['POLYMARKET_KEY'] #This is your Private Key. Export from reveal.polymarket.com or from your Web3 Application
    POLYMARKET_CHAIN_ID: int = 137 #No need to adjust this
    POLYMARKET_PROXY_ADDRESS: str = os.environ['POLYMARKET_PROXY_ADDRESS'] #This is the address you deposit/send USDC to to FUND your Polymarket account.
    try:
        if not side or side not in [BUY, SELL]:
            log.error(f"[pm_limit_order] Invalid side: {side}")
            return None
        client = ClobClient(POLYMARKET_HOST, key=POLYMARKET_KEY, chain_id=POLYMARKET_CHAIN_ID, signature_type=1, funder=POLYMARKET_PROXY_ADDRESS)
        client.set_api_creds(client.create_or_derive_api_creds())
        order_args = OrderArgs(
            price=price,
            side=side,
            size=size,
            token_id=token_id,
        )
        signed_order = client.create_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)
        return resp
    except Exception as e:
        log.error(f"[pm_limit_order] Error: {e}")
        return None

async def pm_market_order_buy(token_id="", amount=1, retry=0, logger_custom=None):
    log = logger_custom if logger_custom else logging.getLogger(__name__)
    log.debug(f"[pm_market_order_buy] token_id={token_id} amount={amount}")
    POLYMARKET_HOST: str = "https://clob.polymarket.com"
    POLYMARKET_KEY: str = os.environ['POLYMARKET_KEY'] #This is your Private Key. Export from reveal.polymarket.com or from your Web3 Application
    POLYMARKET_CHAIN_ID: int = 137 #No need to adjust this
    POLYMARKET_PROXY_ADDRESS: str = os.environ['POLYMARKET_PROXY_ADDRESS'] #This is the address you deposit/send USDC to to FUND your Polymarket account.
    try:
        client = ClobClient(POLYMARKET_HOST, key=POLYMARKET_KEY, chain_id=POLYMARKET_CHAIN_ID, signature_type=1, funder=POLYMARKET_PROXY_ADDRESS)
        client.set_api_creds(client.create_or_derive_api_creds())
        order_args = MarketOrderArgs(
            side=BUY,
            amount=amount, # $$$
            token_id=token_id,
        )
        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)
        log.info(f"[pm_market_order_buy] resp={resp}")
        return resp
    except PolyApiException as e:
        log.error(f"[pm_market_order_buy] #{retry} ({token_id}) Error: {e}")
        # retry once
        time.sleep(0.5)
        resp = None
        while retry < 1 and resp is None:
            retry += 1
            resp = await pm_market_order_buy(token_id=token_id, amount=amount, retry=retry, logger_custom=logger_custom)
            log.info(f"[pm_market_order_buy] #{retry} ({token_id}) resp={resp}")
        return resp
    except Exception as e:
        log.error(f"[pm_market_order_buy] Error: {e}")
        return None

async def pm_market_order_sell(token_id="", size=1, retry=0, logger_custom=None):
    log = logger_custom if logger_custom else logging.getLogger(__name__)
    log.debug(f"[pm_market_order_sell] token_id={token_id} size={size}")
    POLYMARKET_HOST: str = "https://clob.polymarket.com"
    POLYMARKET_KEY: str = os.environ['POLYMARKET_KEY'] #This is your Private Key. Export from reveal.polymarket.com or from your Web3 Application
    POLYMARKET_CHAIN_ID: int = 137 #No need to adjust this
    POLYMARKET_PROXY_ADDRESS: str = os.environ['POLYMARKET_PROXY_ADDRESS'] #This is the address you deposit/send USDC to to FUND your Polymarket account.
    try:
        client = ClobClient(POLYMARKET_HOST, key=POLYMARKET_KEY, chain_id=POLYMARKET_CHAIN_ID, signature_type=1, funder=POLYMARKET_PROXY_ADDRESS)
        client.set_api_creds(client.create_or_derive_api_creds())
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=size,  # SHARES
            side=SELL,
        )
        signed_order = client.create_market_order(order_args)
        orderType = OrderType.FOK if retry == 0 else OrderType.FAK
        resp = client.post_order(signed_order, orderType=orderType)
        log.info(f"[pm_market_order_sell] ({token_id}) resp={resp}")
        return resp
    except PolyApiException as e:
        log.error(f"[pm_market_order_sell] #{retry} ({token_id}) Error: {e}")
        # retry 3 times
        time.sleep(0.5)
        resp = None
        while retry < 3 and resp is None:
            retry += 1
            resp = await pm_market_order_sell(token_id=token_id, size=size, retry=retry, logger_custom=logger_custom)
            log.info(f"[pm_market_order_sell] #{retry} ({token_id}) resp={resp}")
        return resp
    except Exception as e:
        log.error(f"[pm_market_order_sell] Error: {e}")
        return None


def pm_get_positions(conditionId: str = None, sizeThreshold=1, logger_custom=None):
    """
    Get Polymarket positions for a given user.
    Optionally filter by market.
    """
    log = logger_custom if logger_custom else logging.getLogger(__name__)
    log.debug(f"[pm_get_positions] conditionId={conditionId} sizeThreshold={sizeThreshold}")

    POLYMARKET_PROXY_ADDRESS: str = os.environ['POLYMARKET_PROXY_ADDRESS'] #This is the address you deposit/send USDC to to FUND your Polymarket account.

    url = "https://data-api.polymarket.com/positions"
    params = {"user": POLYMARKET_PROXY_ADDRESS}
    if conditionId:
        params["market"] = conditionId
        params["sizeThreshold"] = sizeThreshold
    else:
        log.warning("[pm_get_positions] No conditionId provided")
        return None
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        log.debug(f"[pm_get_positions] resp={resp.json()}")
        return resp.json()
    except Exception as e:
        log.error(f"[pm_get_positions] Error: {e}")
        return None


# TODO: Get Polymarket trades
# def pm_get_trades()


# %%
####################################
#        SCRAPING FUNCTIONS
####################################
def fetch_pm_events(slug="", id=None, tag_slug="", start_date_min=None, end_date_max=None, exclude_tags=[], active=None, limit=20, offset=0, proxy_provider=None, logger_custom=None):
    log = logger_custom if logger_custom else logging.getLogger(__name__)
    log.debug(f"[fetch_pm_events] slug={slug} id={id} tag_slug={tag_slug} start_date_min={start_date_min} end_date_max={end_date_max} active={active} limit={limit} offset={offset}")
    
    REQUEST_TIMEOUT = 5
    proxy_dict = {}
    proxy = fetch_proxy(provider=proxy_provider)
    if proxy:
        proxy_dict = {'http': proxy, 'https': proxy}
    
    try:
        url = "https://gamma-api.polymarket.com/events"
        params = {
            "limit": limit,
            "offset": offset
        }
        if slug:
            params["slug"] = slug
            log.debug(f"Fetching Polymarket event with slug={slug}")
        elif id:
            params["id"] = id
            log.debug(f"Fetching Polymarket event with id={id}")
        elif tag_slug:
            params["tag_slug"] = tag_slug # (Example: economy)
            log.debug(f"Fetching Polymarket event with tag_slug={tag_slug}")
        else:
            log.warning("[fetch_pm_events] No slug, id, or tag_slug provided")
            return None
        
        if start_date_min:
            params["end_date_min"] = start_date_min # (Example: 2025-06-01)
        
        if end_date_max:
            params["end_date_max"] = end_date_max # (Example: 2025-06-30)
        
        if exclude_tags and len(exclude_tags):
            params["exclude_tag_id"] = exclude_tags
        if active in [1, True]:
            params["active"] = 1
            params["closed"] = 0
        log.info(f"Fetching Polymarket events with params: {params}")

        headers = {}
        response = requests.request("GET", url, headers=headers, params=params, proxies=proxy_dict, timeout=REQUEST_TIMEOUT)
        if response.status_code in [403, 429, 202, 101]:
            log.error(f"Request failed. " + str(response.status_code))
            time.sleep(1)
            response = requests.get(url, headers=headers, params=params, proxies=proxy_dict, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

        response.raise_for_status()

        if response.json():
            response_json = response.json()
            log.debug(f"Fetched {len(response_json)}")
            return response_json
    except Exception as e:
        log.error(f"Error fetching Polymarket events: {e}")
        return None

async def fetch_polymarket_prices(tokens=[], worker=True, proxy_provider=None, logger_custom=None):
    log = logger_custom if logger_custom else logging.getLogger(__name__)
    # log.debug(f"[fetch_polymarket_prices] tokens={tokens}")
    REQUEST_TIMEOUT = 2
    proxy_dict = {}
    proxy = fetch_proxy(provider=proxy_provider)
    if proxy:
        proxy_dict = {'http': proxy, 'https': proxy}

    headers = {"Content-Type": "application/json"}

    if worker:
        url = "https://pm.pryv.online/prices"
    else:
        url = "https://clob.polymarket.com/prices"
    
    if tokens and len(tokens):
        payload = []
        if worker:
            payload = {"tokens": tokens}
        else:
            for token in tokens:
                payload.append({
                    "token_id": token,
                    "side": "BUY"
                })
                payload.append({
                    "token_id": token,
                    "side": "SELL"
                })
    else:
        log.warning("[fetch_polymarket_prices] No tokens provided")
        return None
    if payload and len(payload):
        # log.debug(f"[fetch_polymarket_prices] POST {url} payload={payload}")
        pass
    else:
        log.warning("[fetch_polymarket_prices] No payload")
        return None

    try:
        response = requests.post(url, headers=headers, json=payload, proxies=proxy_dict, timeout=REQUEST_TIMEOUT)
        if response.status_code in [403, 429, 202, 101]:
            log.error(f"Request failed. " + str(response.status_code))
            time.sleep(1)
            response = requests.post(url, headers=headers, json=payload, proxies=proxy_dict, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        response.raise_for_status()
        if response.json():
            response_json = response.json()
            # log.debug(f"Fetched {len(response_json)} prices")
            return response_json
    except requests.exceptions.RequestException as e:
        log.error(f"Error fetching Polymarket prices: {e}")
        return None
    except Exception as e:
        log.error(f"Error fetching Polymarket prices: {e}")
        return None


# %%
###
# IP
###
def fetch_ip_requests(provider=""):
    proxy_dict = {}
    proxy = fetch_proxy(provider=provider)
    if proxy:
        proxy_dict = {'http': proxy, 'https': proxy}
    else:
        print("No proxy available.")
        proxy_dict = {}

    try:
        response = requests.get('https://ipinfo.io/json', proxies=proxy_dict, timeout=10)
        response.raise_for_status()
        ip_info = response.json()
        print("IP Information:")
        print(json.dumps(ip_info, indent=2))
    except requests.RequestException as e:
        print(f"Error fetching IP information: {e}")


###
# MATH
###
def round_up(num, dec=0):
    mult = 10 ** dec
    return math.ceil(num * mult) / mult


###
# LOGGING COLORS
###
class AnsiColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        no_style = '\033[0m'
        bold = '\033[1m'
        underline = '\033[4m'
        grey = '\033[90m'
        yellow = '\033[93m'
        red = '\033[31m'
        red_light = '\033[91m'
        green = '\033[32m'
        start_style = {
            'DEBUG': grey,
            'INFO': no_style + bold,
            'WARNING': yellow,
            'ERROR': red + underline,
            'CRITICAL': red_light + bold,
        }.get(record.levelname, no_style)
        end_style = no_style
        return f'{start_style}{super().format(record)}{end_style}'

