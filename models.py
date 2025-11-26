from sqlalchemy import Column, Integer, BigInteger, String, Date, DateTime, ForeignKey, JSON, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()



class SportEventFootystats(Base):
    __tablename__ = 'sport_event_footystats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_uid = Column(String, default=None) # FootyStats match UID
    match_slug = Column(String, default=None) # FootyStats match slug
    time = Column(DateTime, default=None) # Match start time (format: datetime)
    time_str = Column(String, default=None)
    country = Column(String, default=None) # Competition country/continent
    competition = Column(String, default=None) # Competition name
    team_a = Column(String, default=None) # Home team name
    team_b = Column(String, default=None) # Away team name
    ppg_a = Column(Float, default=None) # Home team PPG (points per game / form)
    ppg_b = Column(Float, default=None) # Away team PPG (points per game / form)
    team_a_xg_prematch = Column(Float, default=None) # Home team XG (expected goals)
    team_b_xg_prematch = Column(Float, default=None) # Away team XG (expected goals)
    avg_potential = Column(Float, default=None) # Average goals (probability in average)
    btts_potential = Column(Float, default=None) # BTTS (Both teams to score) (probability in %)
    o05_potential = Column(Float, default=None) # Over 0.5 goals at full-time (probability in %)
    o15_potential = Column(Float, default=None) # Over 1.5 goals at full-time (probability in %)
    o25_potential = Column(Float, default=None) # Over 2.5 goals at full-time (probability in %)
    o35_potential = Column(Float, default=None) # Over 3.5 goals at full-time (probability in %)
    o05HT_potential = Column(Float, default=None) # Over 0.5 goals at half-time (probability in %)
    btts_fhg_potential = Column(Float, default=None) # BTTS (Both teams to score) at half-time (probability in %)
    btts_2hg_potential = Column(Float, default=None) # BTTS (Both teams to score) in 2nd half (probability in %)
    offsides_potential = Column(Float, default=None) # Offsides (probability in %)
    odds_ft_1 = Column(Float, default=None) # Decimal odds for Home team to win (full-time)
    odds_ft_x = Column(Float, default=None) # Decimal odds for Draw (full-time)
    odds_ft_2 = Column(Float, default=None) # Decimal odds for Away team to win (full-time)
    odds_btts_yes = Column(Float, default=None) # Decimal odds for BTTS (Both teams to score) (full-time)
    odds_ft_over05 = Column(Float, default=None) # Decimal odds for Over 0.5 goals (full-time)
    odds_ft_over15 = Column(Float, default=None) # Decimal odds for Over 1.5 goals (full-time)
    odds_ft_over25 = Column(Float, default=None) # Decimal odds for Over 2.5 goals (full-time)
    odds_ft_over35 = Column(Float, default=None) # Decimal odds for Over 3.5 goals (full-time)
    odds_ft_under15 = Column(Float, default=None) # Decimal odds for Under 1.5 goals (full-time)
    odds_ft_under25 = Column(Float, default=None) # Decimal odds for Under 2.5 goals (full-time)
    odds_ft_under35 = Column(Float, default=None) # Decimal odds for Under 3.5 goals (full-time)
    score_fh = Column(String, default=None) # Score at half-time (format= "1:0")
    score_2h = Column(String, default=None) # Score at 2nd half (format= "1:0")
    score_ft = Column(String, default=None) # Score at full-time (format= "1:0")
    date_scraped = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    date_updated = Column(DateTime, default=None)
    scrape_script = Column(String, default=None)

    def __repr__(self):
        return f"<SportEventFootystats(match_uid={self.match_uid}, time={self.time}, country={self.country}, competition={self.competition}, team_a={self.team_a}, team_b={self.team_b}>"


class SportPotentialFootystats(Base):
    __tablename__ = 'sport_potential_footystats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    stat = Column(String, default=None)
    sport_event_footystats_id = Column(Integer, ForeignKey('sport_event_footystats.id'), default=None)
    match_uid = Column(String, default=None)
    match_slug = Column(String, default=None)
    time_start_str = Column(String, default=None)
    time_start = Column(DateTime, default=None) # Only updated when confirmed mapping found
    date_start = Column(Date, default=None)
    country = Column(String, default=None)
    competition = Column(String, default=None)
    team_a = Column(String, default=None) # Used if stat concerns the match (both teams)
    team_b = Column(String, default=None) # Used if stat concerns the match (both teams)
    team_potential = Column(String, default=None) # Used if stat concerns a specific team
    average = Column(Float, default=None)
    matches_count = Column(Integer, default=None)
    probability = Column(Float, default=None)
    odd = Column(Float, default=None) # used for single probability
    probabilities = Column(JSON, default=None) # used for multiple probabilities (win-draw-wn)
    odds = Column(JSON, default=None) # used for multiple odds (win-draw-wn)

    date_scraped = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    date_updated = Column(DateTime, default=None)
    scrape_script = Column(String, default=None)

    # Relationship
    sport_event_footystats = relationship("SportEventFootystats", backref="sport_potential_footystats")

    def __repr__(self):
        return f"<SportEventFootystats(match_uid={self.match_uid}, time={self.time}, country={self.country}, competition={self.competition}, team_a={self.team_a}, team_b={self.team_b}>"


    
class SportEventBookmaker(Base):
    __tablename__ = 'sport_event_bookmaker'
    id = Column(BigInteger, primary_key=True)
    bookmaker = Column(String, nullable=False)
    league_id = Column(String, nullable=False)
    game_id = Column(String, nullable=False)
    match_url = Column(String, default=None)
    time = Column(DateTime, default=None)
    time_str = Column(String, default=None)
    country = Column(String, default=None)
    competition = Column(String, default=None)
    team_a = Column(String, default=None)
    team_b = Column(String, default=None)
    location = Column(String, default=None)
    round = Column(String, default=None)
    odds_ft_1 = Column(Float, default=None)
    odds_ft_x = Column(Float, default=None)
    odds_ft_2 = Column(Float, default=None)
    odds_btts_yes = Column(Float, default=None)
    odds_btts_no = Column(Float, default=None)
    odds_dc_1x = Column(Float, default=None)
    odds_dc_12 = Column(Float, default=None)
    odds_dc_x2 = Column(Float, default=None)
    odds_ft_over15 = Column(Float, default=None)
    odds_ft_over25 = Column(Float, default=None)
    odds_ft_over35 = Column(Float, default=None)
    odds_ft_over45 = Column(Float, default=None)
    odds_ft_under15 = Column(Float, default=None)
    odds_ft_under25 = Column(Float, default=None)
    odds_ft_under35 = Column(Float, default=None)
    odds_ft_under45 = Column(Float, default=None)
    score_fh = Column(String, default=None)
    score_2h = Column(String, default=None)
    score_ft = Column(String, default=None)
    score = Column(String, default=None)
    corners_fh = Column(String, default=None)
    corners_2h = Column(String, default=None)
    corners_ft = Column(String, default=None)
    subgames = Column(JSON, default={})
    status = Column(String, default=None)
    date_scraped = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    date_updated = Column(DateTime, default=None)

    def __repr__(self):
        return f"<SportEventBookmaker(id={self.id}, bookmaker={self.bookmaker}, league_id={self.league_id}, game_id={self.game_id}, time={self.time}, country={self.country}, competition={self.competition}, team_a={self.team_a}, team_b={self.team_b}>"



class SportEventMapping(Base):
    __tablename__ = 'sport_event_mapping'
    id = Column(BigInteger, primary_key=True)
    sport_event_footystats_id = Column(Integer, ForeignKey('sport_event_footystats.match_uid'))
    sport_event_bookmaker_id = Column(Integer, ForeignKey('sport_event_bookmaker.id'))
    start_time = Column(DateTime, default=None)
    status = Column(String, default=None)
    step = Column(String, default=None)
    country = Column(String, default=None)
    competition = Column(String, default=None)
    team_a = Column(String, default=None)
    team_b = Column(String, default=None)
    date_added = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    date_updated = Column(DateTime, default=None)
    
    # Relationships for easier analysis
    sport_event_footystats = relationship("SportEventFootystats", backref="mappings")
    sport_event_bookmaker = relationship("SportEventBookmaker", backref="mappings")
    
    def __repr__(self):
        return f"<SportEventMapping(sport_event_footystats_id={self.sport_event_footystats_id}, sport_event_bookmaker_id={self.sport_event_bookmaker_id}>"
    
    






