
CREATE TABLE sport_event_footystats (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    match_uid VARCHAR(255) DEFAULT NULL,
    match_slug VARCHAR(255) DEFAULT NULL,
    time DATETIME DEFAULT NULL,
    time_str VARCHAR(255) DEFAULT NULL,
    country VARCHAR(255) DEFAULT NULL,
    competition VARCHAR(255) DEFAULT NULL,
    team_a VARCHAR(255) DEFAULT NULL,
    team_b VARCHAR(255) DEFAULT NULL,
    ppg_a FLOAT DEFAULT NULL,
    ppg_b FLOAT DEFAULT NULL,
    team_a_xg_prematch FLOAT DEFAULT NULL,
    team_b_xg_prematch FLOAT DEFAULT NULL,
    avg_potential FLOAT DEFAULT NULL,
    btts_potential FLOAT DEFAULT NULL,
    o15_potential FLOAT DEFAULT NULL,
    o25_potential FLOAT DEFAULT NULL,
    o05_potential FLOAT DEFAULT NULL,
    o35_potential FLOAT DEFAULT NULL,
    o05HT_potential FLOAT DEFAULT NULL,
    btts_fhg_potential FLOAT DEFAULT NULL,
    btts_2hg_potential FLOAT DEFAULT NULL,
    offsides_potential FLOAT DEFAULT NULL,
    odds_ft_1 FLOAT DEFAULT NULL,
    odds_ft_x FLOAT DEFAULT NULL,
    odds_ft_2 FLOAT DEFAULT NULL,
    odds_btts_yes FLOAT DEFAULT NULL,
    odds_ft_over05 FLOAT DEFAULT NULL,
    odds_ft_over15 FLOAT DEFAULT NULL,
    odds_ft_over25 FLOAT DEFAULT NULL,
    odds_ft_over35 FLOAT DEFAULT NULL,
    odds_ft_under15 FLOAT DEFAULT NULL,
    odds_ft_under25 FLOAT DEFAULT NULL,
    odds_ft_under35 FLOAT DEFAULT NULL,
    score_fh VARCHAR(255) DEFAULT NULL,
    score_2h VARCHAR(255) DEFAULT NULL,
    score_ft VARCHAR(255) DEFAULT NULL,
    scrape_script VARCHAR(100) DEFAULT NULL,
    date_scraped DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    date_updated DATETIME DEFAULT NULL,
);

-- Add scrape_script column
ALTER TABLE sport_event_footystats ADD COLUMN scrape_script VARCHAR(100) DEFAULT NULL AFTER date_updated;
UPDATE sport_event_footystats SET scrape_script = 'scrape_sports_data';

-- Add the id column as the first column
ALTER TABLE sport_event_footystats ADD COLUMN id BIGINT FIRST;
-- First, populate the id column with sequential values
SET @row_number = 0;
UPDATE sport_event_footystats 
SET id = (@row_number := @row_number + 1)
ORDER BY date_scraped, match_uid;
-- Drop the existing foreign key constraint
ALTER TABLE sport_event_mapping DROP FOREIGN KEY sport_event_mapping_ibfk_2;
-- Change the column type to match the new primary key
ALTER TABLE sport_event_mapping MODIFY sport_event_footystats_id BIGINT DEFAULT NULL;
-- Drop the old primary key if it exists (match_uid)
ALTER TABLE sport_event_footystats DROP PRIMARY KEY;
-- Set id as the new primary key
ALTER TABLE sport_event_footystats ADD PRIMARY KEY (id);
-- Make the id column NOT NULL and AUTO_INCREMENT
ALTER TABLE sport_event_footystats MODIFY id BIGINT NOT NULL AUTO_INCREMENT;
-- Update any existing references from match_uid to id
UPDATE sport_event_mapping sem 
JOIN sport_event_footystats sef ON sem.sport_event_footystats_id = sef.match_uid 
SET sem.sport_event_footystats_id = sef.id;
-- Add the new foreign key constraint
ALTER TABLE sport_event_mapping ADD CONSTRAINT fk_sport_event_footystats 
    FOREIGN KEY (sport_event_footystats_id) REFERENCES sport_event_footystats(id);

-- Change column match_url to match_slug
ALTER TABLE sport_event_footystats CHANGE match_url match_slug VARCHAR(255) DEFAULT NULL;

-- Update match_slug to remove the URL part
UPDATE sport_event_footystats SET match_slug = SUBSTRING_INDEX(match_slug, '/', -1);




CREATE TABLE sport_event_bookmaker (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    bookmaker VARCHAR(255) DEFAULT NULL,
    league_id VARCHAR(255) DEFAULT NULL,
    game_id VARCHAR(255) DEFAULT NULL,
    match_url VARCHAR(255) DEFAULT NULL,
    time DATETIME DEFAULT NULL,
    time_str VARCHAR(255) DEFAULT NULL,
    country VARCHAR(255) DEFAULT NULL,
    competition VARCHAR(255) DEFAULT NULL,
    team_a VARCHAR(255) DEFAULT NULL,
    team_b VARCHAR(255) DEFAULT NULL,
    location VARCHAR(255) DEFAULT NULL,
    round VARCHAR(255) DEFAULT NULL,
    odds_ft_1 FLOAT DEFAULT NULL,
    odds_ft_x FLOAT DEFAULT NULL,
    odds_ft_2 FLOAT DEFAULT NULL,
    odds_btts_yes FLOAT DEFAULT NULL,
    odds_ft_over05 FLOAT DEFAULT NULL,
    odds_ft_over15 FLOAT DEFAULT NULL,
    odds_ft_over25 FLOAT DEFAULT NULL,
    odds_ft_over35 FLOAT DEFAULT NULL,
    odds_ft_under15 FLOAT DEFAULT NULL,
    odds_ft_under25 FLOAT DEFAULT NULL,
    odds_ft_under35 FLOAT DEFAULT NULL,
    score_fh VARCHAR(30) DEFAULT NULL,
    score_2h VARCHAR(30) DEFAULT NULL,
    score_ft VARCHAR(30) DEFAULT NULL,
    score VARCHAR(30) DEFAULT NULL,
    corners_fh VARCHAR(30) DEFAULT NULL,
    corners_2h VARCHAR(30) DEFAULT NULL,
    corners_ft VARCHAR(30) DEFAULT NULL,
    subgames JSON DEFAULT NULL,
    date_scraped DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    date_updated DATETIME DEFAULT NULL
);
-- ALTER TABLE queries for new columns added to sport_event_bookmaker
ALTER TABLE sport_event_bookmaker ADD COLUMN score VARCHAR(30) DEFAULT NULL AFTER score_ft;
ALTER TABLE sport_event_bookmaker ADD COLUMN subgames JSON DEFAULT NULL AFTER corners_ft;
ALTER TABLE sport_event_bookmaker ADD COLUMN status VARCHAR(255) DEFAULT NULL AFTER subgames;
ALTER TABLE sport_event_bookmaker
    MODIFY corners_fh VARCHAR(30),
    MODIFY corners_2h VARCHAR(30),
    MODIFY corners_ft VARCHAR(30),
    MODIFY score_ft VARCHAR(30),
    MODIFY score_fh VARCHAR(30),
    MODIFY score_2h VARCHAR(30);
ALTER TABLE sport_event_bookmaker ADD COLUMN location VARCHAR(255) DEFAULT NULL AFTER team_b;
ALTER TABLE sport_event_bookmaker ADD COLUMN round VARCHAR(255) DEFAULT NULL AFTER location;

CREATE TABLE sport_event_mapping (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    sport_event_footystats_id VARCHAR(255) DEFAULT NULL,
    sport_event_bookmaker_id BIGINT DEFAULT NULL,
    start_time DATETIME,
    status VARCHAR(255),
    step VARCHAR(50),
    country VARCHAR(255),
    competition VARCHAR(255),
    team_a VARCHAR(255),
    team_b VARCHAR(255),
    date_added DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    date_updated DATETIME DEFAULT NULL,
    -- FOREIGN KEY (sport_event_id) REFERENCES sport_event(id),
    FOREIGN KEY (sport_event_footystats_id) REFERENCES sport_event_footystats(id),
    FOREIGN KEY (sport_event_bookmaker_id) REFERENCES sport_event_bookmaker(id)
);
ALTER TABLE sport_event_mapping ADD COLUMN step VARCHAR(50) DEFAULT NULL AFTER status;
ALTER TABLE sport_event_mapping DROP FOREIGN KEY sport_event_mapping_ibfk_1;
ALTER TABLE sport_event_mapping DROP COLUMN sport_event_id;



CREATE TABLE sport_potential_footystats (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    stat VARCHAR(255) DEFAULT NULL,
    sport_event_footystats_id BIGINT DEFAULT NULL,
    match_uid VARCHAR(255) DEFAULT NULL,
    match_slug VARCHAR(255) DEFAULT NULL,
    time_start_str VARCHAR(255) DEFAULT NULL,
    time_start DATETIME DEFAULT NULL,
    date_start DATE DEFAULT NULL,
    country VARCHAR(255) DEFAULT NULL,
    competition VARCHAR(255) DEFAULT NULL,
    team_a VARCHAR(255) DEFAULT NULL,
    team_b VARCHAR(255) DEFAULT NULL,
    team_potential VARCHAR(255) DEFAULT NULL,
    average FLOAT DEFAULT NULL,
    matches_count INT DEFAULT NULL,
    probability FLOAT DEFAULT NULL,
    odd FLOAT DEFAULT NULL,
    probabilities JSON DEFAULT NULL,
    odds JSON DEFAULT NULL,
    date_scraped DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    date_updated DATETIME DEFAULT NULL,
    scrape_script VARCHAR(255) DEFAULT NULL,
    FOREIGN KEY (sport_event_footystats_id) REFERENCES sport_event_footystats(id)
);