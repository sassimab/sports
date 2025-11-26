# Sports Performance Tracker

A comprehensive Streamlit interface for analyzing sports matches performance using footystats potentials, bookmaker odds, and actual results. Build and test betting strategies based on statistical potentials and historical outcomes.

## Features

### 📊 Overview Dashboard
- Key performance metrics (total matches, average potential, countries covered)
- Interactive charts showing match distribution by country and competition
- Potential distribution analysis
- Competition performance profiling

### 🔍 Match Browser
- Searchable table of all matched events
- Team name search functionality
- Sortable columns (date, potential values, competition)
- CSV export for filtered datasets
- Comprehensive match information display

### 🎯 Strategy Backtesting
- Test different potential thresholds (0.5-0.9 range)
- Analyze historical success rates for:
  - Over 2.5 goals markets
  - Both Teams to Score (BTTS) markets
  - Home win markets
- Performance recommendations based on historical data
- ROI and success rate calculations

### 📈 Potential Analysis
- Distribution analysis of various potential types
- Correlation analysis between potentials and actual results
- Team-specific performance tracking
- Time-based trend analysis (daily/weekly/monthly)

## Data Sources

The interface analyzes data from multiple sources through the central `SportEventMapping` table:

- **SportEventFootystats**: Statistical potentials and odds from footystats
- **SportEventBookmaker**: Bookmaker odds and match results
- **SportPotentialFootystats**: Detailed team-specific statistical potentials
- **SportEventMapping**: Central mapping table linking all data sources

Only events that have been successfully matched in `SportEventMapping` are included in the analysis.

## Installation & Setup

1. **Environment Setup**
   ```bash
   # Ensure you're in the polysmart root directory
   cd /home/mab/Lab/polysmart
   
   # Install required packages
   pip install streamlit pandas numpy plotly sqlalchemy mysql-connector-python python-dotenv
   ```

2. **Database Configuration**
   - Ensure MySQL database is running
   - Verify database credentials in `settings.env`
   - Confirm sports tables are populated with data

3. **Run the Application**
   ```bash
   cd sports/interface
   streamlit run sports_performance_app.py
   ```

   The application will open in your browser at `http://localhost:8501`

## Usage Guide

### Getting Started

1. **Apply Filters**: Use the sidebar filters to load matched events data:
   - Select country and competition
   - Set date range for analysis
   - Configure potential thresholds
   - Choose data quality options

2. **Explore Tabs**: Navigate through the four main analysis tabs:
   - Start with **Overview** for high-level insights
   - Use **Match Browser** for detailed event inspection
   - Test strategies in **Strategy Backtesting**
   - Deep dive into **Potential Analysis**

### Filter Options

- **Country**: Filter by specific countries or show all
- **Competition**: League/competition level filtering
- **Date Range**: Analyze specific time periods
- **Potential Thresholds**: 
  - Min/Max Average Potential (0.0-1.0)
  - Min BTTS Potential for goal-based strategies
- **Data Quality**: Require score data from footystats/bookmaker sources

### Strategy Development

1. **Backtesting**: Use the Strategy Backtesting tab to:
   - Test potential thresholds (0.5-0.9 recommended range)
   - Analyze success rates for different market types
   - Identify optimal threshold levels

2. **Performance Analysis**: In Potential Analysis tab:
   - Examine potential distributions
   - Correlate potentials with actual results
   - Track team-specific performance over time

3. **Data Export**: Use Match Browser to:
   - Filter events based on your criteria
   - Export to CSV for external analysis
   - Build custom datasets for strategy testing

## Key Metrics Explained

### Potential Values
- **Average Potential**: Overall match potential (0.0-1.0 scale)
- **BTTS Potential**: Probability both teams score
- **O15/O25/O35 Potential**: Over 1.5/2.5/3.5 goals potential
- **Team Potentials**: Individual team statistical potentials

### Success Rates
- Calculated from actual match results vs. predictions
- Higher potential values should correlate with higher success rates
- Use backtesting to find optimal thresholds

### Performance Indicators
- **Sample Size**: Number of matches analyzed
- **Correlation**: Statistical relationship between potential and outcomes
- **ROI Potential**: Based on potential vs. bookmaker odds comparison

## Technical Details

### Database Schema
The application queries through `SportEventMapping` to join:
- `SportEventFootystats` (via `sport_event_footystats_id`)
- `SportEventBookmaker` (via `sport_event_bookmaker_id`)
- `SportPotentialFootystats` (detailed stats)

### Performance Optimizations
- Database connection pooling
- Streamlit caching for expensive queries
- Efficient SQLAlchemy joins with proper indexing
- Lazy loading of large datasets

### Data Processing
- Score parsing for multiple formats (2-1, 2:1, etc.)
- Goal calculations for total/over/under markets
- Team performance attribution (home/away)
- Time-based aggregation for trend analysis

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify MySQL is running on correct port
   - Check credentials in `settings.env`
   - Ensure database and tables exist

2. **Empty Data Results**
   - Apply filters in sidebar to load data
   - Check if SportEventMapping has records
   - Verify date range contains events

3. **Performance Issues**
   - Reduce date range for faster queries
   - Use specific country/competition filters
   - Clear browser cache if needed

### Data Quality
- Events require successful mapping in SportEventMapping
- Score data needed for result analysis
- Potential values may be null for some events
- Bookmaker odds vary by source and time

## File Structure

```
sports/interface/
├── sports_performance_app.py    # Main Streamlit application
├── utils_sports_db.py          # Database utility functions
└── README.md                   # This documentation
```

## Contributing

To extend the interface:

1. **Add New Analysis**: Create new functions in `utils_sports_db.py`
2. **Enhance Filters**: Modify sidebar filter section in main app
3. **Add Charts**: Use Plotly for new visualizations
4. **Export Options**: Extend CSV export functionality

## Support

For technical issues:
1. Check database connectivity
2. Verify environment variables
3. Review Streamlit console output
4. Test with smaller date ranges

---

**Built for sports analytics and strategy development using footystats potentials and bookmaker data.**
