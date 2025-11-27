#!/usr/bin/env python3
"""
Streamlit Sports Matches Performance Tracker
Analyze footystats potentials, bookmaker odds, and match results
Build strategies based on statistical potentials and actual outcomes
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from utils_sports_db import (
    get_matched_events, get_potential_stats, get_filter_options,
    calculate_potential_accuracy, parse_score, parse_all_scores, analyze_potential_performance,
    export_to_csv
)

# Load environment
env_file = Path(__file__).parent.parent.parent / 'settings.env'
load_dotenv(dotenv_path=env_file)

# Page configuration
st.set_page_config(
    page_title="Sports Performance Tracker",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #00cc00;
    }
    .negative {
        color: #ff0000;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
    }
    .highlight-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 10px;
        margin: 10px 0;
    }
    .stElementContainer.st-key-btn-simulate {
        position: fixed;
        width: auto;
        top: 1em;
        box-shadow: 0 0 50px rgba(0,0,0,0.9);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
# st.title("⚽ Sports Performance Tracker")
# st.markdown("Analyze footystats potentials, bookmaker odds, and match results to build winning strategies")

# Initialize session state
if 'filters' not in st.session_state:
    st.session_state.filters = {}
if 'current_data' not in st.session_state:
    st.session_state.current_data = pd.DataFrame()
if 'last_filters_str' not in st.session_state:
    st.session_state.last_filters_str = ''
if 'should_run_simulation' not in st.session_state:
    st.session_state.should_run_simulation = False

# Initialize default odds values to prevent NULL filter issues
default_odds_values = {
    'footystats_odds_btts_min': 1.01, 'footystats_odds_btts_max': 20.0,
    'footystats_odds_1_min': 1.01, 'footystats_odds_1_max': 20.0,
    'footystats_odds_x_min': 1.01, 'footystats_odds_x_max': 20.0,
    'footystats_odds_2_min': 1.01, 'footystats_odds_2_max': 20.0,
    'footystats_odds_over15_min': 1.01, 'footystats_odds_over15_max': 20.0,
    'footystats_odds_over25_min': 1.01, 'footystats_odds_over25_max': 20.0,
    'footystats_odds_over35_min': 1.01, 'footystats_odds_over35_max': 20.0,
    'footystats_odds_over05_min': 1.01, 'footystats_odds_over05_max': 20.0
}

for key, value in default_odds_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

def render_overview_tab(df):
    """Render overview dashboard with key metrics and betting insights"""
    st.header("📊 Performance Overview")
    
    # Get bet stake from session state
    bet_stake = st.session_state.filters.get('bet_stake', 10.0) if 'filters' in st.session_state else 10.0
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_matches = len(df)
        st.metric("Total Matches", total_matches)
    
    with col2:
        matches_with_scores = len(df[df['bookmaker_score_ft'].notna()])
        st.metric("Matches with Scores", matches_with_scores)
    
    with col3:
        avg_potential = df['avg_potential'].mean() if 'avg_potential' in df.columns else 0
        st.metric("Avg Potential", f"{avg_potential:.3f}")
    
    with col4:
        countries_count = df['country'].nunique() if 'country' in df.columns else 0
        st.metric("Countries", countries_count)
    
    # Betting metrics row
    st.subheader("💰 Betting Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Calculate theoretical bets if all matches had odds
        matches_with_btts_odds = len(df[df['odds_btts_yes'].notna()])
        st.metric("Matches with BTTS Odds", matches_with_btts_odds)
    
    with col2:
        # Average BTTS odds
        avg_btts_odds = df['odds_btts_yes'].mean() if 'odds_btts_yes' in df.columns else 0
        if avg_btts_odds > 0:
            st.metric("Avg BTTS Odds", f"{avg_btts_odds:.2f}")
        else:
            st.metric("Avg BTTS Odds", "N/A")
    
    with col3:
        # Theoretical max single bet return
        max_btts_odds = df['odds_btts_yes'].max() if 'odds_btts_yes' in df.columns else 0
        if max_btts_odds > 0:
            max_return = bet_stake * (max_btts_odds - 1)
            st.metric(f"Max Single Bet Return (${bet_stake} stake)", f"${max_return:.2f}")
        else:
            st.metric(f"Max Single Bet Return (${bet_stake} stake)", "N/A")
    
    with col4:
        # High potential matches (>0.8)
        high_potential_matches = len(df[df['avg_potential'] > 0.8]) if 'avg_potential' in df.columns else 0
        st.metric("High Potential Matches (>0.8)", high_potential_matches)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matches by Country")
        if 'country' in df.columns:
            country_counts = df['country'].value_counts().head(10)
            fig = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                title="Top 10 Countries by Match Count"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Potential Distribution")
        if 'avg_potential' in df.columns:
            fig = px.histogram(
                df, 
                x='avg_potential',
                nbins=20,
                title="Average Potential Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
    
    # Performance by competition
    st.subheader("Performance by Competition")
    if 'competition' in df.columns and not df.empty:
        comp_stats = df.groupby('competition').agg({
            'avg_potential': 'mean',
            'btts_potential': 'mean',
            'o25_potential': 'mean'
        }).round(3)
        
        comp_stats = comp_stats.sort_values('avg_potential', ascending=False).head(10)
        
        fig = px.scatter(
            comp_stats.reset_index(),
            x='avg_potential',
            y='btts_potential',
            size='o25_potential',
            hover_name='competition',
            title="Competition Potential Profile"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    # Betting odds distribution
    if 'odds_btts_yes' in df.columns:
        st.subheader("📈 BTTS Odds Distribution")
        odds_data = df['odds_btts_yes'].dropna()
        
        if not odds_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    odds_data,
                    nbins=30,
                    title="BTTS Odds Distribution",
                    marginal="box"
                )
                fig.add_vline(x=odds_data.mean(), line_dash="dash", line_color="red", 
                             annotation_text=f"Avg: {odds_data.mean():.2f}")
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Odds ranges analysis
                odds_ranges = pd.cut(odds_data, bins=[0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0], 
                                   labels=['<1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-5.0', '>5.0'])
                range_counts = odds_ranges.value_counts().sort_index()
                
                fig = px.pie(
                    values=range_counts.values,
                    names=range_counts.index,
                    title="BTTS Odds Range Distribution"
                )
                st.plotly_chart(fig, width='stretch')

def render_match_browser_tab(df):
    """Render searchable match browser"""
    st.header("🔍 Match Browser")
    
    # Check if we have simulation results and show only simulation matches
    if 'last_simulation_results' in st.session_state:
        sim_results = st.session_state.last_simulation_results
        if sim_results is None:
            st.info("👈 Enable auto-run in sidebar and change filters to see simulation results")
            return
        valid_indices = sim_results.get('valid_row_indices', [])
        
        if valid_indices:
            # Filter valid indices to only those that exist in current dataframe
            valid_indices = [idx for idx in valid_indices if idx in df.index]
            if not valid_indices:
                st.warning("⚠️ Simulation results are outdated. Please run simulation again with current data.")
                return
            
            # Extract only the matches used in simulation
            df = df.loc[valid_indices].copy()
            
            # Add simulation-specific columns for display
            df['Sim Odds'] = [sim_results['valid_odds_list'][i] for i in range(len(valid_indices))]
            df['Sim Stake'] = [round(stake, 1) for stake in sim_results['stake_history']]
            df['Sim Outcome'] = ['Win' if sim_results['outcomes_list'][i] else 'Loss' for i in range(len(valid_indices))]
            df['Sim PnL'] = sim_results['actual_pnl_per_bet']
            
            # Show simulation info
            betting_method = sim_results['betting_method']
            betting_config = sim_results['betting_config']
            
            # Format stake display based on staking method
            if betting_config['method'] == 'Fixed Amount':
                stake_display = f"Stake: ${betting_config['stake']:.2f}"
            else:
                stake_display = f"Stake: {betting_config['stake']:.1f}%"
            
            st.success(f"🎯 Showing {len(df)} matches from **{betting_method}** simulation ({stake_display})")
            
            # Add simulation summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Valid Bets", sim_results['valid_bets'])
            with col2:
                st.metric("Win Rate", f"{sim_results['win_rate']:.1%}")
            with col3:
                st.metric("Total PnL", f"${sim_results['total_pnl']:.2f}")
            with col4:
                st.metric("ROI", f"{sim_results['roi']:.1%}")
            
            st.markdown("---")
        else:
            st.warning("⚠️ No valid matches found in last simulation")
            df = pd.DataFrame()
    else:
        st.info("ℹ️ Run a simulation in the Strategy Backtesting tab to see simulation matches here")
        df = pd.DataFrame()
    
    if df.empty:
        st.info("No matches to display")
        return
    
    # Get current betting method for outcome validation
    betting_method = st.session_state.filters.get('betting_method', 'BTTS at Fulltime')
    
    # Betting method config for outcome validation
    method_config = {
        "BTTS at Fulltime": {
            'outcome_func': lambda df: (df['home_ft'] > 0) & (df['away_ft'] > 0),
            'label': 'BTTS'
        },
        "Home Win": {
            'outcome_func': lambda df: df['home_ft'] > df['away_ft'],
            'label': 'Home Win'
        },
        "Draw": {
            'outcome_func': lambda df: df['home_ft'] == df['away_ft'],
            'label': 'Draw'
        },
        "Away Win": {
            'outcome_func': lambda df: df['home_ft'] < df['away_ft'],
            'label': 'Away Win'
        },
        "Over 1.5 Goals": {
            'outcome_func': lambda df: df['total_ft'] > 1.5,
            'label': 'Over 1.5'
        },
        "Over 2.5 Goals": {
            'outcome_func': lambda df: df['total_ft'] > 2.5,
            'label': 'Over 2.5'
        },
        "Over 3.5 Goals": {
            'outcome_func': lambda df: df['total_ft'] > 3.5,
            'label': 'Over 3.5'
        },
        "Over 0.5 Goals HT": {
            'outcome_func': lambda df: df['total_fh'] > 0.5,
            'label': 'Over 0.5 HT'
        }
    }
    
    # Show current betting method being validated
    st.info(f"🎯 Validating outcomes for: **{betting_method}**")
    
    # Search and filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_team = st.text_input("Search Team", placeholder="Team name...")
    
    with col2:
        sort_by = st.selectbox(
            "Sort By",
            options=df.columns.tolist(),
            key="sort_by"
        )
    
    with col3:
        sort_order = st.selectbox(
            "Order",
            options=["Descending", "Ascending"],
            key="sort_order"
        )
    
    # Apply additional filters
    filtered_df = df.copy()
    
    if search_team:
        filtered_df = filtered_df[
            filtered_df['team_a'].str.contains(search_team, case=False, na=False) |
            filtered_df['team_b'].str.contains(search_team, case=False, na=False)
        ]
    
    # Sort
    ascending = sort_order == "Ascending"
    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
    
    # Add outcome validation column
    config = method_config.get(betting_method)
    if config and all(col in filtered_df.columns for col in ['home_ft', 'away_ft']):
        try:
            # Calculate outcomes
            outcomes = config['outcome_func'](filtered_df)
            
            # Handle cases where scores are missing/NULL
            outcome_status = []
            for idx, outcome in enumerate(outcomes):
                if pd.isna(filtered_df.iloc[idx]['home_ft']) or pd.isna(filtered_df.iloc[idx]['away_ft']):
                    outcome_status.append("⏳ Pending")
                elif outcome:
                    outcome_status.append("✅ Correct")
                else:
                    outcome_status.append("❌ Incorrect")
            
            filtered_df['Outcome'] = outcome_status
            
            # Add summary statistics
            correct_count = outcome_status.count("✅ Correct")
            incorrect_count = outcome_status.count("❌ Incorrect")
            pending_count = outcome_status.count("⏳ Pending")
            
            st.subheader("📊 Outcome Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ Correct", correct_count)
            with col2:
                st.metric("❌ Incorrect", incorrect_count)
            with col3:
                st.metric("⏳ Pending", pending_count)
            with col4:
                if correct_count + incorrect_count > 0:
                    accuracy = (correct_count / (correct_count + incorrect_count)) * 100
                    st.metric("📈 Accuracy", f"{accuracy:.1f}%")
                else:
                    st.metric("📈 Accuracy", "N/A")
            
        except Exception as e:
            st.warning(f"Could not calculate outcomes: {e}")
            filtered_df['Outcome'] = "❓ Unknown"
    
    # Display summary
    st.info(f"Showing {len(filtered_df)} of {len(df)} matches")
    
    # Data table with column selection
    if not filtered_df.empty:
        # Integrate simulation results if available
        simulation_results = st.session_state.get('last_simulation_results')
        if simulation_results and simulation_results.get('valid_row_indices'):
            st.info(f"🎯 Showing simulation results for {simulation_results['betting_method']} with {simulation_results['valid_bets']} valid bets")
            
            # Create simulation data mapping
            sim_data = {}
            for i, row_idx in enumerate(simulation_results['valid_row_indices']):
                sim_data[row_idx] = {
                    'Bet #': i + 1,
                    'Sim Stake': round(simulation_results['stake_history'][i], 1),
                    'Sim Outcome': 'Win' if simulation_results['outcomes_list'][i] else 'Loss',
                    'Sim PnL': simulation_results['actual_pnl_per_bet'][i],
                    'Sim Odds': simulation_results['valid_odds_list'][i]
                }
            
            # Add simulation columns to filtered_df
            filtered_df['Bet #'] = filtered_df.index.map(lambda x: sim_data.get(x, {}).get('Bet #', ''))
            filtered_df['Sim Stake'] = filtered_df.index.map(lambda x: sim_data.get(x, {}).get('Sim Stake', float('nan')))
            filtered_df['Sim Outcome'] = filtered_df.index.map(lambda x: sim_data.get(x, {}).get('Sim Outcome', ''))
            filtered_df['Sim PnL'] = filtered_df.index.map(lambda x: sim_data.get(x, {}).get('Sim PnL', float('nan')))
            filtered_df['Sim Odds'] = filtered_df.index.map(lambda x: sim_data.get(x, {}).get('Sim Odds', float('nan')))
        
        # Column selection expander
        with st.expander("🔧 Column Selection"):
            all_columns = filtered_df.columns.tolist()
            
            # Build default_selected list only with columns that actually exist
            base_columns = ['start_time', 'country', 'competition', 'team_a', 'team_b', 'home_ft', 'away_ft', 'Outcome']
            # Add simulation columns if available
            if simulation_results and simulation_results.get('valid_row_indices'):
                base_columns.extend(['Bet #', 'Sim Stake', 'Sim Outcome', 'Sim PnL', 'Sim Odds'])
            else:
                # If no simulation data, ensure Outcome is prominently displayed
                if 'Outcome' in all_columns:
                    base_columns = ['Outcome', 'start_time', 'country', 'competition', 'team_a', 'team_b', 'home_ft', 'away_ft']
            
            odds_columns = [col for col in all_columns if 'odds_' in col][:5]
            default_selected = [col for col in base_columns if col in all_columns] + odds_columns
            
            selected_columns = st.multiselect(
                "Select columns to display:",
                options=all_columns,
                default=default_selected,
                key="column_selection"
            )
        
        # Use selected columns or default if none selected
        if selected_columns:
            display_df = filtered_df[selected_columns].copy()
        else:
            # Show first 15 columns by default if nothing selected
            display_df = filtered_df[all_columns[:15]].copy()
        
        # Format datetime
        if 'start_time' in display_df.columns:
            display_df['start_time'] = display_df['start_time'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Format numeric columns (potentials, odds, scores)
        numeric_cols = display_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col not in ['home_ft', 'away_ft', 'total_ft', 'total_fh']:  # Don't round scores
                display_df[col] = display_df[col].round(3)
        
        # Enhanced formatting for simulation outcome column
        if 'Sim Outcome' in display_df.columns:
            # Add emoji indicators for outcomes
            display_df['Sim Outcome'] = display_df['Sim Outcome'].apply(
                lambda x: '✅ Win' if x == 'Win' else '❌ Loss' if x == 'Loss' else x
            )
        
        st.dataframe(display_df, width='stretch')
        
        # Export functionality - browser download
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"match_browser_filtered_{timestamp}.csv"
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="download_match_browser_csv"
            )
        except Exception as e:
            st.error(f"Error preparing export data: {str(e)}")
    else:
        st.warning("No matches found with current filters")

def reset_filters_to_defaults(betting_method):
    """Reset all filters to realistic defaults based on betting method"""
    # Clear only betting simulation filter keys from session state
    betting_filter_keys = [
        'ppg_a_min', 'ppg_a_max', 'ppg_b_min', 'ppg_b_max', 'ppg_diff_min', 'ppg_diff_max',
        'team_a_xg_min', 'team_a_xg_max', 'team_b_xg_min', 'team_b_xg_max', 'xg_diff_min', 'xg_diff_max',
        'avg_potential_min', 'avg_potential_max',
        'btts_potential_min', 'btts_potential_max',
        'btts_fhg_min', 'btts_fhg_max', 'btts_2hg_min', 'btts_2hg_max',
        'o15_potential_min', 'o15_potential_max',
        'o25_potential_min', 'o25_potential_max',
        'o35_potential_min', 'o35_potential_max',
        'o05ht_potential_min', 'o05ht_potential_max',
        # Odds filters
        'footystats_odds_btts_min', 'footystats_odds_btts_max',
        'footystats_odds_1_min', 'footystats_odds_1_max',
        'footystats_odds_x_min', 'footystats_odds_x_max',
        'footystats_odds_2_min', 'footystats_odds_2_max',
        'footystats_odds_over15_min', 'footystats_odds_over15_max',
        'footystats_odds_over25_min', 'footystats_odds_over25_max',
        'footystats_odds_over35_min', 'footystats_odds_over35_max',
        'footystats_odds_over05_min', 'footystats_odds_over05_max'
    ]
    
    for key in betting_filter_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear filter change detection to trigger auto-run after reset
    st.session_state.last_filters_str = ''
    
    # Set realistic defaults based on method
    defaults = {
        'ppg_a_min': 0.0, 'ppg_a_max': 2.5,
        'ppg_b_min': 0.0, 'ppg_b_max': 2.5,
        'ppg_diff_min': -3.0, 'ppg_diff_max': 3.0,
        'team_a_xg_min': 0.0, 'team_a_xg_max': 3.0,
        'team_b_xg_min': 0.0, 'team_b_xg_max': 3.0,
        'xg_diff_min': -5.0, 'xg_diff_max': 5.0,
        'avg_potential_min': 0.0, 'avg_potential_max': 10.0,
        'btts_potential_min': 0.0, 'btts_potential_max': 1.0,
        'btts_fhg_min': 0.0, 'btts_fhg_max': 1.0,
        'btts_2hg_min': 0.0, 'btts_2hg_max': 1.0,
        'o15_potential_min': 0.0, 'o15_potential_max': 1.0,
        'o25_potential_min': 0.0, 'o25_potential_max': 1.0,
        'o35_potential_min': 0.0, 'o35_potential_max': 1.0,
        'o05ht_potential_min': 0.0, 'o05ht_potential_max': 1.0,
    }
    # if betting_method == "BTTS at Fulltime":
    #     defaults = {
    #         'ppg_a_min': 0, 'ppg_a_max': 2.5,
    #         'ppg_b_min': 0, 'ppg_b_max': 2.5,
    #         'team_a_xg_min': 0, 'team_a_xg_max': 3.0,
    #         'team_b_xg_min': 0, 'team_b_xg_max': 3.0,
    #         'avg_potential_min': 0, 'avg_potential_max': 3.0,
    #         'btts_potential_min': 0, 'btts_potential_max': 3.0,
    #         'btts_fhg_min': 0, 'btts_fhg_max': 1.0,
    #         'btts_2hg_min': 0, 'btts_2hg_max': 1.0,
    #         'o15_potential_min': 0, 'o15_potential_max': 1.0,
    #         'o25_potential_min': 0, 'o25_potential_max': 1.0,
    #         'o35_potential_min': 0, 'o35_potential_max': 1.0,
    #         'o05ht_potential_min': 0, 'o05ht_potential_max': 1.0,
    #     }
    # elif betting_method == "Home Win":
    #     defaults = {
    #         'ppg_a_min': 0, 'ppg_a_max': 2.5,
    #         'ppg_b_min': 0, 'ppg_b_max': 2.5,
    #         'team_a_xg_min': 0, 'team_a_xg_max': 3.0,
    #         'team_b_xg_min': 0, 'team_b_xg_max': 3.0,
    #         'avg_potential_min': 0, 'avg_potential_max': 3.0,
    #         'btts_potential_min': 0, 'btts_potential_max': 3.0,
    #         'o15_potential_min': 0.3, 'o15_potential_max': 2.0,
    #         'o25_potential_min': 0.3, 'o25_potential_max': 2.0,
    #         # 'footystats_odds_1_min': 1.3, 'footystats_odds_1_max': 2.5
    #     }
    # elif betting_method == "Draw":
    #     defaults = {
    #         'ppg_a_min': 0, 'ppg_a_max': 2.5,
    #         'ppg_b_min': 0, 'ppg_b_max': 2.5,
    #         'team_a_xg_min': 0.5, 'team_a_xg_max': 3.0,
    #         'team_b_xg_min': 0.5, 'team_b_xg_max': 3.0,
    #         'avg_potential_min': 0.5, 'avg_potential_max': 3.0,
    #         'btts_potential_min': 0.5, 'btts_potential_max': 3.0,
    #         'o15_potential_min': 0.3, 'o15_potential_max': 2.0,
    #         'o25_potential_min': 0.3, 'o25_potential_max': 2.0,
    #         # 'footystats_odds_x_min': 2.5, 'footystats_odds_x_max': 4.0
    #     }
    # elif betting_method == "Away Win":
    #     defaults = {
    #         'ppg_a_min': 0.5, 'ppg_a_max': 2.5,
    #         'ppg_b_min': 0.5, 'ppg_b_max': 2.5,
    #         'team_a_xg_min': 0.5, 'team_a_xg_max': 3.0,
    #         'team_b_xg_min': 0.5, 'team_b_xg_max': 3.0,
    #         'avg_potential_min': 0.5, 'avg_potential_max': 3.0,
    #         'btts_potential_min': 0.5, 'btts_potential_max': 3.0,
    #         'o15_potential_min': 0.3, 'o15_potential_max': 2.0,
    #         'o25_potential_min': 0.3, 'o25_potential_max': 2.0,
    #         # 'footystats_odds_2_min': 2.0, 'footystats_odds_2_max': 5.0
    #     }
    # elif betting_method == "Over 1.5 Goals":
    #     defaults = {
    #         'ppg_a_min': 0.5, 'ppg_a_max': 2.5,
    #         'ppg_b_min': 0.5, 'ppg_b_max': 2.5,
    #         'team_a_xg_min': 0.5, 'team_a_xg_max': 3.0,
    #         'team_b_xg_min': 0.5, 'team_b_xg_max': 3.0,
    #         'avg_potential_min': 0.5, 'avg_potential_max': 3.0,
    #         'o15_potential_min': 0.3, 'o15_potential_max': 2.0,
    #         'o25_potential_min': 0.3, 'o25_potential_max': 2.0,
    #         'o35_potential_min': 0.2, 'o35_potential_max': 1.5,
    #         'footystats_odds_over15_min': 1.2, 'footystats_odds_over15_max': 2.0
    #     }
    # elif betting_method == "Over 2.5 Goals":
    #     defaults = {
    #         'ppg_a_min': 0.5, 'ppg_a_max': 2.5,
    #         'ppg_b_min': 0.5, 'ppg_b_max': 2.5,
    #         'team_a_xg_min': 0.5, 'team_a_xg_max': 3.0,
    #         'team_b_xg_min': 0.5, 'team_b_xg_max': 3.0,
    #         'avg_potential_min': 0.5, 'avg_potential_max': 3.0,
    #         'o15_potential_min': 0.3, 'o15_potential_max': 2.0,
    #         'o25_potential_min': 0.3, 'o25_potential_max': 2.0,
    #         'o35_potential_min': 0.2, 'o35_potential_max': 1.5,
    #         'footystats_odds_over25_min': 1.4, 'footystats_odds_over25_max': 2.5
    #     }
    # elif betting_method == "Over 3.5 Goals":
    #     defaults = {
    #         'ppg_a_min': 0.5, 'ppg_a_max': 2.5,
    #         'ppg_b_min': 0.5, 'ppg_b_max': 2.5,
    #         'team_a_xg_min': 0.5, 'team_a_xg_max': 3.0,
    #         'team_b_xg_min': 0.5, 'team_b_xg_max': 3.0,
    #         'avg_potential_min': 0.5, 'avg_potential_max': 3.0,
    #         'o15_potential_min': 0.3, 'o15_potential_max': 2.0,
    #         'o25_potential_min': 0.3, 'o25_potential_max': 2.0,
    #         'o35_potential_min': 0.2, 'o35_potential_max': 1.5,
    #         'footystats_odds_over35_min': 1.8, 'footystats_odds_over35_max': 3.5
    #     }
    # elif betting_method == "Over 0.5 Goals HT":
    #     defaults = {
    #         'ppg_a_min': 0.5, 'ppg_a_max': 2.5,
    #         'ppg_b_min': 0.5, 'ppg_b_max': 2.5,
    #         'team_a_xg_min': 0.5, 'team_a_xg_max': 3.0,
    #         'team_b_xg_min': 0.5, 'team_b_xg_max': 3.0,
    #         'avg_potential_min': 0.5, 'avg_potential_max': 3.0,
    #         'o05ht_potential_min': 0.3, 'o05ht_potential_max': 2.0,
    #         'btts_fhg_min': 0.3, 'btts_fhg_max': 2.0,
    #         'footystats_odds_over05_min': 1.1, 'footystats_odds_over05_max': 1.8
    #     }
    
    # Update session state with defaults
    for key, value in defaults.items():
        st.session_state[key] = value

def render_strategy_backtesting_tab(df):
    """Render comprehensive betting simulation interface"""
    st.header(f"🎯 Simulation: {st.session_state.filters.get('betting_method', 'BTTS at Fulltime')}")
    
    # Get betting method and stake from filters
    betting_method = st.session_state.filters.get('betting_method', 'BTTS at Fulltime')
    bet_stake = st.session_state.filters.get('bet_stake', 10.0)
    betting_config = st.session_state.betting_config
    
    # Display current filter summary
    st.subheader("📊 Current Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Betting Method", betting_method)
        # Show correct stake value based on staking method
        if betting_config['method'] == 'Fixed Amount':
            st.metric("Bet Stake", f"${bet_stake:.2f}")
        else:
            # Calculate initial stake for percentage staking
            initial_stake = betting_config['base_capital'] * (bet_stake / 100.0)
            st.metric("Bet Stake", f"{bet_stake:.1f}% (${initial_stake:.2f})")
    
    with col2:
        # Show active range filters
        active_filters = []
        filter_labels = {
            'ppg_a_min': 'Home PPG', 'ppg_a_max': 'Home PPG',
            'ppg_b_min': 'Away PPG', 'ppg_b_max': 'Away PPG',
            'ppg_diff_min': 'PPG Difference', 'ppg_diff_max': 'PPG Difference',
            'team_a_xg_min': 'Home XG', 'team_a_xg_max': 'Home XG',
            'team_b_xg_min': 'Away XG', 'team_b_xg_max': 'Away XG',
            'xg_diff_min': 'XG Difference', 'xg_diff_max': 'XG Difference',
            'avg_potential_min': 'Avg Potential', 'avg_potential_max': 'Avg Potential',
            'btts_potential_min': 'BTTS Potential', 'btts_potential_max': 'BTTS Potential',
            'o15_potential_min': 'Over 1.5 Potential', 'o15_potential_max': 'Over 1.5 Potential',
            'o25_potential_min': 'Over 2.5 Potential', 'o25_potential_max': 'Over 2.5 Potential',
            'o35_potential_min': 'Over 3.5 Potential', 'o35_potential_max': 'Over 3.5 Potential',
            'o05ht_potential_min': 'Over 0.5 HT Potential', 'o05ht_potential_max': 'Over 0.5 HT Potential',
            'btts_fhg_min': 'BTTS 1H Potential', 'btts_fhg_max': 'BTTS 1H Potential',
            'btts_2hg_min': 'BTTS 2H Potential', 'btts_2hg_max': 'BTTS 2H Potential'
        }
        
        for filter_key, filter_value in st.session_state.filters.items():
            if filter_key.endswith(('_min', '_max')) and filter_value is not None:
                label = filter_labels.get(filter_key, filter_key)
                if filter_key not in active_filters:
                    if f"{filter_key[:-4]}_min" in st.session_state.filters and f"{filter_key[:-4]}_max" in st.session_state:
                        min_val = st.session_state.filters.get(f"{filter_key[:-4]}_min")
                        max_val = st.session_state.filters.get(f"{filter_key[:-4]}_max")
                        if min_val is not None and max_val is not None:
                            active_filters.append(f"{label}: {min_val:.2f}-{max_val:.2f}")
        
        st.write("**Active Range Filters:**")
        for filter_desc in active_filters[:5]:  # Show first 5 filters
            st.write(f"• {filter_desc}")
        if len(active_filters) > 5:
            st.write(f"• ... and {len(active_filters) - 5} more")
    
    with col3:
        st.metric("Total Matches", len(df))
        matches_with_odds = len(df[df['odds_btts_yes'].notna()]) if 'odds_btts_yes' in df.columns else 0
        st.metric("Matches with Odds", matches_with_odds)
    
    # Auto-display existing simulation results if available
    simulation_results = st.session_state.get('last_simulation_results')
    betting_method_from_results = st.session_state.get('last_simulation_betting_method')
    
    # Display results if we have recent results that match current method
    if simulation_results and betting_method_from_results == betting_method:
        st.info("📊 Showing simulation results...")
        display_simulation_results(simulation_results, betting_method, st.session_state.get('last_simulation_betting_config', {}))
    elif not simulation_results:
        st.info("👈 Click '🚀 Simulate' in sidebar to run simulation")

def run_betting_simulation(df, betting_method, betting_config):
    """Run enhanced betting simulation with percentage staking and martingale"""
    
    # Filter to only consider finished matches
    if 'match_status' in df.columns:
        original_count = len(df)
        df = df[df['match_status'] == 'finished']
        filtered_count = len(df)
        if filtered_count < original_count:
            st.info(f"📊 Filtered to {filtered_count} finished matches from {original_count} total matches")
    else:
        st.warning("⚠️ Match status column not found - including all matches")
    
    # Map betting method to outcome calculation and odds column
    method_config = {
        "BTTS at Fulltime": {
            'outcome_func': lambda df: (df['home_ft'] > 0) & (df['away_ft'] > 0),
            'odds_col': 'odds_btts_yes',
            'label': 'BTTS'
        },
        "Home Win": {
            'outcome_func': lambda df: df['home_ft'] > df['away_ft'],
            'odds_col': 'odds_ft_1',
            'label': 'Home Win'
        },
        "Draw": {
            'outcome_func': lambda df: df['home_ft'] == df['away_ft'],
            'odds_col': 'odds_ft_x',
            'label': 'Draw'
        },
        "Away Win": {
            'outcome_func': lambda df: df['home_ft'] < df['away_ft'],
            'odds_col': 'odds_ft_2',
            'label': 'Away Win'
        },
        "Over 1.5 Goals": {
            'outcome_func': lambda df: df['total_ft'] > 1.5,
            'odds_col': 'odds_ft_over15',
            'label': 'Over 1.5'
        },
        "Over 2.5 Goals": {
            'outcome_func': lambda df: df['total_ft'] > 2.5,
            'odds_col': 'odds_ft_over25',
            'label': 'Over 2.5'
        },
        "Over 3.5 Goals": {
            'outcome_func': lambda df: df['total_ft'] > 3.5,
            'odds_col': 'odds_ft_over35',
            'label': 'Over 3.5'
        },
        "Over 0.5 Goals HT": {
            'outcome_func': lambda df: df['total_fh'] > 0.5,
            'odds_col': 'odds_ft_over05',
            'label': 'Over 0.5 HT'
        }
    }
    
    config = method_config.get(betting_method)
    if not config:
        st.error(f"Betting method '{betting_method}' not configured")
        return None
    
    # Calculate outcomes
    outcomes = config['outcome_func'](df)
    odds_col = config['odds_col']
    
    # Check if odds column exists
    if odds_col not in df.columns:
        st.error(f"Odds column '{odds_col}' not found in data")
        return None
    
    # Filter valid bets (with odds and outcomes)
    valid_mask = df[odds_col].notna() & outcomes.notna()
    valid_outcomes = outcomes[valid_mask]
    valid_odds = df[odds_col][valid_mask]
    
    # Track original row indices for valid bets
    valid_row_indices = df.index[valid_mask].tolist()
    
    if valid_outcomes.empty:
        st.warning("No matches with valid odds found for current filters")
        # Clear simulation results from session state to prevent display in other tabs
        st.session_state.last_simulation_results = None
        st.session_state.last_simulation_betting_method = None
        st.session_state.last_simulation_betting_config = None
        return None
    
    # Initialize bankroll tracking
    current_capital = betting_config['base_capital']
    base_capital = betting_config['base_capital']
    bankroll_history = [current_capital]
    stake_history = []
    consecutive_losses = 0
    max_consecutive_losses = 0
    busted = False
    
    # Helper function to calculate stake
    def calculate_stake(capital, config, consecutive_losses):
        if config['method'] == 'Fixed Amount':
            base_stake = config['stake']
        else:
            base_stake = capital * (config['stake'] / 100.0)
        
        # Apply martingale if enabled
        if config['use_martingale'] and consecutive_losses > 0:
            martingale_stake = base_stake * (2 ** consecutive_losses)
            max_allowed = capital * (config['max_bet_percentage'] / 100.0)
            return min(martingale_stake, max_allowed)
        
        return base_stake
    
    # Process each bet
    actual_pnl_per_bet = []
    actual_stakes = []
    
    for i, (outcome, odds) in enumerate(zip(valid_outcomes, valid_odds)):
        # Check if busted (90% loss - capital <= 10% of base)
        if current_capital <= (base_capital * 0.1):
            busted = True
            break
        
        # Calculate stake for this bet
        current_stake = calculate_stake(current_capital, betting_config, consecutive_losses)
        
        # Ensure stake doesn't exceed available capital
        current_stake = min(current_stake, current_capital)
        
        # Record stake
        actual_stakes.append(current_stake)
        stake_history.append(current_stake)
        
        # Calculate PnL
        if outcome:  # Win
            pnl = current_stake * (odds - 1)
            consecutive_losses = 0  # Reset martingale
        else:  # Loss
            pnl = -current_stake
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        actual_pnl_per_bet.append(pnl)
        current_capital += pnl
        bankroll_history.append(current_capital)
    
    # Calculate final metrics
    total_bets = len(actual_pnl_per_bet)
    wins = sum(1 for outcome in valid_outcomes[:total_bets] if outcome)
    losses = total_bets - wins
    total_pnl = sum(actual_pnl_per_bet)
    win_rate = wins / total_bets if total_bets > 0 else 0
    
    # Calculate ROI based on base capital
    roi = total_pnl / base_capital if base_capital > 0 else 0
    
    # Additional metrics
    avg_odds = valid_odds[:total_bets].mean() if total_bets > 0 else 0
    avg_winning_odds = valid_odds[:total_bets][valid_outcomes[:total_bets]].mean() if wins > 0 else 0
    max_odds = valid_odds[:total_bets].max() if total_bets > 0 else 0
    min_odds = valid_odds[:total_bets].min() if total_bets > 0 else 0
    
    # Bankroll metrics
    peak_bankroll = max(bankroll_history)
    lowest_bankroll = min(bankroll_history)
    max_drawdown = (peak_bankroll - lowest_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
    final_bankroll = current_capital
    
    # Calculate profit distribution and streaks
    profit_distribution = []
    winning_streaks = []
    losing_streaks = []
    current_streak = 0
    current_type = None
    
    for i, outcome in enumerate(valid_outcomes[:total_bets]):
        if current_type is None:
            current_type = outcome
            current_streak = 1
        elif outcome == current_type:
            current_streak += 1
        else:
            # Store completed streak
            if current_type:  # Winning streak
                winning_streaks.append(current_streak)
            else:  # Losing streak
                losing_streaks.append(current_streak)
            # Start new streak
            current_type = outcome
            current_streak = 1
    
    # Store final streak
    if current_type:  # Winning streak
        winning_streaks.append(current_streak)
    else:  # Losing streak
        losing_streaks.append(current_streak)
    
    max_winning_streak = max(winning_streaks) if winning_streaks else 0
    max_losing_streak = max(losing_streaks) if losing_streaks else 0
    avg_winning_streak = sum(winning_streaks) / len(winning_streaks) if winning_streaks else 0
    avg_losing_streak = sum(losing_streaks) / len(losing_streaks) if losing_streaks else 0
    
    # Calculate additional metrics for compatibility
    total_stake = sum(stake_history)
    total_return = total_stake + total_pnl
    
    return {
        'betting_method': betting_method,
        'total_matches': len(df),
        'valid_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'base_capital': base_capital,
        'final_bankroll': final_bankroll,
        'roi': roi,
        'avg_odds': avg_odds,
        'avg_winning_odds': avg_winning_odds,
        'max_odds': max_odds,
        'min_odds': min_odds,
        'profit_distribution': profit_distribution,
        'actual_pnl_per_bet': actual_pnl_per_bet,
        'valid_odds_list': valid_odds[:total_bets].tolist(),
        'outcomes_list': valid_outcomes[:total_bets].tolist(),
        'valid_row_indices': valid_row_indices[:total_bets],
        'winning_streaks': winning_streaks,
        'losing_streaks': losing_streaks,
        'max_winning_streak': max_winning_streak,
        'max_losing_streak': max_losing_streak,
        'avg_winning_streak': avg_winning_streak,
        'avg_losing_streak': avg_losing_streak,
        'bankroll_history': bankroll_history,
        'stake_history': stake_history,
        'peak_bankroll': peak_bankroll,
        'lowest_bankroll': lowest_bankroll,
        'max_drawdown': max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'busted': busted,
        'betting_config': betting_config,
        # Additional metrics for display compatibility
        'total_stake': total_stake,
        'total_return': total_return
    }

def display_simulation_results(results, betting_method, betting_config):
    """Display comprehensive simulation results with enhanced betting features"""
    
    st.subheader(f"📈 {betting_method} Simulation Results")
    
    # Show bust warning if applicable
    if results.get('busted', False):
        st.error("💥 **BANKROLL BUSTED!** Simulation stopped when bankroll reached zero.")
    
    # Enhanced metrics with bankroll analysis
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Valid Bets", results['valid_bets'])
        st.metric("Win Rate", f"{results['win_rate']:.1%}")
    
    with col2:
        pnl_color = "normal" if results['total_pnl'] >= 0 else "inverse"
        st.metric("Total PnL", f"${results['total_pnl']:.2f}", delta=f"{results['roi']:.1%} ROI", delta_color=pnl_color)
        st.metric("Final Bankroll", f"${results['final_bankroll']:.2f}")
    
    with col3:
        st.metric("Wins", results['wins'])
        st.metric("Losses", results['losses'])
    
    with col4:
        st.metric("Peak Bankroll", f"${results['peak_bankroll']:.2f}")
        st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
    
    with col5:
        st.metric("Max Win Streak", results['max_winning_streak'])
        st.metric("Max Consec. Losses", results.get('max_consecutive_losses', 0))
    
    # Betting configuration summary
    st.markdown("### 💰 Betting Configuration")
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.info(f"**Method:** {betting_config['method']}")
        if betting_config['method'] == 'Fixed Amount':
            st.info(f"**Stake:** ${betting_config['stake']:.2f}")
        else:
            st.info(f"**Stake:** {betting_config['stake']:.1f}% of capital")
    
    with config_col2:
        st.info(f"**Base Capital:** ${betting_config['base_capital']:.2f}")
        if betting_config['use_martingale']:
            st.info(f"**Martingale:** Enabled (max {betting_config['max_bet_percentage']:.0f}% of capital)")
        else:
            st.info(f"**Martingale:** Disabled")
    
    with config_col3:
        if results.get('busted', False):
            st.error("🚨 **Status:** BUSTED")
        else:
            profit_amount = results['final_bankroll'] - betting_config['base_capital']
            profit_pct = (profit_amount / betting_config['base_capital']) * 100
            if profit_amount >= 0:
                st.success(f"📈 **Status:** +${profit_amount:.2f} ({profit_pct:+.1f}%)")
            else:
                st.warning(f"📉 **Status:** ${profit_amount:.2f} ({profit_pct:+.1f}%)")
    
    # Bankroll Evolution Chart
    st.markdown("### 📊 Bankroll Evolution")
    bankroll_history = results.get('bankroll_history', [])
    if bankroll_history:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(bankroll_history))),
            y=bankroll_history,
            mode='lines+markers',
            name='Bankroll',
            line=dict(color='blue', width=2)
        ))
        
        # Add reference lines
        fig.add_hline(y=betting_config['base_capital'], line_dash="dash", line_color="green", 
                     annotation_text="Starting Capital")
        if results.get('busted', False):
            fig.add_hline(y=base_capital * 0.1, line_dash="dash", line_color="red", annotation_text="Bust Level")
        
        fig.update_layout(
            title="Bankroll Evolution Throughout Simulation",
            xaxis_title="Bet Number",
            yaxis_title="Bankroll ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
    
    # Enhanced PnL Analysis
    st.subheader("💰 Enhanced Profit & Loss Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cumulative PnL chart
        cumulative_pnl = []
        running_total = 0
        for pnl in results['actual_pnl_per_bet']:
            running_total += pnl
            cumulative_pnl.append(running_total)
        
        fig = px.line(
            x=list(range(1, len(cumulative_pnl) + 1)),
            y=cumulative_pnl,
            title="📈 Cumulative PnL Progression",
            labels={'x': 'Bet Number', 'y': 'Cumulative PnL ($)'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Win/Loss distribution
        outcomes_data = ['Win' if outcome else 'Loss' for outcome in results['outcomes_list']]
        outcome_counts = pd.Series(outcomes_data).value_counts()
        
        fig = px.pie(
            values=outcome_counts.values,
            names=outcome_counts.index,
            title="🎯 Win/Loss Distribution",
            color_discrete_map={'Win': '#2E8B57', 'Loss': '#DC143C'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, width='stretch')
    
    # Detailed PnL Distribution
    st.subheader("📊 Detailed PnL Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PnL breakdown (existing)
        pnl_data = {
            'Wins': results['wins'] * bet_stake * (results['avg_winning_odds'] - 1) if results['wins'] > 0 else 0,
            'Losses': -results['losses'] * bet_stake
        }
        
        fig = px.bar(
            x=list(pnl_data.keys()),
            y=list(pnl_data.values()),
            title="PnL Breakdown",
            labels={'x': 'Result Type', 'y': 'PnL ($)'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Individual bet PnL distribution
        wins_pnl = [pnl for pnl in results['actual_pnl_per_bet'] if pnl > 0]
        losses_pnl = [pnl for pnl in results['actual_pnl_per_bet'] if pnl < 0]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=wins_pnl,
            name='Profits',
            nbinsx=20,
            marker_color='#2E8B57'
        ))
        fig.add_trace(go.Histogram(
            x=losses_pnl,
            name='Losses',
            nbinsx=20,
            marker_color='#DC143C'
        ))
        
        fig.update_layout(
            title="Individual Bet PnL Distribution",
            xaxis_title="PnL ($)",
            yaxis_title="Count",
            barmode='overlay'
        )
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig, width='stretch')
    
    # Detailed betting analysis
    st.subheader("🔍 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Betting Performance:**
        - Total Stake: ${results['total_stake']:.2f}
        - Total Return: ${results['total_stake'] + results['total_pnl']:.2f}
        - Net Profit: ${results['total_pnl']:.2f}
        - ROI: {results['roi']:.1%}
        - Win Rate: {results['win_rate']:.1%}
        
        **Odds Analysis:**
        - Average Odds: {results['avg_odds']:.2f}
        - Winning Odds: {results['avg_winning_odds']:.2f}
        - Min Odds: {results['min_odds']:.2f}
        - Max Odds: {results['max_odds']:.2f}
        """)
    
    with col2:
        # Profit by odds ranges
        odds_ranges = [
            (1.0, 1.5, "Low (1.0-1.5)"),
            (1.5, 2.0, "Medium (1.5-2.0)"),
            (2.0, 3.0, "High (2.0-3.0)"),
            (3.0, 10.0, "Very High (3.0+)")
        ]
        
        range_analysis = []
        for min_odds, max_odds, label in odds_ranges:
            range_mask = [(odds >= min_odds) & (odds < max_odds) for odds in results['valid_odds_list']]
            range_outcomes = [results['outcomes_list'][i] for i, mask in enumerate(range_mask) if mask]
            range_odds = [results['valid_odds_list'][i] for i, mask in enumerate(range_mask) if mask]
            
            if range_outcomes:
                range_wins = sum(range_outcomes)
                range_bets = len(range_outcomes)
                range_win_rate = range_wins / range_bets
                range_pnl = sum([bet_stake * (odds - 1) if win else -bet_stake for odds, win in zip(range_odds, range_outcomes)])
                
                range_analysis.append({
                    'Odds Range': label,
                    'Bets': range_bets,
                    'Win Rate': f"{range_win_rate:.1%}",
                    'PnL': f"${range_pnl:.2f}"
                })
        
        if range_analysis:
            range_df = pd.DataFrame(range_analysis)
            st.write("**Performance by Odds Range:**")
            st.dataframe(range_df, width='stretch')
    
    # Recommendations
    st.subheader("💡 Strategy Recommendations")
    
    if results['roi'] > 0:
        st.success(f"🎉 Profitable Strategy! ROI of {results['roi']:.1%} indicates this filter combination works well.")
        
        if results['win_rate'] > 0.5:
            st.info("✅ High win rate suggests consistent performance. Consider increasing stake size.")
        else:
            st.info("📈 Lower win rate but profitable due to good odds value. Focus on odds optimization.")
    else:
        st.warning(f"⚠️ Unprofitable Strategy: ROI of {results['roi']:.1%}. Consider adjusting filters.")
        
        if results['win_rate'] < 0.3:
            st.info("🔄 Low win rate suggests filters may be too restrictive. Try relaxing constraints.")
        else:
            st.info("💰 Decent win rate but poor odds value. Focus on finding better odds or stake management.")
    
    # Export functionality - prepare data and show download button directly
    try:
        # Create detailed results dataframe
        detailed_data = []
        for i, (odds, outcome) in enumerate(zip(results['valid_odds_list'], results['outcomes_list'])):
            pnl = bet_stake * (odds - 1) if outcome else -bet_stake
            detailed_data.append({
                'Bet #': i + 1,
                'Odds': odds,
                'Outcome': 'Win' if outcome else 'Loss',
                'PnL': pnl,
                'Return': bet_stake + pnl if outcome else 0
            })
        
        export_df = pd.DataFrame(detailed_data)
        
        # Add summary rows
        summary_row = {
            'Bet #': 'SUMMARY',
            'Odds': results['avg_odds'],
            'Outcome': f"{results['win_rate']:.1%} Win Rate",
            'PnL': results['total_pnl'],
            'Return': results['total_stake'] + results['total_pnl']
        }
        
        export_df = pd.concat([export_df, pd.DataFrame([summary_row])], ignore_index=True)
        
        # Convert to CSV for download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_{betting_method.replace(' ', '_').lower()}_{timestamp}.csv"
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        # Show download button directly in UI
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key="download_simulation_csv"
        )
        
    except Exception as e:
        st.error(f"Error preparing export data: {str(e)}")

def render_potential_analysis_tab(df):
    """Render detailed potential analysis with goal timing and HT/FT patterns"""
    st.header("📈 Advanced Potential Analysis")
    
    # Parse all scores for comprehensive analysis
    analysis_df = parse_all_scores(df.copy())
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Analysis Type",
        options=["Goal Timing Analysis", "HT/FT Patterns", "Comeback Analysis", "Team Performance", "Time Trends"],
        key="analysis_type"
    )
    
    if analysis_type == "Goal Timing Analysis":
        st.subheader("⏰ Goal Timing Distribution")
        
        if not analysis_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Goal distribution by half
                fig = px.histogram(
                    analysis_df,
                    x='total_fh',
                    nbins=10,
                    title="First Half Goals Distribution",
                    marginal="box"
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                fig = px.histogram(
                    analysis_df,
                    x='total_2h',
                    nbins=10,
                    title="Second Half Goals Distribution",
                    marginal="box"
                )
                st.plotly_chart(fig, width='stretch')
            
            # Goal timing ratio analysis
            st.subheader("Goal Timing Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Goals by half comparison
                avg_goals = analysis_df[['total_fh', 'total_2h']].mean()
                fig = px.bar(
                    x=['First Half', 'Second Half'],
                    y=[avg_goals['total_fh'], avg_goals['total_2h']],
                    title="Average Goals by Half"
                )
                fig.update_layout(yaxis_title="Average Goals")
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Goal timing ratio distribution
                fig = px.histogram(
                    analysis_df,
                    x='goals_1h_half',
                    nbins=20,
                    title="First Half Goal Ratio (1H Goals / Total Goals)",
                    marginal="box"
                )
                fig.update_layout(xaxis_title="Ratio", yaxis_title="Count")
                st.plotly_chart(fig, width='stretch')
            
            # Potential vs goal timing
            st.subheader("Potential vs Goal Timing")
            
            fig = px.scatter(
                analysis_df,
                x='avg_potential',
                y='goals_1h_half',
                color='total_ft',
                title="Average Potential vs First Half Goal Ratio",
                labels={
                    'avg_potential': 'Average Potential',
                    'goals_1h_half': 'First Half Goal Ratio',
                    'total_ft': 'Total Goals'
                }
            )
            st.plotly_chart(fig, width='stretch')
    
    elif analysis_type == "HT/FT Patterns":
        st.subheader("🔄 Half-Time/Full-Time Analysis")
        
        if not analysis_df.empty:
            # HT/FT result matrix
            ht_ft_counts = analysis_df['ht_ft_result'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # HT/FT distribution
                fig = px.bar(
                    x=ht_ft_counts.index,
                    y=ht_ft_counts.values,
                    title="HT/FT Double Result Distribution",
                    labels={'x': 'HT/FT Result', 'y': 'Count'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # HT vs FT winning patterns
                ht_results = analysis_df[['ht_home_winning', 'ht_away_winning', 'ht_draw']].sum()
                ft_results = analysis_df[['ft_home_winning', 'ft_away_winning', 'ft_draw']].sum()
                
                result_comparison = pd.DataFrame({
                    'Half-Time': ht_results,
                    'Full-Time': ft_results
                }).reset_index().melt(id_vars='index', var_name='Time', value_name='Count')
                result_comparison.rename(columns={'index': 'Result'}, inplace=True)
                
                fig = px.bar(
                    result_comparison,
                    x='Result',
                    y='Count',
                    color='Time',
                    title="HT vs FT Result Comparison",
                    barmode='group'
                )
                st.plotly_chart(fig, width='stretch')
            
            # HT lead conversion rates
            st.subheader("Lead Conversion Analysis")
            
            # Calculate conversion rates
            ht_home_leads = analysis_df[analysis_df['ht_home_winning']]
            ht_away_leads = analysis_df[analysis_df['ht_away_winning']]
            ht_draws = analysis_df[analysis_df['ht_draw']]
            
            if not ht_home_leads.empty:
                home_conversion = ht_home_leads['ft_home_winning'].mean()
                home_draw_conversion = ht_home_leads['ft_draw'].mean()
                home_loss_conversion = ht_home_leads['ft_away_winning'].mean()
                
                st.markdown(f"""
                **Home Team HT Lead Conversion:**
                - Win FT: {home_conversion:.1%}
                - Draw FT: {home_draw_conversion:.1%}
                - Lose FT: {home_loss_conversion:.1%}
                """)
            
            if not ht_away_leads.empty:
                away_conversion = ht_away_leads['ft_away_winning'].mean()
                away_draw_conversion = ht_away_leads['ft_draw'].mean()
                away_loss_conversion = ht_away_leads['ft_home_winning'].mean()
                
                st.markdown(f"""
                **Away Team HT Lead Conversion:**
                - Win FT: {away_conversion:.1%}
                - Draw FT: {away_draw_conversion:.1%}
                - Lose FT: {away_loss_conversion:.1%}
                """)
    
    elif analysis_type == "Comeback Analysis":
        st.subheader("🔄 Comeback Patterns Analysis")
        
        if not analysis_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Comeback rates
                comeback_data = {
                    'Home Comebacks': analysis_df['home_comeback'].sum(),
                    'Away Comebacks': analysis_df['away_comeback'].sum(),
                    'No Comeback': (~analysis_df['any_comeback']).sum()
                }
                
                fig = px.pie(
                    values=list(comeback_data.values()),
                    names=list(comeback_data.keys()),
                    title="Comeback Distribution"
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Comeback rates by competition
                if 'competition' in analysis_df.columns:
                    comp_comebacks = analysis_df.groupby('competition')['any_comeback'].agg(['mean', 'count']).reset_index()
                    comp_comebacks = comp_comebacks[comp_comebacks['count'] >= 10]  # Filter for sample size
                    
                    fig = px.bar(
                        comp_comebacks.sort_values('mean', ascending=False).head(10),
                        x='competition',
                        y='mean',
                        title="Comeback Rate by Competition",
                        labels={'mean': 'Comeback Rate'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, width='stretch')
            
            # Comeback vs potential
            st.subheader("Comeback vs Potential Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Potential distribution for comebacks vs non-comebacks
                comeback_potential = analysis_df[analysis_df['any_comeback']]['avg_potential']
                non_comeback_potential = analysis_df[~analysis_df['any_comeback']]['avg_potential']
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=comeback_potential, name='Comebacks', opacity=0.7))
                fig.add_trace(go.Histogram(x=non_comeback_potential, name='No Comebacks', opacity=0.7))
                fig.update_layout(
                    title="Potential Distribution: Comebacks vs Non-Comebacks",
                    barmode='overlay',
                    xaxis_title='Average Potential',
                    yaxis_title='Count'
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Comeback rate by potential threshold
                potential_bins = pd.cut(analysis_df['avg_potential'], bins=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
                comeback_by_potential = analysis_df.groupby(potential_bins)['any_comeback'].mean()
                
                fig = px.bar(
                    x=comeback_by_potential.index,
                    y=comeback_by_potential.values,
                    title="Comeback Rate by Potential Level",
                    labels={'x': 'Potential Level', 'y': 'Comeback Rate'}
                )
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, width='stretch')
    
    elif analysis_type == "Team Performance":
        st.subheader("👥 Team Performance Analysis")
        
        # Team selector
        all_teams = pd.concat([
            analysis_df['team_a'].dropna(),
            analysis_df['team_b'].dropna()
        ]).unique()
        
        selected_team = st.selectbox("Select Team", sorted(all_teams))
        
        if selected_team:
            # Filter data for selected team
            team_df = analysis_df[
                (analysis_df['team_a'] == selected_team) | (analysis_df['team_b'] == selected_team)
            ].copy()
            
            # Determine if team was home or away
            team_df['is_home'] = team_df['team_a'] == selected_team
            team_df['team_goals_fh'] = np.where(
                team_df['is_home'],
                team_df['home_fh'],
                team_df['away_fh']
            )
            team_df['team_goals_2h'] = np.where(
                team_df['is_home'],
                team_df['home_2h'],
                team_df['away_2h']
            )
            team_df['team_goals_ft'] = np.where(
                team_df['is_home'],
                team_df['home_ft'],
                team_df['away_ft']
            )
            
            # Display team stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_matches = len(team_df)
                st.metric("Total Matches", total_matches)
            
            with col2:
                avg_goals_ft = team_df['team_goals_ft'].mean()
                st.metric("Avg Goals FT", f"{avg_goals_ft:.2f}")
            
            with col3:
                comeback_rate = team_df['any_comeback'].mean()
                st.metric("Comeback Rate", f"{comeback_rate:.1%}")
            
            # Goal timing by team
            st.subheader(f"{selected_team} Goal Timing Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Goals by half
                avg_goals = team_df[['team_goals_fh', 'team_goals_2h']].mean()
                fig = px.bar(
                    x=['First Half', 'Second Half'],
                    y=[avg_goals['team_goals_fh'], avg_goals['team_goals_2h']],
                    title=f"{selected_team} Goals by Half"
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # HT/FT performance
                if team_df['is_home'].any():
                    home_df = team_df[team_df['is_home']]
                    away_df = team_df[~team_df['is_home']]
                    
                    ht_ft_comparison = pd.DataFrame({
                        'Home': [home_df['ht_home_winning'].mean(), home_df['ft_home_winning'].mean()],
                        'Away': [away_df['ht_away_winning'].mean(), away_df['ft_away_winning'].mean()]
                    }, index=['HT Winning', 'FT Winning'])
                    
                    fig = px.bar(
                        ht_ft_comparison,
                        x=ht_ft_comparison.index,
                        y=['Home', 'Away'],
                        title=f"{selected_team} HT vs FT Winning Rate",
                        barmode='group'
                    )
                    fig.update_layout(yaxis_tickformat='.1%')
                    st.plotly_chart(fig, width='stretch')
    
    elif analysis_type == "Time Trends":
        st.subheader("📅 Temporal Analysis")
        
        # Time period selector
        time_period = st.selectbox(
            "Time Period",
            options=["Daily", "Weekly", "Monthly"],
            key="time_period"
        )
        
        if 'start_time' in analysis_df.columns:
            analysis_df['date'] = pd.to_datetime(analysis_df['start_time']).dt.date
            
            # Group by time period
            if time_period == "Daily":
                time_group = analysis_df.groupby('date')
            elif time_period == "Weekly":
                analysis_df['week'] = pd.to_datetime(analysis_df['date']).dt.isocalendar().week
                analysis_df['year'] = pd.to_datetime(analysis_df['date']).dt.year
                time_group = analysis_df.groupby(['year', 'week'])
            else:  # Monthly
                analysis_df['month'] = pd.to_datetime(analysis_df['date']).dt.month
                analysis_df['year'] = pd.to_datetime(analysis_df['date']).dt.year
                time_group = analysis_df.groupby(['year', 'month'])
            
            # Calculate trends
            trends = time_group.agg({
                'avg_potential': 'mean',
                'btts_potential': 'mean',
                'o25_potential': 'mean',
                'total_fh': 'mean',
                'total_2h': 'mean',
                'any_comeback': 'mean'
            }).round(3)
            
            trends['match_count'] = time_group.size()
            
            # Display trends
            st.dataframe(trends, width='stretch')
            
            # Plot trends
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Average Potential Over Time", "Goal Timing Over Time", "Comeback Rate Over Time"),
                vertical_spacing=0.1
            )
            
            # Reset index for plotting
            trends_reset = trends.reset_index()
            
            # Potential trends
            fig.add_trace(
                go.Scatter(
                    x=range(len(trends_reset)),
                    y=trends_reset['avg_potential'],
                    mode='markers+lines',
                    name='Avg Potential'
                ),
                row=1, col=1
            )
            
            # Goal timing trends
            fig.add_trace(
                go.Scatter(
                    x=range(len(trends_reset)),
                    y=trends_reset['total_fh'],
                    mode='markers+lines',
                    name='1H Goals'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=range(len(trends_reset)),
                    y=trends_reset['total_2h'],
                    mode='markers+lines',
                    name='2H Goals'
                ),
                row=2, col=1
            )
            
            # Comeback trends
            fig.add_trace(
                go.Scatter(
                    x=range(len(trends_reset)),
                    y=trends_reset['any_comeback'],
                    mode='markers+lines',
                    name='Comeback Rate'
                ),
                row=3, col=1
            )
            
            fig.update_layout(height=900, title_text=f"{time_period} Trends Analysis")
            st.plotly_chart(fig, width='stretch')

def render_country_breakdown(simulated_matches, simulation_results):
    """Render country-based breakdown analysis"""
    st.subheader("🌍 Country Performance Analysis")
    
    if 'country' not in simulated_matches.columns:
        st.info("ℹ️ No country data available")
        return
    
    # Group by country and calculate metrics
    country_analysis = []
    for country in simulated_matches['country'].unique():
        if pd.isna(country):
            continue
            
        country_data = simulated_matches[simulated_matches['country'] == country]
        total_bets = len(country_data)
        wins = len(country_data[country_data['Sim Outcome'] == 'Win'])
        win_rate = wins / total_bets if total_bets > 0 else 0
        total_pnl = country_data['Sim PnL'].sum()
        
        # Calculate ROI using base capital from simulation results
        base_capital = simulation_results.get('base_capital', 100)
        roi = (total_pnl / (base_capital * total_bets)) * 100 if total_bets > 0 else 0
        
        country_analysis.append({
            'Country': country,
            'Total Bets': total_bets,
            'Wins': wins,
            'Losses': total_bets - wins,
            'Win Rate': win_rate,
            'Total PnL': total_pnl,
            'ROI (%)': roi
        })
    
    if not country_analysis:
        st.info("ℹ️ No country data available for analysis")
        return
    
    # Create DataFrame
    country_df = pd.DataFrame(country_analysis).sort_values('Total Bets', ascending=False)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Countries Analyzed", len(country_df))
    with col2:
        if not country_df.empty:
            best_country = country_df.loc[country_df['Win Rate'].idxmax(), 'Country']
            st.metric("Best Win Rate", f"{country_df['Win Rate'].max():.1%}", help=best_country)
    with col3:
        if not country_df.empty:
            most_profitable = country_df.loc[country_df['Total PnL'].idxmax(), 'Country']
            st.metric("Most Profitable", f"${country_df['Total PnL'].max():.2f}", help=most_profitable)
    
    # Bar chart for win rate by country
    st.subheader("Win Rate by Country")
    fig = px.bar(
        country_df.head(15),  # Top 15 countries by bets
        x='Country',
        y='Win Rate',
        title='Win Rate by Country (Top 15 by Volume)',
        labels={'Win Rate': 'Win Rate', 'Country': 'Country'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width='stretch')
    
    # Bar chart for PnL by country
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total PnL by Country")
        fig_pnl = px.bar(
            country_df.head(15),
            x='Country',
            y='Total PnL',
            title='Total PnL by Country (Top 15 by Volume)',
            labels={'Total PnL': 'Total PnL ($)', 'Country': 'Country'}
        )
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="red")
        fig_pnl.update_xaxes(tickangle=45)
        st.plotly_chart(fig_pnl, width='stretch')
    
    with col2:
        st.subheader("ROI by Country")
        fig_roi = px.bar(
            country_df.head(15),
            x='Country',
            y='ROI (%)',
            title='ROI by Country (Top 15 by Volume)',
            labels={'ROI (%)': 'ROI (%)', 'Country': 'Country'}
        )
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red")
        fig_roi.update_xaxes(tickangle=45)
        st.plotly_chart(fig_roi, width='stretch')
    
    # Detailed table
    st.subheader("Detailed Country Analysis")
    display_df = country_df.copy()
    display_df['Win Rate'] = display_df['Win Rate'].map('{:.1%}'.format)
    display_df['ROI (%)'] = display_df['ROI (%)'].map('{:.1f}%'.format)
    display_df['Total PnL'] = display_df['Total PnL'].map('${:.2f}'.format)
    
    st.dataframe(display_df, width='stretch')

def render_competition_breakdown(simulated_matches, simulation_results):
    """Render competition-based breakdown analysis"""
    st.subheader("🏆 Competition Performance Analysis")
    
    if 'competition' not in simulated_matches.columns:
        st.info("ℹ️ No competition data available")
        return
    
    # Group by competition and calculate metrics
    competition_analysis = []
    for _, row in simulated_matches[['competition', 'country']].drop_duplicates().iterrows():
        competition = row['competition']
        country = row['country']
        
        if pd.isna(competition) or pd.isna(country):
            continue
            
        # Create unique competition identifier with country
        competition_with_country = f"{competition} ({country})"
        
        # Filter data for this specific competition-country combination
        comp_data = simulated_matches[
            (simulated_matches['competition'] == competition) & 
            (simulated_matches['country'] == country)
        ]
        
        total_bets = len(comp_data)
        wins = len(comp_data[comp_data['Sim Outcome'] == 'Win'])
        win_rate = wins / total_bets if total_bets > 0 else 0
        total_pnl = comp_data['Sim PnL'].sum()
        
        # Calculate ROI using base capital from simulation results
        base_capital = simulation_results.get('base_capital', 100)
        roi = (total_pnl / (base_capital * total_bets)) * 100 if total_bets > 0 else 0
        
        competition_analysis.append({
            'Competition': competition_with_country,
            'Total Bets': total_bets,
            'Wins': wins,
            'Losses': total_bets - wins,
            'Win Rate': win_rate,
            'Total PnL': total_pnl,
            'ROI (%)': roi
        })
    
    if not competition_analysis:
        st.info("ℹ️ No competition data available for analysis")
        return
    
    # Create DataFrame
    competition_df = pd.DataFrame(competition_analysis).sort_values('Total Bets', ascending=False)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Competitions Analyzed", len(competition_df))
    with col2:
        if not competition_df.empty:
            best_comp = competition_df.loc[competition_df['Win Rate'].idxmax(), 'Competition']
            st.metric("Best Win Rate", f"{competition_df['Win Rate'].max():.1%}", help=best_comp)
    with col3:
        if not competition_df.empty:
            most_profitable = competition_df.loc[competition_df['Total PnL'].idxmax(), 'Competition']
            st.metric("Most Profitable", f"${competition_df['Total PnL'].max():.2f}", help=most_profitable)
    
    # Bar chart for win rate by competition
    st.subheader("Win Rate by Competition")
    fig = px.bar(
        competition_df.head(15),  # Top 15 competitions by bets
        x='Competition',
        y='Win Rate',
        title='Win Rate by Competition (Top 15 by Volume)',
        labels={'Win Rate': 'Win Rate', 'Competition': 'Competition'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width='stretch')
    
    # Bar chart for PnL by competition
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total PnL by Competition")
        fig_pnl = px.bar(
            competition_df.head(15),
            x='Competition',
            y='Total PnL',
            title='Total PnL by Competition (Top 15 by Volume)',
            labels={'Total PnL': 'Total PnL ($)', 'Competition': 'Competition'}
        )
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="red")
        fig_pnl.update_xaxes(tickangle=45)
        st.plotly_chart(fig_pnl, width='stretch')
    
    with col2:
        st.subheader("ROI by Competition")
        fig_roi = px.bar(
            competition_df.head(15),
            x='Competition',
            y='ROI (%)',
            title='ROI by Competition (Top 15 by Volume)',
            labels={'ROI (%)': 'ROI (%)', 'Competition': 'Competition'}
        )
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red")
        fig_roi.update_xaxes(tickangle=45)
        st.plotly_chart(fig_roi, width='stretch')
    
    # Detailed table
    st.subheader("Detailed Competition Analysis")
    display_df = competition_df.copy()
    display_df['Win Rate'] = display_df['Win Rate'].map('{:.1%}'.format)
    display_df['ROI (%)'] = display_df['ROI (%)'].map('{:.1f}%'.format)
    display_df['Total PnL'] = display_df['Total PnL'].map('${:.2f}'.format)
    
    st.dataframe(display_df, width='stretch')

def render_breakdown_analysis_tab(df):
    """Render comprehensive breakdown analysis with range-based metrics"""
    st.header("📊 Breakdown Analysis")
    
    # Check if simulation results exist
    simulation_results = st.session_state.get('last_simulation_results')
    if not simulation_results or not simulation_results.get('valid_row_indices'):
        st.info("👈 Enable auto-run in sidebar and change filters to see breakdown analysis")
        return
    
    # Get simulated matches only
    simulated_matches = df.loc[simulation_results['valid_row_indices']].copy()
    
    # Add simulation data to simulated matches
    sim_data = {}
    for i, row_idx in enumerate(simulation_results['valid_row_indices']):
        sim_data[row_idx] = {
            'Bet #': i + 1,
            'Sim Outcome': 'Win' if simulation_results['outcomes_list'][i] else 'Loss',
            'Sim PnL': simulation_results['actual_pnl_per_bet'][i],
            'Sim Odds': simulation_results['valid_odds_list'][i]
        }
    
    for col in ['Bet #', 'Sim Outcome', 'Sim PnL', 'Sim Odds']:
        simulated_matches[col] = simulated_matches.index.map(lambda x: sim_data.get(x, {}).get(col, ''))
    
    # Overall summary
    st.subheader("📈 Overall Simulation Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Bets", simulation_results['valid_bets'])
        st.metric("Win Rate", f"{simulation_results['win_rate']:.1%}")
    
    with col2:
        st.metric("Wins", simulation_results['wins'])
        st.metric("Losses", simulation_results['losses'])
    
    with col3:
        st.metric("Total PnL", f"${simulation_results['total_pnl']:.2f}")
        st.metric("ROI", f"{simulation_results['roi']:.1%}")
    
    with col4:
        st.metric("Max Win Streak", simulation_results['max_winning_streak'])
        st.metric("Max Lose Streak", simulation_results['max_losing_streak'])
    
    # Create breakdown sub-tabs
    breakdown_tabs = st.tabs([
        "📈 Odds Analysis", 
        "⚡ PPG Analysis", 
        "🎯 XG Analysis", 
        "📊 Potential Analysis",
        "🌍 Country Analysis",
        "🏆 Competition Analysis"
    ])
    
    with breakdown_tabs[0]:
        render_odds_breakdown(simulated_matches, simulation_results)
    with breakdown_tabs[1]:
        render_ppg_breakdown(simulated_matches, simulation_results)
    with breakdown_tabs[2]:
        render_xg_breakdown(simulated_matches, simulation_results)
    with breakdown_tabs[3]:
        render_potential_breakdown(simulated_matches, simulation_results)
    with breakdown_tabs[4]:
        render_country_breakdown(simulated_matches, simulation_results)
    with breakdown_tabs[5]:
        render_competition_breakdown(simulated_matches, simulation_results)

def render_odds_breakdown(simulated_matches, simulation_results):
    """Render odds range analysis"""
    st.subheader("📈 Odds Range Analysis")
    
    # Create odds buckets
    odds_bins = [1.01, 1.2, 1.5, 1.8, 1.9, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, float('inf')]
    odds_labels = ["1.01-1.2","1.2-1.5", "1.5-1.8", "1.8-1.9", "1.9-2.0", "2.0-2.5", "2.5-3.0", "3.0-4.0", "4.0-5.0", "5.0-7.0", "7.0-10.0", "10.0+"]
    
    # Helper function to analyze odds column
    def analyze_odds_column(data, column_name, title):
        if column_name not in data.columns:
            return None
            
        # Filter out rows with missing odds
        valid_data = data[data[column_name].notna()].copy()
        if valid_data.empty:
            return None
            
        # Create odds ranges
        valid_data['Odds Range'] = pd.cut(
            valid_data[column_name], 
            bins=odds_bins, 
            labels=odds_labels,
            right=False
        )
        
        # Analyze by odds range
        odds_analysis = []
        for odds_range in odds_labels:
            range_data = valid_data[valid_data['Odds Range'] == odds_range]
            
            if not range_data.empty:
                wins = sum(range_data['Sim Outcome'] == 'Win')
                total = len(range_data)
                win_rate = wins / total if total > 0 else 0
                avg_pnl = range_data['Sim PnL'].mean()
                total_pnl = range_data['Sim PnL'].sum()
                
                odds_analysis.append({
                    'Odds Range': odds_range,
                    'Bets': total,
                    'Wins': wins,
                    'Losses': total - wins,
                    'Win Rate': f"{win_rate:.1%}",
                    'Avg PnL': f"${avg_pnl:.2f}",
                    'Total PnL': f"${total_pnl:.2f}",
                    'Avg Odds': f"{range_data[column_name].mean():.2f}"
                })
        
        return odds_analysis
    
    # 1. Sim Odds Analysis (current simulation odds)
    st.markdown("#### 🎯 Simulation Odds Analysis")
    sim_analysis = analyze_odds_column(simulated_matches, 'Sim Odds', 'Simulation Odds')
    
    if sim_analysis:
        sim_df = pd.DataFrame(sim_analysis)
        st.dataframe(sim_df, width='stretch')
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            sim_df['Win Rate Numeric'] = sim_df['Win Rate'].str.rstrip('%').astype(float) / 100
            fig = px.bar(
                sim_df, 
                x='Odds Range', 
                y='Win Rate',
                title="Win Rate by Simulation Odds Range",
                labels={'Win Rate': 'Win Rate', 'Odds Range': 'Odds Range'}
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.bar(
                sim_df,
                x='Odds Range',
                y='Total PnL',
                title="Total PnL by Simulation Odds Range",
                labels={'Total PnL': 'Total PnL ($)', 'Odds Range': 'Odds Range'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # 2. Home Win Odds Analysis (odds_ft_1)
    st.markdown("#### 🏠 Home Win Odds Analysis (odds_ft_1)")
    home_analysis = analyze_odds_column(simulated_matches, 'odds_ft_1', 'Home Win Odds')
    
    if home_analysis:
        home_df = pd.DataFrame(home_analysis)
        st.dataframe(home_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            home_df['Win Rate Numeric'] = home_df['Win Rate'].str.rstrip('%').astype(float) / 100
            fig = px.bar(
                home_df, 
                x='Odds Range', 
                y='Win Rate',
                title="Win Rate by Home Win Odds Range"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.bar(
                home_df,
                x='Odds Range',
                y='Total PnL',
                title="Total PnL by Home Win Odds Range"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("ℹ️ No Home Win odds data available")
    
    st.markdown("---")
    
    # 3. Draw Odds Analysis (odds_ft_x)
    st.markdown("#### 🤝 Draw Odds Analysis (odds_ft_x)")
    draw_analysis = analyze_odds_column(simulated_matches, 'odds_ft_x', 'Draw Odds')
    
    if draw_analysis:
        draw_df = pd.DataFrame(draw_analysis)
        st.dataframe(draw_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            draw_df['Win Rate Numeric'] = draw_df['Win Rate'].str.rstrip('%').astype(float) / 100
            fig = px.bar(
                draw_df, 
                x='Odds Range', 
                y='Win Rate',
                title="Win Rate by Draw Odds Range"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.bar(
                draw_df,
                x='Odds Range',
                y='Total PnL',
                title="Total PnL by Draw Odds Range"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("ℹ️ No Draw odds data available")
    
    st.markdown("---")
    
    # 4. Away Win Odds Analysis (odds_ft_2)
    st.markdown("#### ✈️ Away Win Odds Analysis (odds_ft_2)")
    away_analysis = analyze_odds_column(simulated_matches, 'odds_ft_2', 'Away Win Odds')
    
    if away_analysis:
        away_df = pd.DataFrame(away_analysis)
        st.dataframe(away_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            away_df['Win Rate Numeric'] = away_df['Win Rate'].str.rstrip('%').astype(float) / 100
            fig = px.bar(
                away_df, 
                x='Odds Range', 
                y='Win Rate',
                title="Win Rate by Away Win Odds Range"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.bar(
                away_df,
                x='Odds Range',
                y='Total PnL',
                title="Total PnL by Away Win Odds Range"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("ℹ️ No Away Win odds data available")
    
    st.markdown("---")
    
    # 5. BTTS Yes Odds Analysis (odds_btts_yes)
    st.markdown("#### ⚽ BTTS Yes Odds Analysis (odds_btts_yes)")
    btts_analysis = analyze_odds_column(simulated_matches, 'odds_btts_yes', 'BTTS Yes Odds')
    
    if btts_analysis:
        btts_df = pd.DataFrame(btts_analysis)
        st.dataframe(btts_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            btts_df['Win Rate Numeric'] = btts_df['Win Rate'].str.rstrip('%').astype(float) / 100
            fig = px.bar(
                btts_df, 
                x='Odds Range', 
                y='Win Rate',
                title="Win Rate by BTTS Yes Odds Range"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.bar(
                btts_df,
                x='Odds Range',
                y='Total PnL',
                title="Total PnL by BTTS Yes Odds Range"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("ℹ️ No BTTS Yes odds data available")

def render_ppg_breakdown(simulated_matches, simulation_results):
    """Render PPG analysis with difference ranges and individual team analysis"""
    st.subheader("⚡ PPG Analysis")
    
    # Check if required columns exist
    if 'ppg_a' not in simulated_matches.columns or 'ppg_b' not in simulated_matches.columns:
        st.warning("⚠️ PPG data not available in current dataset")
        return
    
    # Section 1: PPG Difference Analysis (existing)
    st.markdown("### 📊 PPG Difference Analysis")
    
    # Calculate PPG difference
    simulated_matches['PPG Difference'] = simulated_matches['ppg_a'] - simulated_matches['ppg_b']
    
    # PPG difference buckets (-3 to 3)
    ppg_diff_bins = [-float('inf'), -2, -1, 0, 1, 2, float('inf')]
    ppg_diff_labels = ['<-2', '-2 to -1', '-1 to 0', '0 to 1', '1 to 2', '>2']
    
    simulated_matches['PPG Diff Range'] = pd.cut(
        simulated_matches['PPG Difference'],
        bins=ppg_diff_bins,
        labels=ppg_diff_labels,
        right=False
    )
    
    # Analyze PPG difference
    ppg_analysis = []
    for diff_range in ppg_diff_labels:
        range_data = simulated_matches[simulated_matches['PPG Diff Range'] == diff_range]
        
        if not range_data.empty:
            wins = sum(range_data['Sim Outcome'] == 'Win')
            total = len(range_data)
            win_rate = wins / total if total > 0 else 0
            avg_pnl = range_data['Sim PnL'].mean()
            total_pnl = range_data['Sim PnL'].sum()
            
            ppg_analysis.append({
                'PPG Diff Range': diff_range,
                'Bets': total,
                'Wins': wins,
                'Losses': total - wins,
                'Win Rate': f"{win_rate:.1%}",
                'Avg PnL': f"${avg_pnl:.2f}",
                'Total PnL': f"${total_pnl:.2f}",
                'Avg PPG A': f"{range_data['ppg_a'].mean():.2f}",
                'Avg PPG B': f"{range_data['ppg_b'].mean():.2f}"
            })
    
    if ppg_analysis:
        ppg_df = pd.DataFrame(ppg_analysis)
        st.dataframe(ppg_df, width='stretch')
        
        # Visualization
        fig = px.bar(
            ppg_df,
            x='PPG Diff Range',
            y='Win Rate',
            title="Win Rate by PPG Difference",
            labels={'Win Rate': 'Win Rate', 'PPG Diff Range': 'PPG A - PPG B'}
        )
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Section 2: Individual Team PPG Analysis
    st.markdown("### 🏠 Individual Team PPG Analysis")
    
    # PPG individual buckets (0 to 3)
    ppg_individual_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, float('inf')]
    ppg_individual_labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0+']
    
    # Team A PPG Analysis
    st.markdown("#### Team A (Home) PPG Analysis")
    simulated_matches['PPG A Range'] = pd.cut(
        simulated_matches['ppg_a'],
        bins=ppg_individual_bins,
        labels=ppg_individual_labels,
        right=False
    )
    
    ppg_a_analysis = []
    for ppg_range in ppg_individual_labels:
        range_data = simulated_matches[simulated_matches['PPG A Range'] == ppg_range]
        
        if not range_data.empty:
            wins = sum(range_data['Sim Outcome'] == 'Win')
            total = len(range_data)
            win_rate = wins / total if total > 0 else 0
            avg_pnl = range_data['Sim PnL'].mean()
            total_pnl = range_data['Sim PnL'].sum()
            
            ppg_a_analysis.append({
                'PPG A Range': ppg_range,
                'Bets': total,
                'Wins': wins,
                'Losses': total - wins,
                'Win Rate': f"{win_rate:.1%}",
                'Avg PnL': f"${avg_pnl:.2f}",
                'Total PnL': f"${total_pnl:.2f}",
                'Avg PPG': f"{range_data['ppg_a'].mean():.2f}"
            })
    
    if ppg_a_analysis:
        ppg_a_df = pd.DataFrame(ppg_a_analysis)
        st.dataframe(ppg_a_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                ppg_a_df,
                x='PPG A Range',
                y='Win Rate',
                title="Win Rate by Team A PPG"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            ppg_a_df['Win Rate Numeric'] = ppg_a_df['Win Rate'].str.rstrip('%').astype(float) / 100
            fig = px.bar(
                ppg_a_df,
                x='PPG A Range',
                y='Total PnL',
                title="Total PnL by Team A PPG"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')
    
    # Team B PPG Analysis
    st.markdown("#### Team B (Away) PPG Analysis")
    simulated_matches['PPG B Range'] = pd.cut(
        simulated_matches['ppg_b'],
        bins=ppg_individual_bins,
        labels=ppg_individual_labels,
        right=False
    )
    
    ppg_b_analysis = []
    for ppg_range in ppg_individual_labels:
        range_data = simulated_matches[simulated_matches['PPG B Range'] == ppg_range]
        
        if not range_data.empty:
            wins = sum(range_data['Sim Outcome'] == 'Win')
            total = len(range_data)
            win_rate = wins / total if total > 0 else 0
            avg_pnl = range_data['Sim PnL'].mean()
            total_pnl = range_data['Sim PnL'].sum()
            
            ppg_b_analysis.append({
                'PPG B Range': ppg_range,
                'Bets': total,
                'Wins': wins,
                'Losses': total - wins,
                'Win Rate': f"{win_rate:.1%}",
                'Avg PnL': f"${avg_pnl:.2f}",
                'Total PnL': f"${total_pnl:.2f}",
                'Avg PPG': f"{range_data['ppg_b'].mean():.2f}"
            })
    
    if ppg_b_analysis:
        ppg_b_df = pd.DataFrame(ppg_b_analysis)
        st.dataframe(ppg_b_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                ppg_b_df,
                x='PPG B Range',
                y='Win Rate',
                title="Win Rate by Team B PPG"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            ppg_b_df['Win Rate Numeric'] = ppg_b_df['Win Rate'].str.rstrip('%').astype(float) / 100
            fig = px.bar(
                ppg_b_df,
                x='PPG B Range',
                y='Total PnL',
                title="Total PnL by Team B PPG"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')

def render_xg_breakdown(simulated_matches, simulation_results):
    """Render XG analysis with individual and difference ranges"""
    st.subheader("🎯 XG Analysis")
    
    # Check if required columns exist
    if 'team_a_xg_prematch' not in simulated_matches.columns or 'team_b_xg_prematch' not in simulated_matches.columns:
        st.warning("⚠️ XG data not available in current dataset")
        return
    
    # Section 1: XG Difference Analysis (existing)
    st.markdown("### 📊 XG Difference Analysis")
    
    # Calculate XG difference using correct column names
    simulated_matches['XG Difference'] = simulated_matches['team_a_xg_prematch'] - simulated_matches['team_b_xg_prematch']
    
    # XG difference buckets (-5 to 5)
    xg_diff_bins = [-float('inf'), -2, -1, 0, 1, 2, float('inf')]
    xg_diff_labels = ['<-2', '-2 to -1', '-1 to 0', '0 to 1', '1 to 2', '>2']
    
    simulated_matches['XG Diff Range'] = pd.cut(
        simulated_matches['XG Difference'],
        bins=xg_diff_bins,
        labels=xg_diff_labels,
        right=False
    )
    
    # Analyze XG difference
    xg_analysis = []
    for diff_range in xg_diff_labels:
        range_data = simulated_matches[simulated_matches['XG Diff Range'] == diff_range]
        
        if not range_data.empty:
            wins = sum(range_data['Sim Outcome'] == 'Win')
            total = len(range_data)
            win_rate = wins / total if total > 0 else 0
            avg_pnl = range_data['Sim PnL'].mean()
            total_pnl = range_data['Sim PnL'].sum()
            
            xg_analysis.append({
                'XG Diff Range': diff_range,
                'Bets': total,
                'Wins': wins,
                'Losses': total - wins,
                'Win Rate': f"{win_rate:.1%}",
                'Avg PnL': f"${avg_pnl:.2f}",
                'Total PnL': f"${total_pnl:.2f}",
                'Avg XG A': f"{range_data['team_a_xg_prematch'].mean():.2f}",
                'Avg XG B': f"{range_data['team_b_xg_prematch'].mean():.2f}"
            })
    
    if xg_analysis:
        xg_df = pd.DataFrame(xg_analysis)
        st.dataframe(xg_df, width='stretch')
        
        # Visualization
        fig = px.bar(
            xg_df,
            x='XG Diff Range',
            y='Win Rate',
            title="Win Rate by XG Difference",
            labels={'Win Rate': 'Win Rate', 'XG Diff Range': 'XG A - XG B'}
        )
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Section 2: Individual Team XG Analysis
    st.markdown("### 🏠 Individual Team XG Analysis")
    
    # XG individual buckets (0 to 3)
    xg_individual_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, float('inf')]
    xg_individual_labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0+']
    
    # Team A XG Analysis
    st.markdown("#### Team A (Home) XG Analysis")
    simulated_matches['XG A Range'] = pd.cut(
        simulated_matches['team_a_xg_prematch'],
        bins=xg_individual_bins,
        labels=xg_individual_labels,
        right=False
    )
    
    xg_a_analysis = []
    for xg_range in xg_individual_labels:
        range_data = simulated_matches[simulated_matches['XG A Range'] == xg_range]
        
        if not range_data.empty:
            wins = sum(range_data['Sim Outcome'] == 'Win')
            total = len(range_data)
            win_rate = wins / total if total > 0 else 0
            avg_pnl = range_data['Sim PnL'].mean()
            total_pnl = range_data['Sim PnL'].sum()
            
            xg_a_analysis.append({
                'XG A Range': xg_range,
                'Bets': total,
                'Wins': wins,
                'Losses': total - wins,
                'Win Rate': f"{win_rate:.1%}",
                'Avg PnL': f"${avg_pnl:.2f}",
                'Total PnL': f"${total_pnl:.2f}",
                'Avg XG': f"{range_data['team_a_xg_prematch'].mean():.2f}"
            })
    
    if xg_a_analysis:
        xg_a_df = pd.DataFrame(xg_a_analysis)
        st.dataframe(xg_a_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                xg_a_df,
                x='XG A Range',
                y='Win Rate',
                title="Win Rate by Team A XG"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            xg_a_df['Win Rate Numeric'] = xg_a_df['Win Rate'].str.rstrip('%').astype(float) / 100
            fig = px.bar(
                xg_a_df,
                x='XG A Range',
                y='Total PnL',
                title="Total PnL by Team A XG"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')
    
    # Team B XG Analysis
    st.markdown("#### Team B (Away) XG Analysis")
    simulated_matches['XG B Range'] = pd.cut(
        simulated_matches['team_b_xg_prematch'],
        bins=xg_individual_bins,
        labels=xg_individual_labels,
        right=False
    )
    
    xg_b_analysis = []
    for xg_range in xg_individual_labels:
        range_data = simulated_matches[simulated_matches['XG B Range'] == xg_range]
        
        if not range_data.empty:
            wins = sum(range_data['Sim Outcome'] == 'Win')
            total = len(range_data)
            win_rate = wins / total if total > 0 else 0
            avg_pnl = range_data['Sim PnL'].mean()
            total_pnl = range_data['Sim PnL'].sum()
            
            xg_b_analysis.append({
                'XG B Range': xg_range,
                'Bets': total,
                'Wins': wins,
                'Losses': total - wins,
                'Win Rate': f"{win_rate:.1%}",
                'Avg PnL': f"${avg_pnl:.2f}",
                'Total PnL': f"${total_pnl:.2f}",
                'Avg XG': f"{range_data['team_b_xg_prematch'].mean():.2f}"
            })
    
    if xg_b_analysis:
        xg_b_df = pd.DataFrame(xg_b_analysis)
        st.dataframe(xg_b_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                xg_b_df,
                x='XG B Range',
                y='Win Rate',
                title="Win Rate by Team B XG"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            xg_b_df['Win Rate Numeric'] = xg_b_df['Win Rate'].str.rstrip('%').astype(float) / 100
            fig = px.bar(
                xg_b_df,
                x='XG B Range',
                y='Total PnL',
                title="Total PnL by Team B XG"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')

def render_potential_breakdown(simulated_matches, simulation_results):
    """Render potential analysis organized by categories in tabs"""
    st.subheader("📊 Potential Analysis")
    
    # Create tabs for the three potential categories
    avg_tab, btts_tab, over_tab = st.tabs(["📊 Average Goals", "🎯 BTTS Potentials", "⚽ Over/Under Potentials"])
    
    # Tab 1: Average Goals Analysis
    with avg_tab:
        if 'avg_potential' in simulated_matches.columns:
            # Avg Potential: 0-10 scale with 1-unit buckets
            potential_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
            potential_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10+']
            potential_values = simulated_matches['avg_potential']
            
            simulated_matches['avg_potential_Range'] = pd.cut(
                potential_values,
                bins=potential_bins,
                labels=potential_labels,
                right=False
            )
            
            # Analyze avg potential
            potential_analysis = []
            for potential_range in potential_labels:
                range_data = simulated_matches[simulated_matches['avg_potential_Range'] == potential_range]
                
                if not range_data.empty:
                    wins = sum(range_data['Sim Outcome'] == 'Win')
                    total = len(range_data)
                    win_rate = wins / total if total > 0 else 0
                    avg_pnl = range_data['Sim PnL'].mean()
                    total_pnl = range_data['Sim PnL'].sum()
                    
                    potential_analysis.append({
                        'Potential Range': potential_range,
                        'Bets': total,
                        'Wins': wins,
                        'Losses': total - wins,
                        'Win Rate': f"{win_rate:.1%}",
                        'Avg PnL': f"${avg_pnl:.2f}",
                        'Total PnL': f"${total_pnl:.2f}",
                        'Avg Potential': f"{range_data['avg_potential'].mean():.2f}"
                    })
            
            if potential_analysis:
                potential_df = pd.DataFrame(potential_analysis)
                st.dataframe(potential_df, width='stretch')
                
                col1, col2 = st.columns(2)
                with col1:
                    potential_df['Win Rate Numeric'] = potential_df['Win Rate'].str.rstrip('%').astype(float) / 100
                    fig = px.bar(
                        potential_df,
                        x='Potential Range',
                        y='Win Rate',
                        title="Win Rate by Average Potential"
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    fig = px.bar(
                        potential_df,
                        x='Potential Range',
                        y='Total PnL',
                        title="Total PnL by Average Potential"
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, width='stretch')
        else:
            st.info("ℹ️ No Average Goals potential data available")
    
    # Tab 2: BTTS Potentials Analysis
    with btts_tab:
        btts_metrics = ['btts_potential', 'btts_fhg_potential', 'btts_2hg_potential']
        available_btts = [metric for metric in btts_metrics if metric in simulated_matches.columns]
        
        if available_btts:
            for metric in available_btts:
                st.markdown(f"**{metric.replace('_', ' ').title()} Analysis**")
                
                # BTTS potentials: 0-1 scale converted to percentages
                potential_values = simulated_matches[metric] * 100
                potential_bins = [0, 20, 40, 50, 60, 70, 80, 90, 100]
                potential_labels = ['0-20%', '20-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
                
                simulated_matches[f'{metric}_Range'] = pd.cut(
                    potential_values,
                    bins=potential_bins,
                    labels=potential_labels,
                    right=False
                )
                
                # Analyze by potential range
                potential_analysis = []
                for potential_range in potential_labels:
                    range_data = simulated_matches[simulated_matches[f'{metric}_Range'] == potential_range]
                    
                    if not range_data.empty:
                        wins = sum(range_data['Sim Outcome'] == 'Win')
                        total = len(range_data)
                        win_rate = wins / total if total > 0 else 0
                        avg_pnl = range_data['Sim PnL'].mean()
                        total_pnl = range_data['Sim PnL'].sum()
                        
                        potential_analysis.append({
                            'Potential Range': potential_range,
                            'Bets': total,
                            'Wins': wins,
                            'Losses': total - wins,
                            'Win Rate': f"{win_rate:.1%}",
                            'Avg PnL': f"${avg_pnl:.2f}",
                            'Total PnL': f"${total_pnl:.2f}",
                            'Avg Potential': f"{range_data[metric].mean():.3f}"
                        })
                
                if potential_analysis:
                    potential_df = pd.DataFrame(potential_analysis)
                    st.dataframe(potential_df, width='stretch')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        potential_df['Win Rate Numeric'] = potential_df['Win Rate'].str.rstrip('%').astype(float) / 100
                        fig = px.bar(
                            potential_df,
                            x='Potential Range',
                            y='Win Rate',
                            title=f"Win Rate by {metric.replace('_', ' ').title()}"
                        )
                        st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        fig = px.bar(
                            potential_df,
                            x='Potential Range',
                            y='Total PnL',
                            title=f"Total PnL by {metric.replace('_', ' ').title()}"
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, width='stretch')
                
                st.markdown("---")
        else:
            st.info("ℹ️ No BTTS potential data available")
    
    # Tab 3: Over/Under Potentials Analysis
    with over_tab:
        over_metrics = ['o15_potential', 'o25_potential', 'o35_potential', 'o05ht_potential']
        available_over = [metric for metric in over_metrics if metric in simulated_matches.columns]
        
        if available_over:
            for metric in available_over:
                st.markdown(f"**{metric.replace('_', ' ').title()} Analysis**")
                
                # Over potentials: 0-1 scale converted to percentages
                potential_values = simulated_matches[metric] * 100
                potential_bins = [0, 20, 40, 50, 60, 70, 80, 90, 100]
                potential_labels = ['0-20%', '20-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
                
                simulated_matches[f'{metric}_Range'] = pd.cut(
                    potential_values,
                    bins=potential_bins,
                    labels=potential_labels,
                    right=False
                )
                
                # Analyze by potential range
                potential_analysis = []
                for potential_range in potential_labels:
                    range_data = simulated_matches[simulated_matches[f'{metric}_Range'] == potential_range]
                    
                    if not range_data.empty:
                        wins = sum(range_data['Sim Outcome'] == 'Win')
                        total = len(range_data)
                        win_rate = wins / total if total > 0 else 0
                        avg_pnl = range_data['Sim PnL'].mean()
                        total_pnl = range_data['Sim PnL'].sum()
                        
                        potential_analysis.append({
                            'Potential Range': potential_range,
                            'Bets': total,
                            'Wins': wins,
                            'Losses': total - wins,
                            'Win Rate': f"{win_rate:.1%}",
                            'Avg PnL': f"${avg_pnl:.2f}",
                            'Total PnL': f"${total_pnl:.2f}",
                            'Avg Potential': f"{range_data[metric].mean():.3f}"
                        })
                
                if potential_analysis:
                    potential_df = pd.DataFrame(potential_analysis)
                    st.dataframe(potential_df, width='stretch')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        potential_df['Win Rate Numeric'] = potential_df['Win Rate'].str.rstrip('%').astype(float) / 100
                        fig = px.bar(
                            potential_df,
                            x='Potential Range',
                            y='Win Rate',
                            title=f"Win Rate by {metric.replace('_', ' ').title()}"
                        )
                        st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        fig = px.bar(
                            potential_df,
                            x='Potential Range',
                            y='Total PnL',
                            title=f"Total PnL by {metric.replace('_', ' ').title()}"
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, width='stretch')
                
                st.markdown("---")
        else:
            st.info("ℹ️ No Over/Under potential data available")
        
        st.markdown("---")

# Sidebar filters
st.sidebar.header("")

# Strategies section
st.sidebar.markdown("### 🎯 Strategies")

# Initialize strategies in session state
if 'saved_strategies' not in st.session_state:
    st.session_state.saved_strategies = {}

# Load strategies from file
def load_strategies():
    """Load saved strategies from JSON file"""
    try:
        import json
        import os
        strategies_file = os.path.join(os.path.dirname(__file__), 'strategies.json')
        if os.path.exists(strategies_file):
            with open(strategies_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading strategies: {e}")
    return {}

# Save strategies to file
def save_strategies():
    """Save strategies to JSON file"""
    try:
        import json
        import os
        strategies_file = os.path.join(os.path.dirname(__file__), 'strategies.json')
        with open(strategies_file, 'w') as f:
            json.dump(st.session_state.saved_strategies, f, indent=2)
        return True
    except Exception as e:
        st.sidebar.error(f"Error saving strategies: {e}")
        return False

# Load strategies on startup
if 'strategies_loaded' not in st.session_state:
    st.session_state.saved_strategies = load_strategies()
    st.session_state.strategies_loaded = True

# Strategy management UI
if st.session_state.saved_strategies:
    strategy_names = list(st.session_state.saved_strategies.keys())
    selected_strategy = st.sidebar.selectbox("Load Strategy:", ["None"] + strategy_names, key="strategy_selector")
    
    if selected_strategy != "None":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.sidebar.button("📥 Load", key="load_strategy"):
                # Apply strategy filters to session state
                strategy_filters = st.session_state.saved_strategies[selected_strategy]
                for key, value in strategy_filters.items():
                    st.session_state[key] = value
                st.sidebar.success(f"Loaded strategy: {selected_strategy}")
                # Trigger simulation with loaded strategy
                st.session_state.should_run_simulation = True
                st.session_state.auto_run_simulation = True
                st.rerun()
        
        with col2:
            if st.sidebar.button("🗑️ Delete", key="delete_strategy"):
                del st.session_state.saved_strategies[selected_strategy]
                save_strategies()
                st.sidebar.success(f"Deleted strategy: {selected_strategy}")
                st.rerun()
else:
    st.sidebar.info("No saved strategies yet")

# Save current strategy
with st.sidebar.expander("💾 Save Current Strategy"):
    strategy_name = st.text_input("Strategy name:", key="new_strategy_name")
    if st.button("Save Strategy", key="save_strategy_btn"):
        if strategy_name.strip():
            # Collect current filter values
            current_filters = {
                'betting_method': st.session_state.get('betting_method', 'BTTS at Fulltime'),
                'bet_stake': st.session_state.get('bet_stake', 10.0),
                # Betting configuration
                'staking_method': st.session_state.get('staking_method', 'Fixed Amount'),
                'base_capital': st.session_state.get('base_capital', 1000.0),
                'bet_percentage': st.session_state.get('bet_percentage', 3.0),
                'use_martingale': st.session_state.get('use_martingale', False),
                'max_bet_percentage': st.session_state.get('max_bet_percentage', 20.0),
                # Range filters
                'ppg_a_min': st.session_state.get('ppg_a_min', 0.0),
                'ppg_a_max': st.session_state.get('ppg_a_max', 3.0),
                'ppg_b_min': st.session_state.get('ppg_b_min', 0.0),
                'ppg_b_max': st.session_state.get('ppg_b_max', 3.0),
                'ppg_diff_min': st.session_state.get('ppg_diff_min', -3.0),
                'ppg_diff_max': st.session_state.get('ppg_diff_max', 3.0),
                'team_a_xg_min': st.session_state.get('team_a_xg_min', 0.0),
                'team_a_xg_max': st.session_state.get('team_a_xg_max', 3.0),
                'team_b_xg_min': st.session_state.get('team_b_xg_min', 0.0),
                'team_b_xg_max': st.session_state.get('team_b_xg_max', 3.0),
                'xg_diff_min': st.session_state.get('xg_diff_min', -5.0),
                'xg_diff_max': st.session_state.get('xg_diff_max', 5.0),
                'avg_potential_min': st.session_state.get('avg_potential_min', 0.0),
                'avg_potential_max': st.session_state.get('avg_potential_max', 10.0),
                'btts_potential_min': st.session_state.get('btts_potential_min', 0.0),
                'btts_potential_max': st.session_state.get('btts_potential_max', 1.0),
                'o15_potential_min': st.session_state.get('o15_potential_min', 0.0),
                'o15_potential_max': st.session_state.get('o15_potential_max', 1.0),
                'o25_potential_min': st.session_state.get('o25_potential_min', 0.0),
                'o25_potential_max': st.session_state.get('o25_potential_max', 1.0),
                'o35_potential_min': st.session_state.get('o35_potential_min', 0.0),
                'o35_potential_max': st.session_state.get('o35_potential_max', 1.0),
                'o05ht_potential_min': st.session_state.get('o05ht_potential_min', 0.0),
                'o05ht_potential_max': st.session_state.get('o05ht_potential_max', 1.0),
                'btts_fhg_min': st.session_state.get('btts_fhg_min', 0.0),
                'btts_fhg_max': st.session_state.get('btts_fhg_max', 2.0),
                'btts_2hg_min': st.session_state.get('btts_2hg_min', 0.0),
                'btts_2hg_max': st.session_state.get('btts_2hg_max', 2.0),
            }
            
            # Add odds filters if they exist
            odds_filters = [
                'footystats_odds_btts_min', 'footystats_odds_btts_max',
                'footystats_odds_1_min', 'footystats_odds_1_max',
                'footystats_odds_x_min', 'footystats_odds_x_max',
                'footystats_odds_2_min', 'footystats_odds_2_max',
                'footystats_odds_over15_min', 'footystats_odds_over15_max',
                'footystats_odds_over25_min', 'footystats_odds_over25_max',
                'footystats_odds_over35_min', 'footystats_odds_over35_max',
                'footystats_odds_over05_min', 'footystats_odds_over05_max'
            ]
            
            for filter_name in odds_filters:
                if filter_name in st.session_state:
                    current_filters[filter_name] = st.session_state[filter_name]
            
            # Save strategy
            st.session_state.saved_strategies[strategy_name.strip()] = current_filters
            if save_strategies():
                st.sidebar.success(f"Strategy '{strategy_name.strip()}' saved!")
                st.rerun()
        else:
            st.sidebar.error("Please enter a strategy name")


st.sidebar.markdown("---")


# Betting Configuration in a box
with st.sidebar.container():
    st.sidebar.markdown("### 💰 Betting Configuration")
    
    # Base capital (always shown for ROI calculations)
    base_capital = st.sidebar.number_input(
        "Base Capital ($)",
        min_value=50,
        max_value=10000,
        value=1000,
        step=10,
        key="base_capital",
        help="Initial bankroll amount used for ROI calculations and bust detection"
    )
    
    # Staking method selection
    staking_method = st.sidebar.radio(
        "Staking Method:",
        ["Fixed Amount", "% of Capital"],
        key="staking_method",
        help="Choose between fixed stake amount or percentage of current bankroll"
    )
    
    if staking_method == "Fixed Amount":
        bet_stake = st.sidebar.slider(
            "Stake Amount ($)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            key="bet_stake"
        )
        st.sidebar.markdown(f"*Fixed stake: ${bet_stake:.2f} per bet*")
    else:
        bet_percentage = st.sidebar.slider(
            "Stake Percentage (%)",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5,
            key="bet_percentage"
        )
        bet_stake = bet_percentage  # Will be used as percentage in simulation
        st.sidebar.markdown(f"*Variable stake: {bet_percentage:.1f}% of current bankroll*")
    
    # Martingale option (only available for Fixed Amount staking)
    if staking_method == "Fixed Amount":
        use_martingale = st.sidebar.checkbox(
            "Use Martingale on Loss",
            key="use_martingale",
            help="Double stake after each loss until win, then reset to base stake"
        )
        
        if use_martingale:
            max_bet_percentage = st.sidebar.slider(
                "Max Bet % of Capital",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=5.0,
                key="max_bet_percentage",
                help="Maximum bet size as percentage of current capital (limits martingale risk)"
            )
            st.sidebar.markdown(f"*Martingale protection: Max bet = {max_bet_percentage:.0f}% of capital*")
        else:
            max_bet_percentage = 20.0  # Default value when martingale is disabled
    else:
        # Disable martingale for percentage-based staking
        use_martingale = False
        max_bet_percentage = 20.0
        st.sidebar.info("💡 *Martingale is only available with Fixed Amount staking to prevent excessive risk*")
    
    # Store betting configuration in session state
    st.session_state.betting_config = {
        'method': staking_method,
        'stake': bet_stake,
        'base_capital': base_capital,
        'use_martingale': use_martingale,
        'max_bet_percentage': max_bet_percentage
    }

st.sidebar.markdown("---")


# Sidebar filters
st.sidebar.header("🔍 Filters")

# Get filter options
try:
    filter_options = get_filter_options()
    
    # Country filter
    selected_country = st.sidebar.selectbox(
        "Country",
        options=["All"] + filter_options['countries'],
        key="country"
    )
    
    # Competition filter
    if selected_country != "All":
        competitions = [comp for comp in filter_options['competitions'] 
                       if selected_country in comp]
        selected_competition = st.sidebar.selectbox(
            "Competition",
            options=["All"] + competitions,
            key="competition"
        )
    else:
        selected_competition = st.sidebar.selectbox(
            "Competition",
            options=["All"] + filter_options['competitions'],
            key="competition"
        )
    
    # Date range filter
    if 'min_date' in filter_options and 'max_date' in filter_options:
        min_date = filter_options['min_date']
        max_date = filter_options['max_date']
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            date_from = st.sidebar.date_input(
                "From Date",
                value=min_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date(),
                key="date_from"
            )
        with col2:
            date_to = st.sidebar.date_input(
                "To Date",
                value=max_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date(),
                key="date_to"
            )
    else:
        date_from = None
        date_to = None
    
    # Betting simulation configuration
    st.sidebar.subheader("🎯 Betting Simulation")
    
    # Method selector
    betting_method = st.sidebar.selectbox(
        "Betting Method",
        options=[
            "BTTS at Fulltime",
            "Home Win", 
            "Draw",
            "Away Win",
            "Over 1.5 Goals",
            "Over 2.5 Goals", 
            "Over 3.5 Goals",
            "Over 0.5 Goals HT"
        ],
        key="betting_method"
    )
    
    # Method-specific range filters
    st.sidebar.subheader("📊 Range Filters")
    
    def create_range_filter(label, key_prefix, min_val=0.0, max_val=1.0, default_min=0.0, default_max=1.0):
        """Create min/max range filter inputs"""
        col1, col2 = st.columns(2)
        with col1:
            min_value = st.number_input(
                f"{label} Min",
                min_value=min_val,
                max_value=max_val,
                value=default_min,
                step=0.01,
                key=f"{key_prefix}_min"
            )
        with col2:
            max_value = st.number_input(
                f"{label} Max",
                min_value=min_val,
                max_value=max_val,
                value=default_max,
                step=0.01,
                key=f"{key_prefix}_max"
            )
        
        # Validation warning
        if min_value > max_value:
            st.sidebar.warning(f"⚠️ {label}: Min should be ≤ Max")
        
        return min_value, max_value
    
    # Show all odds filters grouped logically
    st.sidebar.markdown("**Odds**")
    
    # Match Result Odds (1X2)
    with st.sidebar.expander("🏠 Match Result Odds (Home/Draw/Away)", expanded=betting_method in ["Home Win", "Draw", "Away Win"]):
        footystats_odds_1_min, footystats_odds_1_max = create_range_filter("Home Win Odds", "footystats_odds_1", 1.01, 20.0, 1.01, 3.0)
        footystats_odds_x_min, footystats_odds_x_max = create_range_filter("Draw Odds", "footystats_odds_x", 1.01, 20.0, 1.01, 4.0)
        footystats_odds_2_min, footystats_odds_2_max = create_range_filter("Away Win Odds", "footystats_odds_2", 1.01, 20.0, 1.01, 5.0)
    
    # BTTS Odds
    with st.sidebar.expander("⚽ BTTS Odds", expanded=betting_method == "BTTS at Fulltime"):
        footystats_odds_btts_min, footystats_odds_btts_max = create_range_filter("BTTS Odds", "footystats_odds_btts", 1.01, 20.0, 1.01, 3.5)
    
    # Goals Odds (Overs)
    with st.sidebar.expander("🥅 Goals Odds (Overs)", expanded=betting_method.startswith("Over")):
        footystats_odds_over15_min, footystats_odds_over15_max = create_range_filter("Over 1.5 Odds", "footystats_odds_over15", 1.01, 20.0, 1.01, 2.0)
        footystats_odds_over25_min, footystats_odds_over25_max = create_range_filter("Over 2.5 Odds", "footystats_odds_over25", 1.01, 20.0, 1.01, 2.5)
        footystats_odds_over35_min, footystats_odds_over35_max = create_range_filter("Over 3.5 Odds", "footystats_odds_over35", 1.01, 20.0, 1.01, 3.5)
        footystats_odds_over05_min, footystats_odds_over05_max = create_range_filter("Over 0.5 HT Odds", "footystats_odds_over05", 1.01, 20.0, 1.01, 1.8)
        
    st.sidebar.markdown("**Performance**")
    # Team Performance
    with st.sidebar.expander("👥 Team Performance", expanded=False):
        ppg_a_min, ppg_a_max = create_range_filter("Home PPG", "ppg_a", 0.0, 3.0, 0.0, 3.0)
        ppg_b_min, ppg_b_max = create_range_filter("Away PPG", "ppg_b", 0.0, 3.0, 0.0, 3.0)
        ppg_diff_min, ppg_diff_max = create_range_filter("PPG Difference", "ppg_diff", -3.0, 3.0, -3.0, 3.0)
    
    # Expected Goals
    with st.sidebar.expander("⚽ Expected Goals", expanded=False):
        xg_a_min, xg_a_max = create_range_filter("Home XG", "team_a_xg", 0.0, 5.0, 0.0, 3.0)
        xg_b_min, xg_b_max = create_range_filter("Away XG", "team_b_xg", 0.0, 5.0, 0.0, 3.0)
        xg_diff_min, xg_diff_max = create_range_filter("XG Difference", "xg_diff", -5.0, 5.0, -5.0, 5.0)
    
    st.sidebar.markdown("**Potentials**")
    # Match Potentials
    with st.sidebar.expander("📊 Match Potentials", expanded=False):
        avg_potential_min, avg_potential_max = create_range_filter("Avg Potential", "avg_potential", 0.0, 10.0, 0.0, 10.0)
        btts_potential_min, btts_potential_max = create_range_filter("BTTS Potential", "btts_potential", 0.0, 1.0, 0.0, 1.0)
        o15_min, o15_max = create_range_filter("Over 1.5 Potential", "o15_potential", 0.0, 1.0, 0.0, 1.0)
        o25_min, o25_max = create_range_filter("Over 2.5 Potential", "o25_potential", 0.0, 1.0, 0.0, 1.0)
        o35_min, o35_max = create_range_filter("Over 3.5 Potential", "o35_potential", 0.0, 1.0, 0.0, 1.0)

    # Half-Time Potentials
    with st.sidebar.expander("⏰ Half-Time Potentials", expanded=False):
        o05ht_min, o05ht_max = create_range_filter("Over 0.5 HT Potential", "o05ht_potential", 0.0, 1.0, 0.0, 1.0)
    
    # BTTS-Specific Potentials
    with st.sidebar.expander("⚽ BTTS-Specific Potentials", expanded=betting_method == "BTTS at Fulltime"):
        btts_fhg_min, btts_fhg_max = create_range_filter("BTTS 1H Potential", "btts_fhg", 0.0, 8.5, 0.0, 1.0)
        btts_2hg_min, btts_2hg_max = create_range_filter("BTTS 2H Potential", "btts_2hg", 0.0, 8.5, 0.0, 1.0)
    
    st.sidebar.markdown("___")
    
    # Build enhanced filters dictionary
    # Parse competition name from formatted string
    parsed_competition = None
    if selected_competition != "All":
        parsed_competition = selected_competition.rsplit(' (', 1)[0]  # Extract name before " ("
    
    filters = {
        'country': selected_country if selected_country != "All" else None,
        'competition': parsed_competition,
        'date_from': datetime.combine(date_from, datetime.min.time()) if date_from else None,
        'date_to': datetime.combine(date_to, datetime.max.time()) if date_to else None,
        'bet_stake': bet_stake,
        'betting_method': betting_method,
        
        # Range filters (only include relevant ones based on method)
        'ppg_a_min': ppg_a_min if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'ppg_a_max': ppg_a_max if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'ppg_b_min': ppg_b_min if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'ppg_b_max': ppg_b_max if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'ppg_diff_min': ppg_diff_min if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'ppg_diff_max': ppg_diff_max if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'team_a_xg_min': xg_a_min if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'team_a_xg_max': xg_a_max if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'team_b_xg_min': xg_b_min if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'team_b_xg_max': xg_b_max if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'xg_diff_min': xg_diff_min if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'xg_diff_max': xg_diff_max if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals", "Over 0.5 Goals HT"] else None,
        'avg_potential_min': avg_potential_min,
        'avg_potential_max': avg_potential_max,
        'btts_potential_min': btts_potential_min if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals"] else None,
        'btts_potential_max': btts_potential_max if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals"] else None,
        'btts_fhg_min': st.session_state.get('btts_fhg_min') if betting_method in ["BTTS at Fulltime", "Over 0.5 Goals HT"] else None,
        'btts_fhg_max': st.session_state.get('btts_fhg_max') if betting_method in ["BTTS at Fulltime", "Over 0.5 Goals HT"] else None,
        'btts_2hg_min': st.session_state.get('btts_2hg_min') if betting_method == "BTTS at Fulltime" else None,
        'btts_2hg_max': st.session_state.get('btts_2hg_max') if betting_method == "BTTS at Fulltime" else None,
        'o15_potential_min': o15_min if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals"] else None,
        'o15_potential_max': o15_max if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals"] else None,
        'o25_potential_min': o25_min if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals"] else None,
        'o25_potential_max': o25_max if betting_method in ["BTTS at Fulltime", "Home Win", "Draw", "Away Win", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals"] else None,
        'o35_potential_min': st.session_state.get('o35_min') if betting_method in ["BTTS at Fulltime", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals"] else None,
        'o35_potential_max': st.session_state.get('o35_max') if betting_method in ["BTTS at Fulltime", "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals"] else None,
        'o05ht_potential_min': st.session_state.get('o05ht_min') if betting_method in ["BTTS at Fulltime", "Over 0.5 Goals HT"] else None,
        'o05ht_potential_max': st.session_state.get('o05ht_max') if betting_method in ["BTTS at Fulltime", "Over 0.5 Goals HT"] else None,
    }
    
    # Add ALL odds filters regardless of betting method
    filters.update({
        # Match Result Odds (1X2)
        'footystats_odds_1_min': footystats_odds_1_min,
        'footystats_odds_1_max': footystats_odds_1_max,
        'footystats_odds_x_min': footystats_odds_x_min,
        'footystats_odds_x_max': footystats_odds_x_max,
        'footystats_odds_2_min': footystats_odds_2_min,
        'footystats_odds_2_max': footystats_odds_2_max,
        
        # BTTS Odds
        'footystats_odds_btts_min': footystats_odds_btts_min,
        'footystats_odds_btts_max': footystats_odds_btts_max,
        
        # Goals Odds (Overs)
        'footystats_odds_over15_min': footystats_odds_over15_min,
        'footystats_odds_over15_max': footystats_odds_over15_max,
        'footystats_odds_over25_min': footystats_odds_over25_min,
        'footystats_odds_over25_max': footystats_odds_over25_max,
        'footystats_odds_over35_min': footystats_odds_over35_min,
        'footystats_odds_over35_max': footystats_odds_over35_max,
        'footystats_odds_over05_min': footystats_odds_over05_min,
        'footystats_odds_over05_max': footystats_odds_over05_max,
    })
    
    # Update session state
    st.session_state.filters = filters
    
    # Data loading - only run when simulation button is clicked
    if st.session_state.should_run_simulation:
        with st.spinner("Loading matched events..."):
            try:
                df = get_matched_events(filters)
                st.session_state.data = df
                st.session_state.current_data = df
                st.session_state.should_run_simulation = False  # Reset the flag
                # Clear stale simulation results to prevent index mismatch
                st.session_state.last_simulation_results = None
                st.session_state.last_simulation_betting_method = None
                st.session_state.last_simulation_betting_config = None
                st.success(f"Loaded {len(df)} matches with current filters")
                
                # Automatically run simulation after data loads
                if not df.empty:
                    with st.spinner("Running betting simulation..."):
                        try:
                            # Parse scores for analysis
                            analysis_df = parse_all_scores(df.copy())
                            
                            if analysis_df.empty:
                                st.warning("No matches found with current filters. Try relaxing the constraints.")
                            else:
                                # Calculate simulation results based on betting method
                                betting_method = st.session_state.filters.get('betting_method', 'BTTS at Fulltime')
                                betting_config = st.session_state.betting_config
                                simulation_results = run_betting_simulation(analysis_df, betting_method, betting_config)
                                
                                if simulation_results:
                                    # Store simulation results in session state
                                    st.session_state.last_simulation_results = simulation_results
                                    st.session_state.last_simulation_betting_method = betting_method
                                    st.session_state.last_simulation_betting_config = betting_config
                                    
                                    st.success(f"✅ Simulation complete! {simulation_results['valid_bets']} valid bets found.")
                                else:
                                    st.warning("No valid bets found with current filters and odds.")
                                    # Clear simulation results from session state
                                    st.session_state.last_simulation_results = None
                                    st.session_state.last_simulation_betting_method = None
                                    st.session_state.last_simulation_betting_config = None
                                    
                        except Exception as e:
                            st.error(f"Error running simulation: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            
            except Exception as e:
                st.sidebar.error(f"Error loading data: {str(e)}")
                st.session_state.current_data = pd.DataFrame()
                st.session_state.should_run_simulation = False
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset Filters"):
        # Reset to realistic defaults based on current method
        reset_filters_to_defaults(betting_method)
        st.session_state.should_run_simulation = True
        st.rerun()
    
    # Manual rerun button
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Simulate", type="primary", key="btn-simulate", width="stretch"):
        st.session_state.should_run_simulation = True
        st.rerun()

except Exception as e:
    st.sidebar.error(f"Error loading filter options: {str(e)}")
    st.session_state.filters = {}
    st.session_state.current_data = pd.DataFrame()
    st.rerun()
    filters = {}

# Main content area
if st.session_state.current_data.empty:
    st.info("👈 Apply filters in the sidebar to load matched events data")
else:
    df = st.session_state.current_data
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Strategy Backtesting", 
        "📊 Overview", 
        "🔍 Match Browser", 
        "📈 Potential Analysis",
        "📊 Breakdown Analysis"
    ])
    
    # Tab 1: Strategy Backtesting
    with tab1:
        render_strategy_backtesting_tab(df)
    
    # Tab 2: Overview
    with tab2:
        render_overview_tab(df)
    
    # Tab 3: Match Browser
    with tab3:
        render_match_browser_tab(df)
    
    # Tab 4: Potential Analysis
    with tab4:
        render_potential_analysis_tab(df)
    
    # Tab 5: Breakdown Analysis
    with tab5:
        render_breakdown_analysis_tab(df)

# Footer
st.markdown("---")
st.markdown("📊 Sports Performance Tracker - Analyze footystats potentials and build winning strategies")
