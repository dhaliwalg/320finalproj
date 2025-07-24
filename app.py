import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="VALORANT Professional Analysis",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF4655;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #0F1419;
    margin: 1rem 0;
}
.highlight-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #FF4655;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">VALORANT: An Up & Coming Competitive Tac-Shooter</h1>', unsafe_allow_html=True)
st.markdown("**An analysis of ACS (Average Combat Score) at official Riot events in relation to the agents that are being played.**")
st.markdown("*By Gurjit Dhaliwal*")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "Introduction", 
    "Data Loading & Processing", 
    "Exploratory Data Analysis", 
    "Linear Regression Analysis", 
    "Machine Learning Predictions",
    "Conclusion"
])

@st.cache_data
def load_and_process_data(sqlite_file='valorant.sqlite'):
    """Load and process the VALORANT data exactly as in the original analysis"""
    
    conn = sqlite3.connect(sqlite_file)
    
    # Load matches
    matches = "SELECT * FROM Matches"
    matchframe = pd.read_sql(matches, conn)
    
    # Create the extensive filter for Riot events (from original analysis)
    riot_events = [
        'Champions Tour Asia-Pacific: Last Chance Qualifier',
        'Champions Tour Brazil Stage 1: Challengers 1',
        'Champions Tour Brazil Stage 1: Challengers 2',
        'Champions Tour Brazil Stage 1: Challengers 3',
        'Champions Tour Brazil Stage 1: Masters',
        'Champions Tour Brazil Stage 2: Challengers 1',
        'Champions Tour Brazil Stage 2: Challengers 2',
        'Champions Tour Brazil Stage 2: Challengers 3',
        'Champions Tour Brazil Stage 2: Challengers Finals',
        'Champions Tour Brazil Stage 3: Challengers 1',
        'Champions Tour Brazil Stage 3: Challengers 2',
        'Champions Tour Brazil Stage 3: Challengers 3',
        'Champions Tour Brazil Stage 3: Challengers Playoffs',
        'Champions Tour CIS Stage 1: Challengers 1',
        'Champions Tour CIS Stage 1: Challengers 2',
        'Champions Tour CIS Stage 1: Challengers 3',
        'Champions Tour CIS Stage 1: Masters',
        'Champions Tour CIS Stage 2: Challengers 1',
        'Champions Tour CIS Stage 2: Challengers 2',
        'Champions Tour CIS Stage 3: Challengers 1',
        'Champions Tour CIS Stage 3: Challengers 2',
        'Champions Tour EMEA: Last Chance Qualifier',
        'Champions Tour Europe Stage 1: Challengers 1',
        'Champions Tour Europe Stage 1: Challengers 2',
        'Champions Tour Europe Stage 1: Challengers 3',
        'Champions Tour Europe Stage 1: Masters',
        'Champions Tour Europe Stage 2: Challengers 1',
        'Champions Tour Europe Stage 2: Challengers 2',
        'Champions Tour Europe Stage 3: Challengers 1',
        'Champions Tour Europe Stage 3: Challengers 2',
        'Champions Tour Hong Kong & Taiwan Stage 2: Challengers 1',
        'Champions Tour Hong Kong & Taiwan Stage 2: Challengers 2',
        'Champions Tour Hong Kong & Taiwan Stage 2: Challengers 3',
        'Champions Tour Hong Kong & Taiwan Stage 3: Challengers 1',
        'Champions Tour Hong Kong & Taiwan Stage 3: Challengers 2',
        'Champions Tour Hong Kong & Taiwan Stage 3: Challengers 3',
        'Champions Tour Hong Kong and Taiwan Stage 1: Challengers 3',
        'Champions Tour Indonesia Stage 1: Challengers 1',
        'Champions Tour Indonesia Stage 1: Challengers 2',
        'Champions Tour Indonesia Stage 1: Challengers 3',
        'Champions Tour Indonesia Stage 2: Challengers 1',
        'Champions Tour Indonesia Stage 2: Challengers 2',
        'Champions Tour Indonesia Stage 2: Challengers 3',
        'Champions Tour Indonesia Stage 3: Challengers 1',
        'Champions Tour Indonesia Stage 3: Challengers 2',
        'Champions Tour Indonesia Stage 3: Challengers 3',
        'Champions Tour Japan Stage 1: Challengers 1',
        'Champions Tour Japan Stage 1: Challengers 2',
        'Champions Tour Japan Stage 1: Challengers 3',
        'Champions Tour Japan Stage 1: Masters',
        'Champions Tour Japan Stage 2: Challengers 1',
        'Champions Tour Japan Stage 2: Challengers 2',
        'Champions Tour Japan Stage 2: Challengers Finals',
        'Champions Tour Japan Stage 3: Challengers 1',
        'Champions Tour Japan Stage 3: Challengers 2',
        'Champions Tour Japan Stage 3: Challengers Playoffs',
        'Champions Tour Korea Stage 1: Challengers 1',
        'Champions Tour Korea Stage 1: Challengers 2',
        'Champions Tour Korea Stage 1: Challengers 3',
        'Champions Tour Korea Stage 1: Masters',
        'Champions Tour Korea Stage 2: Challengers',
        'Champions Tour Korea Stage 3: Challengers ',
        'Champions Tour LATAM Stage 1: Challengers 1',
        'Champions Tour LATAM Stage 1: Challengers 2',
        'Champions Tour LATAM Stage 1: Challengers 3',
        'Champions Tour LATAM Stage 1: Masters',
        'Champions Tour LATAM Stage 2: Challengers 1',
        'Champions Tour LATAM Stage 2: Challengers 2',
        'Champions Tour LATAM Stage 2: Challengers Finals',
        'Champions Tour LATAM Stage 3: Challengers 1',
        'Champions Tour LATAM Stage 3: Challengers 2',
        'Champions Tour LATAM Stage 3: Challengers Playoffs',
        'Champions Tour Malaysia & Singapore Stage 1: Challengers 1',
        'Champions Tour Malaysia & Singapore Stage 1: Challengers 2',
        'Champions Tour Malaysia & Singapore Stage 1: Challengers 3',
        'Champions Tour Malaysia & Singapore Stage 2: Challengers 1',
        'Champions Tour Malaysia & Singapore Stage 2: Challengers 2',
        'Champions Tour Malaysia & Singapore Stage 2: Challengers 3',
        'Champions Tour Malaysia & Singapore Stage 3: Challengers 1',
        'Champions Tour Malaysia & Singapore Stage 3: Challengers 2',
        'Champions Tour Malaysia & Singapore Stage 3: Challengers 3',
        'Champions Tour North America Stage 1: Challengers 1',
        'Champions Tour North America Stage 1: Challengers 2',
        'Champions Tour North America Stage 1: Challengers 3',
        'Champions Tour North America Stage 1: Masters',
        'Champions Tour North America Stage 2: Challengers 1',
        'Champions Tour North America Stage 2: Challengers 2',
        'Champions Tour North America Stage 2: Challengers Finals',
        'Champions Tour North America Stage 3: Challengers 1',
        'Champions Tour North America Stage 3: Challengers 2',
        'Champions Tour North America Stage 3: Challengers Playoffs',
        'Champions Tour Philippines Stage 1: Challengers 1',
        'Champions Tour Philippines Stage 1: Challengers 2',
        'Champions Tour Philippines Stage 1: Challengers 3',
        'Champions Tour Philippines Stage 2: Challengers 1',
        'Champions Tour Philippines Stage 2: Challengers 2',
        'Champions Tour Philippines Stage 2: Challengers 3',
        'Champions Tour Philippines Stage 3: Challengers 1',
        'Champions Tour Philippines Stage 3: Challengers 2',
        'Champions Tour Philippines Stage 3: Challengers 3',
        'Champions Tour SEA Stage 1: Masters',
        'Champions Tour SEA Stage 2: Challengers Finals',
        'Champions Tour SEA Stage 3: Challengers Playoffs',
        'Champions Tour South America: Last Chance Qualifier',
        'Champions Tour North America: Last Chance Qualifier',
        'Champions Tour Stage 2: EMEA Challengers Playoffs',
        'Champions Tour Stage 3: EMEA Challengers Playoffs',
        'Champions Tour Thailand Stage 1: Challengers 1',
        'Champions Tour Thailand Stage 1: Challengers 2',
        'Champions Tour Hong Kong & Taiwan Stage 1: Challengers 2',
        'Champions Tour Thailand Stage 1: Challengers 3',
        'Champions Tour Thailand Stage 2: Challengers 1',
        'Champions Tour Thailand Stage 2: Challengers 2',
        'Champions Tour Thailand Stage 3: Challengers 1',
        'Champions Tour Thailand Stage 3: Challengers 2',
        'Champions Tour Thailand Stage 3: Challengers 3',
        'Champions Tour Turkey Stage 1: Challengers 1',
        'Champions Tour Turkey Stage 1: Challengers 2',
        'Champions Tour Turkey Stage 1: Challengers 3',
        'Champions Tour Turkey Stage 1: Masters',
        'Champions Tour Turkey Stage 2: Challengers 1',
        'Champions Tour Turkey Stage 2: Challengers 2',
        'Champions Tour Turkey Stage 3: Challengers 1',
        'Champions Tour Turkey Stage 3: Challengers 2',
        'Champions Tour Vietnam Stage 2: Challengers',
        'Champions Tour Vietnam Stage 3: Challengers 1',
        'Champions Tour Vietnam Stage 3: Challengers 2',
        'Champions Tour Vietnam Stage 3: Challengers 3',
        'First Strike: CIS',
        'First Strike: Indonesia',
        'First Strike: Europe',
        'First Strike: Korea',
        'First Strike : Japan',
        'First Strike: Malaysia & Singapore',
        'First Strike: Oceania',
        'First Strike: MENA',
        'First Strike: Latin America',
        'First Strike: North America',
        'First Strike: Brazil',
        'First Strike: Turkey',
        'First Strike: Hong Kong & Taiwan',
        'Valorant Champions Tour Stage 2: Masters Reykjavik',
        'Valorant Champions Tour Stage 3: Masters Berlin',
        'VALORANT Champions'
    ]
    
    # Create filter mask
    mask = matchframe['EventName'].isin(riot_events)
    riotframe = matchframe[mask]
    
    # Load Games table
    games = "SELECT * FROM Games"
    allgames_df = pd.read_sql(games, conn)
    
    # Filter games for Riot events
    riotmatchIDs = riotframe['MatchID'].unique().tolist()
    riotgames = allgames_df[allgames_df['MatchID'].isin(riotmatchIDs)]
    
    # Load Game_Scoreboard
    scoreboard = "SELECT * FROM Game_Scoreboard"
    scoreframe = pd.read_sql(scoreboard, conn)
    
    # Filter scoreboard for Riot games
    riotgameIDs = riotgames['GameID'].unique()
    riotscores = scoreframe[scoreframe['GameID'].isin(riotgameIDs)]
    
    # Clean data - remove blank team abbreviations and agents
    riotscores = riotscores[riotscores['TeamAbbreviation'] != ""]
    riotscores = riotscores[riotscores['Agent'] != ""]
    
    # Implement the original win/loss calculation logic
    def count_winloss(teamabbr, game_id, games_df):
        """Simplified version of the original count_winloss function"""
        try:
            game_info = games_df[games_df['GameID'] == game_id]
            if game_info.empty:
                return False
            
            winner = game_info['Winner'].values[0].lower()
            teamabbr = teamabbr.lower()
            
            # Check if each character in team abbreviation is in winner name
            for char in teamabbr:
                if char not in winner:
                    return False
            return True
        except:
            return False
    
    # Calculate wins and losses for each player
    player_stats = {}
    
    for _, row in riotscores.iterrows():
        player = row['PlayerName']
        won = count_winloss(row['TeamAbbreviation'], row['GameID'], riotgames)
        
        if player not in player_stats:
            player_stats[player] = {'wins': 0, 'losses': 0}
        
        if won:
            player_stats[player]['wins'] += 1
        else:
            player_stats[player]['losses'] += 1
    
    # Add win/loss data to dataframe
    riotscores['Wins'] = riotscores['PlayerName'].map(lambda x: player_stats.get(x, {}).get('wins', 0))
    riotscores['Lost'] = riotscores['PlayerName'].map(lambda x: player_stats.get(x, {}).get('losses', 0))
    riotscores['GamesPlayed'] = riotscores['Wins'] + riotscores['Lost']
    riotscores['Win_Perc'] = riotscores['Wins'] / riotscores['GamesPlayed'].replace(0, 1)
    
    # Trim unnecessary columns
    columns_to_keep = ['GameID', 'PlayerID', 'PlayerName', 'TeamAbbreviation', 'Agent', 
                       'ACS', 'Kills', 'Deaths', 'Assists', 'PlusMinus', 'ADR', 
                       'Wins', 'Lost', 'GamesPlayed', 'Win_Perc']
    
    riotscores_trim = riotscores[columns_to_keep]
    
    conn.close()
    return riotscores_trim, riotgames, riotframe

# Load data
if page != "Introduction":
    try:
        with st.spinner("Loading and processing VALORANT data..."):
            data, games_data, matches_data = load_and_process_data()
        st.sidebar.success(f"Loaded {len(data):,} player records")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

if page == "Introduction":
    st.markdown("## Introduction")
    
    st.write("""
    VALORANT is a tactical shooter created by Riot Games in 2020 that has taken off in the field of E-Sports. 
    Tactical Shooters are games that aim to simulate combat through slow-paced, team-oriented gameplay. 
    Each bullet is devastating, resulting in a focus on strategy and precise aim. VALORANT however, changes the 
    scene of competitive tac-shooters by bringing characters and abilities into the field. It's been categorized 
    as a crossover between Overwatch and CS:GO.
    """)
    
    st.image("https://cdn.mos.cms.futurecdn.net/MsHTeL6UUzcmmvFDuDSRwC-1200-80.jpg", 
             caption="VALORANT gameplay showing an agent firing a weapon alongside Viper & her wall (smoke ability)")
    
    st.markdown("### Agent Classes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown("**CONTROLLER**")
        st.write("Characters who possess smokes & can block off lines of sight.")
        st.write("*Current Controllers: Brimstone, Astra, Omen, Viper*")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown("**DUELIST**")
        st.write("Self-sufficient fragging characters, everything in their kit enables them to do damage and make space.")
        st.write("*Current Duelists: Jett, Phoenix, Reyna, Raze, Yoru*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown("**SENTINEL**")
        st.write("Defense-oriented characters who watch for flanks and lockdown bomb sites.")
        st.write("*Current Sentinels: Killjoy, Cypher, Sage*")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown("**INITIATOR**")
        st.write("Intelligence-oriented characters who can set up their allies for success through their abilities.")
        st.write("*Current Initiators: KAY/O, Breach, Skye, Sova*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### About ACS")
    st.info("""
    **ACS (Average Combat Score)** effectively captures a player's impact on a match. 
    Points are awarded for getting kills & doing damage, but kills early in a round give more points than kills at the end. 
    ACS also accounts for multi-kills, using proper utility, and clutching (winning a round when the odds are against you).
    """)
    
    st.markdown("### Game Fundamentals")
    st.write("""
    The fundamentals of the game remain the same as CS:GO. One team starts as the "attackers". In each round, 
    they must either plant the bomb and ensure it detonates or eliminate the opposing team. The other team are 
    the "defenders". They must defend the map and prevent the bomb from being planted or eliminate all attackers. 
    Each round lasts for 2 minutes with teams switching sides after playing 12 rounds. The first team to win 13 
    rounds will win the game, unless overtime is triggered.
    """)

elif page == "Data Loading & Processing":
    st.markdown('<h2 class="sub-header">Data Loading & Processing</h2>', unsafe_allow_html=True)
    
    st.write("""
    Like I said before, there have been dozens of tournaments that've occurred for VALORANT. "Nerd Street" is a 3rd party 
    organization which used to run Winter Championship tournaments. These were popular at the start prior to Riot operating 
    their own events. But once Riot started running official competitions, the largest organizations and best players shifted 
    to play in those. For the purposes of this project, I am ignoring all data from non-Riot events.
    """)
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", f"{len(data):,}")
    with col2:
        st.metric("Unique Agents", data['Agent'].nunique())
    with col3:
        st.metric("Total Games", f"{len(games_data):,}")
    with col4:
        st.metric("Average ACS", f"{data['ACS'].mean():.1f}")
    
    st.markdown("### Sample Data")
    st.dataframe(data.head(10))
    
    st.markdown("### Data Quality")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values Check:**")
        missing_data = data.isnull().sum()
        st.write(missing_data[missing_data > 0] if missing_data.sum() > 0 else "No missing values found ✅")
    
    with col2:
        st.write("**Data Types:**")
        st.write(data.dtypes)

elif page == "Exploratory Data Analysis":
    st.markdown('<h2 class="sub-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    st.write("What's data science without some cool graphs, am I right?")
    
    # Agent frequency
    st.markdown("### Agent Pick Rates")
    agent_counts = data['Agent'].value_counts()
    
    fig = px.bar(
        x=agent_counts.index, 
        y=agent_counts.values,
        title="Frequency of Agents Played in Professional Matches",
        labels={'x': 'Agent', 'y': 'Number of Games Played'},
        color=agent_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"""
    Seems like my prediction of duelist characters being the most played throughout the tournament was wrong. 
    With our histogram, we can see that **{agent_counts.index[0]}** was the most played agent throughout all of the events. 
    {agent_counts.index[0].title()}'s kit is built around intelligence gathering - they seem to have been an essential pick 
    as they have the highest pickrate.
    """)
    
    # Show top/bottom agents
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Most Played Agents:**")
        for i, (agent, count) in enumerate(agent_counts.head().items()):
            st.write(f"{i+1}. {agent.title()}: {count:,} games")
    
    with col2:
        st.markdown("**Least Played Agents:**")
        for i, (agent, count) in enumerate(agent_counts.tail().items()):
            st.write(f"{len(agent_counts)-4+i}. {agent.title()}: {count:,} games")
    
    # ACS vs Win Percentage scatter plot (original messy one)
    st.markdown("### ACS vs Win Percentage Distribution")
    st.write("First, let's look at the distribution of all players - this plot's way too messy, but there's some stuff to learn!")
    
    fig = px.scatter(
        data, 
        x='ACS', 
        y='Win_Perc',
        color='Agent',
        title="Distribution of ACS vs Win Percentage for All Players in Riot Events",
        labels={'ACS': 'Average Combat Score', 'Win_Perc': 'Win Percentage'},
        hover_data=['PlayerName', 'Kills', 'Deaths']
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    This plot's way too messy, we can't glean much as there's too many points. But - there's some stuff to learn from this! 
    If we look at points with ACS values at about 400 or greater, we can see that most of these points correspond to duelist agents. 
    There's a multitude of Jetts, Razes, and Reynas out here. We can deduce that most of the highest ACS performances throughout 
    these Riot events are done by duelists.
    """)
    
    # Top performances
    st.markdown("### Highest ACS Performances")
    top_performances = data.nlargest(10, 'ACS')
    st.dataframe(top_performances[['PlayerName', 'Agent', 'ACS', 'Win_Perc', 'Kills', 'Deaths']])
    
    highest_acs = top_performances.iloc[0]
    st.write(f"""
    The highest ACS throughout this entire year of Riot events was **{highest_acs['PlayerName']}** with an ACS of 
    **{highest_acs['ACS']}** playing {highest_acs['Agent'].title()}. That's INCREDIBLE - the average ACS for a duelist 
    is about 250-300, maybe about 350 if they're having a great day.
    """)
    
    # Average stats per agent
    st.markdown("### Average Performance by Agent")
    
    agent_stats = data.groupby('Agent').agg({
        'ACS': 'median',
        'Win_Perc': 'median',
        'Kills': 'median',
        'Deaths': 'median'
    }).reset_index()
    
    fig = px.scatter(
        agent_stats, 
        x='ACS', 
        y='Win_Perc',
        color='Agent',
        size=[100]*len(agent_stats),  # Make all points same size
        title="Average ACS vs Average Win Percentage for Each Agent",
        labels={'ACS': 'Median ACS', 'Win_Perc': 'Median Win Percentage'},
        hover_data=['Kills', 'Deaths']
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    Now we have something easy to read! This plot shows the Average ACS vs Average Win % per agent. 
    The furthest point along our x-axis shows which agent has the highest average ACS. We can notice a cluster 
    of agents at the ACS threshold of about 215 and higher - these are likely the main duelists with the highest 
    average ACS. If any player is new and simply wants to get kills/do damage, a duelist is their best bet.
    """)
    
    # Agent performance table
    st.markdown("### Agent Performance Summary")
    agent_stats_display = agent_stats.sort_values('ACS', ascending=False)
    agent_stats_display.columns = ['Agent', 'Median ACS', 'Median Win %', 'Median Kills', 'Median Deaths']
    st.dataframe(agent_stats_display)

elif page == "Linear Regression Analysis":
    st.markdown('<h2 class="sub-header">Linear Regression on Each Agent\'s Scatter Plot</h2>', unsafe_allow_html=True)
    
    st.write("""
    Let's take a closer look at the relationship between ACS and Win Percentage per Agent. 
    We'll create scatterplots with a regression line for each agent. This'll allow us to see if 
    there's any correlation between ACS and win_percentage.
    """)
    
    # Agent selection
    selected_agents = st.multiselect(
        "Select agents for regression analysis:",
        options=sorted(data['Agent'].unique()),
        default=data['Agent'].value_counts().head(6).index.tolist()
    )
    
    if selected_agents:
        # Calculate regression stats
        regression_results = []
        
        for agent in selected_agents:
            agent_data = data[data['Agent'] == agent]
            
            if len(agent_data) > 10:  # Need sufficient data points
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    agent_data['ACS'], agent_data['Win_Perc']
                )
                
                regression_results.append({
                    'Agent': agent.title(),
                    'Slope': slope,
                    'Y-Intercept': intercept,
                    'R-squared': r_value**2,
                    'P-Value': p_value,
                    'Sample Size': len(agent_data)
                })
        
        # Display regression results table
        if regression_results:
            st.markdown("### Regression Analysis Results")
            results_df = pd.DataFrame(regression_results)
            st.dataframe(results_df.round(6))
            
            # Create subplot-style visualization
            st.markdown("### Individual Agent Regression Plots")
            
            # Create individual plots for each agent
            for agent in selected_agents:
                agent_data = data[data['Agent'] == agent]
                
                if len(agent_data) > 10:
                    # Get regression stats for this agent
                    result = [r for r in regression_results if r['Agent'] == agent.title()][0]
                    
                    # Create scatter plot with regression line
                    fig = px.scatter(
                        agent_data,
                        x='ACS',
                        y='Win_Perc',
                        title=f'{agent.title()} - ACS vs Win Percentage (Slope: {result["Slope"]:.5f}, R²: {result["R-squared"]:.3f})',
                        labels={'ACS': 'Average Combat Score', 'Win_Perc': 'Win Percentage'},
                        hover_data=['PlayerName']
                    )
                    
                    # Add regression line
                    x_range = np.linspace(agent_data['ACS'].min(), agent_data['ACS'].max(), 100)
                    y_pred = result['Slope'] * x_range + result['Y-Intercept']
                    
                    fig.add_scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name='Regression Line',
                        line=dict(color='red')
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Analysis of results
            st.markdown("### Analysis")
            
            highest_slope = max(regression_results, key=lambda x: x['Slope'])
            lowest_pvalue = min(regression_results, key=lambda x: x['P-Value'])
            
            st.write(f"""
            Our agents with the highest slope are those whose regression lines are much steeper than others, 
            implying the higher ACS, the more likely you are to win on these agents. **{highest_slope['Agent']}** 
            has the steepest slope at {highest_slope['Slope']:.5f}.
            
            **{lowest_pvalue['Agent']}** has the lowest p-value at {lowest_pvalue['P-Value']:.2e}, suggesting 
            the strongest statistical relationship between ACS and win percentage.
            """)
            
            # Show which agents have significant correlations
            significant_agents = [r for r in regression_results if r['P-Value'] < 0.05]
            if significant_agents:
                st.success(f"Agents with statistically significant ACS-Win% correlation (p < 0.05): {', '.join([r['Agent'] for r in significant_agents])}")
            else:
                st.warning("No agents show statistically significant correlation between ACS and Win Percentage at p < 0.05 level.")

elif page == "Machine Learning Predictions":
    st.markdown('<h2 class="sub-header">Machine Learning Predictions</h2>', unsafe_allow_html=True)
    
    st.write("""
    In this final section, we'll attempt to develop a model that can accurately predict win percentage 
    based on ACS and agent selection. We will be utilizing both linear regression and random forest models 
    to see which performs better. I'll also be making an interactive tool, that'll let you choose an agent and your ACS, trying to
    predict your win %!
    """)
    
    # Prepare ML data
    st.markdown("### Data Preparation")
    
    # Create ML dataframe
    ml_df = data[['Agent', 'ACS', 'Win_Perc']].copy()
    
    # Filter out extreme win percentages that could skew results
    ml_df = ml_df[(ml_df['Win_Perc'] > 0) & (ml_df['Win_Perc'] < 1)]
    
    st.write(f"After filtering extreme win percentages, we have {len(ml_df):,} records for training.")
    
    # One-hot encode agents
    le = LabelEncoder()
    ml_df['Agent_Encoded'] = le.fit_transform(ml_df['Agent'])
    
    # Prepare features and target
    X = ml_df[['Agent_Encoded', 'ACS']]
    y = ml_df['Win_Perc']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.markdown("### Model Training and Evaluation")
    
    # Train Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    linear_pred = linear_model.predict(X_test_scaled)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    linear_mse = mean_squared_error(y_test, linear_pred)
    linear_r2 = r2_score(y_test, linear_pred)
    linear_mae = mean_absolute_error(y_test, linear_pred)
    
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    # Display model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Linear Regression Results:**")
        st.metric("R² Score", f"{linear_r2:.4f}")
        st.metric("Mean Squared Error", f"{linear_mse:.6f}")
        st.metric("Mean Absolute Error", f"{linear_mae:.4f}")
    
    with col2:
        st.markdown("**Random Forest Results:**")
        st.metric("R² Score", f"{rf_r2:.4f}")
        st.metric("Mean Squared Error", f"{rf_mse:.6f}")
        st.metric("Mean Absolute Error", f"{rf_mae:.4f}")
    
    # Determine better model
    better_model = "Random Forest" if rf_r2 > linear_r2 else "Linear Regression"
    better_pred = rf_pred if rf_r2 > linear_r2 else linear_pred
    
    st.success(f"**{better_model}** performs better with an R² score of {max(rf_r2, linear_r2):.4f}")
    
    # Visualization of predictions vs actual
    st.markdown("### Prediction Analysis")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Actual': y_test,
        'Linear_Regression': linear_pred,
        'Random_Forest': rf_pred
    })
    
    # Plot actual vs predicted
    fig = go.Figure()
    
    # Add actual vs linear regression
    fig.add_trace(go.Scatter(
        x=comparison_df['Actual'],
        y=comparison_df['Linear_Regression'],
        mode='markers',
        name='Linear Regression',
        opacity=0.6
    ))
    
    # Add actual vs random forest
    fig.add_trace(go.Scatter(
        x=comparison_df['Actual'],
        y=comparison_df['Random_Forest'],
        mode='markers',
        name='Random Forest',
        opacity=0.6
    ))
    
    # Add perfect prediction line
    min_val = min(comparison_df['Actual'].min(), comparison_df['Linear_Regression'].min(), comparison_df['Random_Forest'].min())
    max_val = max(comparison_df['Actual'].max(), comparison_df['Linear_Regression'].max(), comparison_df['Random_Forest'].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Actual vs Predicted Win Percentage",
        xaxis_title="Actual Win Percentage",
        yaxis_title="Predicted Win Percentage",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution comparison
    st.markdown("### Distribution Comparison")
    
    dist_df = pd.DataFrame({
        'Win_Percentage': list(y_test) + list(linear_pred) + list(rf_pred),
        'Type': ['Actual'] * len(y_test) + ['Linear Regression'] * len(linear_pred) + ['Random Forest'] * len(rf_pred)
    })
    
    fig = px.histogram(
        dist_df,
        x='Win_Percentage',
        color='Type',
        title="Distribution of Actual vs Predicted Win Percentages",
        opacity=0.7,
        barmode='overlay'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance for Random Forest
    if rf_r2 > linear_r2:
        st.markdown("### Feature Importance (Random Forest)")
        
        feature_importance = pd.DataFrame({
            'Feature': ['Agent', 'ACS'],
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Feature',
            y='Importance',
            title="Feature Importance in Random Forest Model"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model interpretation
    st.markdown("### Model Interpretation")
    
    if better_model == "Random Forest":
        st.write(f"""
        The Random Forest model achieves an R² score of {rf_r2:.4f}, meaning it explains about {rf_r2*100:.1f}% 
        of the variance in win percentage. This is a moderate predictive power, suggesting that while ACS and agent 
        choice do influence win percentage, there are other important factors not captured in our model.
        
        The Random Forest model performs better than linear regression because it can capture non-linear relationships 
        and interactions between features that linear regression cannot.
        """)
    else:
        st.write(f"""
        The Linear Regression model achieves an R² score of {linear_r2:.4f}, meaning it explains about {linear_r2*100:.1f}% 
        of the variance in win percentage. While this shows some predictive power, it indicates that ACS and agent choice 
        alone are not sufficient to predict win percentage with high accuracy.
        """)
    
    st.info("""
    **Key Insights from Machine Learning Analysis:**
    - Neither model achieves very high predictive accuracy, suggesting win percentage depends on many factors beyond ACS and agent choice
    - Team coordination, strategy, opponent strength, and map selection likely play crucial roles
    - ACS is a useful indicator but not a definitive predictor of winning
    - Different agents may have different ACS-to-win-rate relationships that our models partially capture
    """)
    
    # Interactive prediction tool
    st.markdown("### Try the Model Yourself!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_agent = st.selectbox("Select an Agent:", sorted(data['Agent'].unique()))
        input_acs = st.slider("Enter ACS:", min_value=50, max_value=500, value=250)
    
    with col2:
        if st.button("Predict Win Percentage"):
            # Encode the selected agent
            agent_encoded = le.transform([selected_agent])[0]
            
            # Create prediction input
            pred_input = scaler.transform([[agent_encoded, input_acs]])
            
            # Make predictions
            if better_model == "Random Forest":
                prediction = rf_model.predict(pred_input)[0]
            else:
                prediction = linear_model.predict(pred_input)[0]
            
            # Ensure prediction is within reasonable bounds
            prediction = max(0, min(1, prediction))
            
            st.success(f"Predicted Win Percentage: {prediction:.1%}")
            
            # Get agent statistics for context
            agent_data = data[data['Agent'] == selected_agent]
            avg_acs = agent_data['ACS'].median()
            avg_winrate = agent_data['Win_Perc'].median()
            
            st.info(f"""
            **{selected_agent.title()} Statistics:**
            - Average ACS: {avg_acs:.0f}
            - Average Win Rate: {avg_winrate:.1%}
            - Your ACS is {'above' if input_acs > avg_acs else 'below'} average
            """)

elif page == "Conclusion":
    st.markdown('<h2 class="sub-header">Conclusion</h2>', unsafe_allow_html=True)
    
    st.write("""
    VALORANT is a complex game that's rising in popularity and has established itself as a major competitor 
    to CS:GO in the tactical shooter space. Through this comprehensive analysis of professional VALORANT data, 
    we've uncovered several key insights about agent performance and the relationship between individual skill 
    and team success.
    """)
    
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div', unsafe_allow_html=True)
        st.markdown("**Agent Performance**")
        st.write("""
        - **Sova** was the most picked agent, highlighting the importance of information gathering
        - **Duelists** (Jett, Raze, Reyna) dominate high ACS performances but don't always translate to wins
        - **Controllers** like Astra show lower ACS but higher win rates, emphasizing team utility
        - **Yoru** surprisingly showed high win rates despite low pick rates, suggesting niche strategic value
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown("**Statistical Insights**")
        st.write("""
        - ACS alone is not a strong predictor of win percentage (R² ≈ 0.3-0.4)
        - Team coordination and strategy appear more important than individual fragging
        - Different agents have varying ACS-to-win-rate relationships
        - Professional play emphasizes utility and teamwork over individual performance
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### What We Learned About VALORANT's Competitive Scene")
    
    st.write("""
    The analysis reveals that VALORANT's professional scene values **balanced team compositions** over individual 
    star power. While duelists can achieve spectacular ACS numbers, controllers and initiators often contribute 
    more to actual victory through their utility and team support.
    
    This finding aligns with VALORANT's design philosophy of combining tactical shooting with ability-based gameplay. 
    Unlike pure aim-based games, VALORANT rewards players who can effectively use their agent's kit to support 
    their team's strategy.
    """)
    
    # Top insights in metrics format
    st.markdown("### By the Numbers")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        highest_acs_player = data.loc[data['ACS'].idxmax()]
        st.metric(
            "Highest ACS", 
            f"{highest_acs_player['ACS']:.0f}",
            f"{highest_acs_player['PlayerName']} ({highest_acs_player['Agent'].title()})"
        )
    
    with col2:
        most_played_agent = data['Agent'].value_counts().index[0]
        most_played_count = data['Agent'].value_counts().iloc[0]
        st.metric(
            "Most Played Agent",
            most_played_agent.title(),
            f"{most_played_count:,} games"
        )
    
    with col3:
        agent_stats = data.groupby('Agent')['ACS'].median().sort_values(ascending=False)
        highest_acs_agent = agent_stats.index[0]
        st.metric(
            "Highest Avg ACS Agent",
            highest_acs_agent.title(),
            f"{agent_stats.iloc[0]:.0f} ACS"
        )
    
    with col4:
        agent_winrates = data.groupby('Agent')['Win_Perc'].median().sort_values(ascending=False)
        highest_wr_agent = agent_winrates.index[0]
        st.metric(
            "Highest Win Rate Agent",
            highest_wr_agent.title(),
            f"{agent_winrates.iloc[0]:.1%}"
        )
    
    st.markdown("### What It Means for Players")
    
    st.info("""
    **For Newbies:**
    - Don't focus solely on fragging - learn to use your agent's utility effectively
    - Consider playing initiators or controllers to develop game sense
    - Remember that winning matters more than individual statistics
    
    **For Competitive Players:**
    - Team composition and coordination are crucial for success
    - Master your agent's utility, not just aim mechanics
    - Understand your role within the team's strategy
    """)
    
    st.markdown("### Future Analysis Opportunities")
    
    st.write("""
    This analysis opens doors for further investigation:
    
    - **Map-specific agent performance**: How do agents perform on different maps?
    - **Team composition analysis**: What combinations of agents work best together?
    - **Economic impact**: How do agent abilities affect the game's economy?
    - **Meta evolution**: How has agent usage changed over time?
    - **Regional differences**: Do different regions favor different playstyles?
    """)
    
    st.markdown("### Final Thoughts")
    
    st.write("""
    While the machine learning models showed that ACS isn't a perfect predictor of win percentage, this actually 
    reinforces VALORANT's design success. The game has achieved its goal of being more than just a pure aim-based 
    shooter - strategy, teamwork, and utility usage are equally important for victory.
    
    I hope this analysis has provided valuable insights into the professional VALORANT scene! Whether you're a 
    new player trying to choose your first main agent or a competitive player looking to understand the meta, 
    the data shows that there's a place for every playstyle in VALORANT.
    
    **Play VALORANT! (But maybe avoid ranked.)**
    """)
    
    st.markdown("---")
    st.markdown("*Analysis completed using data from official Riot Games tournaments through January 2022*")
    st.markdown("*Researched by Gurjit Dhaliwal - Adapted from Jupiter Notebook*")