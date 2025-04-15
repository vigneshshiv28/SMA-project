import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Blinkit Brand Reputation Analysis",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00b300;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .kpi-box {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #00b300;
        padding: 10px;
        margin-bottom: 10px;
    }
    .negative {
        color: #ff4b4b;
    }
    .positive {
        color: #00b300;
    }
    .neutral {
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    # Import the two CSV files
    df1 = pd.read_csv('tweets/blinkit-tweets.csv')
    df2 = pd.read_csv('final_tweets/blinkit-final.csv')
    
    # Use the second file (paste-2.txt) as it contains sentiment data
    df = df2
    
    # If there are any columns in df1 that aren't in df2, you could merge them
    # But in this case, df2 seems to have all the columns from df1 plus sentiment data
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract date
    df['date'] = df['timestamp'].dt.date
    
    # Convert has_image based on image_urls
    df['has_image'] = df['image_urls'].notna() & (df['image_urls'] != '')
    
    return df

# Function to extract issues from tweets
# Function to extract issues from tweets
def extract_issues(df):
    # First, check if 'text' column exists
    if 'text' not in df.columns:
        # Check for alternative column names that might contain tweet text
        possible_text_columns = ['tweet_text', 'content', 'tweet', 'message']
        found = False
        for col in possible_text_columns:
            if col in df.columns:
                df['text'] = df[col]
                found = True
                break
        
        if not found:
            print("Warning: No text column found. Available columns:", df.columns.tolist())
            # Create an empty text column to avoid errors
            df['text'] = ""
    
    # Handle null values in the text column
    df['text'] = df['text'].fillna("")
    
    # Define issue categories and related keywords
    issue_categories = {
        'Product Quality': ['expired', 'spoilt', 'rotten', 'damaged', 'quality'],
        'Delivery Issues': ['delivery', 'missing', 'wrong item', 'wrong product', 'ordered', 'sent'],
        'Refund Problems': ['refund', 'money', 'return', 'payment'],
        'Customer Support': ['support', 'customer service', 'resolution', 'response', 'help'],
        'App/Technical': ['app', 'not working', 'technical', 'website', 'down', 'serving'],
    }
    
    # Initialize new columns for each issue category
    for category in issue_categories:
        df[category] = 0
    
    # Check for issues in each tweet
    for idx, row in df.iterrows():
        text = row['text'].lower()
        for category, keywords in issue_categories.items():
            if any(keyword.lower() in text for keyword in keywords):
                df.at[idx, category] = 1
    
    return df

# Load the data
df = load_data()
df = extract_issues(df)

# Dashboard Title
st.markdown("<h1 class='main-header'>Blinkit Brand Reputation Analysis Dashboard</h1>", unsafe_allow_html=True)

# Date filter
st.sidebar.markdown("## Date Range Filter")
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Filter data by date
filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

# Sentiment filter
st.sidebar.markdown("## Sentiment Filter")
sentiment_options = ['All'] + list(filtered_df['sentiment'].unique())
selected_sentiment = st.sidebar.multiselect("Select Sentiment", sentiment_options, default='All')

if 'All' not in selected_sentiment and selected_sentiment:
    filtered_df = filtered_df[filtered_df['sentiment'].isin(selected_sentiment)]

# Issue filter
st.sidebar.markdown("## Issue Type Filter")
issue_types = ['All', 'Product Quality', 'Delivery Issues', 'Refund Problems', 'Customer Support', 'App/Technical']
selected_issues = st.sidebar.multiselect("Select Issue Type", issue_types, default='All')

if 'All' not in selected_issues and selected_issues:
    # Filter by selected issues
    issue_filter = filtered_df[selected_issues].sum(axis=1) > 0
    filtered_df = filtered_df[issue_filter]

# Sidebar - About
st.sidebar.markdown("---")
st.sidebar.markdown("## About")
st.sidebar.info(
    """
    This dashboard analyzes customer sentiment and feedback for Blinkit 
    based on Twitter mentions. Identify common issues, track sentiment trends, 
    and gain insights to improve customer satisfaction.
    """
)

# Main dashboard
# Row 1 - KPIs
st.markdown("<h2 class='sub-header'>Key Performance Indicators</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
    st.metric(
        "Total Mentions", 
        len(filtered_df),
        delta=f"{len(filtered_df) - len(df)} from all time" if len(filtered_df) != len(df) else None
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    sentiment_counts = filtered_df['sentiment'].value_counts()
    sentiment_percentages = sentiment_counts / sentiment_counts.sum() * 100
    
    negative_pct = sentiment_percentages.get('negative', 0)
    st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
    st.metric(
        "Negative Sentiment", 
        f"{negative_pct:.1f}%",
        delta=f"{negative_pct - (df['sentiment'].value_counts().get('negative', 0) / len(df) * 100):.1f}%" if len(filtered_df) != len(df) else None,
        delta_color="inverse"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    avg_engagement = (filtered_df['likes'].sum() + filtered_df['retweets'].sum() + filtered_df['replies'].sum()) / len(filtered_df) if len(filtered_df) > 0 else 0
    st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
    st.metric(
        "Avg. Engagement per Tweet", 
        f"{avg_engagement:.1f}"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    most_common_issue = filtered_df[['Product Quality', 'Delivery Issues', 'Refund Problems', 'Customer Support', 'App/Technical']].sum().idxmax()
    issue_count = filtered_df[most_common_issue].sum()
    st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
    st.metric(
        "Top Issue", 
        f"{most_common_issue} ({issue_count})"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Row 2 - Sentiment Analysis
st.markdown("<h2 class='sub-header'>Sentiment Analysis</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Sentiment Distribution
    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Set color map
    color_map = {'positive': '#00b300', 'negative': '#ff4b4b', 'neutral': '#888888'}
    
    fig = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment',
        title='Sentiment Distribution',
        color='Sentiment',
        color_discrete_map=color_map,
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Sentiment Over Time
    sentiment_time = filtered_df.groupby([pd.Grouper(key='timestamp', freq='D'), 'sentiment']).size().reset_index(name='count')
    
    fig = px.line(
        sentiment_time, 
        x='timestamp', 
        y='count', 
        color='sentiment',
        title='Sentiment Trend Over Time',
        color_discrete_map=color_map
    )
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Tweets')
    st.plotly_chart(fig, use_container_width=True)

# Row 3 - Issue Analysis
st.markdown("<h2 class='sub-header'>Issue Analysis</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Issue Distribution
    issue_cols = ['Product Quality', 'Delivery Issues', 'Refund Problems', 'Customer Support', 'App/Technical']
    issue_counts = filtered_df[issue_cols].sum().reset_index()
    issue_counts.columns = ['Issue', 'Count']
    issue_counts = issue_counts.sort_values('Count', ascending=False)
    
    fig = px.bar(
        issue_counts,
        x='Issue',
        y='Count',
        title='Issue Distribution',
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Issues by Sentiment
    issue_sentiment = pd.DataFrame()
    
    for issue in issue_cols:
        temp_df = filtered_df[filtered_df[issue] == 1]
        if not temp_df.empty:
            sentiment_count = temp_df['sentiment'].value_counts().reset_index()
            sentiment_count.columns = ['sentiment', 'count']
            sentiment_count['issue'] = issue
            issue_sentiment = pd.concat([issue_sentiment, sentiment_count])
    
    if not issue_sentiment.empty:
        fig = px.bar(
            issue_sentiment,
            x='issue',
            y='count',
            color='sentiment',
            title='Issues by Sentiment',
            barmode='stack',
            color_discrete_map=color_map
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available for the selected filters.")

# Row 4 - Content Analysis
st.markdown("<h2 class='sub-header'>Content Analysis</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Word Cloud
    if not filtered_df.empty:
        # Combine all text
        all_text = ' '.join(filtered_df['text'].str.lower())
        
        # Remove mentions, URLs, and common stop words
        all_text = re.sub(r'@\w+', '', all_text)
        all_text = re.sub(r'http\S+', '', all_text)
        stop_words = ['and', 'the', 'is', 'in', 'it', 'to', 'for', 'with', 'from', 'letsblinkit']
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             colormap='viridis', max_words=100).generate(all_text)
        
        # Display word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write("No data available for the selected filters.")

with col2:
    # Most Engaged Tweets
    if not filtered_df.empty:
        filtered_df['total_engagement'] = filtered_df['likes'] + filtered_df['retweets'] + filtered_df['replies']
        top_tweets = filtered_df.sort_values('total_engagement', ascending=False).head(5)
        
        st.subheader("Most Engaged Tweets")
        for _, tweet in top_tweets.iterrows():
            engagement = tweet['likes'] + tweet['retweets'] + tweet['replies']
            sentiment_class = tweet['sentiment']
            
            st.markdown(f"""
            <div style='border-left: 5px solid {"#00b300" if sentiment_class == "positive" else "#ff4b4b" if sentiment_class == "negative" else "#888888"}; padding-left: 10px; margin-bottom: 10px;'>
                <p><strong>{tweet['author_name']}</strong> (@{tweet['author_handle']}) - {tweet['timestamp'].strftime('%Y-%m-%d %H:%M')}</p>
                <p>{tweet['text']}</p>
                <p style='color: #666;'>❤️ {tweet['likes']} | 🔄 {tweet['retweets']} | 💬 {tweet['replies']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No data available for the selected filters.")

# Row 5 - Key Insights
st.markdown("<h2 class='sub-header'>Key Insights</h2>", unsafe_allow_html=True)

# Calculate insights
total_tweets = len(filtered_df)
if total_tweets > 0:
    negative_percentage = filtered_df['sentiment'].value_counts().get('negative', 0) / total_tweets * 100
    
    # Top issues
    issue_cols = ['Product Quality', 'Delivery Issues', 'Refund Problems', 'Customer Support', 'App/Technical']
    top_issues = filtered_df[issue_cols].sum().sort_values(ascending=False).head(3)
    
    # Most engaged tweet
    filtered_df['engagement'] = filtered_df['likes'] + filtered_df['retweets'] + filtered_df['replies']
    most_engaged = filtered_df.loc[filtered_df['engagement'].idxmax()] if not filtered_df['engagement'].empty else None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown(f"**Sentiment Overview:**")
        st.markdown(f"* {negative_percentage:.1f}% of tweets have negative sentiment")
        st.markdown(f"* The most common negative topics are about: {', '.join(top_issues.index[:2])}")
        st.markdown(f"* The most engaged tweet ({most_engaged['engagement']} engagements) was about: '{most_engaged['text'][:100]}...'")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Recommendations based on the data
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("**Recommendations:**")
        
        # Customize recommendations based on top issues
        if 'Product Quality' in top_issues.index[:2]:
            st.markdown("* **Improve Quality Control:** Address issues with expired/damaged products, especially eggs, refrigerated items")
        
        if 'Refund Problems' in top_issues.index[:2]:
            st.markdown("* **Speed Up Refund Process:** Current 7-day refund policy causing frustration")
        
        if 'Customer Support' in top_issues.index[:2]:
            st.markdown("* **Enhance Customer Support:** More responsive and empathetic service agents")
        
        if 'Delivery Issues' in top_issues.index[:2]:
            st.markdown("* **Review Order Accuracy:** Ensure customers receive exactly what they ordered")
        
        # General recommendation
        st.markdown("* **Monitor Cold Chain:** Check warehouse refrigeration, especially in Yemalur, Bangalore")
        
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.write("No data available for the selected filters.")

# Row 6 - Detailed Data Table
st.markdown("<h2 class='sub-header'>Detailed Tweet Data</h2>", unsafe_allow_html=True)

# Prepare data for display
display_df = filtered_df[['author_name', 'author_handle', 'text', 'timestamp', 'sentiment', 'likes', 'retweets', 'replies']]
display_df = display_df.rename(columns={
    'author_name': 'Name',
    'author_handle': 'Handle',
    'text': 'Tweet Text',
    'timestamp': 'Date & Time',
    'sentiment': 'Sentiment',
    'likes': 'Likes',
    'retweets': 'Retweets',
    'replies': 'Replies'
})

# Show data table
st.dataframe(display_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Blinkit Brand Reputation Analysis Dashboard** | Data from Twitter API | Last updated: April 16, 2025")