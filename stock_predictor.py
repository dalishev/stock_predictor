import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import os
import joblib
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# ________________config_____________________________
GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY")

if not GUARDIAN_API_KEY:
    st.error("Please set the GUARDIAN_API_KEY environment variable to run this application.")
    st.stop()


TARGET_STOCKS = {
    'XLE': 'Energy ETF',
    'XOM': 'ExxonMobil',
    'CVX': 'Chevron',
    'SHEL': 'Shell',
    'BP': 'BP plc'
}
CACHE_FILE = 'oil_news_cache.csv'

#page layout
st.set_page_config(page_title="Energy Sector ML Predictor", layout="wide")

#_____________main functions___________________________
def fetch_guardian_news():
    """fetches up to 200 articles from the Guardian API."""
    url = "https://content.guardianapis.com/search"
    all_articles = []

    for page in range(1, 5): #4 pages of 50 = 200 articles
        params = {
            'q': "('crude oil' OR 'OPEC' OR 'brent' OR 'wti' OR 'oil prices') NOT 'cooking' NOT 'olive'",
            'section': 'business',
            'show-fields': 'bodyText,headline',
            'page-size': 50,
            'page': page,
            'api-key': GUARDIAN_API_KEY
        }
        response = requests.get(url, params=params).json()

        #extracting results
        results = response.get('response', {}).get('results', [])
        if not results:
            break

        for r in results:
            fields = r.get('fields', {})
            all_articles.append({
                'Date': r['webPublicationDate'],
                'Headline': fields.get('headline', ''),
                'Body': fields.get('bodyText', '')
            })

    df = pd.DataFrame(all_articles)
    #fix for timezones
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    #sorting newest to oldest
    df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
    return df

def get_stock_data(ticker, start_date, end_date):
    """downloads historical stock data and ensures every calendar day has a price."""
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.droplevel(1)
        
    #forward-fill for weekends/holidays so weekend articles map to Friday's closing price
    idx = pd.date_range(start=start_date, end=end_date)
    
    #getting rid of timezones
    if stock.index.tz is not None:
        stock.index = stock.index.tz_localize(None)
        
    stock = stock.reindex(idx).ffill()
    
    #daily percentage return
    stock['Return'] = stock['Close'].pct_change() * 100 
    stock = stock.dropna()
    
    return stock

#______streamlit UI_______________
st.title("🛢️ Energy Sector News ML Predictor")

#initializing session state
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
    st.session_state.results = {}
    st.session_state.target_article = None
    st.session_state.total_articles = 0

col1, col2 = st.columns([1, 3])
with col1:
    train_clicked = st.button("Train Model!", use_container_width=True)

status_text = st.empty()

#training pipeline(TRIGGERED ON BUTTON CLICK)
if train_clicked:
    st.session_state.is_trained = False
    
    #check for local cache of news articles
    status_text.write("⏳ Checking local cache or downloading Guardian articles...")
    if os.path.exists(CACHE_FILE):
        news_df = pd.read_csv(CACHE_FILE)
        news_df['Date'] = pd.to_datetime(news_df['Date'])
    else:
        news_df = fetch_guardian_news()
        news_df.to_csv(CACHE_FILE, index=False)
        
    #excluding current week and isolating the newest article
    status_text.write("🫩 Splitting data and isolating Verdict Target...")
    #the newewst article
    target_article = news_df.iloc[0] 
    
    #calculating cutoff date
    today = datetime.datetime.today()
    cutoff_date = today - datetime.timedelta(days=7)
    four_months_ago = today - datetime.timedelta(days=120)
    
    #training pool: older than 7 days, newer than 120 days
    train_pool = news_df[(news_df['Date'] < cutoff_date) & (news_df['Date'] >= four_months_ago)].copy()
    
    #1 pool, models training loop x 5
    results = {}
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.85, max_features=1000)
    
    status_text.write("Vectorizing text and training 5 separate company models...")
    
    X_train_tfidf = vectorizer.fit_transform(train_pool['Body'])
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    for ticker, name in TARGET_STOCKS.items():
        #getting stock data
        stock_df = get_stock_data(ticker, four_months_ago.date(), today.date())
        
        #map articles to stock returns by Date
        merged = pd.merge(train_pool, stock_df, left_on='Date', right_index=True, how='inner')
        
        #if we have enough overlap, train the model
        if len(merged) > 10: 
            
            X_mapped = vectorizer.transform(merged['Body'])
            y_mapped = merged['Return']
            
            #linear regression training
            model = LinearRegression()
            model.fit(X_mapped, y_mapped)
            
            #saving model locally(same folder)
            joblib.dump(model, f"{ticker}_model.pkl")
            
            #predicting target article
            target_vector = vectorizer.transform([target_article['Body']])
            target_prediction = model.predict(target_vector)[0]
            
            #predicts historical points for plotting colors
            hist_vectors = vectorizer.transform(merged['Body'])
            merged['Impact'] = model.predict(hist_vectors)
            
            #extracts top keywords
            coefs = model.coef_
            top_pos_idx = coefs.argsort()[-5:][::-1]
            top_neg_idx = coefs.argsort()[:5]
            
            results[ticker] = {
                'stock_df': stock_df,
                'merged_df': merged,
                'target_pred': target_prediction,
                'pos_words': feature_names[top_pos_idx],
                'neg_words': feature_names[top_neg_idx]
            }

    #save to state and finish
    st.session_state.results = results
    st.session_state.target_article = target_article
    st.session_state.is_trained = True
    st.session_state.total_articles = len(news_df)
    status_text.write("✅ Process complete!")

#_____________UI visualization____________
if not st.session_state.is_trained:
    st.warning("Please click 'Train Model'")
else:
    #post-training Summary
    st.info(f"📰 **Verdict Target Article (Recent):** {st.session_state.target_article['Headline']}")
    #scraping results
    st.caption(f"🥹 Successfully scraped and processed **{st.session_state.total_articles}** articles from The Guardian.")
    
    # tabs for different stocks
    tabs = st.tabs([f"🛢️ {ticker}" for ticker in TARGET_STOCKS.keys()])
    
    for tab, ticker in zip(tabs, TARGET_STOCKS.keys()):
        with tab:
            if ticker not in st.session_state.results:
                st.error("Insufficient data to train this specific model.")
                continue
                
            data = st.session_state.results[ticker]
            
            with st.expander(f"View {TARGET_STOCKS[ticker]} Machine Learning Analysis", expanded=True):
                
                #verdict metric
                st.subheader("Verdict on Most Recent Article")
                pred_val = data['target_pred']
                st.metric(
                    label=f"Expected Daily Move for {ticker}",
                    value=f"{pred_val:.2f}%",
                    delta="Positive Impact" if pred_val > 0 else "Negative Impact",
                    delta_color="normal"
                )
                
                st.divider()
                
                #PLOTLY interactive chart
                st.subheader("Historical Price vs. News Sentiment Impact")
                fig = go.Figure()
                
                #plot stock price lines
                fig.add_trace(go.Scatter(
                    x=data['stock_df'].index, 
                    y=data['stock_df']['Close'],
                    mode='lines',
                    name='Closing Price',
                    line=dict(color='white', width=2)
                ))
                
                #assigning colors based on impact
                impact_colors = []
                for imp in data['merged_df']['Impact']:
                    if imp > 0.1: impact_colors.append('green')
                    elif imp < -0.1: impact_colors.append('red')
                    else: impact_colors.append('grey')
                    
                #scattering articles
                fig.add_trace(go.Scatter(
                    x=data['merged_df']['Date'],
                    y=data['merged_df']['Close'],
                    mode='markers',
                    name='News Articles',
                    marker=dict(color=impact_colors, size=10, line=dict(width=1, color='black')),
                    text=data['merged_df']['Headline'],
                    hoverinfo='text'
                ))
                
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode='closest')
                st.plotly_chart(fig, use_container_width=True)
                
                #top keywords
                st.subheader("Model Coefficient Drivers")
                k_col1, k_col2 = st.columns(2)
                with k_col1:
                    st.markdown("**🟢 Top 5 Growth Keywords**")
                    for word in data['pos_words']:
                        st.write(f"- {word.capitalize()}")
                with k_col2:
                    st.markdown("**🔴 Top 5 Decline Keywords**")
                    for word in data['neg_words']:
                        st.write(f"- {word.capitalize()}")
                        
                        
# add emojis?
