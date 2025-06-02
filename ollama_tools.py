import yfinance as yf
import ollama
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import re
import numpy as np
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.express as px
import base64
import uuid

class OllamaFinance:
    def __init__(self, model="llama3.2:3b"):
        self.model = model
    
    def query_llm(self, prompt):
        """Query the Ollama LLM with a prompt."""
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']
    
    def get_stock_data(self, ticker, period="1mo"):
        """Get stock data for a specific ticker."""
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    
    def get_stock_info(self, ticker):
        """Get basic info about a stock."""
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    
    def analyze_stock(self, ticker, period="1mo"):
        """Get a simple analysis of a stock."""
        hist = self.get_stock_data(ticker, period)
        
        # Basic stats
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        change = (end_price - start_price) / start_price * 100
        
        # Get LLM to analyze the stock
        prompt = f"""
        Analyze the following stock data for {ticker}:
        Starting price: ${start_price:.2f}
        Ending price: ${end_price:.2f}
        Change: {change:.2f}%
        Maximum price: ${hist['High'].max():.2f}
        Minimum price: ${hist['Low'].min():.2f}
        Average volume: {hist['Volume'].mean():.0f}
        
        Provide a brief analysis of this stock's performance.
        """
        
        analysis = self.query_llm(prompt)
        return {
            'ticker': ticker,
            'start_price': start_price,
            'end_price': end_price,
            'change_percent': change,
            'analysis': analysis,
            'data': hist
        }
    
    def plot_stock(self, ticker, period="1mo"):
        """Plot stock price for a specific ticker."""
        hist = self.get_stock_data(ticker, period)
        plt.figure(figsize=(10, 6))
        plt.plot(hist.index, hist['Close'])
        plt.title(f"{ticker} Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def compare_stocks(self, tickers, period="1mo"):
        """Compare multiple stocks."""
        plt.figure(figsize=(12, 8))
        all_data = {}
        
        for ticker in tickers:
            hist = self.get_stock_data(ticker, period)
            # Normalize to compare percentage changes
            normalized = hist['Close'] / hist['Close'].iloc[0] * 100
            plt.plot(hist.index, normalized, label=ticker)
            all_data[ticker] = hist
        
        plt.title("Stock Price Comparison (Normalized)")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Get LLM to compare the stocks
        changes = []
        for ticker in tickers:
            hist = all_data[ticker]
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            change = (end_price - start_price) / start_price * 100
            changes.append(f"{ticker}: {change:.2f}%")
        
        prompt = f"""
        Compare the following stocks over the past {period}:
        {"".join([f"{change}\n" for change in changes])}
        
        Which stock performed best and why might that be?
        """
        
        comparison = self.query_llm(prompt)
        print(comparison)
    
    def extract_json_from_text(self, text):
        """Extract JSON from a text response that might contain additional text."""
        # Try to find JSON pattern with regex
        json_pattern = r'({[\s\S]*?})'
        json_matches = re.findall(json_pattern, text)
        
        # Try each potential JSON match
        for potential_json in json_matches:
            try:
                return json.loads(potential_json)
            except:
                continue
        
        # If no valid JSON found, create a fallback structure
        # Try to extract information manually
        action_map = {
            'price': 'get_price',
            'info': 'get_info',
            'analysis': 'analyze',
            'compare': 'compare',
            'plot': 'plot'
        }
        
        # Default values
        parsed = {
            'action': None,
            'tickers': [],
            'period': '1mo'
        }
        
        # Try to extract tickers (common stock symbols)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        tickers = re.findall(ticker_pattern, text)
        if tickers:
            parsed['tickers'] = list(set(tickers))
        
        # Try to extract action
        for key, action in action_map.items():
            if key.lower() in text.lower():
                parsed['action'] = action
                break
        
        # Try to extract time periods
        time_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        for period in time_periods:
            if period in text:
                parsed['period'] = period
                break
        
        # If we still don't have an action, default to get_price
        if not parsed['action'] and parsed['tickers']:
            parsed['action'] = 'get_price'
        
        return parsed
    
    def calculate_technical_indicators(self, ticker, period="6mo"):
        """Calculate technical indicators for a stock."""
        hist = self.get_stock_data(ticker, period)
        
        # Moving Averages
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        hist['MA200'] = hist['Close'].rolling(window=200).mean()
        
        # RSI (Relative Strength Index)
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA12'] - hist['EMA26']
        hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['MACD_Hist'] = hist['MACD'] - hist['Signal']
        
        # Bollinger Bands
        hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
        hist['BB_Upper'] = hist['BB_Middle'] + 2 * hist['Close'].rolling(window=20).std()
        hist['BB_Lower'] = hist['BB_Middle'] - 2 * hist['Close'].rolling(window=20).std()
        
        return hist
    
    def plot_technical_analysis(self, ticker, period="6mo"):
        """Plot technical analysis chart with indicators."""
        data = self.calculate_technical_indicators(ticker, period)
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], 
            high=data['High'],
            low=data['Low'], 
            close=data['Close'],
            name='Price'
        ))
        
        # Add Moving Averages
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='20 Day MA', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='50 Day MA', line=dict(color='orange')))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper',
                                line=dict(color='rgba(250, 0, 0, 0.5)', width=1)))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower',
                                line=dict(color='rgba(250, 0, 0, 0.5)', width=1),
                                fill='tonexty', fillcolor='rgba(250, 0, 0, 0.1)'))
        
        # Layout
        fig.update_layout(
            title=f"{ticker} Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=800,
            xaxis_rangeslider_visible=False
        )
        
        # Create subplot for RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', yaxis="y2"))
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        
        # Add secondary y-axis for RSI
        fig.update_layout(
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="right",
                range=[0, 100]
            )
        )
        
        fig.show()
        
        return data
    
    def get_stock_news(self, ticker, max_news=5):
        """Get latest news for a stock."""
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return "No news found for this ticker."
        
        # Limit the number of news items
        news = news[:max_news]
        
        results = []
        for item in news:
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown source')
            link = item.get('link', '#')
            published = datetime.fromtimestamp(item.get('providerPublishTime', 0))
            
            results.append(f"📰 {title}\n   📆 {published.strftime('%Y-%m-%d')} | 🗞️ {publisher}\n   🔗 {link}\n")
        
        # Ask LLM to summarize sentiment
        news_text = "\n".join([item.get('title', '') for item in news])
        prompt = f"""
        Analyze these news headlines for {ticker} and determine the overall sentiment (positive, negative, or neutral):
        {news_text}
        
        Provide a 2-3 sentence summary of the sentiment and potential market impact.
        """
        
        sentiment = self.query_llm(prompt)
        results.append(f"\n🧠 AI Sentiment Analysis:\n{sentiment}")
        
        return "\n".join(results)
    
    def calculate_portfolio_metrics(self, tickers, weights=None, period="1y"):
        """Calculate portfolio performance metrics."""
        if weights is None:
            # Equal weight if not specified
            weights = [1/len(tickers)] * len(tickers)
        
        if len(tickers) != len(weights):
            raise ValueError("Number of tickers must match number of weights")
        
        if abs(sum(weights) - 1.0) > 0.0001:
            raise ValueError("Weights must sum to 1")
        
        # Get data for all tickers
        all_data = {}
        for ticker in tickers:
            hist = self.get_stock_data(ticker, period)
            all_data[ticker] = hist['Close']
        
        # Combine into a dataframe
        df = pd.DataFrame(all_data)
        
        # Calculate daily returns
        returns = df.pct_change().dropna()
        
        # Portfolio metrics
        portfolio_return = sum(returns.mean() * weights) * 252  # Annualized return
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))  # Annualized volatility
        sharpe_ratio = portfolio_return / portfolio_vol  # Sharpe ratio (assuming risk-free rate of 0)
        
        # Calculate portfolio value over time (starting with $10,000)
        portfolio_value = (1 + returns.dot(weights)).cumprod() * 10000
        
        # Generate some basic stats for display
        metrics = {
            'tickers': tickers,
            'weights': weights,
            'annual_return': portfolio_return * 100,  # As percentage
            'annual_volatility': portfolio_vol * 100,  # As percentage
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': (portfolio_value / portfolio_value.cummax() - 1).min() * 100,  # As percentage
            'portfolio_values': portfolio_value
        }
        
        return metrics
    
    def plot_portfolio_performance(self, tickers, weights=None, period="1y"):
        """Plot portfolio performance."""
        metrics = self.calculate_portfolio_metrics(tickers, weights, period)
        
        # Plot portfolio value over time
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(metrics['portfolio_values'])
        plt.title('Portfolio Value Over Time ($10,000 Initial Investment)')
        plt.grid(True)
        
        # Create a summary text for the bottom subplot
        plt.subplot(2, 1, 2)
        plt.axis('off')
        summary = f"""
        Portfolio Metrics:
        Tickers: {', '.join(metrics['tickers'])}
        Weights: {[f'{w*100:.1f}%' for w in metrics['weights']]}
        
        Annual Return: {metrics['annual_return']:.2f}%
        Annual Volatility: {metrics['annual_volatility']:.2f}%
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Maximum Drawdown: {metrics['max_drawdown']:.2f}%
        """
        plt.text(0.1, 0.5, summary, fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        # Ask LLM for portfolio analysis
        prompt = f"""
        Analyze this investment portfolio:
        Tickers: {', '.join(metrics['tickers'])}
        Weights: {[f'{w*100:.1f}%' for w in metrics['weights']]}
        Annual Return: {metrics['annual_return']:.2f}%
        Annual Volatility: {metrics['annual_volatility']:.2f}%
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        
        Provide a brief analysis of this portfolio's performance, risk level, and suggestions for improvement.
        """
        
        analysis = self.query_llm(prompt)
        print(analysis)
        
        return metrics
    
    def natural_language_query(self, query):
        """Process natural language queries about stocks."""
        prompt = f"""
        You are a financial assistant. Extract the key information from this query:
        '{query}'
        
        Your response must be ONLY valid JSON with these fields:
        - action (get_price, get_info, analyze, compare, plot, technical, indicators, news, portfolio)
        - tickers (list of stock symbols)
        - period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        IMPORTANT: Return ONLY the JSON object without any other text or explanation.
        
        Example response:
        {{
          "action": "get_price",
          "tickers": ["AAPL"],
          "period": "1mo"
        }}
        """
        
        try:
            # Parse the command using LLM
            response = self.query_llm(prompt)
            print(f"LLM Response: {response[:100]}...")  # Debug output
            
            # Try to extract valid JSON from the response
            parsed = self.extract_json_from_text(response)
            
            if not parsed['action'] or not parsed['tickers']:
                return f"Couldn't understand your query. Please specify a stock ticker and what you want to know about it."
            
            # Execute the appropriate action
            if parsed['action'] == 'get_price':
                results = []
                for ticker in parsed['tickers']:
                    data = self.get_stock_data(ticker, parsed.get('period', '1mo'))
                    results.append(f"{ticker} latest price: ${data['Close'].iloc[-1]:.2f}")
                return "\n".join(results)
                
            elif parsed['action'] == 'get_info':
                results = []
                for ticker in parsed['tickers']:
                    info = self.get_stock_info(ticker)
                    results.append(f"{ticker} ({info.get('longName', ticker)}): {info.get('longBusinessSummary', 'No info available')[:300]}...")
                return "\n".join(results)
                
            elif parsed['action'] == 'analyze':
                results = []
                for ticker in parsed['tickers']:
                    analysis = self.analyze_stock(ticker, parsed.get('period', '1mo'))
                    results.append(f"{ticker} Analysis:\n{analysis['analysis']}")
                return "\n".join(results)
                
            elif parsed['action'] == 'compare':
                self.compare_stocks(parsed['tickers'], parsed.get('period', '1mo'))
                return "Comparison plotted and analyzed."
                
            elif parsed['action'] == 'plot':
                for ticker in parsed['tickers']:
                    self.plot_stock(ticker, parsed.get('period', '1mo'))
                return "Plot(s) generated."
            
            elif parsed['action'] == 'technical' or parsed['action'] == 'indicators':
                for ticker in parsed['tickers']:
                    self.plot_technical_analysis(ticker, parsed.get('period', '6mo'))
                return "Technical analysis chart(s) generated."
                
            elif parsed['action'] == 'news':
                results = []
                for ticker in parsed['tickers']:
                    news = self.get_stock_news(ticker)
                    results.append(f"--- {ticker} NEWS ---\n{news}")
                return "\n\n".join(results)
                
            elif parsed['action'] == 'portfolio':
                tickers = parsed['tickers']
                self.plot_portfolio_performance(tickers)
                return "Portfolio analysis complete."
                
            else:
                return "I couldn't understand the requested action."
                
        except Exception as e:
            return f"Error processing your request: {str(e)}"
    
    def analyze_buy_sell_decision(self, ticker, action="buy", amount=1000, period="3mo"):
        """Analyze whether a buy/sell decision is good based on technical indicators."""
        # Get technical data
        data = self.calculate_technical_indicators(ticker, period)
        
        # Extract latest values
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Calculate important signals
        price = latest['Close']
        price_change_1d = (latest['Close'] - prev['Close']) / prev['Close'] * 100
        
        # Basic technical signals
        above_ma50 = latest['Close'] > latest['MA50']
        above_ma200 = latest['Close'] > latest['MA200']
        ma_crossing = (prev['MA50'] < prev['MA200'] and latest['MA50'] > latest['MA200']) or \
                     (prev['MA50'] > prev['MA200'] and latest['MA50'] < latest['MA200'])
        
        rsi_overbought = latest['RSI'] > 70
        rsi_oversold = latest['RSI'] < 30
        
        macd_signal = (prev['MACD'] < prev['Signal'] and latest['MACD'] > latest['Signal']) or \
                      (prev['MACD'] > prev['Signal'] and latest['MACD'] < latest['Signal'])
        
        bollinger_upper_touch = latest['Close'] > latest['BB_Upper'] * 0.98
        bollinger_lower_touch = latest['Close'] < latest['BB_Lower'] * 1.02
        
        # Prepare data for LLM analysis
        prompt = f"""
        Analyze this potential {action.upper()} decision for {ticker} at ${price:.2f} with ${amount:.2f}:
        
        Technical Indicators:
        - Price change (1 day): {price_change_1d:.2f}%
        - Price above 50-day MA: {above_ma50}
        - Price above 200-day MA: {above_ma200}
        - MA crossing: {ma_crossing}
        - RSI: {latest['RSI']:.2f} (Overbought: {rsi_overbought}, Oversold: {rsi_oversold})
        - MACD signal change: {macd_signal}
        - Price near Bollinger upper band: {bollinger_upper_touch}
        - Price near Bollinger lower band: {bollinger_lower_touch}
        
        If this is a BUY decision, analyze if this seems like a good time to buy.
        If this is a SELL decision, analyze if this seems like a good time to sell.
        
        Provide a concise, actionable recommendation with supporting technical rationale.
        Rate the decision on a scale of 1-10, where 10 is extremely favorable.
        """
        
        analysis = self.query_llm(prompt)
        
        # Create a fig of the recent price with MA and Bollinger Bands for reference
        fig = go.Figure()
        recent_data = data[-60:]  # Last ~3 months of trading days
        
        fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], name='Price',
                                line=dict(color='black', width=1)))
        fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MA50'], name='50 MA',
                                line=dict(color='blue', width=1)))
        fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MA200'], name='200 MA',
                                line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['BB_Upper'], name='BB Upper',
                                line=dict(color='red', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['BB_Lower'], name='BB Lower',
                                line=dict(color='green', width=1, dash='dash')))
        
        # Highlight the buy/sell point
        fig.add_trace(go.Scatter(
            x=[recent_data.index[-1]],
            y=[price],
            mode='markers',
            marker=dict(color='red' if action == 'sell' else 'green', size=12, symbol='star'),
            name=f"{action.upper()} point"
        ))
        
        fig.update_layout(
            title=f"{ticker} - {action.upper()} Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            width=700,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        # Try to convert to base64, but provide fallback if kaleido isn't installed
        plot_base64 = None
        try:
            import kaleido
            img_bytes = BytesIO()
            fig.write_image(img_bytes, format="png")
            img_bytes.seek(0)
            plot_base64 = base64.b64encode(img_bytes.read()).decode('ascii')
        except (ImportError, ValueError):
            # If kaleido is not available, return None for the plot
            # This will be handled in the template
            pass
        
        return {
            'ticker': ticker,
            'price': price,
            'action': action,
            'amount': amount,
            'analysis': analysis,
            'plot': plot_base64
        }
    
    def initialize_simulation_portfolio(self, cash=10000):
        """Initialize a simulation portfolio."""
        return {
            'id': str(uuid.uuid4()),
            'cash': cash,
            'stocks': {},  # {ticker: {'shares': 0, 'avg_price': 0}}
            'history': [],  # List of transaction dicts
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'start_cash': cash
        }
    
    def execute_simulated_trade(self, portfolio, ticker, action, amount=None, shares=None):
        """Execute a simulated trade in the portfolio."""
        # Get current price
        current_data = self.get_stock_data(ticker, period="1d")
        current_price = current_data['Close'].iloc[-1]
        
        if portfolio is None:
            portfolio = self.initialize_simulation_portfolio()
        
        # Calculate number of shares or amount
        if action.lower() == 'buy':
            if amount is not None and amount > 0:
                # Buy by amount
                shares = amount / current_price
                cost = amount
            elif shares is not None and shares > 0:
                # Buy by shares
                cost = shares * current_price
                amount = cost
            else:
                return {'success': False, 'message': 'Invalid amount or shares for buy order'}
            
            # Check if enough cash
            if cost > portfolio['cash']:
                return {'success': False, 'message': 'Insufficient funds for this purchase'}
            
            # Execute buy
            if ticker not in portfolio['stocks']:
                portfolio['stocks'][ticker] = {'shares': 0, 'avg_price': 0, 'total_cost': 0}
            
            # Update position
            current_shares = portfolio['stocks'][ticker]['shares']
            current_cost = portfolio['stocks'][ticker]['total_cost']
            
            # New position details
            new_shares = current_shares + shares
            new_cost = current_cost + cost
            new_avg_price = new_cost / new_shares if new_shares > 0 else 0
            
            # Update portfolio
            portfolio['stocks'][ticker]['shares'] = new_shares
            portfolio['stocks'][ticker]['avg_price'] = new_avg_price
            portfolio['stocks'][ticker]['total_cost'] = new_cost
            portfolio['cash'] -= cost
            
        elif action.lower() == 'sell':
            # Check if we have the stock
            if ticker not in portfolio['stocks'] or portfolio['stocks'][ticker]['shares'] <= 0:
                return {'success': False, 'message': 'No shares available to sell'}
            
            available_shares = portfolio['stocks'][ticker]['shares']
            
            if shares is not None and shares > 0:
                # Sell specific number of shares
                if shares > available_shares:
                    return {'success': False, 'message': f'Insufficient shares. You have {available_shares:.2f} shares.'}
                
                sell_value = shares * current_price
                portfolio['stocks'][ticker]['shares'] -= shares
                
            elif amount is not None and amount > 0:
                # Sell by amount value
                max_sell = available_shares * current_price
                if amount > max_sell:
                    return {'success': False, 'message': f'Insufficient value. Max sell value is ${max_sell:.2f}'}
                
                shares_to_sell = amount / current_price
                if shares_to_sell > available_shares:
                    shares_to_sell = available_shares
                    sell_value = shares_to_sell * current_price
                else:
                    sell_value = amount
                
                portfolio['stocks'][ticker]['shares'] -= shares_to_sell
                shares = shares_to_sell
                
            else:
                return {'success': False, 'message': 'Invalid amount or shares for sell order'}
            
            # Update cash
            portfolio['cash'] += sell_value
            amount = sell_value
            
            # If all shares are sold, recalculate or cleanup
            if portfolio['stocks'][ticker]['shares'] <= 0.000001:  # Account for floating point errors
                portfolio['stocks'][ticker] = {'shares': 0, 'avg_price': 0, 'total_cost': 0}
            
        else:
            return {'success': False, 'message': 'Invalid action. Use "buy" or "sell"'}
        
        # Record transaction in history
        transaction = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'action': action,
            'shares': shares,
            'price': current_price,
            'amount': amount,
            'cash_after': portfolio['cash']
        }
        
        portfolio['history'].append(transaction)
        
        # Calculate performance
        portfolio_value = portfolio['cash']
        for stock, details in portfolio['stocks'].items():
            if details['shares'] > 0:
                try:
                    stock_data = self.get_stock_data(stock, period="1d")
                    current_stock_price = stock_data['Close'].iloc[-1]
                    portfolio_value += details['shares'] * current_stock_price
                except:
                    # If can't get current price, use last known price
                    portfolio_value += details['shares'] * details['avg_price']
        
        performance = (portfolio_value / portfolio['start_cash'] - 1) * 100
        
        # Generate analysis
        analysis = self.analyze_buy_sell_decision(ticker, action, amount)
        
        return {
            'success': True,
            'portfolio': portfolio,
            'transaction': transaction,
            'analysis': analysis,
            'performance': performance,
            'portfolio_value': portfolio_value,
            'message': f'Successfully {action}ed {shares:.2f} shares of {ticker} at ${current_price:.2f}'
        }
    
    def get_portfolio_summary(self, portfolio):
        """Get a summary of the current portfolio with latest prices."""
        if not portfolio:
            return None
        
        portfolio_value = portfolio['cash']
        holdings = []
        
        for ticker, details in portfolio['stocks'].items():
            if details['shares'] > 0:
                try:
                    current_data = self.get_stock_data(ticker, period="1d")
                    current_price = current_data['Close'].iloc[-1]
                    
                    position_value = details['shares'] * current_price
                    gain_loss = position_value - details['total_cost']
                    gain_loss_pct = (gain_loss / details['total_cost']) * 100 if details['total_cost'] > 0 else 0
                    
                    holdings.append({
                        'ticker': ticker,
                        'shares': details['shares'],
                        'avg_price': details['avg_price'],
                        'current_price': current_price,
                        'position_value': position_value,
                        'gain_loss': gain_loss,
                        'gain_loss_pct': gain_loss_pct
                    })
                    
                    portfolio_value += position_value
                except:
                    # If error, use average price
                    position_value = details['shares'] * details['avg_price']
                    holdings.append({
                        'ticker': ticker,
                        'shares': details['shares'],
                        'avg_price': details['avg_price'],
                        'current_price': details['avg_price'],
                        'position_value': position_value,
                        'gain_loss': 0,
                        'gain_loss_pct': 0
                    })
                    
                    portfolio_value += position_value
        
        # Calculate performance
        performance = (portfolio_value / portfolio['start_cash'] - 1) * 100
        
        return {
            'id': portfolio['id'],
            'cash': portfolio['cash'],
            'holdings': holdings,
            'total_value': portfolio_value,
            'performance': performance,
            'start_date': portfolio['start_date'],
            'start_cash': portfolio['start_cash'],
        }

# Enhanced example usage
if __name__ == "__main__":
    finance = OllamaFinance()
    
    print("OllamaFinance Tool - Financial Analysis with AI")
    print("================================================")
    print("Example commands:")
    print("- 'Show me Apple's stock price for the last month'")
    print("- 'Compare Tesla, Amazon and Microsoft for 6 months'")
    print("- 'Technical analysis for NVIDIA for 1 year'")
    print("- 'Get recent news for Google'")
    print("- 'Analyze my portfolio of AAPL, MSFT, GOOG, AMZN with equal weights'")
    print("- 'Type 'exit' to quit")
    print("================================================")
    
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() == 'exit':
            break
        
        result = finance.natural_language_query(user_input)
        print("\n" + result + "\n")
