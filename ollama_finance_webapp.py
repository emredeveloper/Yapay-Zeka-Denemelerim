from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from ollama_tools import OllamaFinance
import os
import pickle
import uuid
import json
import traceback
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize OllamaFinance
finance = OllamaFinance()

# Directory to store portfolios
PORTFOLIOS_DIR = "portfolios"
os.makedirs(PORTFOLIOS_DIR, exist_ok=True)

# Helper functions
def save_portfolio(portfolio):
    """Save portfolio to disk"""
    if not portfolio or 'id' not in portfolio:
        return False
    
    filename = os.path.join(PORTFOLIOS_DIR, f"{portfolio['id']}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(portfolio, f)
    
    # Also save in session
    if 'portfolios' not in session:
        session['portfolios'] = {}
    
    session['portfolios'][portfolio['id']] = portfolio['id']
    session.modified = True
    
    return True

def load_portfolio(portfolio_id):
    """Load portfolio from disk"""
    filename = os.path.join(PORTFOLIOS_DIR, f"{portfolio_id}.pkl")
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'rb') as f:
        portfolio = pickle.load(f)
    
    return portfolio

def get_user_portfolios():
    """Get list of user's portfolios"""
    if 'portfolios' not in session:
        return []
    
    portfolios = []
    for portfolio_id in session['portfolios'].values():
        portfolio = load_portfolio(portfolio_id)
        if portfolio:
            summary = finance.get_portfolio_summary(portfolio)
            portfolios.append(summary)
    
    return portfolios

# Routes
@app.route('/')
def index():
    return render_template('index.html', portfolios=get_user_portfolios())

@app.route('/stock/<ticker>')
def stock_details(ticker):
    period = request.args.get('period', '6mo')
    
    try:
        stock_data = finance.get_stock_data(ticker, period)
        stock_info = finance.get_stock_info(ticker)
        
        # Calculate technical indicators
        tech_data = finance.calculate_technical_indicators(ticker, period)
        
        # Get latest price
        latest_price = stock_data['Close'].iloc[-1] if not stock_data.empty else 0
        
        # Get AI analysis
        analysis = finance.analyze_stock(ticker, period)
        
        # Stock historical performance metrics
        if not stock_data.empty:
            returns = stock_data['Close'].pct_change().dropna()
            perf = {
                'return': ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100,
                'volatility': returns.std() * 100 * (252 ** 0.5),  # Annualized volatility
                'max_drawdown': (stock_data['Close'] / stock_data['Close'].cummax() - 1).min() * 100
            }
        else:
            perf = {'return': 0, 'volatility': 0, 'max_drawdown': 0}
        
        # Check if user has portfolios
        portfolios = get_user_portfolios()
        
        return render_template(
            'stock_details.html',
            ticker=ticker,
            price=latest_price,
            info=stock_info,
            analysis=analysis['analysis'],
            period=period,
            performance=perf,
            portfolios=portfolios
        )
    
    except Exception as e:
        traceback.print_exc()  # Debug için
        return render_template(
            'error.html', 
            message=f"Ticker '{ticker}' için veri yüklenirken hata oluştu: {str(e)}",
            suggestion="Lütfen hisse sembolünü kontrol edin ve tekrar deneyin. Örneğin, Tesla'nın sembolü 'TESLA' değil 'TSLA'dır."
        )

@app.route('/portfolio')
def portfolio():
    portfolios = get_user_portfolios()
    return render_template('portfolio.html', portfolios=portfolios)

@app.route('/portfolio/new', methods=['GET', 'POST'])
def new_portfolio():
    if request.method == 'GET':
        return redirect(url_for('portfolio'))
        
    cash = float(request.form.get('initial_cash', 10000))
    portfolio = finance.initialize_simulation_portfolio(cash)
    save_portfolio(portfolio)
    return redirect(url_for('portfolio_details', portfolio_id=portfolio['id']))

@app.route('/portfolio/<portfolio_id>')
def portfolio_details(portfolio_id):
    portfolio = load_portfolio(portfolio_id)
    if not portfolio:
        return redirect(url_for('portfolio'))
    
    summary = finance.get_portfolio_summary(portfolio)
    return render_template('portfolio_details.html', portfolio=summary, history=portfolio['history'])

@app.route('/trade/<portfolio_id>', methods=['GET', 'POST'])
def trade(portfolio_id):
    portfolio = load_portfolio(portfolio_id)
    if not portfolio:
        return redirect(url_for('portfolio'))
    
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper()
        action = request.form.get('action', 'buy')
        
        amount = None
        shares = None
        
        # Determine if trading by amount or shares
        trade_type = request.form.get('trade_type', 'amount')
        if trade_type == 'amount':
            try:
                amount = float(request.form.get('amount', 0))
            except ValueError:
                amount = 0
        else:
            try:
                shares = float(request.form.get('shares', 0))
            except ValueError:
                shares = 0
        
        # Execute trade
        try:
            result = finance.execute_simulated_trade(portfolio, ticker, action, amount, shares)
            
            if result['success']:
                save_portfolio(result['portfolio'])
                return render_template(
                    'trade_result.html',
                    ticker=ticker,
                    action=action,
                    result=result,
                    portfolio_id=portfolio_id
                )
            else:
                # Show error
                return render_template(
                    'trade.html',
                    ticker=ticker,
                    portfolio=finance.get_portfolio_summary(portfolio),
                    error=result['message']
                )
                
        except Exception as e:
            traceback.print_exc()  # Debug için
            return render_template(
                'trade.html',
                ticker=ticker,
                portfolio=finance.get_portfolio_summary(portfolio),
                error=f"İşlem gerçekleştirilemedi: {str(e)}"
            )
    
    # GET request - show trade form
    ticker = request.args.get('ticker', '')
    summary = finance.get_portfolio_summary(portfolio)
    
    return render_template('trade.html', portfolio=summary, ticker=ticker)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    return redirect(url_for('stock_details', ticker=query.upper()))

@app.route('/api/stock_chart/<ticker>')
def stock_chart_data(ticker):
    period = request.args.get('period', '6mo')
    
    try:
        data = finance.calculate_technical_indicators(ticker, period)
        
        # JSON için güvenli veri oluştur (NaN değerleri temizle)
        chart_data = {
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'close': [float(x) if pd.notna(x) else None for x in data['Close'].tolist()],
            'open': [float(x) if pd.notna(x) else None for x in data['Open'].tolist()],
            'high': [float(x) if pd.notna(x) else None for x in data['High'].tolist()],
            'low': [float(x) if pd.notna(x) else None for x in data['Low'].tolist()],
            'volume': [float(x) if pd.notna(x) else None for x in data['Volume'].tolist()]
        }
        return jsonify(chart_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': True,
            'message': f'Graf verisi alınırken hata oluştu: {str(e)}'
        })

@app.route('/api/analyze_trade', methods=['POST'])
def analyze_trade():
    ticker = request.form.get('ticker', '').upper()
    action = request.form.get('action', 'buy')
    try:
        amount = float(request.form.get('amount', 1000))
    except ValueError:
        amount = 1000
    
    try:
        # LLM isteklerini sadeleştirmek için analiz kısmını atla
        analysis = {
            'ticker': ticker,
            'price': 0,
            'action': action,
            'amount': amount,
            'analysis': f"{ticker} için {action} işlemi analiz ediliyor...",
            'plot': None
        }
        
        # Gerçek fiyatı almaya çalış
        try:
            data = finance.get_stock_data(ticker, '1d')
            if not data.empty:
                analysis['price'] = float(data['Close'].iloc[-1])
        except:
            pass
            
        return jsonify(analysis)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': True,
            'analysis': f"İşlem analizi yapılırken hata oluştu: {str(e)}"
        })

if __name__ == '__main__':
    # Make sure to import pandas here for use in the route function
    import pandas as pd
    app.run(debug=True)
