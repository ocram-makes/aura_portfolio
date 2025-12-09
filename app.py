"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║           OTTIMIZZATORE PORTAFOGLIO - Streamlit Web App                       ║
║                     Basato su PyPortfolioOpt                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pypfopt import expected_returns, risk_models, EfficientFrontier, EfficientSemivariance, HRPOpt
from scipy.optimize import minimize, Bounds
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURAZIONE PAGINA
# ================================================================================
st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# COSTANTI E CONFIGURAZIONI
# ================================================================================

BENCHMARK_CANDIDATES = [
    {'ticker': 'SPY', 'name': 'S&P 500'},
    {'ticker': 'QQQ', 'name': 'NASDAQ 100'},
    {'ticker': 'VTI', 'name': 'Total US'},
    {'ticker': 'XLK', 'name': 'Tech SPDR'},
    {'ticker': 'SMH', 'name': 'Semiconductor'},
    {'ticker': 'VUG', 'name': 'Growth'},
]

DEFAULT_SECTOR_MAP = {
    'SXLK': 'Technology', 'XDWT': 'Technology', 'XLK': 'Technology', 
    'VGT': 'Technology', 'IYW': 'Technology',
    'CSNDX': 'NASDAQ', 'NQSE': 'NASDAQ', 'QQQ': 'NASDAQ', 'TQQQ': 'NASDAQ',
    'AIQ': 'AI', 'WTAI': 'AI', 'BOTZ': 'AI', 'ROBO': 'AI', 'IRBO': 'AI',
    'SMH': 'Semiconductor', 'SOXX': 'Semiconductor', 'FTXL': 'Semiconductor',
    'HNSC': 'Semiconductor', 'PSI': 'Semiconductor',
    'WTEC': 'Cloud', 'SKYY': 'Cloud', 'CLOU': 'Cloud', 
    'XNGI': 'Cloud', 'WCLD': 'Cloud',
    'SIXG': 'EmergingTech', 'QTUM': 'EmergingTech', 'ARKQ': 'EmergingTech',
    'CTEK': 'CleanEnergy', 'ICLN': 'CleanEnergy', 'TAN': 'CleanEnergy',
    'SEME': 'EmergingMarkets', 'EEM': 'EmergingMarkets', 'VWO': 'EmergingMarkets',
    'CIBR': 'Cybersecurity', 'HACK': 'Cybersecurity', 'BUG': 'Cybersecurity',
    'FINX': 'Fintech', 'ARKF': 'Fintech',
    'ARKG': 'HealthTech', 'IBB': 'HealthTech', 'XBI': 'HealthTech',
}

DEFAULT_SECTOR_LIMITS = {
    'Technology': 0.35, 'NASDAQ': 0.30, 'AI': 0.25, 'Semiconductor': 0.30,
    'Cloud': 0.25, 'EmergingTech': 0.20, 'CleanEnergy': 0.20,
    'EmergingMarkets': 0.15, 'Cybersecurity': 0.20, 'Fintech': 0.15,
    'HealthTech': 0.20, 'Other': 0.25,
}

DEFAULT_ETFS = """SXLK
XDWT
CSNDX
WTEC
NQSE
AIQ
WTAI
XNGI
SIXG
SEME
SMH
QTUM
FTXL
CTEK
HNSC"""

# ================================================================================
# GLOSSARIO FORMULE
# ================================================================================

GLOSSARIO = """
## 📚 GLOSSARIO DELLE FORMULE

### 1. RENDIMENTO MEDIO ANNUALIZZATO (Mean Return)
**Formula:** μ = (∏(1 + rᵢ))^(52/n) - 1

- rᵢ = rendimento settimanale al tempo i
- n = numero di settimane
- 52 = numero di settimane in un anno

**Scopo:** Misura il guadagno medio annuo atteso dall'investimento.

---

### 2. VOLATILITÀ ANNUALIZZATA (Volatility)
**Formula:** σ = σ_weekly × √52

- σ_weekly = deviazione standard dei rendimenti settimanali
- √52 = fattore di annualizzazione

**Scopo:** Misura il rischio totale del portafoglio. Volatilità alta = rischio alto.

---

### 3. SHARPE RATIO
**Formula:** SR = (μ - Rf) / σ

- μ = rendimento annualizzato del portafoglio
- Rf = tasso risk-free
- σ = volatilità annualizzata

**Scopo:** Misura il rendimento aggiustato per il rischio TOTALE.
- Valori > 1 sono buoni
- Valori > 2 eccellenti

---

### 4. DOWNSIDE DEVIATION (Semideviazione)
**Formula:** DD = √[Σ(min(rᵢ - Rf/52, 0))² / n] × √52

**Scopo:** Misura solo la volatilità "negativa", ovvero le oscillazioni al ribasso.

---

### 5. SORTINO RATIO
**Formula:** SoR = (μ - Rf) / DD

**Scopo:** Come Sharpe ma penalizza solo il rischio di ribasso.

---

### 6. MAXIMUM DRAWDOWN (MDD)
**Formula:** MDD = max[(Peak - Trough) / Peak] × 100

**Scopo:** Indica la massima perdita percentuale dal picco al minimo.

---

### 7. CALMAR RATIO
**Formula:** CR = μ / MDD

**Scopo:** Bilancia rendimento e peggior perdita storica.
- CR > 1 significa che il rendimento annuo supera il peggior drawdown
- CR > 3 è eccellente

---

### 8. BETA (β)
**Formula:** β = Cov(Rp, Rb) / Var(Rb)

- β=1: si muove come il mercato
- β>1: amplifica i movimenti del mercato
- β<1: meno volatile del mercato

---

### 9. TRACKING ERROR (TE)
**Formula:** TE = σ(Rp - Rb) × √52

**Scopo:** Misura quanto il portafoglio devia dal benchmark.

---

### 10. ALPHA (α)
**Formula:** α = Rp_ann - Rb_ann

**Scopo:** Misura l'extra-rendimento rispetto al benchmark.

---

### 11. INFORMATION RATIO (IR)
**Formula:** IR = α / TE

**Scopo:** Misura la capacità di generare alpha per unità di rischio attivo.

---

### 12. TREYNOR RATIO (TR)
**Formula:** TR = (μ - Rf) / β

**Scopo:** Misura il rendimento in eccesso per unità di rischio SISTEMATICO.

---

## 🎯 STRATEGIE DI OTTIMIZZAZIONE

### 1. MAX SHARPE (Mean-Variance Optimization)
- **Obiettivo:** Massimizzare il rapporto rendimento/rischio
- **Pro:** Approccio classico e intuitivo
- **Contro:** Sensibile a errori di stima nei rendimenti attesi

### 2. MAX SORTINO (Semi-Variance Optimization)
- **Obiettivo:** Massimizzare rendimento rispetto al rischio di ribasso
- **Pro:** Migliore per chi vuole minimizzare le perdite
- **Contro:** Richiede più dati per stime accurate

### 3. RISK PARITY
- **Obiettivo:** Ogni asset contribuisce ugualmente al rischio totale
- **Pro:** Portafoglio più diversificato
- **Contro:** Non massimizza il rendimento

### 4. HRP (Hierarchical Risk Parity)
- **Obiettivo:** Allocazione robusta basata su clustering gerarchico
- **Pro:** Robusto a errori di stima
- **Contro:** Può sottoperformare in mercati trending

---

## 🔒 VINCOLI DI OTTIMIZZAZIONE

### Vincolo Settoriale
**Formula:** Σ wᵢ ≤ max_sector_weight per ogni settore

**Scopo:** Evita la concentrazione eccessiva in un singolo settore.

### Vincolo di Volatilità (Volatility Cap)
**Formula:** σ_portfolio ≤ target_volatility

**Scopo:** Limita il rischio complessivo del portafoglio.
"""

# ================================================================================
# FUNZIONI DI DOWNLOAD E CACHING
# ================================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers, start_date, end_date):
    """Scarica e pulisce i dati storici degli ETF con caching."""
    all_data = {}
    failed = []
    
    for t in tickers:
        try:
            data = yf.Ticker(t).history(start=start_date, end=end_date, auto_adjust=True)
            if data.empty or len(data) < 50:
                for sfx in ['.L', '.DE', '.MI']:
                    try:
                        data = yf.Ticker(t + sfx).history(start=start_date, end=end_date, auto_adjust=True)
                        if not data.empty and len(data) >= 50:
                            break
                    except:
                        continue
            
            if data.empty or len(data) < 50:
                failed.append(t)
                continue
                
            prices = data['Close'].squeeze()
            prices.index = pd.to_datetime(prices.index.tz_localize(None) if prices.index.tz else prices.index).normalize()
            all_data[t] = prices
        except Exception as e:
            failed.append(t)
    
    return all_data, failed

@st.cache_data(ttl=3600, show_spinner=False)
def download_benchmark_data(ticker, start_date, end_date):
    """Scarica i dati del benchmark con caching."""
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
        if data.empty or len(data) < 50:
            return None, None
        
        prices = data['Close'].squeeze()
        prices.index = pd.to_datetime(prices.index.tz_localize(None) if prices.index.tz else prices.index).normalize()
        weekly = prices.resample('W').last().dropna()
        if isinstance(weekly, pd.DataFrame):
            weekly = weekly.squeeze()
        
        returns = weekly.pct_change().dropna() * 100
        return weekly, returns
    except:
        return None, None

# ================================================================================
# CLASSE BENCHMARK ANALYZER
# ================================================================================

class BenchmarkAnalyzer:
    def __init__(self, start_date, end_date, risk_free_rate=0.02):
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.benchmark_prices = {}
        self.benchmark_returns = {}
        self.benchmark_metrics = {}
        self.best_benchmark = None

    def download_benchmark_data(self, ticker):
        if ticker in self.benchmark_prices:
            return True
        
        prices, returns = download_benchmark_data(ticker, self.start_date, self.end_date)
        if prices is None:
            return False
        
        self.benchmark_prices[ticker] = prices
        self.benchmark_returns[ticker] = returns
        return len(returns) >= 20

    def calculate_metrics(self, ticker):
        if not self.download_benchmark_data(ticker):
            return None
        
        ret = self.benchmark_returns[ticker]
        prices = self.benchmark_prices[ticker]
        
        mu = float(expected_returns.mean_historical_return(
            pd.DataFrame({ticker: prices}), frequency=52).iloc[0]) * 100
        rd = ret / 100
        vol = float(rd.std() * np.sqrt(52) * 100)
        excess = mu - self.risk_free_rate * 100
        sharpe = excess / vol if vol > 0 else 0
        
        ds = rd[rd < self.risk_free_rate/52]
        dd = float(np.sqrt(((ds - self.risk_free_rate/52)**2).mean()) * np.sqrt(52) * 100) if len(ds) > 0 else vol * 0.7
        sortino = excess / dd if dd > 0 else 0
        
        cum = (1 + rd).cumprod()
        mdd = float(abs(((cum - cum.expanding().max()) / cum.expanding().max()).min()) * 100)
        
        self.benchmark_metrics[ticker] = {
            'ticker': ticker,
            'mean_return': mu,
            'volatility': vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': mdd,
            'calmar': mu/mdd if mdd > 0 else 0
        }
        return self.benchmark_metrics[ticker]

    def find_best(self, port_returns, port_metrics):
        results = []
        
        for b in BENCHMARK_CANDIDATES:
            m = self.calculate_metrics(b['ticker'])
            if not m:
                continue
            
            br = self.benchmark_returns[b['ticker']].copy()
            pr = port_returns.copy()
            pr.index = pd.to_datetime(pr.index).normalize()
            br.index = pd.to_datetime(br.index).normalize()
            common = pr.index.intersection(br.index)
            
            if len(common) < 20:
                continue
            
            corr = float(pr.loc[common].corr(br.loc[common]))
            te = float((pr.loc[common] - br.loc[common]).std() * np.sqrt(52))
            cov = np.cov(pr.loc[common]/100, br.loc[common]/100)
            beta = cov[0,1]/cov[1,1] if cov[1,1] > 0 else 1.0
            
            score = corr * 0.5 + (1/(1+te/10)) * 0.35 + (1/(1+abs(m['volatility']-port_metrics['vol'])/10)) * 0.15
            results.append({**m, 'name': b['name'], 'correlation': corr, 
                          'tracking_error': te, 'beta': beta, 'score': score})
        
        if not results:
            return None
        
        self.best_benchmark = max(results, key=lambda x: x['score'])
        return self.best_benchmark

# ================================================================================
# CLASSE PORTFOLIO OPTIMIZER
# ================================================================================

class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, min_weight=0.01,
                 risk_free_rate=0.02, max_concentration=0.25,
                 sector_map=None, sector_limits=None, target_volatility=None):
        
        self.tickers = [t.upper() for t in tickers]
        self.n_assets = len(tickers)
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.min_weight = min_weight
        self.max_concentration = max_concentration
        self.risk_free_rate = risk_free_rate
        self.prices = None
        self.returns = None
        self.mu = None
        self.S = None
        self.bench = BenchmarkAnalyzer(start_date, self.end_date, risk_free_rate)
        self.best_benchmark = None
        self.results = {}
        
        self.sector_map = sector_map if sector_map else DEFAULT_SECTOR_MAP.copy()
        
        if sector_limits is False:
            self.sector_limits = None
            self.use_sector_constraints = False
        elif sector_limits is None:
            self.sector_limits = DEFAULT_SECTOR_LIMITS.copy()
            self.use_sector_constraints = True
        else:
            self.sector_limits = sector_limits
            self.use_sector_constraints = True
        
        self.target_volatility = target_volatility
        self.use_volatility_constraint = target_volatility is not None

    def download_data(self):
        all_data, failed = download_data(self.tickers, self.start_date, self.end_date)
        
        self.tickers = list(all_data.keys())
        self.n_assets = len(self.tickers)
        
        if self.n_assets < 2:
            return False, failed
        
        df = pd.DataFrame(all_data).ffill(limit=5).bfill(limit=5).dropna()
        self.prices = df.resample('W').last().dropna()
        self.returns = self.prices.pct_change().dropna() * 100
        self.mu = expected_returns.mean_historical_return(self.prices, frequency=52)
        self.S = risk_models.sample_cov(self.prices, frequency=52)
        
        return True, failed

    def _build_sector_mapper(self):
        return {ticker: self.sector_map.get(ticker, 'Other') for ticker in self.tickers}

    def _get_active_sectors(self):
        sector_mapper = self._build_sector_mapper()
        sectors = {}
        for ticker, sector in sector_mapper.items():
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)
        return sectors

    def _apply_sector_constraints(self, ef):
        if not self.use_sector_constraints:
            return ef
        
        sector_mapper = self._build_sector_mapper()
        sector_lower = {}
        sector_upper = {}
        active_sectors = set(sector_mapper.values())
        
        for sector in active_sectors:
            sector_lower[sector] = 0
            sector_upper[sector] = self.sector_limits.get(sector, self.sector_limits.get('Other', 1.0))
        
        try:
            ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        except:
            pass
        
        return ef

    def stats(self, w):
        pr = (self.returns * w).sum(axis=1) / 100
        tot = (1 + pr).prod()
        n = len(pr) / 52
        ret = (tot ** (1/n) - 1) * 100 if n > 0 else 0
        vol = float(pr.std() * np.sqrt(52) * 100)
        excess = ret - self.risk_free_rate * 100
        sharpe = excess / vol if vol > 0 else 0
        
        ds = pr[pr < self.risk_free_rate/52]
        dd = float(np.sqrt(((ds - self.risk_free_rate/52)**2).mean()) * np.sqrt(52) * 100) if len(ds) > 0 else vol * 0.7
        sortino = excess / dd if dd > 0 else 0
        
        cum = (1 + pr).cumprod()
        mdd = float(abs(((cum - cum.expanding().max()) / cum.expanding().max()).min()) * 100)
        
        return {
            'ret': ret, 'vol': vol, 'sharpe': sharpe,
            'sortino': sortino, 'mdd': mdd,
            'calmar': ret/mdd if mdd > 0 else 0
        }

    def optimize_max_sharpe(self):
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=(self.min_weight, self.max_concentration))
        
        if self.use_sector_constraints:
            ef = self._apply_sector_constraints(ef)
        
        if self.use_volatility_constraint:
            try:
                ef.efficient_risk(target_volatility=self.target_volatility)
            except:
                ef = EfficientFrontier(self.mu, self.S, weight_bounds=(self.min_weight, self.max_concentration))
                if self.use_sector_constraints:
                    ef = self._apply_sector_constraints(ef)
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        
        w = np.array([ef.clean_weights().get(t, 0) for t in self.tickers])
        s = self.stats(w)
        self.results['sharpe'] = {'weights': w, **s}
        return w

    def optimize_max_sortino(self):
        es = EfficientSemivariance(self.mu, self.prices.pct_change().dropna(), frequency=52,
                                   weight_bounds=(self.min_weight, self.max_concentration))
        
        if self.use_sector_constraints:
            es = self._apply_sector_constraints(es)
        
        if self.use_volatility_constraint:
            try:
                target_semidev = self.target_volatility * 0.8
                es.efficient_risk(target_semideviation=target_semidev)
            except:
                es = EfficientSemivariance(self.mu, self.prices.pct_change().dropna(), frequency=52,
                                          weight_bounds=(self.min_weight, self.max_concentration))
                if self.use_sector_constraints:
                    es = self._apply_sector_constraints(es)
                es.max_quadratic_utility(risk_aversion=1)
        else:
            es.max_quadratic_utility(risk_aversion=1)
        
        w = np.array([es.clean_weights().get(t, 0) for t in self.tickers])
        s = self.stats(w)
        self.results['sortino'] = {'weights': w, **s}
        return w

    def optimize_risk_parity(self):
        cov = self.S.values * 10000
        cov_annual = self.S.values
        
        def obj(w):
            pv = np.sqrt(w.T @ cov @ w)
            if pv < 1e-10:
                return 1e10
            rc = w * (cov @ w) / pv
            return np.sum((rc - pv/self.n_assets)**2)
        
        iv = 1 / np.sqrt(np.diag(cov))
        init = iv / iv.sum()
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if self.use_sector_constraints:
            active_sectors = self._get_active_sectors()
            for sector, tickers_in_sector in active_sectors.items():
                max_weight = self.sector_limits.get(sector, self.sector_limits.get('Other', 1.0))
                indices = [self.tickers.index(t) for t in tickers_in_sector if t in self.tickers]
                if indices:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=indices, mw=max_weight: mw - np.sum(w[idx])
                    })
        
        if self.use_volatility_constraint:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.target_volatility**2 - (w.T @ cov_annual @ w)
            })
        
        res = minimize(obj, init, method='SLSQP',
                      bounds=Bounds([self.min_weight]*self.n_assets, [self.max_concentration]*self.n_assets),
                      constraints=constraints)
        
        w = res.x
        s = self.stats(w)
        self.results['rp'] = {'weights': w, **s}
        return w

    def optimize_hrp(self):
        hrp = HRPOpt(self.prices.pct_change().dropna())
        hrp.optimize()
        w = np.array([hrp.clean_weights().get(t, 0) for t in self.tickers])
        
        if self.use_volatility_constraint:
            cov_annual = self.S.values
            port_vol = np.sqrt(w.T @ cov_annual @ w)
            if port_vol > self.target_volatility:
                scale_factor = self.target_volatility / port_vol
                w = w * scale_factor
                w = w / w.sum()
        
        s = self.stats(w)
        self.results['hrp'] = {'weights': w, **s}
        return w

    def find_benchmark(self):
        eq = np.array([1/self.n_assets] * self.n_assets)
        pr = (self.returns * eq).sum(axis=1)
        s = self.stats(eq)
        self.best_benchmark = self.bench.find_best(pr, {'vol': s['vol'], 'ret': s['ret']})

    def calc_bench_metrics(self):
        if not self.best_benchmark:
            return
        
        br = self.bench.benchmark_returns[self.best_benchmark['ticker']].copy()
        br.index = pd.to_datetime(br.index).normalize()
        
        for name, data in self.results.items():
            pr = (self.returns * data['weights']).sum(axis=1)
            pr.index = pd.to_datetime(pr.index).normalize()
            common = pr.index.intersection(br.index)
            
            if len(common) < 20:
                continue
            
            pa, ba = pr.loc[common], br.loc[common]
            cov = np.cov(pa/100, ba/100)
            beta = cov[0,1]/cov[1,1] if cov[1,1] > 0 else 1.0
            te = float((pa - ba).std() * np.sqrt(52))
            
            n = len(pa)/52
            pc = ((1+pa/100).prod()**(1/n)-1)*100
            bc = ((1+ba/100).prod()**(1/n)-1)*100
            alpha = pc - bc
            ir = alpha/te if te > 0 else 0
            
            excess_return = data['ret'] - self.risk_free_rate * 100
            treynor = excess_return / beta if beta != 0 else 0
            calmar = data['ret'] / data['mdd'] if data['mdd'] > 0 else 0
            
            self.results[name].update({
                'beta': beta, 'te': te, 'alpha': alpha,
                'ir': ir, 'treynor': treynor, 'calmar': calmar,
                'n_weeks': len(common)
            })

    def run_all_optimizations(self):
        self.optimize_max_sharpe()
        self.optimize_max_sortino()
        self.optimize_risk_parity()
        self.optimize_hrp()
        self.find_benchmark()
        self.calc_bench_metrics()

# ================================================================================
# FUNZIONI DI PLOTTING
# ================================================================================

def plot_frontier(optimizer):
    pts = []
    for t in np.linspace(float(optimizer.mu.min()), float(optimizer.mu.max()), 40):
        try:
            ef = EfficientFrontier(optimizer.mu, optimizer.S, 
                                  weight_bounds=(optimizer.min_weight, optimizer.max_concentration))
            ef.efficient_return(t)
            w = np.array([ef.clean_weights().get(tk, 0) for tk in optimizer.tickers])
            s = optimizer.stats(w)
            pts.append((s['vol'], s['ret'], s['sharpe']))
        except:
            pass
    
    if not pts:
        return None
    
    v, r, sh = zip(*pts)
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(v, r, c=sh, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')
    
    colors = {'sharpe': ('red', '*'), 'sortino': ('purple', 'P'), 
              'rp': ('orange', 'D'), 'hrp': ('cyan', 'H')}
    
    for name, data in optimizer.results.items():
        c, m = colors.get(name, ('gray', 'o'))
        ax.scatter(data['vol'], data['ret'], c=c, s=300, marker=m,
                  edgecolors='black', linewidth=2, label=name.upper(), zorder=10)
    
    ax.set_xlabel('Volatilità Annualizzata (%)')
    ax.set_ylabel('Rendimento Annualizzato (%)')
    ax.set_title('FRONTIERA EFFICIENTE\n(Ogni punto = portafoglio ottimale per quel livello di rischio)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_cumulative(optimizer):
    fig, ax = plt.subplots(figsize=(12, 7))
    norm = optimizer.prices / optimizer.prices.iloc[0] * 100
    
    colors = {'sharpe': '#2ecc71', 'sortino': '#9b59b6', 'rp': '#f39c12', 'hrp': '#00bcd4'}
    
    for name, data in optimizer.results.items():
        ax.plot((norm * data['weights']).sum(axis=1),
               color=colors.get(name, 'gray'), linewidth=2, label=name.upper())
    
    if optimizer.best_benchmark and optimizer.best_benchmark['ticker'] in optimizer.bench.benchmark_prices:
        bp = optimizer.bench.benchmark_prices[optimizer.best_benchmark['ticker']]
        common = norm.index.intersection(bp.index)
        if len(common) > 0:
            ax.plot(bp.loc[common]/bp.loc[common].iloc[0]*100,
                   'b--', linewidth=2, label=f"Bench:{optimizer.best_benchmark['ticker']}")
    
    ax.axhline(100, color='gray', linestyle=':', label='Base 100')
    ax.set_xlabel('Data')
    ax.set_ylabel('Valore Portafoglio (Base 100€)')
    ax.set_title('RENDIMENTI CUMULATIVI\n(Evoluzione di 100€ investiti)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_allocation(optimizer):
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    colors = {'sharpe': '#2ecc71', 'sortino': '#9b59b6', 'rp': '#f39c12', 'hrp': '#00bcd4'}
    
    for idx, (name, data) in enumerate(optimizer.results.items()):
        ax = axes[idx]
        si = np.argsort(data['weights'])[::-1]
        tks = [optimizer.tickers[i] for i in si if data['weights'][i] > 0.005]
        wts = [data['weights'][i]*100 for i in si if data['weights'][i] > 0.005]
        ax.barh(tks, wts, color=colors.get(name, 'gray'))
        ax.set_xlabel('Peso (%)')
        ax.set_title(f"{name.upper()}\n(Sharpe: {data['sharpe']:.2f})")
    
    plt.suptitle('ALLOCAZIONI PER STRATEGIA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_drawdown(optimizer):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'sharpe': '#2ecc71', 'sortino': '#9b59b6', 'rp': '#f39c12', 'hrp': '#00bcd4'}
    
    for name, data in optimizer.results.items():
        pr = (optimizer.returns * data['weights']).sum(axis=1)/100
        cum = (1+pr).cumprod()
        dd = (cum - cum.expanding().max())/cum.expanding().max()*100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=colors.get(name, 'gray'))
        ax.plot(dd.index, dd.values, color=colors.get(name, 'gray'),
               label=f"{name.upper()} (Max: {dd.min():.1f}%)")
    
    ax.set_xlabel('Data')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('DRAWDOWN NEL TEMPO\n(Perdita dal massimo storico)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_risk_return_map(optimizer):
    if not optimizer.best_benchmark:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = {
        'sharpe': ('#2ecc71', '*', 'MAX SHARPE'),
        'sortino': ('#9b59b6', 'P', 'MAX SORTINO'),
        'rp': ('#f39c12', 'D', 'RISK PARITY'),
        'hrp': ('#00bcd4', 'H', 'HRP')
    }
    
    ax.scatter(0, optimizer.risk_free_rate * 100, c='gray', s=200, marker='s',
              edgecolors='black', linewidth=2, 
              label=f'Risk-Free ({optimizer.risk_free_rate*100:.1f}%)', zorder=5)
    
    b = optimizer.best_benchmark
    ax.scatter(b['volatility'], b['mean_return'], c='blue', s=400, marker='X',
              edgecolors='black', linewidth=2, label=f"BENCHMARK ({b['ticker']})", zorder=10)
    
    if b['sharpe'] > 0:
        vol_range = np.linspace(0, max(b['volatility'] * 1.5, 40), 100)
        cml_returns = optimizer.risk_free_rate * 100 + b['sharpe'] * vol_range
        ax.plot(vol_range, cml_returns, 'b--', alpha=0.5, linewidth=1.5,
               label=f"CML (Sharpe Bench = {b['sharpe']:.2f})")
    
    for name, data in optimizer.results.items():
        c, m, label = colors.get(name, ('gray', 'o', name.upper()))
        ax.scatter(data['vol'], data['ret'], c=c, s=350, marker=m,
                  edgecolors='black', linewidth=2, label=label, zorder=10)
    
    ax.set_xlabel('Volatilità Annualizzata (%)', fontsize=12)
    ax.set_ylabel('Rendimento Annualizzato (%)', fontsize=12)
    ax.set_title('MAPPA RISK-RETURN vs BENCHMARK', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ================================================================================
# SIDEBAR - INPUT UTENTE
# ================================================================================

st.sidebar.markdown("## ⚙️ Configurazione")

# Input ETF
st.sidebar.markdown("### 📋 Lista ETF")
etf_input = st.sidebar.text_area(
    "Inserisci i ticker (uno per riga):",
    value=DEFAULT_ETFS,
    height=200,
    help="Inserisci i ticker degli ETF da analizzare, uno per ogni riga"
)

# Date
st.sidebar.markdown("### 📅 Periodo di Analisi")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Data Inizio",
        value=datetime(2022, 1, 1),
        min_value=datetime(2010, 1, 1),
        max_value=datetime.today() - timedelta(days=365)
    )
with col2:
    end_date = st.date_input(
        "Data Fine",
        value=datetime.today(),
        min_value=start_date + timedelta(days=365),
        max_value=datetime.today()
    )

# Parametri Base
st.sidebar.markdown("### 💰 Parametri Base")
risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0, max_value=10.0, value=3.7, step=0.1,
    help="Tasso dei titoli di stato (es. BTP)"
) / 100

min_weight = st.sidebar.slider(
    "Peso Minimo per ETF (%)",
    min_value=0, max_value=10, value=1, step=1,
    help="Allocazione minima per ogni ETF"
) / 100

max_concentration = st.sidebar.slider(
    "Peso Massimo per ETF (%)",
    min_value=10, max_value=50, value=25, step=5,
    help="Allocazione massima per ogni ETF"
) / 100

# Vincoli Avanzati
st.sidebar.markdown("### 🔒 Vincoli Avanzati")

use_sector_constraints = st.sidebar.checkbox(
    "Abilita Vincoli Settoriali",
    value=True,
    help="Limita l'esposizione massima per settore"
)

use_volatility_cap = st.sidebar.checkbox(
    "Abilita Volatility Cap",
    value=True,
    help="Limita la volatilità massima del portafoglio"
)

target_volatility = None
if use_volatility_cap:
    target_volatility = st.sidebar.slider(
        "Volatilità Target Massima (%)",
        min_value=10, max_value=40, value=22, step=1,
        help="Volatilità annuale massima consentita"
    ) / 100

# Pulsante di esecuzione
run_button = st.sidebar.button("🚀 ESEGUI OTTIMIZZAZIONE", type="primary", use_container_width=True)

# ================================================================================
# MAIN CONTENT
# ================================================================================

st.markdown('<h1 class="main-header">📊 Portfolio Optimizer Pro</h1>', unsafe_allow_html=True)
st.markdown("---")

# Tabs principali
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Metriche & Risultati", 
    "📊 Grafici", 
    "💼 Allocazioni",
    "📋 Confronto Benchmark",
    "📚 Glossario"
])

# Tab Glossario (sempre disponibile)
with tab5:
    st.markdown(GLOSSARIO)

# Esecuzione principale
if run_button:
    # Parse ETF list
    tickers = [t.strip().upper() for t in etf_input.strip().split('\n') if t.strip()]
    
    if len(tickers) < 2:
        st.error("❌ Inserisci almeno 2 ETF per l'ottimizzazione!")
    else:
        with st.spinner("🔄 Download dati e ottimizzazione in corso..."):
            # Crea optimizer
            sector_limits_param = DEFAULT_SECTOR_LIMITS if use_sector_constraints else False
            
            optimizer = PortfolioOptimizer(
                tickers=tickers,
                start_date=str(start_date),
                end_date=str(end_date),
                min_weight=min_weight,
                max_concentration=max_concentration,
                risk_free_rate=risk_free_rate,
                sector_limits=sector_limits_param,
                target_volatility=target_volatility
            )
            
            # Download data
            success, failed = optimizer.download_data()
            
            if not success:
                st.error("❌ Impossibile scaricare dati sufficienti. Verifica i ticker inseriti.")
            else:
                if failed:
                    st.warning(f"⚠️ ETF non trovati o dati insufficienti: {', '.join(failed)}")
                
                # Esegui ottimizzazioni
                optimizer.run_all_optimizations()
                
                # Salva in session state
                st.session_state['optimizer'] = optimizer
                st.session_state['run_complete'] = True
                
                st.success(f"✅ Ottimizzazione completata! Analizzati {optimizer.n_assets} ETF su {len(optimizer.returns)} settimane.")

# Mostra risultati se disponibili
if st.session_state.get('run_complete', False):
    optimizer = st.session_state['optimizer']
    
    # TAB 1: Metriche
    with tab1:
        st.markdown("## 📊 Riepilogo Performance")
        
        # Info generali
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📅 Settimane Analizzate", len(optimizer.returns))
        with col2:
            st.metric("📋 ETF nel Portafoglio", optimizer.n_assets)
        with col3:
            st.metric("💵 Risk-Free Rate", f"{optimizer.risk_free_rate*100:.1f}%")
        with col4:
            if optimizer.best_benchmark:
                st.metric("🎯 Benchmark", optimizer.best_benchmark['ticker'])
        
        st.markdown("---")
        
        # Metriche per strategia
        st.markdown("### 🎯 Metriche Assolute per Strategia")
        
        metrics_data = []
        for name, data in optimizer.results.items():
            metrics_data.append({
                'Strategia': name.upper(),
                'Rendimento %': f"{data['ret']:.2f}",
                'Volatilità %': f"{data['vol']:.2f}",
                'Max Drawdown %': f"{data['mdd']:.2f}",
                'Sharpe': f"{data['sharpe']:.3f}",
                'Sortino': f"{data['sortino']:.3f}",
                'Calmar': f"{data.get('calmar', 0):.3f}"
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        # Confronto visivo con metric cards
        st.markdown("### 📊 Confronto Rapido Strategie")
        
        cols = st.columns(4)
        strategy_names = ['sharpe', 'sortino', 'rp', 'hrp']
        strategy_labels = ['MAX SHARPE', 'MAX SORTINO', 'RISK PARITY', 'HRP']
        colors = ['🟢', '🟣', '🟠', '🔵']
        
        for i, (name, label, color) in enumerate(zip(strategy_names, strategy_labels, colors)):
            if name in optimizer.results:
                data = optimizer.results[name]
                with cols[i]:
                    st.markdown(f"#### {color} {label}")
                    st.metric("Rendimento", f"{data['ret']:.2f}%")
                    st.metric("Volatilità", f"{data['vol']:.2f}%")
                    st.metric("Sharpe", f"{data['sharpe']:.3f}")
        
        # Metriche relative al benchmark
        if optimizer.best_benchmark and any('beta' in d for d in optimizer.results.values()):
            st.markdown("---")
            st.markdown(f"### 📈 Metriche Relative al Benchmark ({optimizer.best_benchmark['ticker']})")
            
            rel_data = []
            for name, data in optimizer.results.items():
                if 'beta' in data:
                    rel_data.append({
                        'Strategia': name.upper(),
                        'Beta': f"{data['beta']:.3f}",
                        'Alpha %': f"{data['alpha']:+.2f}",
                        'Tracking Error %': f"{data['te']:.2f}",
                        'Info Ratio': f"{data['ir']:.3f}",
                        'Treynor': f"{data.get('treynor', 0):.3f}"
                    })
            
            df_rel = pd.DataFrame(rel_data)
            st.dataframe(df_rel, use_container_width=True, hide_index=True)
    
    # TAB 2: Grafici
    with tab2:
        st.markdown("## 📊 Grafici di Analisi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Frontiera Efficiente")
            fig_frontier = plot_frontier(optimizer)
            if fig_frontier:
                st.pyplot(fig_frontier)
                plt.close()
        
        with col2:
            st.markdown("### 📈 Rendimenti Cumulativi")
            fig_cumulative = plot_cumulative(optimizer)
            st.pyplot(fig_cumulative)
            plt.close()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 📉 Drawdown")
            fig_drawdown = plot_drawdown(optimizer)
            st.pyplot(fig_drawdown)
            plt.close()
        
        with col4:
            st.markdown("### 🗺️ Mappa Risk-Return")
            fig_map = plot_risk_return_map(optimizer)
            if fig_map:
                st.pyplot(fig_map)
                plt.close()
    
    # TAB 3: Allocazioni
    with tab3:
        st.markdown("## 💼 Allocazioni per Strategia")
        
        fig_alloc = plot_allocation(optimizer)
        st.pyplot(fig_alloc)
        plt.close()
        
        st.markdown("---")
        st.markdown("### 📋 Dettaglio Pesi")
        
        weights_data = {'ETF': optimizer.tickers}
        for name, data in optimizer.results.items():
            weights_data[name.upper()] = [f"{w*100:.2f}%" for w in data['weights']]
        
        df_weights = pd.DataFrame(weights_data)
        st.dataframe(df_weights, use_container_width=True, hide_index=True)
        
        # Export CSV
        csv = df_weights.to_csv(index=False)
        st.download_button(
            label="📥 Scarica Allocazioni (CSV)",
            data=csv,
            file_name="allocazioni_portafoglio.csv",
            mime="text/csv"
        )
    
    # TAB 4: Confronto Benchmark
    with tab4:
        st.markdown("## 📋 Confronto Dettagliato vs Benchmark")
        
        if optimizer.best_benchmark:
            b = optimizer.best_benchmark
            
            st.markdown(f"### 🎯 Benchmark Selezionato: **{b['ticker']}** ({b['name']})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rendimento Benchmark", f"{b['mean_return']:.2f}%")
            with col2:
                st.metric("Volatilità Benchmark", f"{b['volatility']:.2f}%")
            with col3:
                st.metric("Sharpe Benchmark", f"{b['sharpe']:.3f}")
            
            st.markdown("---")
            
            # Tabella completa
            st.markdown("### 📊 Tabella Comparativa Completa")
            
            metrics_names = ['Rendimento %', 'Volatilità %', 'Max Drawdown %', 
                           'Sharpe', 'Sortino', 'Calmar', 'Beta', 'Alpha %', 
                           'Tracking Error %', 'Info Ratio', 'Treynor']
            
            bench_values = [
                b['mean_return'], b['volatility'], b['max_drawdown'],
                b['sharpe'], b['sortino'], b['calmar'],
                1.0, 0.0, 0.0, 0.0, 
                (b['mean_return'] - optimizer.risk_free_rate*100)
            ]
            
            comparison_data = {'Metrica': metrics_names, b['ticker']: bench_values}
            
            for name, data in optimizer.results.items():
                strategy_values = [
                    data['ret'], data['vol'], data['mdd'],
                    data['sharpe'], data['sortino'], data.get('calmar', 0),
                    data.get('beta', 1.0), data.get('alpha', 0.0),
                    data.get('te', 0.0), data.get('ir', 0.0),
                    data.get('treynor', 0.0)
                ]
                comparison_data[name.upper()] = strategy_values
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Formatta i numeri
            for col in df_comparison.columns[1:]:
                df_comparison[col] = df_comparison[col].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Legenda:**
            - 🟢 **Verde**: Valore migliore del benchmark
            - 🔴 **Rosso**: Valore peggiore del benchmark
            - Per Volatilità, MDD e TE: valori più bassi sono migliori
            - Per tutte le altre metriche: valori più alti sono migliori
            """)
        else:
            st.warning("⚠️ Nessun benchmark disponibile per il confronto.")

else:
    # Messaggio iniziale
    with tab1:
        st.info("👈 Configura i parametri nella sidebar e clicca **ESEGUI OTTIMIZZAZIONE** per iniziare.")
        
        st.markdown("""
        ### 🎯 Cosa fa questa applicazione?
        
        Questa Web App implementa un **ottimizzatore di portafoglio professionale** basato su 4 strategie:
        
        1. **MAX SHARPE**: Massimizza il rapporto rendimento/rischio totale
        2. **MAX SORTINO**: Massimizza il rendimento rispetto al rischio di ribasso
        3. **RISK PARITY**: Equalizza il contributo al rischio di ogni asset
        4. **HRP**: Allocazione gerarchica robusta basata su clustering
        
        ### 📊 Funzionalità principali:
        - ✅ Download automatico dati da Yahoo Finance
        - ✅ Calcolo di tutte le metriche finanziarie (Sharpe, Sortino, Calmar, Alpha, Beta, Treynor, etc.)
        - ✅ Vincoli settoriali personalizzabili
        - ✅ Vincolo di volatilità target (volatility cap)
        - ✅ Selezione automatica del benchmark più appropriato
        - ✅ Grafici interattivi e tabelle comparative
        - ✅ Export dei risultati in CSV
        
        ### 🚀 Come iniziare:
        1. Inserisci la lista di ETF nella sidebar (uno per riga)
        2. Seleziona il periodo di analisi
        3. Configura i parametri (risk-free rate, pesi min/max)
        4. Abilita/disabilita i vincoli avanzati
        5. Clicca **ESEGUI OTTIMIZZAZIONE**
        """)
