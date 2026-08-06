# EVE Online Price Prediction Engine

> **Design document, not a status report.** This file describes the intended
> architecture. Substantial parts of it are still roadmap. Where the two
> disagree, **README.md describes what actually exists**; see
> **AUDIT_REMEDIATION.md** for the current implementation status.
>
> Known to be design-only at the time of writing:
> ARIMA/XGBoost and other alternative models, hyperparameter search
> (`ParameterGrid`), `vectorbt` backtesting and trading metrics,
> email/Slack/PagerDuty alerting beyond the single webhook hook,
> whale/large-order detection, ±0.5/1/2/5% order-depth levels, patch-cycle and
> regional-hub features, the data-validation layer (gap detection, >3σ anomaly
> flags), and the Grafana/Prometheus/ELK observability stack.
>
> Since implemented: the Redis caching layer, automated weekly retraining,
> prediction persistence, `actual_price` backfill, the `model_performance`
> table, prediction-error monitoring, and an MLflow registered model per item.

## Executive Summary

A machine learning system for predicting EVE Online market prices using historical data and real-time order books from the ESI API. This document outlines a production-ready approach leveraging LSTM neural networks to forecast price movements for high-liquidity items across major trade hubs.

**Key Objectives:**
- Predict short to medium-term price movements (1 hour to 7 days)
- Enable algorithmic trading strategies (station trading, arbitrage, speculation)
- Analyze market dynamics and economic patterns
- Achieve >60% directional accuracy on liquid items

**Success Metrics:**
- Directional accuracy (% of correct up/down predictions)
- RMSE/MAE on price predictions
- Simulated trading profit factor
- Sharpe ratio of strategy returns

---

## Data Sources & API Integration

### ESI Market Endpoints

#### 1. Historical Market Data
**Endpoint:** `/markets/{region_id}/history/?type_id={type_id}`

**Returns:**
```json
{
  "date": "2025-10-26",
  "average": 1234567.89,
  "highest": 1300000.00,
  "lowest": 1200000.00,
  "volume": 150,
  "order_count": 45
}
```

**Characteristics:**
- Daily aggregated statistics
- Historical depth: ~365 days
- Cache duration: 3600 seconds
- Use case: Long-term trend analysis, seasonality detection

#### 2. Real-Time Order Book
**Endpoint:** `/markets/{region_id}/orders/?type_id={type_id}`

**Returns:**
```json
{
  "order_id": 5598641910,
  "type_id": 34,
  "location_id": 60003760,
  "volume_remain": 100,
  "volume_total": 150,
  "min_volume": 1,
  "price": 1250000.00,
  "is_buy_order": false,
  "duration": 90,
  "issued": "2025-10-26T12:00:00Z",
  "range": "region"
}
```

**Characteristics:**
- Real-time order snapshots
- Cache duration: 300 seconds (5-minute updates)
- Pagination required for high-volume items
- Use case: Order book analysis, spread calculation, short-term prediction

#### 3. Reference Prices
**Endpoint:** `/markets/prices/`

**Returns:**
```json
{
  "type_id": 34,
  "average_price": 1225000.00,
  "adjusted_price": 1230000.00
}
```

**Characteristics:**
- Universe-wide aggregate prices
- Used for industry calculations
- Cache duration: 3600 seconds
- Use case: Baseline price references, outlier detection

### Data Collection Strategy

**Target Items:**
- Top 100 most-traded items by volume
- Focus on: PLEX, skill extractors/injectors, popular T1/T2 ships, faction modules
- Criteria: Daily volume >1B ISK, order count >50

**Target Regions:**
- **Primary:** The Forge (region 10000002) - Jita trade hub
- **Secondary:** Domain (10000043) - Amarr, Sinq Laison (10000032) - Dodixie
- **Expansion:** Nullsec market hubs after validation

**Collection Frequency:**
- Order book: Every 5 minutes (aligned with ESI cache)
- Historical data: Daily at 00:00 EVE time
- Reference prices: Hourly

**Storage Volume Estimates (per 90 days):**
- Historical: 100 items × 90 days × 6 fields ≈ 500 KB
- Order snapshots: 100 items × 25,920 snapshots × 50 avg orders ≈ 15 GB compressed
- Total with indices: ~20 GB per quarter

---

## Machine Learning Approaches

### Recommended: LSTM Neural Networks

**Why LSTM for EVE Markets:**
1. **Temporal Dependencies**: Captures how price changes today affect tomorrow
2. **Non-Linear Patterns**: Handles complex market dynamics (pumps, crashes, manipulation)
3. **Memory Cells**: Retains long-term context (patch impacts, seasonal events)
4. **Proven Performance**: State-of-the-art for financial time series

**Architecture:**
```
Input Sequence (lookback window)
    ↓
LSTM Layer 1 (128 units, dropout 0.2)
    ↓
LSTM Layer 2 (64 units, dropout 0.2)
    ↓
Dense Layer (32 units, ReLU activation)
    ↓
Output Layer (1 unit, linear activation)
    ↓
Predicted Price (t+1)
```

**Hyperparameters:**
- Lookback window: 14-30 days for daily predictions, 24-72 time steps for hourly
- Batch size: 32-64
- Learning rate: 0.001 (Adam optimizer)
- Epochs: 50-100 with early stopping
- Loss function: MSE or Huber loss (robust to outliers)

**Training Process:**
1. Normalize features using MinMaxScaler or StandardScaler
2. Create sequences: `[t-30, t-29, ..., t-1]` → `[t]`
3. Split: 70% train, 15% validation, 15% test (chronological)
4. Train with early stopping (patience=10 epochs)
5. Evaluate on hold-out test set

### Alternative: Hybrid ARIMA-LSTM

**When to Use:**
- Items with strong linear trends and seasonality
- Longer prediction horizons (7-30 days)
- When interpretability is important

**Process:**
```python
# Step 1: Fit ARIMA to capture linear components
from statsmodels.tsa.arima.model import ARIMA
model_arima = ARIMA(prices, order=(5,1,2))
arima_fit = model_arima.fit()
linear_predictions = arima_fit.predict()

# Step 2: Train LSTM on residuals
residuals = prices - linear_predictions
lstm_model.fit(residuals)

# Step 3: Combine predictions
final_prediction = arima_prediction + lstm_prediction
```

**Advantages:**
- ARIMA handles trend/seasonality efficiently
- LSTM captures non-linear residuals
- Often outperforms single-method approaches

### Alternative: Gradient Boosting (XGBoost)

**When to Use:**
- Short-term predictions (1-6 hours)
- When order book features are primary signals
- Need for fast training/inference

**Feature Engineering:**
```python
features = [
    'current_price',
    'price_change_1h', 'price_change_24h',
    'volume_1h', 'volume_24h',
    'order_book_imbalance',
    'bid_ask_spread',
    'top_5_buy_volume', 'top_5_sell_volume',
    'hour_of_day', 'day_of_week'
]
```

**Model:**
```python
import xgboost as xgb

params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'objective': 'reg:squarederror',
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)
```

---

## Feature Engineering

### Time Series Features

**Price-Based:**
- `price_t-1, price_t-2, ..., price_t-N`: Historical prices (lag features)
- `price_sma_7, price_sma_14, price_sma_30`: Simple moving averages
- `price_ema_7, price_ema_14`: Exponential moving averages
- `price_std_7, price_std_14`: Rolling standard deviation (volatility)
- `price_return_1d`: Daily returns `(price_t - price_t-1) / price_t-1`

**Technical Indicators:**
```python
def compute_rsi(prices, period=14):
    """Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(prices, period=20, std_dev=2):
    """Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def compute_macd(prices, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line
```

**Volume Features:**
- `volume_sma_7, volume_sma_14`: Volume moving averages
- `order_count_trend`: Change in number of active orders
- `volume_price_ratio`: Volume / price (liquidity indicator)

### Order Book Features

**Spread Analysis:**
```python
def compute_spread_metrics(orders_df):
    """Calculate bid-ask spread and related metrics"""
    buy_orders = orders_df[orders_df['is_buy_order']]
    sell_orders = orders_df[~orders_df['is_buy_order']]

    best_bid = buy_orders['price'].max()
    best_ask = sell_orders['price'].min()

    spread_absolute = best_ask - best_bid
    spread_percentage = (spread_absolute / best_ask) * 100
    mid_price = (best_bid + best_ask) / 2

    return {
        'best_bid': best_bid,
        'best_ask': best_ask,
        'spread_abs': spread_absolute,
        'spread_pct': spread_percentage,
        'mid_price': mid_price
    }
```

**Order Book Imbalance:**
```python
def compute_order_book_imbalance(orders_df, depth_levels=5):
    """
    Imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    Range: -1 (all sells) to +1 (all buys)
    """
    buy_orders = orders_df[orders_df['is_buy_order']].nlargest(depth_levels, 'price')
    sell_orders = orders_df[~orders_df['is_buy_order']].nsmallest(depth_levels, 'price')

    buy_volume = buy_orders['volume_remain'].sum()
    sell_volume = sell_orders['volume_remain'].sum()

    imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-9)
    return imbalance
```

**Order Depth:**
```python
def compute_order_depth(orders_df, price_levels=10):
    """Volume at various price levels from best bid/ask"""
    spread_metrics = compute_spread_metrics(orders_df)
    mid_price = spread_metrics['mid_price']

    # Calculate depth within ±X% of mid price
    for pct in [0.5, 1.0, 2.0, 5.0]:
        lower = mid_price * (1 - pct/100)
        upper = mid_price * (1 + pct/100)

        buy_depth = orders_df[
            (orders_df['is_buy_order']) &
            (orders_df['price'] >= lower)
        ]['volume_remain'].sum()

        sell_depth = orders_df[
            (~orders_df['is_buy_order']) &
            (orders_df['price'] <= upper)
        ]['volume_remain'].sum()

        yield f'buy_depth_{pct}pct', buy_depth
        yield f'sell_depth_{pct}pct', sell_depth
```

**Whale Order Detection:**
```python
def detect_large_orders(orders_df, threshold_percentile=95):
    """Identify unusually large orders that may move market"""
    volume_threshold = orders_df['volume_remain'].quantile(threshold_percentile)

    large_buy_volume = orders_df[
        (orders_df['is_buy_order']) &
        (orders_df['volume_remain'] >= volume_threshold)
    ]['volume_remain'].sum()

    large_sell_volume = orders_df[
        (~orders_df['is_buy_order']) &
        (orders_df['volume_remain'] >= volume_threshold)
    ]['volume_remain'].sum()

    return {
        'large_buy_volume': large_buy_volume,
        'large_sell_volume': large_sell_volume,
        'whale_imbalance': large_buy_volume - large_sell_volume
    }
```

### Game-Specific Features

**Patch & Update Tracking:**
- `days_since_patch`: Days since last major game update
- `is_patch_week`: Binary flag for week of patch release
- `affected_by_patch`: Binary flag if item mentioned in patch notes

**Event Seasonality:**
- `is_summer_period`: June-August (PLEX prices spike)
- `is_winter_holiday`: December (seasonal events)
- `days_to_next_expansion`: Countdown to major content release

**Item Category Indicators:**
- `item_category`: One-hot encoded (ships, modules, materials, PLEX)
- `manufacturing_tier`: T1, T2, T3, faction, officer
- `is_npc_loot`: Binary flag for mission/ratting drops
- `blueprint_availability`: Market-only vs. manufacturable

### Regional Features

**Hub Classification:**
```python
MAJOR_HUBS = {
    60003760: 'jita',      # Jita 4-4
    60008494: 'amarr',     # Amarr VIII
    60011866: 'dodixie',   # Dodixie IX
    60004588: 'rens',      # Rens VI
    60005686: 'hek'        # Hek VIII
}

def classify_location(location_id):
    if location_id in MAJOR_HUBS:
        return MAJOR_HUBS[location_id], 'major_hub'
    elif location_id in MINOR_HUBS:
        return 'minor_hub'
    else:
        return 'other'
```

**Security Status:**
- `system_security`: Highsec (>0.5), Lowsec (0.1-0.4), Nullsec (<0.0)
- `jumps_to_jita`: Distance from major trade hub

**Production Indices:**
- `manufacturing_cost_index`: From `/industry/systems/`
- `invention_cost_index`: Research cost in region
- `copying_cost_index`: Blueprint copying cost

### Temporal Features

**Cyclic Encoding:**
```python
import numpy as np

def encode_cyclic_time(hour=None, day_of_week=None, day_of_month=None):
    """Encode time cyclically to preserve continuity"""
    features = {}

    if hour is not None:
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    if day_of_week is not None:
        features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    if day_of_month is not None:
        features['dom_sin'] = np.sin(2 * np.pi * day_of_month / 30)
        features['dom_cos'] = np.cos(2 * np.pi * day_of_month / 30)

    return features
```

**Peak Trading Hours:**
- `is_peak_hour`: 18:00-22:00 EVE time (European prime time)
- `is_weekend`: Saturday-Sunday (different trading patterns)

---

## Implementation Roadmap

### Phase 1: Infrastructure Setup (Week 1)

**Objectives:**
- Set up development environment
- Configure database and caching
- Implement ESI API client

**Tasks:**
1. Create PostgreSQL database with TimescaleDB extension
   ```sql
   CREATE EXTENSION IF NOT EXISTS timescaledb;

   CREATE TABLE market_history (
       time TIMESTAMPTZ NOT NULL,
       region_id INTEGER NOT NULL,
       type_id INTEGER NOT NULL,
       average NUMERIC,
       highest NUMERIC,
       lowest NUMERIC,
       volume BIGINT,
       order_count INTEGER,
       PRIMARY KEY (time, region_id, type_id)
   );

   SELECT create_hypertable('market_history', 'time');
   CREATE INDEX ON market_history (region_id, type_id, time DESC);
   ```

2. Set up Redis for API caching
   ```python
   import redis
   cache = redis.Redis(host='localhost', port=6379, db=0)
   ```

3. Implement rate-limited ESI client
   ```python
   import httpx
   from ratelimit import limits, sleep_and_retry

   class ESIClient:
       BASE_URL = "https://esi.evetech.net/latest"

       def __init__(self):
           self.client = httpx.AsyncClient(timeout=10.0)

       @sleep_and_retry
       @limits(calls=100, period=60)
       async def get_market_history(self, region_id, type_id):
           url = f"{self.BASE_URL}/markets/{region_id}/history/"
           params = {'type_id': type_id, 'datasource': 'tranquility'}

           response = await self.client.get(url, params=params)
           response.raise_for_status()
           return response.json()
   ```

4. Create data models and ORM schemas
5. Set up logging and monitoring

**Deliverables:**
- Working ESI API client with error handling
- Database schema and connection pooling
- Basic data collection script

### Phase 2: Data Collection (Week 2-3)

**Objectives:**
- Collect 90+ days of historical data
- Implement continuous order book scraping
- Validate data quality

**Tasks:**
1. Backfill historical market data
   ```python
   async def backfill_history(region_id, type_ids, days=365):
       for type_id in type_ids:
           history = await esi_client.get_market_history(region_id, type_id)
           # Store in database
   ```

2. Schedule order book collection (every 5 minutes)
   ```python
   from apscheduler.schedulers.asyncio import AsyncIOScheduler

   scheduler = AsyncIOScheduler()
   scheduler.add_job(
       collect_order_books,
       'interval',
       minutes=5,
       args=[region_id, type_ids]
   )
   scheduler.start()
   ```

3. Implement data validation
   - Check for missing timestamps
   - Detect price anomalies (>3 std dev)
   - Verify order book completeness

4. Build data quality dashboard

**Deliverables:**
- 90 days of historical data for top 100 items
- Continuous order book snapshots
- Data quality report

### Phase 3: Exploratory Data Analysis (Week 3-4)

**Objectives:**
- Understand price distributions and volatility
- Identify correlations and patterns
- Characterize item liquidity tiers

**Analysis Tasks:**
1. Price distribution analysis
   - Log-normal vs normal distribution
   - Volatility clustering
   - Outlier detection

2. Correlation studies
   ```python
   import pandas as pd
   import seaborn as sns

   # Calculate correlation matrix
   price_matrix = pivot_table(df, values='average', index='date', columns='type_id')
   correlation = price_matrix.corr()

   # Visualize
   sns.heatmap(correlation, cmap='coolwarm')
   ```

3. Seasonality detection
   ```python
   from statsmodels.tsa.seasonal import seasonal_decompose

   result = seasonal_decompose(prices, model='multiplicative', period=30)
   result.plot()
   ```

4. Liquidity classification
   ```python
   def classify_liquidity(avg_daily_volume, avg_daily_orders):
       if avg_daily_volume > 1e9 and avg_daily_orders > 100:
           return 'high'
       elif avg_daily_volume > 1e8 and avg_daily_orders > 20:
           return 'medium'
       else:
           return 'low'
   ```

5. Identify predictable vs chaotic items
   - Calculate autocorrelation
   - Measure entropy of price changes
   - Classify items by predictability score

**Deliverables:**
- EDA Jupyter notebook
- Item classification taxonomy
- Target item shortlist for initial modeling

### Phase 4: Feature Engineering (Week 4-5)

**Objectives:**
- Compute technical indicators
- Extract order book features
- Create feature store

**Tasks:**
1. Implement feature computation pipeline
   ```python
   class FeatureEngine:
       def __init__(self, db_connection):
           self.db = db_connection

       def compute_daily_features(self, type_id, date):
           # Fetch raw data
           prices = self.get_price_history(type_id, lookback=90)
           orders = self.get_order_snapshots(type_id, date)

           # Compute features
           features = {}
           features.update(self.compute_price_features(prices))
           features.update(self.compute_volume_features(prices))
           features.update(self.compute_technical_indicators(prices))
           features.update(self.compute_order_book_features(orders))
           features.update(self.compute_temporal_features(date))

           return features
   ```

2. Create feature store table
   ```sql
   CREATE TABLE features (
       time TIMESTAMPTZ NOT NULL,
       type_id INTEGER NOT NULL,
       feature_name VARCHAR(100) NOT NULL,
       feature_value NUMERIC,
       PRIMARY KEY (time, type_id, feature_name)
   );
   ```

3. Batch compute historical features
4. Set up real-time feature computation
5. Feature validation and unit tests

**Deliverables:**
- Feature computation library
- Populated feature store (90 days)
- Feature documentation

### Phase 5: Baseline Modeling (Week 5-6)

**Objectives:**
- Establish performance baselines
- Validate train/test split strategy
- Define evaluation metrics

**Models:**
1. **Naive Baseline**: Tomorrow's price = Today's price
   ```python
   baseline_predictions = prices.shift(1)
   baseline_mae = mean_absolute_error(y_true, baseline_predictions)
   ```

2. **Moving Average**:
   ```python
   sma_predictions = prices.rolling(window=7).mean()
   ```

3. **Linear Regression**:
   ```python
   from sklearn.linear_model import LinearRegression

   model = LinearRegression()
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)
   ```

4. **ARIMA**:
   ```python
   from statsmodels.tsa.arima.model import ARIMA

   model = ARIMA(prices, order=(5,1,2))
   fitted = model.fit()
   forecast = fitted.forecast(steps=7)
   ```

**Evaluation:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }
```

**Deliverables:**
- Baseline model performances
- Evaluation framework
- Target metrics for LSTM model

### Phase 6: LSTM Development (Week 6-7)

**Objectives:**
- Implement LSTM architecture
- Hyperparameter tuning
- Achieve >baseline performance

**Implementation:**
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MarketDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class MarketLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(MarketLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        # Feed through dense layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model
```

**Hyperparameter Tuning:**
```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.001, 0.01],
    'lookback_window': [14, 30, 60]
}

best_score = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    model = MarketLSTM(
        input_size=num_features,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )

    # Train and evaluate
    trained_model = train_model(model, train_loader, val_loader, lr=params['learning_rate'])
    val_score = evaluate_model(trained_model, val_loader)

    if val_score < best_score:
        best_score = val_score
        best_params = params
```

**Deliverables:**
- Trained LSTM model
- Hyperparameter tuning results
- Model performance exceeding baseline

### Phase 7: Backtesting (Week 7-8)

**Objectives:**
- Simulate trading strategies
- Calculate risk-adjusted returns
- Validate model profitability

**Trading Strategy:**
```python
class SimpleStrategy:
    def __init__(self, model, initial_capital=1_000_000_000):  # 1B ISK
        self.model = model
        self.capital = initial_capital
        self.position = 0
        self.trades = []

    def generate_signal(self, features):
        """Generate buy/sell signal from prediction"""
        current_price = features['current_price']
        predicted_price = self.model.predict(features)

        expected_return = (predicted_price - current_price) / current_price

        # Buy if predicted return > 2% (accounting for fees)
        if expected_return > 0.02 and self.position == 0:
            return 'BUY'
        # Sell if in position and predicted decline or target reached
        elif self.position > 0 and (expected_return < -0.01 or expected_return > 0.05):
            return 'SELL'
        else:
            return 'HOLD'

    def execute_trade(self, signal, price, quantity, timestamp):
        """Execute trade with transaction costs"""
        BROKER_FEE = 0.025  # 2.5%
        SALES_TAX = 0.01    # 1%

        if signal == 'BUY':
            cost = price * quantity * (1 + BROKER_FEE)
            if cost <= self.capital:
                self.capital -= cost
                self.position += quantity
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'cost': cost
                })

        elif signal == 'SELL' and self.position > 0:
            revenue = price * quantity * (1 - BROKER_FEE - SALES_TAX)
            self.capital += revenue
            self.position -= quantity
            self.trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': price,
                'quantity': quantity,
                'revenue': revenue
            })

    def calculate_metrics(self):
        """Calculate trading performance metrics"""
        if len(self.trades) < 2:
            return None

        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']

        total_profit = sum(t['revenue'] for t in sell_trades) - sum(t['cost'] for t in buy_trades)

        # Calculate returns
        returns = []
        for i in range(len(buy_trades)):
            if i < len(sell_trades):
                buy_price = buy_trades[i]['price']
                sell_price = sell_trades[i]['price']
                returns.append((sell_price - buy_price) / buy_price)

        # Sharpe ratio (assuming daily returns, 365 trading days)
        if len(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (mean_return / std_return) * np.sqrt(365) if std_return > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        return {
            'total_profit': total_profit,
            'total_return': total_profit / (self.initial_capital - self.capital + total_profit),
            'num_trades': len(self.trades),
            'win_rate': np.mean([r > 0 for r in returns]),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
```

**Backtesting Framework:**
```python
def run_backtest(model, test_data, initial_capital=1_000_000_000):
    strategy = SimpleStrategy(model, initial_capital)

    for idx, row in test_data.iterrows():
        features = extract_features(row)
        signal = strategy.generate_signal(features)

        if signal in ['BUY', 'SELL']:
            strategy.execute_trade(
                signal,
                price=row['price'],
                quantity=calculate_position_size(strategy.capital),
                timestamp=row['timestamp']
            )

    metrics = strategy.calculate_metrics()
    return metrics, strategy.trades
```

**Deliverables:**
- Backtesting framework
- Performance metrics report
- Trade log analysis

### Phase 8: Production Deployment (Week 8+)

**Objectives:**
- Deploy model serving API
- Implement monitoring and alerting
- Set up automated retraining

**API Implementation:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

app = FastAPI()

class PredictionRequest(BaseModel):
    type_id: int
    region_id: int
    horizon: int = 1  # days ahead

class PredictionResponse(BaseModel):
    type_id: int
    current_price: float
    predicted_price: float
    confidence_interval: tuple
    timestamp: str

# Load model at startup
model = MarketLSTM(input_size=50, hidden_size=128, num_layers=2)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    try:
        # Fetch latest features
        features = await fetch_latest_features(request.type_id, request.region_id)

        # Prepare input sequence
        sequence = prepare_sequence(features, lookback=30)

        # Make prediction
        with torch.no_grad():
            prediction = model(sequence).item()

        # Calculate confidence interval (using historical prediction errors)
        ci = calculate_confidence_interval(prediction, request.type_id)

        return PredictionResponse(
            type_id=request.type_id,
            current_price=features['current_price'],
            predicted_price=prediction,
            confidence_interval=ci,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

**Monitoring Dashboard:**
```python
import plotly.graph_objects as go
from dash import Dash, dcc, html

app = Dash(__name__)

app.layout = html.Div([
    html.H1("EVE Market Prediction Monitor"),

    dcc.Graph(id='prediction-accuracy'),
    dcc.Graph(id='model-drift'),
    dcc.Graph(id='feature-importance'),

    dcc.Interval(id='interval-component', interval=60*1000)  # Update every minute
])

@app.callback(
    Output('prediction-accuracy', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_accuracy_chart(n):
    # Fetch recent predictions vs actuals
    data = fetch_recent_predictions()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['actual'], name='Actual'))
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['predicted'], name='Predicted'))

    return fig
```

**Automated Retraining:**
```python
from apscheduler.schedulers.blocking import BlockingScheduler

def retrain_model():
    """Weekly model retraining"""
    print("Starting model retraining...")

    # Fetch latest data (last 90 days)
    data = fetch_training_data(days=90)

    # Compute features
    features = compute_features(data)

    # Prepare sequences
    X, y = prepare_training_data(features)

    # Train new model
    model = MarketLSTM(input_size=features.shape[1])
    trained_model = train_model(model, X, y)

    # Evaluate on hold-out set
    metrics = evaluate_model(trained_model, X_test, y_test)

    # Deploy if performance acceptable
    if metrics['RMSE'] < PERFORMANCE_THRESHOLD:
        torch.save(trained_model.state_dict(), 'best_model.pth')
        print(f"Model retrained successfully. RMSE: {metrics['RMSE']}")
    else:
        print(f"Retraining failed. RMSE {metrics['RMSE']} exceeds threshold.")

scheduler = BlockingScheduler()
scheduler.add_job(retrain_model, 'cron', day_of_week='sun', hour=2)
scheduler.start()
```

**Deliverables:**
- Production API
- Monitoring dashboard
- Automated retraining pipeline
- Documentation and runbooks

---

## Evaluation Framework

### Standard Regression Metrics

**Mean Absolute Error (MAE):**
```python
MAE = mean(|y_true - y_pred|)
```
- Interpretation: Average prediction error in ISK
- Target: < 5% of average item price

**Root Mean Square Error (RMSE):**
```python
RMSE = sqrt(mean((y_true - y_pred)^2))
```
- Interpretation: Penalizes large errors more heavily
- Target: < 7% of average item price

**Mean Absolute Percentage Error (MAPE):**
```python
MAPE = mean(|y_true - y_pred| / y_true) * 100
```
- Interpretation: Error as percentage of true value
- Target: < 10%

### Trading-Specific Metrics

**Directional Accuracy:**
```python
def directional_accuracy(y_true, y_pred):
    """Percentage of correct up/down predictions"""
    direction_true = (y_true[1:] - y_true[:-1]) > 0
    direction_pred = (y_pred[1:] - y_pred[:-1]) > 0
    return np.mean(direction_true == direction_pred)
```
- **Target:** >60% (baseline is 50%)
- **Critical for:** Station trading strategies

**Profit Factor:**
```python
def profit_factor(trades):
    """Gross profit / gross loss"""
    profits = sum([t['profit'] for t in trades if t['profit'] > 0])
    losses = sum([abs(t['profit']) for t in trades if t['profit'] < 0])
    return profits / losses if losses > 0 else float('inf')
```
- **Target:** >1.5 (profit exceeds losses by 50%)
- **Interpretation:** 2.0 means you make $2 for every $1 lost

**Sharpe Ratio:**
```python
def sharpe_ratio(returns, risk_free_rate=0):
    """Risk-adjusted return"""
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
```
- **Target:** >1.0 (good), >2.0 (excellent)
- **Interpretation:** Return per unit of risk

**Maximum Drawdown:**
```python
def max_drawdown(equity_curve):
    """Largest peak-to-trough decline"""
    cumulative = np.cumsum(equity_curve)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)
```
- **Target:** < -20%
- **Interpretation:** Worst-case loss from peak

**Win Rate:**
```python
def win_rate(trades):
    """Percentage of profitable trades"""
    profitable = sum([1 for t in trades if t['profit'] > 0])
    return profitable / len(trades)
```
- **Target:** >55%
- **Note:** High win rate doesn't guarantee profitability

### Model Drift Detection

**Prediction Error Monitoring:**
```python
def detect_drift(recent_errors, historical_errors, threshold=2.0):
    """Detect if model performance is degrading"""
    recent_mean = np.mean(recent_errors)
    historical_mean = np.mean(historical_errors)
    historical_std = np.std(historical_errors)

    # Z-score of recent performance
    z_score = (recent_mean - historical_mean) / historical_std

    if z_score > threshold:
        return True, f"Model drift detected: Z-score = {z_score:.2f}"
    return False, "Model performance stable"
```

**Feature Distribution Shift:**
```python
from scipy.stats import ks_2samp

def check_feature_drift(feature_current, feature_historical):
    """Kolmogorov-Smirnov test for distribution shift"""
    statistic, p_value = ks_2samp(feature_current, feature_historical)

    if p_value < 0.05:
        return True, f"Feature drift detected: p-value = {p_value:.4f}"
    return False, "Feature distribution stable"
```

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Collection Layer                    │
├─────────────────────────────────────────────────────────────┤
│  ESI API Client (async, rate-limited)                       │
│  ├─ Market History Fetcher (daily)                          │
│  ├─ Order Book Scraper (5-min intervals)                    │
│  └─ Reference Price Fetcher (hourly)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     Caching Layer (Redis)                    │
├─────────────────────────────────────────────────────────────┤
│  - API response caching (TTL based on endpoint)             │
│  - Feature computation results (hourly TTL)                 │
│  - Model predictions (5-min TTL)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Storage Layer (PostgreSQL + Timescale)          │
├─────────────────────────────────────────────────────────────┤
│  Tables:                                                     │
│  ├─ market_history (time-series)                            │
│  ├─ order_snapshots (time-series)                           │
│  ├─ features (computed indicators)                          │
│  ├─ predictions (model outputs)                             │
│  ├─ items (reference data)                                  │
│  └─ model_performance (metrics tracking)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Engineering Layer                   │
├─────────────────────────────────────────────────────────────┤
│  - Technical indicator computation                          │
│  - Order book feature extraction                            │
│  - Temporal encoding                                        │
│  - Feature scaling/normalization                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      ML Training Layer                       │
├─────────────────────────────────────────────────────────────┤
│  - LSTM model training (PyTorch)                            │
│  - Hyperparameter tuning                                    │
│  - Cross-validation                                         │
│  - Model versioning (MLflow)                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Prediction Serving Layer                   │
├─────────────────────────────────────────────────────────────┤
│  FastAPI REST API:                                          │
│  ├─ POST /predict - Get price prediction                    │
│  ├─ GET /features/{type_id} - Current features              │
│  ├─ GET /performance - Model metrics                        │
│  └─ GET /health - Health check                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Monitoring & Alerting Layer                  │
├─────────────────────────────────────────────────────────────┤
│  - Dash/Plotly dashboard (real-time metrics)                │
│  - Model drift detection                                    │
│  - Prediction accuracy tracking                             │
│  - Alert system (email/Slack on drift)                      │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Core Languages:**
- Python 3.11+ (primary development language)
- SQL (database queries, TimescaleDB optimizations)

**ML Frameworks:**
- PyTorch 2.0+ (LSTM implementation)
- scikit-learn (preprocessing, metrics, baseline models)
- NumPy, pandas (data manipulation)

**Web Framework:**
- FastAPI (prediction serving API)
- Pydantic (data validation)
- Uvicorn (ASGI server)

**Database:**
- PostgreSQL 15+ (relational data)
- TimescaleDB extension (time-series optimization)
- Redis 7+ (caching, rate limiting)

**Data Collection:**
- httpx or aiohttp (async HTTP client)
- APScheduler (job scheduling)
- ratelimit (API rate limiting)

**Monitoring & Visualization:**
- Dash/Plotly (interactive dashboards)
- Jupyter Lab (exploratory analysis)
- Grafana (optional, time-series monitoring)

**DevOps:**
- Docker & Docker Compose (containerization)
- GitHub Actions (CI/CD)
- MLflow (model versioning and experiment tracking)

**Optional Enhancements:**
- Airflow/Prefect (workflow orchestration)
- Prometheus (metrics collection)
- ELK Stack (logging)

### Database Schema

```sql
-- Market history (daily aggregates)
CREATE TABLE market_history (
    time TIMESTAMPTZ NOT NULL,
    region_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    average NUMERIC(20,2),
    highest NUMERIC(20,2),
    lowest NUMERIC(20,2),
    volume BIGINT,
    order_count INTEGER,
    PRIMARY KEY (time, region_id, type_id)
);

SELECT create_hypertable('market_history', 'time');
CREATE INDEX idx_mh_region_type ON market_history (region_id, type_id, time DESC);

-- Order book snapshots
CREATE TABLE order_snapshots (
    snapshot_time TIMESTAMPTZ NOT NULL,
    order_id BIGINT NOT NULL,
    type_id INTEGER NOT NULL,
    region_id INTEGER NOT NULL,
    location_id BIGINT NOT NULL,
    price NUMERIC(20,2),
    volume_remain INTEGER,
    volume_total INTEGER,
    is_buy_order BOOLEAN,
    duration INTEGER,
    issued TIMESTAMPTZ,
    PRIMARY KEY (snapshot_time, order_id)
);

SELECT create_hypertable('order_snapshots', 'snapshot_time');
CREATE INDEX idx_os_type_time ON order_snapshots (type_id, snapshot_time DESC);

-- Computed features
CREATE TABLE features (
    time TIMESTAMPTZ NOT NULL,
    type_id INTEGER NOT NULL,
    region_id INTEGER NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value NUMERIC,
    PRIMARY KEY (time, type_id, region_id, feature_name)
);

SELECT create_hypertable('features', 'time');

-- Model predictions
CREATE TABLE predictions (
    prediction_time TIMESTAMPTZ NOT NULL,
    type_id INTEGER NOT NULL,
    region_id INTEGER NOT NULL,
    current_price NUMERIC(20,2),
    predicted_price NUMERIC(20,2),
    prediction_horizon INTEGER,  -- hours ahead
    model_version VARCHAR(50),
    actual_price NUMERIC(20,2),  -- filled in later for evaluation
    PRIMARY KEY (prediction_time, type_id, region_id, prediction_horizon)
);

SELECT create_hypertable('predictions', 'prediction_time');

-- Item reference data
CREATE TABLE items (
    type_id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    category VARCHAR(100),
    liquidity_tier VARCHAR(20),  -- high, medium, low
    avg_daily_volume BIGINT,
    avg_daily_orders INTEGER,
    is_tradeable BOOLEAN
);

-- Model performance tracking
CREATE TABLE model_performance (
    evaluation_time TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50),
    type_id INTEGER,
    metric_name VARCHAR(50),
    metric_value NUMERIC,
    PRIMARY KEY (evaluation_time, model_version, type_id, metric_name)
);

SELECT create_hypertable('model_performance', 'evaluation_time');
```

---

## Risk Factors & Mitigation

### Market Manipulation

**Risk:**
- Coordinated pump-and-dump schemes
- Whale orders manipulating thin markets
- False signals from spoofing (placing/canceling large orders)

**Mitigation:**
1. **Focus on liquid items**: >1B ISK daily volume, >100 orders
2. **Anomaly detection**: Flag price movements >3σ from mean
3. **Order stability filtering**: Ignore orders <5 minutes old
4. **Conservative position sizing**: Never risk >5% capital on single trade
5. **Outlier removal in training**: Clean historical manipulation events

### Game Patches & Meta Shifts

**Risk:**
- Balance changes instantly alter item values
- New ship/module releases disrupt established patterns
- Skill changes affect training demand

**Mitigation:**
1. **Patch monitoring**: Scrape dev blogs and patch notes
2. **Rapid retraining**: Weekly model updates post-patch
3. **Feature flags**: `days_since_patch`, `is_affected_item`
4. **Manual overrides**: Disable predictions for patched items (7 days)
5. **Ensemble models**: Combine LSTM with rule-based adjustments

### Data Quality Issues

**Risk:**
- ESI API downtime (historical occurrence rate ~99.5% uptime)
- Missing order snapshots due to rate limiting
- Structure location opacity
- Trade inference errors

**Mitigation:**
1. **Redundancy**: Multiple data collection processes
2. **Gap detection**: Alert on missing timestamps
3. **Interpolation**: Fill small gaps (<15 min) with linear interpolation
4. **Data validation**: Check for nonsensical prices (negative, extreme spikes)
5. **Fallback sources**: Use community APIs (evemarketer.com) as backup

### Overfitting

**Risk:**
- Model memorizes training data patterns
- Poor generalization to new market conditions
- Excellent backtest, poor live performance

**Mitigation:**
1. **Regularization**: Dropout (20%), L2 regularization
2. **Cross-validation**: Time-series split, walk-forward validation
3. **Ensemble methods**: Average predictions from multiple models
4. **Feature selection**: Remove low-importance features
5. **Simplicity bias**: Start with simple models, add complexity only if justified

### Model Drift

**Risk:**
- Market dynamics change over time
- Feature distributions shift
- Prediction accuracy degrades silently

**Mitigation:**
1. **Continuous monitoring**: Track rolling 7-day MAE
2. **Drift detection**: Statistical tests (KS test) on feature distributions
3. **Automated retraining**: Weekly schedule
4. **Performance alerts**: Email/Slack if RMSE > threshold
5. **Model versioning**: MLflow tracking, easy rollback

### Execution Risk

**Risk:**
- Predicted price ≠ executable price (slippage)
- Transaction costs eat profits
- Order execution lag (5-minute API snapshots)

**Mitigation:**
1. **Spread awareness**: Only trade items with spread <2%
2. **Fee incorporation**: Model includes 2.5% broker + 1% tax
3. **Liquidity filtering**: Ensure sufficient volume for position size
4. **Limit orders**: Never market orders; use limit orders at predicted price
5. **Partial fills**: Accept partial order fulfillment

### Infrastructure Failures

**Risk:**
- Database downtime
- API service crashes
- Model serving unavailable

**Mitigation:**
1. **Health checks**: `/health` endpoint monitoring
2. **Database backups**: Daily automated backups to S3
3. **Redundancy**: Multi-container deployment
4. **Graceful degradation**: Fallback to simple moving average if LSTM fails
5. **Alerting**: PagerDuty/Slack on service failures

---

## Resource Estimates

### Computational Requirements

**Data Collection:**
- CPU: 2 cores (async I/O bound, not CPU intensive)
- RAM: 4 GB (caching API responses)
- Storage: 20 GB/quarter (order snapshots)
- Network: <1 Mbps average (5-min polling)

**Feature Engineering:**
- CPU: 4 cores (parallel feature computation)
- RAM: 8 GB (holding 90-day datasets)
- Storage: Additional 10 GB for feature store

**Model Training:**
- GPU: 1x RTX 3060 or better (CUDA 11+)
  - Alternative: CPU-only (slower, ~10x training time)
- RAM: 16 GB (large batch processing)
- Storage: 5 GB (model checkpoints, experiments)

**Production Serving:**
- CPU: 2 cores (inference is lightweight)
- RAM: 4 GB (loaded model + cache)
- Storage: 2 GB (current model only)

**Total Recommended Setup:**
- CPU: 8 cores
- RAM: 32 GB
- Storage: 100 GB SSD
- GPU: Optional but recommended for training

**Cloud Cost Estimate (AWS):**
- EC2 g4dn.xlarge (4 vCPU, 16GB, T4 GPU): ~$0.50/hour
- RDS PostgreSQL (db.t3.large): ~$0.15/hour
- ElastiCache Redis (cache.t3.medium): ~$0.07/hour
- **Total: ~$525/month** (assuming 24/7 operation)

### Development Time

**Phase 1 (Infrastructure):** 1 week
- Database setup: 1 day
- ESI client: 2 days
- Data models: 1 day
- Testing: 1 day

**Phase 2 (Data Collection):** 2 weeks
- Backfill historical: 2 days
- Order scraping: 3 days
- Validation: 2 days
- Monitoring dashboard: 2 days
- Buffer for issues: 3 days

**Phase 3 (EDA):** 1-2 weeks
- Statistical analysis: 3 days
- Correlation studies: 2 days
- Visualization: 2 days
- Documentation: 1 day

**Phase 4 (Features):** 1-2 weeks
- Technical indicators: 3 days
- Order book features: 2 days
- Game-specific features: 2 days
- Testing: 1 day

**Phase 5 (Baseline):** 1 week
- Implement models: 2 days
- Evaluation framework: 2 days
- Benchmarking: 1 day

**Phase 6 (LSTM):** 2 weeks
- Architecture: 2 days
- Training pipeline: 3 days
- Hyperparameter tuning: 4 days
- Evaluation: 2 days
- Optimization: 2 days

**Phase 7 (Backtesting):** 1-2 weeks
- Strategy implementation: 3 days
- Simulation: 2 days
- Metrics calculation: 2 days
- Analysis: 2 days

**Phase 8 (Production):** 2 weeks
- API development: 4 days
- Monitoring: 3 days
- Retraining automation: 2 days
- Documentation: 2 days
- Testing & deployment: 2 days

**Total Estimate: 10-13 weeks (2.5-3 months)**

---

## Next Steps

### Immediate Actions

1. **Set up development environment**
   ```bash
   # Create Python virtual environment
   python3.11 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install torch torchvision pandas numpy scikit-learn
   pip install httpx aiohttp ratelimit
   pip install psycopg2-binary redis
   pip install fastapi uvicorn pydantic
   pip install plotly dash jupyter
   ```

2. **Initialize database**
   ```bash
   # Install PostgreSQL + TimescaleDB
   sudo apt install postgresql-15 timescaledb-2-postgresql-15

   # Create database
   createdb eve_markets
   psql eve_markets -c "CREATE EXTENSION timescaledb;"
   ```

3. **Register ESI application**
   - Visit https://developers.eveonline.com/applications
   - Create new application
   - Note `client_id` and `client_secret`

4. **Collect initial data**
   - Identify top 100 traded items in Jita
   - Backfill 90 days historical data
   - Start 5-minute order scraping

### Validation Milestones

**Milestone 1 (Week 2):**
- ✓ 30+ days historical data collected
- ✓ Order snapshots running continuously
- ✓ Data quality >95% (no gaps, valid prices)

**Milestone 2 (Week 4):**
- ✓ EDA complete, item classification done
- ✓ Top 20 predictable items identified
- ✓ Feature engineering library tested

**Milestone 3 (Week 6):**
- ✓ Baseline models achieving MAE < 10%
- ✓ Evaluation framework validated
- ✓ Training pipeline automated

**Milestone 4 (Week 8):**
- ✓ LSTM model exceeding baseline by >20%
- ✓ Directional accuracy >60%
- ✓ Backtesting showing profit factor >1.5

**Milestone 5 (Week 10):**
- ✓ Production API deployed
- ✓ Monitoring dashboard live
- ✓ Automated retraining tested

### Success Criteria

**Minimum Viable Product (MVP):**
- Predicts prices for top 20 items in Jita
- Medium-term forecasts (1-7 days)
- Directional accuracy >55%
- MAE <8% of average price
- Backtested profit factor >1.2

**Production-Ready:**
- Covers top 100 items across 3 hubs
- Multi-horizon predictions (1h, 1d, 7d)
- Directional accuracy >60%
- MAE <5% of average price
- Backtested Sharpe ratio >1.0
- Automated retraining and monitoring

**Stretch Goals:**
- Real-time predictions (<1 min latency)
- Cross-item correlation modeling
- Arbitrage opportunity detection
- Alert system for trading signals
- Public API for community use

---

## References & Further Reading

**EVE Online API Documentation:**
- ESI Swagger Spec: https://esi.evetech.net/ui/
- Developer Resources: https://developers.eveonline.com/

**Academic Research:**
- "Stock Price Prediction Using LSTM" - various papers on arXiv
- "Deep Learning for Time Series Forecasting" - survey papers
- "Limit Order Book Analysis" - financial mathematics literature

**EVE Market Analysis:**
- EVE Market Strategies (book): https://orbitalenterprises.github.io/eve-market-strategies/
- EVE University Trading Guide: https://wiki.eveuniversity.org/Trading

**Technical Resources:**
- TimescaleDB Docs: https://docs.timescale.com/
- PyTorch Tutorials: https://pytorch.org/tutorials/
- FastAPI Docs: https://fastapi.tiangolo.com/

**Community Tools:**
- EveMarketer: https://evemarketer.com/
- Fuzzwork: https://www.fuzzwork.co.uk/
- Eve-Central (deprecated): Historical reference for market APIs

---

**Document Version:** 1.0
**Last Updated:** 2025-10-26
**Author:** EVE Discovery Project

