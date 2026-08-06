"""EVE Online Market Dashboard with Dash/Plotly.

Runs synchronously (Dash callbacks are sync) but talks to the async SQLAlchemy
engine through :func:`src.database.run_async`, which owns one long-lived event
loop. Calling ``asyncio.run()`` per callback used to create a new loop each
time while the pooled asyncpg connections stayed bound to the first one,
producing intermittent "attached to a different loop" errors on refresh.
"""

import logging
import os
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html
from sqlalchemy import select

from src.config import settings
from src.constants import TRACKED_ITEMS, region_name
from src.database import Item, MarketHistory, Prediction, get_session, run_async
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Initialize Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="EVE Price Prediction Dashboard"
)
#: WSGI entry point for gunicorn: ``gunicorn src.dashboard.app:server``
server = app.server

# Define color scheme
COLORS = {
    'background': '#0a0e27',
    'card': '#1e2139',
    'text': '#ffffff',
    'primary': '#00d4ff',
    'secondary': '#ff6b6b',
    'success': '#51cf66',
    'warning': '#ffd43b',
    'prediction': '#c77dff',
}

# Items and regions come from the shared constants / settings so the dropdowns
# can never advertise something the collector does not gather.
ITEMS = [{'label': name, 'value': type_id} for type_id, name in TRACKED_ITEMS.items()]
DEFAULT_ITEM = ITEMS[0]['value']

REGION_OPTIONS = [
    {'label': region_name(region_id), 'value': region_id}
    for region_id in settings.region_ids
]
DEFAULT_REGION = settings.region_ids[0]

ALL_TIME = 9999

# Short-lived query cache: daily data does not change between page refreshes.
_CACHE_TTL_SECONDS = max(settings.dashboard_refresh_seconds // 2, 30)
_cache: dict = {}
_cache_lock = threading.Lock()


# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🚀 EVE Online Price Prediction Dashboard",
                   className="text-center mb-4 mt-4",
                   style={'color': COLORS['primary']}),
            html.P("Market analysis and LSTM price forecasting for EVE Online",
                  className="text-center text-muted mb-4")
        ])
    ]),

    # Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Select Item:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='item-dropdown',
                        options=ITEMS,
                        value=DEFAULT_ITEM,
                        clearable=False,
                        style={'color': '#000'}
                    ),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Time Range:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='timerange-dropdown',
                        options=[
                            {'label': 'Last 30 Days', 'value': 30},
                            {'label': 'Last 90 Days', 'value': 90},
                            {'label': 'Last 6 Months', 'value': 180},
                            {'label': 'Last Year', 'value': 365},
                            {'label': 'All Time', 'value': ALL_TIME},
                        ],
                        value=90,
                        clearable=False,
                        style={'color': '#000'}
                    ),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Region:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=REGION_OPTIONS,
                        value=DEFAULT_REGION,
                        clearable=False,
                        style={'color': '#000'}
                    ),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=4),
    ]),

    # Stats Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Current Price", className="text-muted"),
                    html.H3(id='current-price', className="text-success"),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("30-Day Change", className="text-muted"),
                    html.H3(id='price-change', className="text-warning"),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Average Price", className="text-muted"),
                    html.H3(id='avg-price', className="text-info"),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Volume", className="text-muted"),
                    html.H3(id='total-volume', className="text-primary"),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=3),
    ]),

    # Price Chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Price History & Predictions", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='price-chart', config={'displayModeBar': False}),
                    html.Div(id='prediction-note', className="text-muted small mt-2"),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=12),
    ]),

    # Volume Chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Trading Volume", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='volume-chart', config={'displayModeBar': False}),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Price Distribution", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='distribution-chart', config={'displayModeBar': False}),
                ])
            ], className="mb-4", style={'backgroundColor': COLORS['card']})
        ], md=6),
    ]),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("EVE Online Price Prediction Engine | Powered by LSTM Neural Networks",
                  className="text-center text-muted small mb-4")
        ])
    ]),

    # Interval component for auto-refresh. Market history updates once a day,
    # so a 60s poll just re-queried and re-rendered the same rows for everyone.
    dcc.Interval(
        id='interval-component',
        interval=settings.dashboard_refresh_seconds * 1000,
        n_intervals=0
    )

], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})


def format_isk(value):
    """Format ISK values with proper notation"""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value/1_000:.2f}K"
    return f"{value:.2f}"


def _cached(key, producer):
    """Memoize a value for ``_CACHE_TTL_SECONDS``."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL_SECONDS:
            return hit[1]

    value = producer()

    with _cache_lock:
        _cache[key] = (now, value)
    return value


async def _fetch_market_data(type_id, region_id, days):
    """Fetch market history, item name and the latest prediction set."""
    query = (
        select(MarketHistory)
        .where(MarketHistory.type_id == type_id)
        .where(MarketHistory.region_id == region_id)
    )
    if days != ALL_TIME:
        # Conditionally omit the clause rather than passing a bare `True`.
        query = query.where(MarketHistory.time >= utcnow() - timedelta(days=days))

    async with get_session() as session:
        records = (
            await session.execute(query.order_by(MarketHistory.time))
        ).scalars().all()

        item = (
            await session.execute(select(Item).where(Item.type_id == type_id))
        ).scalar_one_or_none()
        item_name = item.name if item else TRACKED_ITEMS.get(type_id, "Unknown Item")

        latest_prediction_time = (
            await session.execute(
                select(Prediction.prediction_time)
                .where(Prediction.type_id == type_id)
                .where(Prediction.region_id == region_id)
                .order_by(Prediction.prediction_time.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        predictions = []
        if latest_prediction_time is not None:
            predictions = (
                await session.execute(
                    select(Prediction)
                    .where(Prediction.type_id == type_id)
                    .where(Prediction.region_id == region_id)
                    .where(Prediction.prediction_time == latest_prediction_time)
                    .order_by(Prediction.prediction_horizon)
                )
            ).scalars().all()

    history = pd.DataFrame([{
        'date': r.time,
        'price': float(r.average) if r.average else 0.0,
        'highest': float(r.highest) if r.highest else 0.0,
        'lowest': float(r.lowest) if r.lowest else 0.0,
        'volume': r.volume or 0,
    } for r in records])

    forecast = pd.DataFrame([{
        'date': p.target_time,
        'predicted_price': float(p.predicted_price),
        'actual_price': float(p.actual_price) if p.actual_price is not None else None,
        'model_version': p.model_version,
        'prediction_time': p.prediction_time,
    } for p in predictions if p.target_time is not None])

    return history, forecast, item_name


def get_market_data(type_id, region_id, days):
    """Cached, synchronous accessor used by the callback."""
    return _cached(
        ("market", type_id, region_id, days),
        lambda: run_async(_fetch_market_data(type_id, region_id, days)),
    )


def thirty_day_change(df: pd.DataFrame) -> float:
    """Percent change against the observation closest to 30 days ago.

    The card says "30-Day Change", so it must not silently mean "change over
    whatever range is selected".
    """
    if df.empty:
        return float('nan')

    current_price = df['price'].iloc[-1]
    cutoff = df['date'].iloc[-1] - pd.Timedelta(days=30)
    earlier = df[df['date'] <= cutoff]

    if earlier.empty:
        return float('nan')

    baseline = earlier['price'].iloc[-1]
    if not baseline:
        return float('nan')
    return (current_price - baseline) / baseline * 100


def _empty_figure(message='No data available'):
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        annotations=[{
            'text': message,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 20, 'color': COLORS['text']}
        }]
    )
    return fig


@callback(
    [Output('current-price', 'children'),
     Output('price-change', 'children'),
     Output('avg-price', 'children'),
     Output('total-volume', 'children'),
     Output('price-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('distribution-chart', 'figure'),
     Output('prediction-note', 'children')],
    [Input('item-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('timerange-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(type_id, region_id, days, n):
    """Update all dashboard components"""
    try:
        df, forecast, item_name = get_market_data(type_id, region_id, days)
    except Exception as e:
        logger.exception(f"Dashboard query failed: {e}")
        error_fig = _empty_figure('Database unavailable')
        return ("N/A", "N/A", "N/A", "N/A", error_fig, error_fig, error_fig,
                "Could not reach the database.")

    if df.empty:
        empty_fig = _empty_figure()
        return ("N/A", "N/A", "N/A", "N/A", empty_fig, empty_fig, empty_fig,
                f"No market history collected for {region_name(region_id)} yet.")

    # Metrics
    current_price = df['price'].iloc[-1]
    change = thirty_day_change(df)
    avg_price = df['price'].mean()
    total_volume = df['volume'].sum()

    current_price_str = f"{format_isk(current_price)} ISK"
    price_change_str = "N/A" if pd.isna(change) else f"{change:+.2f}%"
    avg_price_str = f"{format_isk(avg_price)} ISK"
    total_volume_str = format_isk(total_volume)

    # Price Chart
    price_fig = go.Figure()

    price_fig.add_trace(go.Scatter(
        x=df['date'], y=df['highest'], name='High',
        line=dict(color=COLORS['success'], width=1), opacity=0.3, fill=None
    ))
    price_fig.add_trace(go.Scatter(
        x=df['date'], y=df['price'], name='Average',
        line=dict(color=COLORS['primary'], width=3), fill='tonexty'
    ))
    price_fig.add_trace(go.Scatter(
        x=df['date'], y=df['lowest'], name='Low',
        line=dict(color=COLORS['secondary'], width=1), opacity=0.3, fill='tonexty'
    ))

    note = "No stored predictions yet - run scripts/predict_all_items.py."
    if not forecast.empty:
        # Join the forecast onto the last observed point so the line is continuous
        bridge = pd.concat([
            pd.DataFrame([{'date': df['date'].iloc[-1], 'predicted_price': current_price}]),
            forecast[['date', 'predicted_price']],
        ])
        price_fig.add_trace(go.Scatter(
            x=bridge['date'], y=bridge['predicted_price'], name='Predicted',
            line=dict(color=COLORS['prediction'], width=2, dash='dash'),
        ))

        scored = forecast[forecast['actual_price'].notna()]
        if not scored.empty:
            price_fig.add_trace(go.Scatter(
                x=scored['date'], y=scored['actual_price'], name='Actual (scored)',
                mode='markers', marker=dict(color=COLORS['warning'], size=7),
            ))

        version = forecast['model_version'].iloc[0] or 'unknown'
        generated = forecast['prediction_time'].iloc[0]
        note = (
            f"Forecast from model {version}, generated {generated:%Y-%m-%d %H:%M} UTC. "
            f"Multi-day values are autoregressive and compound error with horizon."
        )

    price_fig.update_layout(
        title=f"{item_name} Price History & Forecast",
        xaxis_title="Date",
        yaxis_title="Price (ISK)",
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Volume Chart
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'], name='Volume', marker_color=COLORS['primary']
    ))
    volume_fig.update_layout(
        title=f"{item_name} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        hovermode='x unified'
    )

    # Distribution Chart
    dist_fig = go.Figure()
    dist_fig.add_trace(go.Histogram(
        x=df['price'], nbinsx=30, name='Price Distribution',
        marker_color=COLORS['primary']
    ))
    dist_fig.update_layout(
        title=f"{item_name} Price Distribution",
        xaxis_title="Price (ISK)",
        yaxis_title="Frequency",
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card']
    )

    return (current_price_str, price_change_str, avg_price_str, total_volume_str,
            price_fig, volume_fig, dist_fig, note)


if __name__ == '__main__':
    logging.basicConfig(level=settings.log_level)

    # debug=True enables the Werkzeug interactive debugger. On 0.0.0.0 inside a
    # container that is a remote-code-execution surface, so it is opt-in via
    # DASHBOARD_DEBUG and off by default. For production use gunicorn:
    #   gunicorn -b 0.0.0.0:8050 -w 2 src.dashboard.app:server
    debug = settings.dashboard_debug
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1" if debug else "0.0.0.0")

    if debug:
        logger.warning("Dash debug mode is ON - never expose this beyond localhost")

    app.run(host=host, port=8050, debug=debug)
