"""EVE Online Market Dashboard with Dash/Plotly"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from sqlalchemy import select, func

from src.config import settings
from src.database import get_session, MarketHistory, Item, OrderSnapshot

# Initialize Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="EVE Price Prediction Dashboard"
)

# Define color scheme
COLORS = {
    'background': '#0a0e27',
    'card': '#1e2139',
    'text': '#ffffff',
    'primary': '#00d4ff',
    'secondary': '#ff6b6b',
    'success': '#51cf66',
    'warning': '#ffd43b',
}

# Available items
ITEMS = [
    {'label': 'PLEX', 'value': 44992},
    {'label': 'Large Skill Injector', 'value': 40520},
    {'label': 'Skill Extractor', 'value': 40519},
    {'label': 'Tritanium', 'value': 34},
    {'label': 'Pyerite', 'value': 35},
    {'label': 'Mexallon', 'value': 36},
    {'label': 'Isogen', 'value': 37},
    {'label': 'Nocxium', 'value': 38},
    {'label': 'Zydrine', 'value': 39},
    {'label': 'Megacyte', 'value': 40},
    {'label': 'Morphite', 'value': 11399},
]

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🚀 EVE Online Price Prediction Dashboard",
                   className="text-center mb-4 mt-4",
                   style={'color': COLORS['primary']}),
            html.P("Real-time market analysis and LSTM price forecasting for EVE Online",
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
                        value=40520,  # Large Skill Injector
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
                            {'label': 'All Time', 'value': 9999},
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
                        options=[
                            {'label': 'The Forge (Jita)', 'value': 10000002},
                            {'label': 'Domain (Amarr)', 'value': 10000043},
                            {'label': 'Sinq Laison (Dodixie)', 'value': 10000032},
                        ],
                        value=10000002,
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

    # Interval component for auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every 60 seconds
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


async def get_market_data(type_id, region_id, days):
    """Fetch market data from database"""
    cutoff_date = datetime.now() - timedelta(days=days)

    async with get_session() as session:
        # Get market history
        result = await session.execute(
            select(MarketHistory)
            .where(MarketHistory.type_id == type_id)
            .where(MarketHistory.region_id == region_id)
            .where(MarketHistory.time >= cutoff_date if days != 9999 else True)
            .order_by(MarketHistory.time)
        )
        records = result.scalars().all()

        # Get item name
        item_result = await session.execute(
            select(Item).where(Item.type_id == type_id)
        )
        item = item_result.scalar_one_or_none()
        item_name = item.name if item else "Unknown Item"

    return records, item_name


@callback(
    [Output('current-price', 'children'),
     Output('price-change', 'children'),
     Output('avg-price', 'children'),
     Output('total-volume', 'children'),
     Output('price-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('distribution-chart', 'figure')],
    [Input('item-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('timerange-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(type_id, region_id, days, n):
    """Update all dashboard components"""

    # Get data
    records, item_name = asyncio.run(get_market_data(type_id, region_id, days))

    if not records:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['card'],
            plot_bgcolor=COLORS['card'],
            annotations=[{
                'text': 'No data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20, 'color': COLORS['text']}
            }]
        )
        return "N/A", "N/A", "N/A", "N/A", empty_fig, empty_fig, empty_fig

    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': r.time,
        'price': float(r.average) if r.average else 0,
        'highest': float(r.highest) if r.highest else 0,
        'lowest': float(r.lowest) if r.lowest else 0,
        'volume': r.volume if r.volume else 0,
    } for r in records])

    # Calculate metrics
    current_price = df['price'].iloc[-1]
    price_30d_ago = df['price'].iloc[0] if len(df) > 30 else df['price'].iloc[0]
    price_change = ((current_price - price_30d_ago) / price_30d_ago) * 100
    avg_price = df['price'].mean()
    total_volume = df['volume'].sum()

    # Format metrics
    current_price_str = f"{format_isk(current_price)} ISK"
    price_change_str = f"{price_change:+.2f}%"
    avg_price_str = f"{format_isk(avg_price)} ISK"
    total_volume_str = format_isk(total_volume)

    # Price Chart
    price_fig = go.Figure()

    price_fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['highest'],
        name='High',
        line=dict(color=COLORS['success'], width=1),
        opacity=0.3,
        fill=None
    ))

    price_fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        name='Average',
        line=dict(color=COLORS['primary'], width=3),
        fill='tonexty'
    ))

    price_fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['lowest'],
        name='Low',
        line=dict(color=COLORS['secondary'], width=1),
        opacity=0.3,
        fill='tonexty'
    ))

    price_fig.update_layout(
        title=f"{item_name} Price History",
        xaxis_title="Date",
        yaxis_title="Price (ISK)",
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Volume Chart
    volume_fig = go.Figure()

    volume_fig.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        name='Volume',
        marker_color=COLORS['primary']
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
        x=df['price'],
        nbinsx=30,
        name='Price Distribution',
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
            price_fig, volume_fig, dist_fig)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8050,
        debug=True
    )
