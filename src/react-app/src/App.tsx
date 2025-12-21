/**
 * Antigravity Trading Dashboard
 * 
 * Main App Component with:
 * - Internet Identity login
 * - Live price chart from canister
 * - Trading signals from canister
 * - Portfolio management
 */

import { useState, useEffect, useRef } from 'react';
import { useCanister } from './hooks/useCanister';
import { StatusBadge } from './components/StatusBadge';
import { createChart, IChartApi, ISeriesApi } from 'lightweight-charts';

interface Price {
    symbol: string;
    price: number;
    change?: number;
}

interface Signal {
    pair: string;
    action: string;
    z_score: number;
    adf_p_value: number;
    confidence: number;
    half_life?: number;
}

interface Trade {
    id: number;
    pair: string;
    action: string;
    position_size: number;
    entry_z: number;
    current_pnl?: number;
}

function App() {
    const { isConnected, isAuthenticated, principal, login, logout, queryCanister, updateCanister } = useCanister();

    const [prices, setPrices] = useState<Record<string, number>>({});
    const [signals, setSignals] = useState<Signal[]>([]);
    const [trades, setTrades] = useState<Trade[]>([]);
    const [portfolio, setPortfolio] = useState({ cash: 100000, realized_pnl: 0 });
    const [selectedAsset, setSelectedAsset] = useState('BTC');
    const [loading, setLoading] = useState(true);
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

    // Chart refs
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<'Line'> | null>(null);

    // Initialize chart
    useEffect(() => {
        if (chartContainerRef.current && !chartRef.current) {
            chartRef.current = createChart(chartContainerRef.current, {
                width: chartContainerRef.current.clientWidth,
                height: 350,
                layout: { background: { color: '#0b0e11' }, textColor: '#848e9c' },
                grid: { vertLines: { color: '#1e2329' }, horzLines: { color: '#1e2329' } },
                rightPriceScale: { borderColor: '#2b3139' },
                timeScale: { borderColor: '#2b3139', timeVisible: true },
            });

            seriesRef.current = chartRef.current.addLineSeries({
                color: '#f0b90b',
                lineWidth: 2,
            });
        }

        return () => {
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
            }
        };
    }, []);

    // Fetch data from canister
    useEffect(() => {
        const fetchData = async () => {
            if (!isConnected) return;

            try {
                setLoading(true);

                // Fetch prices from CANISTER (not browser API)
                const pricesResult = await queryCanister('get_live_prices');
                if (pricesResult) {
                    const data = JSON.parse(pricesResult);
                    setPrices(data.prices || {});
                }

                // Fetch signals
                const signalsResult = await queryCanister('get_trading_signals');
                if (signalsResult) {
                    const data = JSON.parse(signalsResult);
                    setSignals(data.signals || []);
                }

                // Fetch trades
                const tradesResult = await queryCanister('get_active_trades');
                if (tradesResult) {
                    const data = JSON.parse(tradesResult);
                    setTrades(data.trades || []);
                }

                // Fetch chart data from CANISTER
                const candlesResult = await queryCanister('get_latest_candles', [selectedAsset, 100]);
                if (candlesResult && seriesRef.current) {
                    const data = JSON.parse(candlesResult);
                    if (data.prices && data.prices.length > 0) {
                        const chartData = data.prices.map((price: number, i: number) => ({
                            time: Math.floor(Date.now() / 1000) - (data.prices.length - i) * 60,
                            value: price,
                        }));
                        seriesRef.current.setData(chartData);
                    }
                }

                // Fetch portfolio if authenticated
                if (isAuthenticated && principal) {
                    const portfolioResult = await queryCanister('get_portfolio', [principal]);
                    if (portfolioResult) {
                        const data = JSON.parse(portfolioResult);
                        setPortfolio({ cash: data.cash || 100000, realized_pnl: data.realized_pnl || 0 });
                    }
                }

                setLastUpdate(new Date());
                setLoading(false);
            } catch (err) {
                console.error('Failed to fetch data:', err);
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 5000); // Poll every 5 seconds

        return () => clearInterval(interval);
    }, [isConnected, isAuthenticated, principal, selectedAsset, queryCanister]);

    // Execute trade
    const executeTrade = async (signalIndex: number) => {
        if (!isAuthenticated) {
            alert('Please login first');
            return;
        }

        try {
            const result = await updateCanister('execute_signal_trade', [signalIndex, 1000.0]);
            const data = JSON.parse(result);
            if (data.success) {
                alert(`Trade executed! Fees: $${data.fees?.total.toFixed(2)}`);
            } else {
                alert(`Trade failed: ${data.error}`);
            }
        } catch (err) {
            console.error('Trade error:', err);
        }
    };

    // Close trade
    const closeTrade = async (tradeId: number) => {
        try {
            const result = await updateCanister('close_trade', [tradeId]);
            const data = JSON.parse(result);
            if (data.success) {
                alert(`Trade closed! P&L: $${data.pnl?.toFixed(2)}`);
            }
        } catch (err) {
            console.error('Close trade error:', err);
        }
    };

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <div className="logo">🚀 ANTIGRAVITY</div>
                <div className="header-right">
                    <StatusBadge />
                    {isAuthenticated ? (
                        <>
                            <span style={{ fontSize: '0.8rem', color: '#848e9c' }}>
                                {principal?.slice(0, 10)}...
                            </span>
                            <button className="btn-login" onClick={logout}>Logout</button>
                        </>
                    ) : (
                        <button className="btn-login" onClick={login}>
                            Login with Internet Identity
                        </button>
                    )}
                </div>
            </header>

            {/* Main Content */}
            <div className="main-content">
                {/* Left Sidebar - Prices */}
                <div className="sidebar">
                    <h3 style={{ marginBottom: '1rem', fontSize: '0.9rem' }}>Markets (from Canister)</h3>
                    <ul className="price-list">
                        {Object.entries(prices).map(([symbol, price]) => (
                            <li
                                key={symbol}
                                className="price-item"
                                onClick={() => setSelectedAsset(symbol)}
                                style={{ background: selectedAsset === symbol ? '#2b3139' : 'transparent' }}
                            >
                                <span className="price-symbol">{symbol}</span>
                                <span className="price-value">${Number(price).toLocaleString()}</span>
                            </li>
                        ))}
                    </ul>
                </div>

                {/* Center - Chart + Signals */}
                <div className="center-panel">
                    <div className="chart-area" ref={chartContainerRef}>
                        {loading && (
                            <div className="loading">
                                <div className="spinner"></div>
                                Loading chart data from canister...
                            </div>
                        )}
                    </div>

                    <div className="signals-area">
                        <h3 style={{ marginBottom: '0.75rem', fontSize: '0.9rem' }}>
                            🤖 AI Signals (ADF p-values visible)
                        </h3>
                        {signals.length === 0 ? (
                            <p style={{ color: '#848e9c', fontSize: '0.85rem' }}>
                                Waiting for cointegration analysis...
                            </p>
                        ) : (
                            signals.map((s, i) => (
                                <div key={i} className={`signal-card ${s.action.includes('LONG') ? 'long' : 'short'}`}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                        <strong>{s.pair}</strong>
                                        <span style={{ color: s.action.includes('LONG') ? '#0ecb81' : '#f6465d' }}>
                                            {s.action}
                                        </span>
                                    </div>
                                    <div style={{ fontSize: '0.75rem', color: '#848e9c', marginBottom: '0.5rem' }}>
                                        Z-Score: <strong>{s.z_score}</strong> |
                                        ADF p-value: <strong>{s.adf_p_value}</strong> |
                                        Confidence: <strong>{s.confidence}%</strong>
                                        {s.half_life && <> | Half-life: <strong>{s.half_life}d</strong></>}
                                    </div>
                                    {isAuthenticated && (
                                        <button
                                            onClick={() => executeTrade(i)}
                                            style={{
                                                padding: '0.375rem 0.75rem',
                                                background: '#f0b90b',
                                                border: 'none',
                                                borderRadius: '4px',
                                                color: '#000',
                                                fontWeight: '600',
                                                fontSize: '0.75rem',
                                                cursor: 'pointer',
                                            }}
                                        >
                                            Execute ($1,000)
                                        </button>
                                    )}
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Right Panel - Portfolio */}
                <div className="right-panel">
                    <h3 style={{ marginBottom: '1rem', fontSize: '0.9rem' }}>Portfolio</h3>
                    <div className="portfolio-summary">
                        <div className="portfolio-stat">
                            <span className="stat-label">Cash</span>
                            <span className="stat-value">${portfolio.cash.toLocaleString()}</span>
                        </div>
                        <div className="portfolio-stat">
                            <span className="stat-label">Realized P&L</span>
                            <span className={`stat-value ${portfolio.realized_pnl >= 0 ? 'positive' : 'negative'}`}>
                                ${portfolio.realized_pnl.toFixed(2)}
                            </span>
                        </div>
                    </div>

                    <h4 style={{ marginBottom: '0.5rem', fontSize: '0.85rem' }}>Active Trades</h4>
                    {trades.length === 0 ? (
                        <p style={{ color: '#848e9c', fontSize: '0.8rem' }}>No active trades</p>
                    ) : (
                        trades.map((t) => (
                            <div
                                key={t.id}
                                style={{
                                    background: '#2b3139',
                                    padding: '0.75rem',
                                    borderRadius: '4px',
                                    marginBottom: '0.5rem',
                                    fontSize: '0.8rem',
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <strong>{t.pair}</strong>
                                    <span className={t.action.includes('LONG') ? 'positive' : 'negative'}>
                                        {t.action}
                                    </span>
                                </div>
                                <div style={{ color: '#848e9c', marginTop: '0.25rem' }}>
                                    Entry Z: {t.entry_z} | Size: ${t.position_size}
                                </div>
                                <button
                                    onClick={() => closeTrade(t.id)}
                                    style={{
                                        marginTop: '0.5rem',
                                        padding: '0.25rem 0.5rem',
                                        background: '#f6465d',
                                        border: 'none',
                                        borderRadius: '3px',
                                        color: '#fff',
                                        fontSize: '0.7rem',
                                        cursor: 'pointer',
                                    }}
                                >
                                    Close
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Status Bar */}
            <footer className="status-bar">
                <span>Canister: {isConnected ? 'Connected' : 'Disconnected'}</span>
                <span>Last Update: {lastUpdate ? lastUpdate.toLocaleTimeString() : '--'}</span>
                <span>Signals: {signals.length}</span>
                <span>Active Trades: {trades.length}</span>
            </footer>
        </div>
    );
}

export default App;
