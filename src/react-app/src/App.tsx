/**
 * Antigravity Trading Dashboard
 * Main Entry Point
 */

import React, { useState, useEffect } from 'react';
import { useCanister } from './hooks/useCanister';
import { StatusBadge } from './components/StatusBadge';
import { AgentPulse } from './components/AgentPulse';
import { SpreadChart } from './components/SpreadChart';
import { RiskGauge } from './components/RiskGauge';

// Types matching Backend `get_dashboard_state`
interface DashboardState {
    system: {
        version: string;
        tick_count: number;
        next_pair_index: number;
        total_pairs: number;
        logs: string[];
    };
    market: {
        prices: Record<string, number>;
        opportunities: Opportunity[];
    };
    cycles: number;
}

interface Opportunity {
    pair: string;
    action: string;
    confidence: number;
    risk_score: number;
    z_score: number;
    est_upside: number;
    adf_p_value: number;
    timestamp: number;
}

function App() {
    const { isConnected, isAuthenticated, principal, login, logout, queryCanister } = useCanister();

    const [state, setState] = useState<DashboardState | null>(null);
    const [selectedOpp, setSelectedOpp] = useState<Opportunity | null>(null);
    const [loading, setLoading] = useState(true);

    // Poll for Dashboard State
    useEffect(() => {
        const fetchState = async () => {
            if (!isConnected) return;
            try {
                const jsonStr = await queryCanister('get_dashboard_state');
                if (jsonStr) {
                    const data = JSON.parse(jsonStr);
                    setState(data);
                    setLoading(false);

                    // Auto-select first opportunity if none selected and opps exist
                    if (!selectedOpp && data.market.opportunities.length > 0) {
                        setSelectedOpp(data.market.opportunities[0]);
                    }
                }
            } catch (err) {
                console.error("Fetch Error:", err);
            }
        };

        fetchState();
        const interval = setInterval(fetchState, 1000); // 1s visual tick (canister limits?)
        return () => clearInterval(interval);
    }, [isConnected, queryCanister, selectedOpp]);

    // Mock chart data generator for now (since we haven't built history API fully yet)
    // In a real app, we'd fetch specific history for the selected pair.
    const chartData = React.useMemo(() => {
        // Just generate some random walk based on Z-Score for visual demo
        // Real implementation would pull from `engine.price_history` via a new endpoint
        const data = [];
        let val = 0;
        const now = Math.floor(Date.now() / 1000);
        for (let i = 0; i < 100; i++) {
            val += (Math.random() - 0.5);
            data.push({ time: now - (100 - i) * 60, value: val });
        }
        return data;
    }, [selectedOpp]); // TODO: Wire up real endpoint

    return (
        <div className="app-container" style={{
            color: '#eaecef',
            background: '#181a20',
            minHeight: '100vh',
            fontFamily: 'Inter, sans-serif'
        }}>
            {/* HEADER */}
            <header style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '1rem 2rem',
                borderBottom: '1px solid #2b3139',
                background: '#0b0e11'
            }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                    🚀 ANTIGRAVITY <span style={{ fontSize: '0.8rem', color: '#f0b90b' }}>AI AGENT</span>
                </div>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <StatusBadge />
                    {isAuthenticated ? (
                        <button onClick={logout} className="btn-secondary">Logout</button>
                    ) : (
                        <button onClick={login} className="btn-primary" style={{
                            background: '#f0b90b', color: '#000', border: 'none', padding: '8px 16px', borderRadius: '4px', fontWeight: 'bold', cursor: 'pointer'
                        }}>Login</button>
                    )}
                </div>
            </header>

            {/* DASHBOARD CONTENT */}
            <div className="dashboard-grid" style={{
                display: 'grid',
                gridTemplateColumns: '250px 1fr 300px',
                gap: '20px',
                padding: '20px'
            }}>

                {/* LEFT: MARKETS / OPPORTUNITIES */}
                <div className="panel-left">
                    <h3>Startups / Signals</h3>
                    <div className="opp-list">
                        {state?.market.opportunities.map((opp, i) => (
                            <div key={i}
                                onClick={() => setSelectedOpp(opp)}
                                style={{
                                    padding: '10px',
                                    background: selectedOpp === opp ? '#2b3139' : '#1e2329',
                                    marginBottom: '10px',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    border: selectedOpp === opp ? '1px solid #f0b90b' : 'none'
                                }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span style={{ fontWeight: 'bold' }}>{opp.pair}</span>
                                    <span style={{ color: opp.action.includes("LONG") ? '#0ecb81' : '#f6465d', fontSize: '0.8rem' }}>
                                        {opp.action}
                                    </span>
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '5px', fontSize: '0.75rem', color: '#848e9c' }}>
                                    <span>Conf: {opp.confidence}%</span>
                                    <span>Z: {opp.z_score}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                    {(!state?.market.opportunities || state.market.opportunities.length === 0) && (
                        <div style={{ color: '#848e9c', fontStyle: 'italic' }}>Scaning Markets...</div>
                    )}
                </div>

                {/* CENTER: CHART & PULSE */}
                <div className="panel-center" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <div style={{ background: '#1e2329', padding: '20px', borderRadius: '8px', minHeight: '350px' }}>
                        <h3 style={{ marginTop: 0 }}>Spread Spectrum: {selectedOpp?.pair || "Select active signal"}</h3>
                        {selectedOpp ? (
                            <SpreadChart data={chartData} pairName={selectedOpp.pair} />
                        ) : (
                            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#848e9c' }}>
                                Waiting for signals...
                            </div>
                        )}
                    </div>

                    <AgentPulse logs={state?.system.logs || []} />
                </div>

                {/* RIGHT: RISK & METRICS */}
                <div className="panel-right" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <div style={{ background: '#1e2329', padding: '20px', borderRadius: '8px' }}>
                        <h3 style={{ marginTop: 0 }}>AI Risk Assessment</h3>
                        {selectedOpp ? (
                            <RiskGauge score={selectedOpp.risk_score} />
                        ) : (
                            <div style={{ textAlign: 'center', color: '#848e9c' }}>--</div>
                        )}
                        <div style={{ marginTop: '20px', fontSize: '0.9rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                                <span>Est. Upside</span>
                                <span style={{ color: '#0ecb81' }}>+{selectedOpp?.est_upside || 0}%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>ADF P-Value</span>
                                <span>{selectedOpp?.adf_p_value || 0}</span>
                            </div>
                        </div>
                    </div>

                    <div style={{ background: '#1e2329', padding: '20px', borderRadius: '8px' }}>
                        <h3 style={{ marginTop: 0 }}>System Status</h3>
                        <div style={{ fontSize: '0.85rem', color: '#848e9c' }}>
                            <p>Version: {state?.system.version}</p>
                            <p>Heartbeat: {state?.system.tick_count}</p>
                            <p>Pair: {state?.system.next_pair_index}/{state?.system.total_pairs}</p>
                        </div>
                        <div style={{ marginTop: '15px', padding: '10px', background: '#0b0e11', borderRadius: '6px', textAlign: 'center' }}>
                            <div style={{ fontSize: '0.7rem', color: '#848e9c', textTransform: 'uppercase' }}>Agent Fuel</div>
                            <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#f0b90b' }}>
                                {state?.cycles ? (state.cycles / 1_000_000_000_000).toFixed(2) : '--'} T Cycles
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}

export default App;
