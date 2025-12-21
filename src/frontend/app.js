/**
 * Antigravity Trading Dashboard
 * JavaScript Application
 */

// Configuration
const CONFIG = {
    // For local development
    canisterId: 'uxrrr-q7777-77774-qaaaq-cai',
    host: 'http://127.0.0.1:8080',

    // For production on ICP
    // host: 'https://ic0.app',

    refreshInterval: 30000, // 30 seconds
};

// Sample data for demo (when canister not available)
const SAMPLE_DATA = {
    prices: {
        BTC: 88319,
        ETH: 2983.42,
        SOL: 126.12,
        XRP: 2.21,
        DOGE: 0.318,
        ADA: 0.91,
        AVAX: 38.45,
        DOT: 7.23,
        LINK: 22.87,
        ICP: 10.52
    },
    signals: [
        {
            pair: "SOL/LINK",
            action: "LONG_SPREAD",
            direction: { SOL: "BUY", LINK: "SELL" },
            timing: "NOW",
            timing_description: "Execute immediately - fast mean reversion",
            z_score: -2.34,
            half_life_days: 5.2,
            confidence_score: 75,
            confidence_level: "HIGH",
            entry_range: { SOL: [124, 128], LINK: [22, 24] },
            targets: { SOL: 135, LINK: 21 },
            potential_upside_pct: 3.5,
            risk_reward_ratio: 0.7
        },
        {
            pair: "ETH/DOT",
            action: "SHORT_SPREAD",
            direction: { ETH: "SELL", DOT: "BUY" },
            timing: "2H",
            timing_description: "Enter within 2 hours",
            z_score: 2.15,
            half_life_days: 8.7,
            confidence_score: 62,
            confidence_level: "MEDIUM",
            entry_range: { ETH: [2950, 3020], DOT: [7.0, 7.5] },
            targets: { ETH: 2850, DOT: 8.0 },
            potential_upside_pct: 2.8,
            risk_reward_ratio: 0.56
        },
        {
            pair: "BTC/ADA",
            action: "LONG_SPREAD",
            direction: { BTC: "BUY", ADA: "SELL" },
            timing: "2D",
            timing_description: "Position over 1-2 days",
            z_score: -1.89,
            half_life_days: 15.3,
            confidence_score: 55,
            confidence_level: "MEDIUM",
            entry_range: { BTC: [87000, 89500], ADA: [0.88, 0.94] },
            targets: { BTC: 92000, ADA: 0.85 },
            potential_upside_pct: 2.1,
            risk_reward_ratio: 0.42
        }
    ],
    cointegratedPairs: [
        { pair: "SOL/LINK", z_score: -2.34, half_life: 5.2, adf_p: 0.023 },
        { pair: "ETH/DOT", z_score: 2.15, half_life: 8.7, adf_p: 0.031 },
        { pair: "BTC/ADA", z_score: -1.89, half_life: 15.3, adf_p: 0.048 },
        { pair: "XRP/DOGE", z_score: 1.45, half_life: 12.1, adf_p: 0.039 },
        { pair: "AVAX/ICP", z_score: -1.12, half_life: 18.5, adf_p: 0.044 }
    ],
    state: {
        pairs_analyzed: 45,
        cointegrated_pairs: 5,
        active_signals: 3
    }
};

// State
let isConnected = false;
let walletPrincipal = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    setupEventListeners();
});

function initializeDashboard() {
    // Load with sample data immediately
    loadSampleData();

    // Try to connect to actual canister
    tryConnectCanister();
}

function setupEventListeners() {
    document.getElementById('connectBtn').addEventListener('click', handleConnect);
    document.getElementById('refreshSignals').addEventListener('click', refreshData);
}

async function handleConnect() {
    const btn = document.getElementById('connectBtn');

    if (isConnected) {
        // Disconnect
        isConnected = false;
        walletPrincipal = null;
        btn.innerHTML = '<span class="wallet-icon">👛</span> Connect Wallet';
        return;
    }

    btn.innerHTML = '<span class="wallet-icon">⏳</span> Connecting...';

    try {
        // Simulate wallet connection
        await new Promise(resolve => setTimeout(resolve, 1500));

        isConnected = true;
        walletPrincipal = 'dftva-uyo64-6nm3b-pdich-6mnrg...';
        btn.innerHTML = `<span class="wallet-icon">✓</span> ${walletPrincipal.slice(0, 8)}...`;
        btn.style.background = 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)';

    } catch (error) {
        console.error('Connection failed:', error);
        btn.innerHTML = '<span class="wallet-icon">👛</span> Connect Wallet';
    }
}

async function tryConnectCanister() {
    try {
        // In production, this would use @dfinity/agent
        console.log('Attempting to connect to canister...');

        // For now, we just use sample data
        // Real implementation would fetch from canister

    } catch (error) {
        console.log('Using sample data mode');
    }
}

function loadSampleData() {
    updateStats(SAMPLE_DATA.state);
    renderPrices(SAMPLE_DATA.prices);
    renderSignals(SAMPLE_DATA.signals);
    renderCointegratedPairs(SAMPLE_DATA.cointegratedPairs);
    updateLastUpdate();
}

async function refreshData() {
    const btn = document.getElementById('refreshSignals');
    btn.classList.add('spinning');

    // Show loading state
    document.getElementById('signalsContainer').innerHTML = `
        <div class="loading-state">
            <div class="spinner"></div>
            <p>Refreshing signals...</p>
        </div>
    `;

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Reload data
    loadSampleData();

    btn.classList.remove('spinning');
}

function updateStats(state) {
    document.getElementById('pairsAnalyzed').textContent = state.pairs_analyzed;
    document.getElementById('cointegratedPairs').textContent = state.cointegrated_pairs;
    document.getElementById('activeSignals').textContent = state.active_signals;
}

function updateLastUpdate() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
    });
    document.getElementById('lastUpdate').textContent = timeStr;
}

function renderPrices(prices) {
    const container = document.getElementById('pricesGrid');
    container.innerHTML = '';

    for (const [symbol, price] of Object.entries(prices)) {
        const card = document.createElement('div');
        card.className = 'price-card';
        card.innerHTML = `
            <div class="price-symbol">${symbol}</div>
            <div class="price-value">${formatPrice(price)}</div>
        `;
        container.appendChild(card);
    }
}

function renderSignals(signals) {
    const container = document.getElementById('signalsContainer');

    if (!signals || signals.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <p>No active trading signals</p>
            </div>
        `;
        return;
    }

    container.innerHTML = signals.map(signal => `
        <div class="signal-card">
            <div class="signal-header">
                <span class="signal-pair">${signal.pair}</span>
                <span class="signal-action ${getActionClass(signal.action)}">${signal.action.replace('_', ' ')}</span>
            </div>
            
            <div class="signal-details">
                <div class="signal-detail">
                    <span class="signal-detail-label">Z-Score</span>
                    <span class="signal-detail-value">${signal.z_score.toFixed(2)}σ</span>
                </div>
                <div class="signal-detail">
                    <span class="signal-detail-label">Half-Life</span>
                    <span class="signal-detail-value">${signal.half_life_days.toFixed(1)}d</span>
                </div>
                <div class="signal-detail">
                    <span class="signal-detail-label">R:R Ratio</span>
                    <span class="signal-detail-value">${signal.risk_reward_ratio}</span>
                </div>
            </div>
            
            <div class="signal-timing">
                <span class="timing-badge">${signal.timing}</span>
                <span class="timing-desc">${signal.timing_description}</span>
            </div>
            
            <div class="confidence-bar-container">
                <div class="confidence-bar-label">
                    <span>Confidence</span>
                    <span>${signal.confidence_level} (${signal.confidence_score}%)</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill" style="width: ${signal.confidence_score}%"></div>
                </div>
            </div>
        </div>
    `).join('');
}

function renderCointegratedPairs(pairs) {
    const container = document.getElementById('pairsContainer');

    if (!pairs || pairs.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🔗</div>
                <p>No cointegrated pairs found</p>
            </div>
        `;
        return;
    }

    container.innerHTML = pairs.map(pair => `
        <div class="pair-item">
            <span class="pair-name">${pair.pair}</span>
            <div class="pair-stats">
                <div class="pair-stat">
                    <span class="pair-stat-label">Z-Score</span>
                    <span class="pair-stat-value ${pair.z_score < 0 ? 'negative' : 'positive'}">${pair.z_score.toFixed(2)}</span>
                </div>
                <div class="pair-stat">
                    <span class="pair-stat-label">p-value</span>
                    <span class="pair-stat-value">${pair.adf_p.toFixed(3)}</span>
                </div>
            </div>
        </div>
    `).join('');
}

// Helpers
function formatPrice(price) {
    if (price >= 1000) {
        return '$' + price.toLocaleString('en-US', { maximumFractionDigits: 0 });
    } else if (price >= 1) {
        return '$' + price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    } else {
        return '$' + price.toFixed(4);
    }
}

function getActionClass(action) {
    if (action.includes('LONG')) return 'long';
    if (action.includes('SHORT')) return 'short';
    if (action.includes('PREPARE')) return 'prepare';
    return '';
}

// Auto-refresh
setInterval(() => {
    updateLastUpdate();
}, 30000);
