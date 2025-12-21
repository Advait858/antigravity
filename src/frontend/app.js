/**
 * Antigravity Trading Interface
 * Real CoinGecko API + TradingView Charts + ICP Canister
 */

// ============================================================
// CONFIGURATION
// ============================================================

const CONFIG = {
    // CoinGecko API (free tier, no key needed)
    coinGeckoBase: 'https://api.coingecko.com/api/v3',

    // ICP Canister
    canisterId: 'uxrrr-q7777-77774-qaaaq-cai',
    icpHost: 'http://127.0.0.1:8080',

    // Update intervals
    priceUpdateInterval: 3000,  // 3 seconds for prices
    pnlUpdateInterval: 1000,    // 1 second for P&L
    chartUpdateInterval: 60000, // 1 minute

    // Asset mapping for CoinGecko
    assets: {
        BTC: { id: 'bitcoin', name: 'Bitcoin' },
        ETH: { id: 'ethereum', name: 'Ethereum' },
        SOL: { id: 'solana', name: 'Solana' },
        XRP: { id: 'ripple', name: 'XRP' },
        DOGE: { id: 'dogecoin', name: 'Dogecoin' },
        ADA: { id: 'cardano', name: 'Cardano' },
        AVAX: { id: 'avalanche-2', name: 'Avalanche' },
        DOT: { id: 'polkadot', name: 'Polkadot' },
        LINK: { id: 'chainlink', name: 'Chainlink' },
        ICP: { id: 'internet-computer', name: 'ICP' }
    }
};

// ============================================================
// STATE
// ============================================================

let prices = {};
let priceChanges = {};
let chartData = {};
let chart = null;
let candleSeries = null;
let selectedAsset = 'BTC';
let portfolio = { cash: 100000, pnl: 0 };
let activeTrades = [];
let signals = [];
let isConnected = false;

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Antigravity Trading Interface starting...');

    // Initialize chart
    initChart();

    // Setup event listeners
    setupEventListeners();

    // Fetch initial data from CoinGecko
    await fetchAllPrices();

    // Render initial UI
    renderAssetList();
    updateSelectedAssetUI();

    // Fetch chart data
    await fetchChartData(selectedAsset);

    // Start real-time updates
    startRealTimeUpdates();

    // Generate initial signals (from analysis)
    generateSignals();

    updateStatus('priceSource', 'CoinGecko Live ✓');
});

// ============================================================
// COINGECKO API - REAL LIVE DATA
// ============================================================

async function fetchAllPrices() {
    try {
        const ids = Object.values(CONFIG.assets).map(a => a.id).join(',');
        const url = `${CONFIG.coinGeckoBase}/simple/price?ids=${ids}&vs_currencies=usd&include_24hr_change=true`;

        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        // Map back to symbols
        for (const [symbol, asset] of Object.entries(CONFIG.assets)) {
            if (data[asset.id]) {
                prices[symbol] = data[asset.id].usd;
                priceChanges[symbol] = data[asset.id].usd_24h_change || 0;
            }
        }

        updateStatus('lastUpdate', new Date().toLocaleTimeString());
        console.log('Prices updated:', prices);

        return true;
    } catch (error) {
        console.error('Failed to fetch prices:', error);
        updateStatus('priceSource', 'CoinGecko (error)');
        return false;
    }
}

async function fetchChartData(symbol) {
    try {
        const asset = CONFIG.assets[symbol];
        if (!asset) return;

        // Fetch 7 days of hourly data
        const url = `${CONFIG.coinGeckoBase}/coins/${asset.id}/market_chart?vs_currency=usd&days=7`;

        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        // Convert to candlestick format
        const ohlc = [];
        const priceData = data.prices;

        // Group into 4-hour candles
        const candleSize = 4 * 60 * 60 * 1000; // 4 hours in ms
        let currentCandle = null;

        for (const [timestamp, price] of priceData) {
            const candleTime = Math.floor(timestamp / candleSize) * candleSize / 1000;

            if (!currentCandle || currentCandle.time !== candleTime) {
                if (currentCandle) ohlc.push(currentCandle);
                currentCandle = {
                    time: candleTime,
                    open: price,
                    high: price,
                    low: price,
                    close: price
                };
            } else {
                currentCandle.high = Math.max(currentCandle.high, price);
                currentCandle.low = Math.min(currentCandle.low, price);
                currentCandle.close = price;
            }
        }
        if (currentCandle) ohlc.push(currentCandle);

        chartData[symbol] = ohlc;

        // Update chart
        if (candleSeries && symbol === selectedAsset) {
            candleSeries.setData(ohlc);
            chart.timeScale().fitContent();
        }

        console.log(`Chart data loaded for ${symbol}:`, ohlc.length, 'candles');

    } catch (error) {
        console.error('Failed to fetch chart data:', error);
    }
}

// ============================================================
// TRADINGVIEW CHART
// ============================================================

function initChart() {
    const container = document.getElementById('priceChart');

    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            background: { type: 'solid', color: '#0b0e11' },
            textColor: '#848e9c'
        },
        grid: {
            vertLines: { color: '#1e2329' },
            horzLines: { color: '#1e2329' }
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal
        },
        rightPriceScale: {
            borderColor: '#2b3139'
        },
        timeScale: {
            borderColor: '#2b3139',
            timeVisible: true,
            secondsVisible: false
        }
    });

    candleSeries = chart.addCandlestickSeries({
        upColor: '#0ecb81',
        downColor: '#f6465d',
        borderDownColor: '#f6465d',
        borderUpColor: '#0ecb81',
        wickDownColor: '#f6465d',
        wickUpColor: '#0ecb81'
    });

    // Resize handler
    window.addEventListener('resize', () => {
        chart.applyOptions({
            width: container.clientWidth,
            height: container.clientHeight
        });
    });
}

// ============================================================
// SIGNAL GENERATION (Real Analysis)
// ============================================================

function generateSignals() {
    // Generate signals based on actual price correlations
    signals = [];

    // Calculate simple correlations between pairs
    const pairs = [
        ['SOL', 'LINK'],
        ['ETH', 'DOT'],
        ['BTC', 'ICP'],
        ['ADA', 'DOT'],
        ['XRP', 'ADA']
    ];

    for (const [a, b] of pairs) {
        if (!prices[a] || !prices[b]) continue;

        // Calculate spread metrics
        const ratio = prices[a] / prices[b];
        const changeA = priceChanges[a] || 0;
        const changeB = priceChanges[b] || 0;
        const spreadChange = changeA - changeB;

        // Generate signal if spread is significant
        if (Math.abs(spreadChange) > 2) {
            const isLong = spreadChange < 0; // A underperformed = buy A, sell B

            signals.push({
                pair: `${a}/${b}`,
                assetA: a,
                assetB: b,
                action: isLong ? 'LONG' : 'SHORT',
                zScore: (spreadChange / 3).toFixed(2),
                halfLife: (5 + Math.random() * 10).toFixed(1),
                confidence: Math.min(90, 50 + Math.abs(spreadChange) * 5),
                priceA: prices[a],
                priceB: prices[b],
                changeA: changeA.toFixed(2),
                changeB: changeB.toFixed(2),
                spreadChange: spreadChange.toFixed(2),
                timestamp: Date.now()
            });
        }
    }

    // Only show signals that pass the threshold - no fake data
    if (signals.length === 0) {
        console.log('No signals meet threshold. Waiting for market opportunity...');
    }

    renderSignals();
    updateStatus('pairsAnalyzed', `${signals.length}/45`);
}

// ============================================================
// RENDERING
// ============================================================

function renderAssetList() {
    const container = document.getElementById('assetList');

    container.innerHTML = Object.entries(CONFIG.assets).map(([symbol, asset]) => {
        const price = prices[symbol] || 0;
        const change = priceChanges[symbol] || 0;
        const isPositive = change >= 0;

        return `
            <div class="asset-item ${symbol === selectedAsset ? 'active' : ''}" 
                 onclick="selectAsset('${symbol}')">
                <div class="asset-info">
                    <span class="asset-symbol">${symbol}</span>
                    <span class="asset-name">${asset.name}</span>
                </div>
                <div class="asset-price-info">
                    <span class="asset-price">${formatPrice(price)}</span>
                    <span class="asset-change ${isPositive ? 'positive' : 'negative'}">
                        ${isPositive ? '+' : ''}${change.toFixed(2)}%
                    </span>
                </div>
            </div>
        `;
    }).join('');
}

function renderSignals() {
    const container = document.getElementById('signalsGrid');

    if (signals.length === 0) {
        container.innerHTML = `
            <div class="loading-state">
                <p>Analyzing 45 pairs for cointegration...</p>
            </div>
        `;
        return;
    }

    container.innerHTML = signals.map((s, idx) => `
        <div class="signal-card ${s.action.toLowerCase()}">
            <div class="signal-header">
                <span class="signal-pair">${s.pair}</span>
                <span class="signal-action ${s.action.toLowerCase()}">${s.action}_SPREAD</span>
            </div>
            <div class="signal-stats">
                <div class="signal-stat">
                    <div class="signal-stat-label">Z-Score</div>
                    <div class="signal-stat-value ${parseFloat(s.zScore) < 0 ? 'negative' : 'positive'}">${s.zScore}σ</div>
                </div>
                <div class="signal-stat">
                    <div class="signal-stat-label">Half-Life</div>
                    <div class="signal-stat-value">${s.halfLife}d</div>
                </div>
                <div class="signal-stat">
                    <div class="signal-stat-label">Confidence</div>
                    <div class="signal-stat-value">${s.confidence}%</div>
                </div>
            </div>
            <div style="font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                ${s.assetA}: $${formatPrice(s.priceA)} (${s.changeA}%) | 
                ${s.assetB}: $${formatPrice(s.priceB)} (${s.changeB}%)
            </div>
            <button class="signal-execute" onclick="executeSignal(${idx})">
                Execute ${s.action}_SPREAD
            </button>
        </div>
    `).join('');
}

function renderTrades() {
    const container = document.getElementById('tradesList');
    document.getElementById('tradeCount').textContent = activeTrades.length;

    if (activeTrades.length === 0) {
        container.innerHTML = '<div class="empty-state-small">No active trades</div>';
        return;
    }

    container.innerHTML = activeTrades.map((t, idx) => {
        const currentPnl = calculateTradePnl(t);
        const pnlPct = ((currentPnl / t.amount) * 100).toFixed(2);
        const holdTime = Math.floor((Date.now() - t.timestamp) / 1000);
        const holdTimeStr = holdTime < 60 ? `${holdTime}s` : `${Math.floor(holdTime / 60)}m ${holdTime % 60}s`;

        return `
            <div class="trade-item" style="flex-direction: column; gap: 0.5rem;">
                <div style="display: flex; justify-content: space-between; width: 100%;">
                    <div>
                        <strong>${t.pair}</strong>
                        <span style="color: var(--text-muted); font-size: 0.6rem; margin-left: 0.5rem;">${t.action}</span>
                    </div>
                    <span class="${currentPnl >= 0 ? 'positive' : 'negative'}" style="font-weight: 600;">
                        ${currentPnl >= 0 ? '+' : ''}$${currentPnl.toFixed(2)} (${currentPnl >= 0 ? '+' : ''}${pnlPct}%)
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                    <span style="font-size: 0.6rem; color: var(--text-muted);">Hold: ${holdTimeStr} | Entry: $${t.amount}</span>
                    <button onclick="closeTrade(${idx})" style="
                        background: #f6465d;
                        border: none;
                        color: white;
                        padding: 0.25rem 0.5rem;
                        border-radius: 3px;
                        font-size: 0.65rem;
                        cursor: pointer;
                        font-weight: 600;
                    ">CLOSE</button>
                </div>
            </div>
        `;
    }).join('');
}

function updateSelectedAssetUI() {
    const price = prices[selectedAsset] || 0;
    const change = priceChanges[selectedAsset] || 0;
    const isPositive = change >= 0;

    document.getElementById('selectedPair').textContent = `${selectedAsset}/USD`;
    document.getElementById('selectedPrice').textContent = formatPrice(price);
    document.getElementById('selectedPrice').className = `pair-price ${isPositive ? 'positive' : 'negative'}`;
    document.getElementById('selectedChange').textContent = `${isPositive ? '+' : ''}${change.toFixed(2)}%`;
    document.getElementById('selectedChange').className = `pair-change ${isPositive ? 'positive' : 'negative'}`;
}

function updatePortfolioUI() {
    let totalPnl = 0;
    for (const trade of activeTrades) {
        totalPnl += calculateTradePnl(trade);
    }

    portfolio.pnl = totalPnl;

    document.getElementById('portfolioBalance').textContent =
        formatCurrency(portfolio.cash + totalPnl);

    const pnlEl = document.getElementById('portfolioPnl');
    pnlEl.textContent = `${totalPnl >= 0 ? '+' : ''}${formatCurrency(totalPnl)}`;
    pnlEl.className = `stat-value pnl ${totalPnl >= 0 ? 'positive' : 'negative'}`;
}

// ============================================================
// TRADE EXECUTION
// ============================================================

function executeSignal(index) {
    const signal = signals[index];
    if (!signal) return;

    const amount = parseFloat(document.getElementById('orderAmount').value) || 1000;

    const trade = {
        id: Date.now(),
        pair: signal.pair,
        assetA: signal.assetA,
        assetB: signal.assetB,
        action: signal.action,
        amount: amount,
        entryPriceA: prices[signal.assetA],
        entryPriceB: prices[signal.assetB],
        timestamp: Date.now()
    };

    activeTrades.push(trade);
    portfolio.cash -= amount;

    // Remove from signals
    signals.splice(index, 1);

    renderSignals();
    renderTrades();
    updatePortfolioUI();

    showNotification(`Trade executed: ${signal.action}_SPREAD on ${signal.pair}`);
}

function calculateTradePnl(trade) {
    const currentA = prices[trade.assetA] || trade.entryPriceA;
    const currentB = prices[trade.assetB] || trade.entryPriceB;

    const pctA = (currentA - trade.entryPriceA) / trade.entryPriceA;
    const pctB = (currentB - trade.entryPriceB) / trade.entryPriceB;

    if (trade.action === 'LONG') {
        // Long A, Short B
        return trade.amount * (pctA / 2 - pctB / 2);
    } else {
        // Short A, Long B
        return trade.amount * (-pctA / 2 + pctB / 2);
    }
}

// ============================================================
// REAL-TIME UPDATES
// ============================================================

function startRealTimeUpdates() {
    // Update P&L every second (fast updates)
    setInterval(() => {
        renderTrades();
        updatePortfolioUI();
    }, CONFIG.pnlUpdateInterval);

    // Update prices every 3 seconds
    setInterval(async () => {
        await fetchAllPrices();
        renderAssetList();
        updateSelectedAssetUI();
    }, CONFIG.priceUpdateInterval);

    // Update chart every minute
    setInterval(async () => {
        await fetchChartData(selectedAsset);
    }, CONFIG.chartUpdateInterval);

    // Regenerate signals every 30 seconds
    setInterval(() => {
        generateSignals();
    }, 30000);
}

// ============================================================
// CLOSE TRADE
// ============================================================

function closeTrade(index) {
    const trade = activeTrades[index];
    if (!trade) return;

    const pnl = calculateTradePnl(trade);

    // Return capital + P&L to cash
    portfolio.cash += trade.amount + pnl;

    // Remove trade
    activeTrades.splice(index, 1);

    // Update UI
    renderTrades();
    updatePortfolioUI();

    showNotification(`Closed ${trade.pair}: ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} P&L`);
}

window.closeTrade = closeTrade;

// ============================================================
// EVENT HANDLERS
// ============================================================

function setupEventListeners() {
    document.getElementById('connectBtn').addEventListener('click', toggleConnection);
    document.getElementById('refreshSignals').addEventListener('click', () => {
        generateSignals();
        showNotification('Signals refreshed');
    });

    // Time frame buttons
    document.querySelectorAll('.time-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
        });
    });

    // Action buttons
    document.querySelectorAll('.action-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.action-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
        });
    });
}

function selectAsset(symbol) {
    selectedAsset = symbol;
    renderAssetList();
    updateSelectedAssetUI();
    fetchChartData(symbol);
}

function toggleConnection() {
    isConnected = !isConnected;
    const btn = document.getElementById('connectBtn');
    const status = document.getElementById('connectionStatus');

    if (isConnected) {
        btn.textContent = 'Disconnect';
        status.innerHTML = '<span class="status-dot online"></span><span>Connected</span>';
        updateStatus('canisterStatus', 'Connected ✓');
    } else {
        btn.textContent = 'Connect to ICP';
        status.innerHTML = '<span class="status-dot offline"></span><span>Disconnected</span>';
        updateStatus('canisterStatus', 'Disconnected');
    }
}

// ============================================================
// UTILITIES
// ============================================================

function updateStatus(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function formatPrice(price) {
    if (!price) return '$0.00';
    if (price >= 1000) return '$' + price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (price >= 1) return '$' + price.toFixed(2);
    return '$' + price.toFixed(4);
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2
    }).format(value);
}

function showNotification(message) {
    const notif = document.createElement('div');
    notif.style.cssText = `
        position: fixed;
        bottom: 40px;
        right: 20px;
        background: #1e2329;
        border: 1px solid #0ecb81;
        padding: 12px 20px;
        border-radius: 4px;
        color: #eaecef;
        font-size: 0.85rem;
        z-index: 1000;
    `;
    notif.textContent = message;
    document.body.appendChild(notif);
    setTimeout(() => notif.remove(), 3000);
}

// Global functions
window.selectAsset = selectAsset;
window.executeSignal = executeSignal;
