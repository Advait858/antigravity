/**
 * Antigravity Trading Dashboard v5.1
 * Trade Proposals with User Approval
 */

// Configuration
const CONFIG = {
    canisterId: 'uxrrr-q7777-77774-qaaaq-cai',
    host: 'http://127.0.0.1:8080',
    refreshInterval: 60000, // 60 seconds
    proposalExpiryMinutes: 5, // Proposals expire after 5 minutes
};

// State
let countdown = 60;
let isConnected = false;
let walletPrincipal = null;
let activeProposals = [];
let activeTrades = [];
let portfolio = { cash: 100000, total_value: 100000, pnl: 0, pnl_pct: 0 };
let currentPrices = {};
let selectedProposal = null;

// Sample data with detailed reasoning
const generateProposals = () => {
    const now = Date.now();
    return [
        {
            id: 1,
            pair: "SOL/LINK",
            asset_a: "SOL",
            asset_b: "LINK",
            action: "LONG_SPREAD",
            direction: { SOL: "BUY", LINK: "SELL" },
            z_score: -2.34,
            half_life_days: 5.2,
            hedge_ratio: 5.52,
            correlation: 0.847,
            r_squared: 0.72,
            adf_p_value: 0.023,
            confidence_score: 78,
            confidence_level: "HIGH",
            entry_prices: { SOL: 126.12, LINK: 22.87 },
            position_size: 10000,
            potential_profit: 350,
            potential_loss: 500,
            risk_reward: 0.7,
            created_at: now,
            expires_at: now + (CONFIG.proposalExpiryMinutes * 60 * 1000),
            reason: "Spread is 2.34σ below historical mean. Strong cointegration (p=0.023) with fast mean reversion (5.2 day half-life). SOL undervalued relative to LINK based on 90-day regression.",
            why_now: "Z-score reached entry threshold. Historical analysis shows 78% win rate at this level."
        },
        {
            id: 2,
            pair: "ETH/DOT",
            asset_a: "ETH",
            asset_b: "DOT",
            action: "SHORT_SPREAD",
            direction: { ETH: "SELL", DOT: "BUY" },
            z_score: 2.15,
            half_life_days: 8.7,
            hedge_ratio: 412.5,
            correlation: 0.912,
            r_squared: 0.83,
            adf_p_value: 0.031,
            confidence_score: 65,
            confidence_level: "MEDIUM",
            entry_prices: { ETH: 2983.42, DOT: 7.23 },
            position_size: 8000,
            potential_profit: 240,
            potential_loss: 400,
            risk_reward: 0.6,
            created_at: now - 120000,
            expires_at: now + ((CONFIG.proposalExpiryMinutes - 2) * 60 * 1000),
            reason: "Spread is 2.15σ above mean. ETH overvalued vs DOT in the short term. Cointegration confirmed across 60 and 90-day windows.",
            why_now: "Divergence peaked. Mean reversion typically begins within 8-9 days at this Z-score level."
        },
        {
            id: 3,
            pair: "BTC/ICP",
            asset_a: "BTC",
            asset_b: "ICP",
            action: "LONG_SPREAD",
            direction: { BTC: "BUY", ICP: "SELL" },
            z_score: -1.89,
            half_life_days: 12.3,
            hedge_ratio: 8401.5,
            correlation: 0.756,
            r_squared: 0.57,
            adf_p_value: 0.048,
            confidence_score: 52,
            confidence_level: "MEDIUM",
            entry_prices: { BTC: 88319, ICP: 10.52 },
            position_size: 6000,
            potential_profit: 180,
            potential_loss: 300,
            risk_reward: 0.6,
            created_at: now - 180000,
            expires_at: now + ((CONFIG.proposalExpiryMinutes - 3) * 60 * 1000),
            reason: "BTC showing relative weakness vs ICP. Approaching entry threshold. Cointegration marginally significant (p=0.048).",
            why_now: "Near entry level. Consider waiting for Z < -2 for stronger signal or enter now for earlier positioning."
        }
    ];
};

const SAMPLE_PRICES = {
    BTC: 88319, ETH: 2983.42, SOL: 126.12, XRP: 2.21, DOGE: 0.318,
    ADA: 0.91, AVAX: 38.45, DOT: 7.23, LINK: 22.87, ICP: 10.52
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    currentPrices = { ...SAMPLE_PRICES };
    activeProposals = generateProposals();

    initializeDashboard();
    setupEventListeners();
    startCountdown();
});

function initializeDashboard() {
    renderPortfolio();
    renderPrices();
    renderProposals();
    renderActiveTrades();
}

function setupEventListeners() {
    document.getElementById('connectBtn').addEventListener('click', handleConnect);
    document.getElementById('refreshProposals').addEventListener('click', refreshData);
    document.getElementById('modalClose').addEventListener('click', closeModal);
    document.getElementById('modalOverlay').addEventListener('click', (e) => {
        if (e.target.id === 'modalOverlay') closeModal();
    });
    document.getElementById('btnReject').addEventListener('click', rejectProposal);
    document.getElementById('btnAccept').addEventListener('click', acceptProposal);
}

function startCountdown() {
    setInterval(() => {
        countdown--;
        if (countdown <= 0) {
            countdown = 60;
            refreshData();
        }
        document.getElementById('nextUpdate').textContent = `${countdown}s`;

        // Update proposal expiry times
        updateExpiryTimes();
    }, 1000);
}

function updateExpiryTimes() {
    const now = Date.now();
    activeProposals = activeProposals.filter(p => p.expires_at > now);

    activeProposals.forEach(p => {
        const remaining = Math.max(0, Math.floor((p.expires_at - now) / 1000));
        const minutes = Math.floor(remaining / 60);
        const seconds = remaining % 60;
        p.timeRemaining = `${minutes}:${String(seconds).padStart(2, '0')}`;
        p.isUrgent = remaining < 60;
    });

    renderProposals();
}

async function handleConnect() {
    const btn = document.getElementById('connectBtn');

    if (isConnected) {
        isConnected = false;
        walletPrincipal = null;
        btn.innerHTML = '<span class="wallet-icon">👛</span> Connect Wallet';
        btn.style.background = '';
        return;
    }

    btn.innerHTML = '⏳ Connecting...';
    await new Promise(r => setTimeout(r, 1500));

    isConnected = true;
    walletPrincipal = 'dftva-uyo64-6nm3b...';
    btn.innerHTML = `✓ ${walletPrincipal}`;
    btn.style.background = 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)';
}

async function refreshData() {
    document.getElementById('agentStatus').textContent = 'Analyzing...';

    // Simulate API refresh
    await new Promise(r => setTimeout(r, 1000));

    // Regenerate proposals with new timestamps
    activeProposals = generateProposals();

    renderProposals();
    renderPrices();

    document.getElementById('agentStatus').textContent = 'Agent Active';
}

function renderPortfolio() {
    document.getElementById('totalValue').textContent = formatCurrency(portfolio.total_value);
    document.getElementById('cashValue').textContent = formatCurrency(portfolio.cash);

    const pnlEl = document.getElementById('pnlValue');
    pnlEl.textContent = (portfolio.pnl >= 0 ? '+' : '') + formatCurrency(portfolio.pnl);
    pnlEl.className = `portfolio-value pnl ${portfolio.pnl >= 0 ? 'positive' : 'negative'}`;

    document.getElementById('positionsCount').textContent = activeTrades.length;
}

function renderPrices() {
    const container = document.getElementById('pricesGrid');
    container.innerHTML = Object.entries(currentPrices).map(([symbol, price]) => `
        <div class="price-card">
            <div class="price-symbol">${symbol}</div>
            <div class="price-value">${formatPrice(price)}</div>
        </div>
    `).join('');
}

function renderProposals() {
    const container = document.getElementById('proposalsContainer');
    document.getElementById('proposalCount').textContent = `${activeProposals.length} active`;

    if (activeProposals.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <p>No trade proposals available</p>
                <p style="font-size: 0.8rem; margin-top: 0.5rem;">Agent is analyzing 45 pairs...</p>
            </div>
        `;
        return;
    }

    container.innerHTML = activeProposals.map(p => `
        <div class="proposal-card ${p.action.includes('LONG') ? 'long' : 'short'}">
            <div class="proposal-header">
                <div class="proposal-pair">
                    <span class="proposal-pair-name">${p.pair}</span>
                    <span class="proposal-action ${p.action.includes('LONG') ? 'long' : 'short'}">
                        ${p.action.replace('_', ' ')}
                    </span>
                </div>
                <div class="proposal-expiry">
                    <span class="expiry-label">Expires in</span>
                    <span class="expiry-time ${p.isUrgent ? 'urgent' : ''}">${p.timeRemaining || '5:00'}</span>
                </div>
            </div>
            
            <div class="proposal-stats">
                <div class="proposal-stat">
                    <span class="proposal-stat-label">Z-Score</span>
                    <span class="proposal-stat-value ${p.z_score < 0 ? 'negative' : 'positive'}">${p.z_score.toFixed(2)}σ</span>
                </div>
                <div class="proposal-stat">
                    <span class="proposal-stat-label">Half-Life</span>
                    <span class="proposal-stat-value">${p.half_life_days.toFixed(1)} days</span>
                </div>
                <div class="proposal-stat">
                    <span class="proposal-stat-label">Correlation</span>
                    <span class="proposal-stat-value">${(p.correlation * 100).toFixed(1)}%</span>
                </div>
                <div class="proposal-stat">
                    <span class="proposal-stat-label">ADF p-value</span>
                    <span class="proposal-stat-value">${p.adf_p_value.toFixed(3)}</span>
                </div>
            </div>
            
            <div class="proposal-reason">
                <div class="proposal-reason-title">🤖 AI Analysis</div>
                <div class="proposal-reason-text">${p.reason}</div>
            </div>
            
            <div class="proposal-pnl">
                <div class="pnl-item">
                    <div class="pnl-item-label">Position Size</div>
                    <div class="pnl-item-value">${formatCurrency(p.position_size)}</div>
                </div>
                <div class="pnl-item">
                    <div class="pnl-item-label">Potential Profit</div>
                    <div class="pnl-item-value profit">+${formatCurrency(p.potential_profit)}</div>
                </div>
                <div class="pnl-item">
                    <div class="pnl-item-label">Max Risk</div>
                    <div class="pnl-item-value loss">-${formatCurrency(p.potential_loss)}</div>
                </div>
            </div>
            
            <div class="confidence-meter">
                <div class="confidence-header">
                    <span class="confidence-label">Confidence</span>
                    <span class="confidence-value ${p.confidence_level.toLowerCase()}">${p.confidence_level} (${p.confidence_score}%)</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill ${p.confidence_level.toLowerCase()}" style="width: ${p.confidence_score}%"></div>
                </div>
            </div>
            
            <div class="proposal-actions">
                <button class="btn-proposal reject" onclick="showModal(${p.id}, 'reject')">
                    ❌ Pass
                </button>
                <button class="btn-proposal accept" onclick="showModal(${p.id}, 'accept')">
                    ✅ Execute Trade
                </button>
            </div>
        </div>
    `).join('');
}

function renderActiveTrades() {
    const container = document.getElementById('tradesContainer');

    if (activeTrades.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <p>No active trades</p>
            </div>
        `;
        return;
    }

    container.innerHTML = activeTrades.map(t => `
        <div class="trade-item">
            <div class="trade-info">
                <span class="trade-pair">${t.pair}</span>
                <span class="trade-meta">${t.action} • Opened ${formatTimeAgo(t.opened_at)}</span>
            </div>
            <span class="trade-pnl ${t.pnl >= 0 ? 'positive' : 'negative'}">
                ${t.pnl >= 0 ? '+' : ''}${formatCurrency(t.pnl)}
            </span>
        </div>
    `).join('');
}

function showModal(proposalId, action) {
    selectedProposal = activeProposals.find(p => p.id === proposalId);
    if (!selectedProposal) return;

    const body = document.getElementById('modalBody');

    if (action === 'accept') {
        body.innerHTML = `
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">
                    ${selectedProposal.action.includes('LONG') ? '📈' : '📉'}
                </div>
                <div style="font-size: 1.25rem; font-weight: 600; font-family: var(--font-mono);">
                    ${selectedProposal.pair}
                </div>
                <div style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.25rem;">
                    ${selectedProposal.action.replace('_', ' ')}
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Position Size</div>
                        <div style="font-size: 1.125rem; font-weight: 600;">${formatCurrency(selectedProposal.position_size)}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Expected Return</div>
                        <div style="font-size: 1.125rem; font-weight: 600; color: var(--success);">+${formatCurrency(selectedProposal.potential_profit)}</div>
                    </div>
                </div>
            </div>
            
            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
                <strong>Trade Details:</strong><br>
                • ${selectedProposal.direction[selectedProposal.asset_a]} ${selectedProposal.asset_a} @ ${formatPrice(selectedProposal.entry_prices[selectedProposal.asset_a])}<br>
                • ${selectedProposal.direction[selectedProposal.asset_b]} ${selectedProposal.asset_b} @ ${formatPrice(selectedProposal.entry_prices[selectedProposal.asset_b])}
            </div>
            
            <div style="font-size: 0.8rem; color: var(--warning); background: var(--warning-bg); padding: 0.75rem; border-radius: 8px;">
                ⚠️ This is paper trading. No real funds will be used.
            </div>
        `;
        document.getElementById('btnAccept').style.display = 'block';
        document.getElementById('btnReject').textContent = '❌ Cancel';
    } else {
        body.innerHTML = `
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🙅</div>
                <p>Skip this trade proposal?</p>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
                    The proposal will be dismissed and won't appear again in this cycle.
                </p>
            </div>
        `;
        document.getElementById('btnAccept').style.display = 'none';
        document.getElementById('btnReject').textContent = '✓ Confirm Skip';
    }

    document.getElementById('modalOverlay').classList.add('active');
}

function closeModal() {
    document.getElementById('modalOverlay').classList.remove('active');
    selectedProposal = null;
}

function acceptProposal() {
    if (!selectedProposal) return;

    // Execute the trade
    const trade = {
        id: Date.now(),
        pair: selectedProposal.pair,
        action: selectedProposal.action,
        position_size: selectedProposal.position_size,
        entry_prices: selectedProposal.entry_prices,
        opened_at: Date.now(),
        pnl: 0
    };

    activeTrades.push(trade);

    // Update portfolio
    portfolio.cash -= selectedProposal.position_size;

    // Remove from proposals
    activeProposals = activeProposals.filter(p => p.id !== selectedProposal.id);

    closeModal();
    renderPortfolio();
    renderProposals();
    renderActiveTrades();

    // Show success notification
    showNotification(`✅ Trade executed: ${trade.action.replace('_', ' ')} on ${trade.pair}`);
}

function rejectProposal() {
    if (selectedProposal) {
        activeProposals = activeProposals.filter(p => p.id !== selectedProposal.id);
    }
    closeModal();
    renderProposals();
}

function showNotification(message) {
    const notif = document.createElement('div');
    notif.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: var(--bg-card);
        border: 1px solid var(--success);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 0.9rem;
        z-index: 2000;
        animation: slideIn 0.3s ease;
    `;
    notif.textContent = message;
    document.body.appendChild(notif);

    setTimeout(() => notif.remove(), 3000);
}

// Helpers
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

function formatPrice(price) {
    if (price >= 1000) return '$' + price.toLocaleString('en-US', { maximumFractionDigits: 0 });
    if (price >= 1) return '$' + price.toFixed(2);
    return '$' + price.toFixed(4);
}

function formatTimeAgo(timestamp) {
    const diff = Date.now() - timestamp;
    const minutes = Math.floor(diff / 60000);
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    return `${Math.floor(minutes / 60)}h ago`;
}

// Make functions globally accessible
window.showModal = showModal;
