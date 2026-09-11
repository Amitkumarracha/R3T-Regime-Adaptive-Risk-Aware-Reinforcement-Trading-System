let apexChart = null;
let currentTicker = null;
let currentInterval = '5m';
let currentType = 'candle';

async function fetchStocks() {
    try {
        const response = await fetch('/api/stocks');
        const data = await response.json();
        
        const stocks = data.stocks || [];
        const global_cash = data.global_cash || 0;
        
        let aggValue = global_cash;
        let positionsPnl = 0;

        stocks.forEach(stock => {
            // Price Update
            const priceEl = document.getElementById(`price-${stock.ticker}`);
            if(priceEl) priceEl.innerText = `₹ ${stock.last_price.toFixed(2)}`;
            
            // PNL Update
            const pnlEl = document.getElementById(`pnl-${stock.ticker}`);
            if(pnlEl) {
                const pnl = stock.net_pnl || 0;
                pnlEl.innerText = pnl >= 0 ? `+ ₹${pnl.toFixed(2)}` : `- ₹${Math.abs(pnl).toFixed(2)}`;
                pnlEl.className = `text-sm font-bold font-mono ${pnl >= 0 ? 'text-pt-success glow-green' : 'text-pt-danger glow-red'}`;
            }

            // Action Update
            const actionEl = document.getElementById(`action-${stock.ticker}`);
            if(actionEl) {
                actionEl.innerText = stock.last_action;
                // dynamic bg colors for action
                if(stock.last_action === 'BUY') {
                    actionEl.className = `px-2 py-1 rounded text-[10px] font-bold font-mono uppercase border border-pt-success text-pt-success bg-pt-success/10`;
                } else if(stock.last_action === 'SELL') {
                    actionEl.className = `px-2 py-1 rounded text-[10px] font-bold font-mono uppercase border border-pt-danger text-pt-danger bg-pt-danger/10`;
                } else {
                    actionEl.className = `px-2 py-1 rounded text-[10px] font-bold font-mono uppercase border border-slate-600 text-slate-400 bg-slate-800/50`;
                }
            }

            // Position Update
            const posEl = document.getElementById(`pos-${stock.ticker}`);
            if(posEl) posEl.innerText = `QTY: ${stock.position}`;

            // Aggregation - Add PositionValue to global cash for Total Value
            aggValue += (stock.total_value || 0);
        });
        
        const INITIAL_GLOBAL_CASH = 15000.0;
        const netAggregatePnl = aggValue - INITIAL_GLOBAL_CASH;

        // Top Header Totals Update
        document.getElementById('agg-value').innerText = `₹ ${aggValue.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        
        const aggPnlEl = document.getElementById('agg-pnl');
        aggPnlEl.innerText = netAggregatePnl >= 0 ? `+ ₹${netAggregatePnl.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : `- ₹${Math.abs(netAggregatePnl).toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        aggPnlEl.className = `text-xl font-bold font-mono ${netAggregatePnl >= 0 ? 'text-pt-success glow-green' : 'text-pt-danger glow-red'}`;

        document.getElementById('last-update').innerText = "Last updated: " + new Date().toLocaleTimeString();

    } catch (e) {
        console.error("Failed to fetch stocks", e);
    }
}

async function fetchMarketVitals() {
    try {
        const response = await fetch('/api/market_vitals');
        const data = await response.json();
        
        // Update Nifty
        if(data.indices['NIFTY 50']) {
            const nifty = data.indices['NIFTY 50'];
            document.getElementById('idx-nifty-price').innerText = `₹ ${nifty.price.toLocaleString()}`;
            const changeEl = document.getElementById('idx-nifty-change');
            changeEl.innerText = nifty.change.toFixed(2);
            changeEl.className = `text-[10px] font-mono ${nifty.change >= 0 ? 'text-pt-success' : 'text-pt-danger'}`;
        }
        
        // Update Bank Nifty
        if(data.indices['BANK NIFTY']) {
            const bank = data.indices['BANK NIFTY'];
            document.getElementById('idx-bank-price').innerText = `₹ ${bank.price.toLocaleString()}`;
            const changeEl2 = document.getElementById('idx-bank-change');
            changeEl2.innerText = bank.change.toFixed(2);
            changeEl2.className = `text-[10px] font-mono ${bank.change >= 0 ? 'text-pt-success' : 'text-pt-danger'}`;
        }

        // Update Expiry Tag
        const tagEl = document.getElementById('market-expiry-tag');
        if(data.expiry.is_expiry_day) {
            tagEl.innerText = `${data.expiry.expiry_type} EXPIRY`;
            tagEl.className = "text-[10px] font-mono text-amber-500 font-bold glow-red";
        } else {
            tagEl.innerText = "REGULAR SESSION";
            tagEl.className = "text-[10px] font-mono text-pt-accent";
        }
        
    } catch (e) {
        console.error("Vitals fetch failed", e);
    }
}

async function fetchPortfolio() {
    try {
        const response = await fetch('/api/portfolio');
        const data = await response.json();
        
        // Update Bars
        const cats = { "ETF": "etf", "MIDCAP": "mid", "HEAVYWEIGHT": "heavy" };
        for(let key in cats) {
            const prefix = cats[key];
            const used = data.usage[key] || 0;
            const limit = data.limits[key] || 1;
            const pct = Math.min(100, (used / limit) * 100);
            
            document.getElementById(`basket-${prefix}-val`).innerText = `₹${used.toLocaleString()}`;
            document.getElementById(`basket-${prefix}-bar`).style.width = `${pct}%`;
        }
    } catch (e) {
        console.error("Portfolio fetch failed", e);
    }
}

async function openModal(ticker, sector) {
    const modal = document.getElementById('stockModal');
    modal.classList.remove('hidden');
    
    // Add small delay to trigger CSS animation
    setTimeout(() => {
        modal.classList.remove('hidden-panel');
        modal.classList.add('show-panel');
        document.body.style.overflow = 'hidden';
    }, 10);
    
    document.getElementById('modal-ticker').innerText = ticker;
    document.getElementById('modal-sector').innerText = sector;
    
    currentTicker = ticker;
    currentInterval = '15m'; // Reset to default when opening another stock
    currentType = 'candle';
    
    // Reset toggle styles
    updateToggleUI();
    
    // Get domain from the clicked card
    const card = document.querySelector(`.glass-card[data-ticker="${ticker}"]`);
    const domain = card ? card.getAttribute('data-domain') : 'nseindia.com';
    
    // Set Logo in modal
    document.getElementById('modal-logo-container').innerHTML = `
        <img src="https://www.google.com/s2/favicons?domain=${domain}&sz=128" alt="${ticker}" class="w-full h-full object-contain p-1" onerror="this.onerror=null; this.parentElement.innerHTML='<span class=\\'text-slate-900 font-bold text-2xl\\'>${ticker[0]}</span>';">
    `;

    // Fetch logs
    try {
        const response = await fetch(`/api/logs/${ticker}`);
        const logs = await response.json();
        
        const tbody = document.getElementById('logs-body');
        tbody.innerHTML = '';
        
        if(logs.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="py-4 text-center text-slate-500 italic">No trading history available</td></tr>`;
        } else {
            logs.reverse().forEach(log => {
                const tr = document.createElement('tr');
                const pnl = parseFloat(log.NetPnL || 0);
                const pnlClass = pnl > 0 ? "text-pt-success" : (pnl < 0 ? "text-pt-danger" : "text-slate-500");
                
                let actionSpan = `<span class="text-slate-500">${log.Action}</span>`;
                if(log.Action === 'BUY') actionSpan = `<span class="text-pt-accent">${log.Action}</span>`;
                if(log.Action === 'SELL') actionSpan = `<span class="text-pt-danger">${log.Action}</span>`;

                tr.innerHTML = `
                    <td class="py-3 pr-4">${log.EntryTime.split(' ')[1]}</td>
                    <td class="py-3 pr-4">${actionSpan}</td>
                    <td class="py-3 pr-4 text-right">${log.Qty}</td>
                    <td class="py-3 pr-4 text-right">₹${parseFloat(log.Entry).toFixed(2)}</td>
                    <td class="py-3 text-right ${pnlClass}">${pnl ? '₹' + pnl.toFixed(2) : '--'}</td>
                `;
                tbody.appendChild(tr);
            });
        }
        
        // Mock Signal Reasons (This would ideally be dynamic API data)
        document.getElementById('signal-reasons').innerHTML = `
           <div class="flex gap-2 items-start"><span class="text-pt-accent">→</span> <span>EMA_5 (${ticker}) checking crossover with EMA_200.</span></div>
           <div class="flex gap-2 items-start"><span class="text-pt-accent">→</span> <span>Chart Signal currently evaluating as <strong>NEUTRAL</strong></span></div>
           <div class="flex gap-2 items-start"><span class="text-pt-accent">→</span> <span>DQN Agent has override authority due to NEUTRAL chart state.</span></div>
        `;

        if (!apexChart) {
            initApexChart();
        }
        
        // Fetch new timeframe data
        await loadChartData();
        
        
    } catch (e) {
        console.error("Failed to fetch logs", e);
    }
}

function closeModal() {
    const modal = document.getElementById('stockModal');
    modal.classList.remove('show-panel');
    modal.classList.add('hidden-panel');
    
    setTimeout(() => {
        modal.classList.add('hidden');
        document.body.style.overflow = '';
    }, 300);
}

function initApexChart() {
    const chartContainer = document.getElementById('tv-chart');
    chartContainer.innerHTML = '';
    
    const options = {
        series: [],
        chart: {
            type: 'candlestick',
            height: '100%',
            background: '#0D1117',
            toolbar: { show: false },
            animations: { enabled: false }
        },
        theme: { mode: 'dark' },
        xaxis: {
            type: 'datetime',
            labels: { style: { colors: '#888', fontFamily: 'JetBrains Mono' } },
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: [
            {
                seriesName: 'Price',
                tooltip: { enabled: true },
                labels: { style: { colors: '#888', fontFamily: 'JetBrains Mono' }, formatter: (value) => { return value ? value.toFixed(2) : value; } }
            },
            { seriesName: 'Price', show: false },
            { seriesName: 'Price', show: false },
            { seriesName: 'Price', show: false },
            {
                seriesName: 'Volume',
                opposite: true,
                show: false,
                max: (max) => { return max * 4; } // Push volume to bottom 25% of chart
            }
        ],
        grid: {
            borderColor: 'rgba(255, 255, 255, 0.05)',
            strokeDashArray: 0,
            padding: { top: 0, right: 0, bottom: 0, left: 10 }
        },
        plotOptions: {
            candlestick: {
                colors: { upward: '#3FB950', downward: '#F85149' },
                wick: { useDataColor: true }
            }
        },
        stroke: {
            width: [2, 1, 1, 2, 0],
            curve: 'smooth'
        },
        colors: ['#4493F8', '#10B981', '#FFFFFF', '#EAB308', 'rgba(255,255,255,0.05)'], // baseLine, EMA5, SMA50, EMA200, Volume
        tooltip: { theme: 'dark' },
        legend: { show: false }
    };

    apexChart = new ApexCharts(chartContainer, options);
    apexChart.render();
}

async function loadChartData() {
    if (!currentTicker || !apexChart) return;
    
    try {
        const response = await fetch(`/api/chart/${currentTicker}?interval=${currentInterval}`);
        if (!response.ok) throw new Error("Data not available");
        const data = await response.json();
        
        const masterSeries = {
            name: 'Price',
            type: currentType === 'candle' ? 'candlestick' : 'line',
            data: currentType === 'candle' ? data.candles : data.lines
        };
        
        apexChart.updateSeries([
            masterSeries,
            { name: 'EMA 5', type: 'line', data: data.ema5 },
            { name: 'SMA 50', type: 'line', data: data.sma50 },
            { name: 'EMA 200', type: 'line', data: data.ema200 },
            { name: 'Volume', type: 'bar', data: data.volume }
        ]);
        
    } catch (e) {
        console.error("Chart data error:", e);
    }
}

function setChartType(type) {
    if (currentType === type) return;
    currentType = type;
    updateToggleUI();
    loadChartData();
}

function setTimeframe(tf) {
    if (currentInterval === tf) return;
    currentInterval = tf;
    updateToggleUI();
    loadChartData();
}

function updateToggleUI() {
    // Type toggles
    document.querySelectorAll('#chart-type-toggles button').forEach(btn => {
        if (btn.getAttribute('data-type') === currentType) {
            btn.className = "px-3 py-1 rounded-lg text-xs font-bold font-mono tracking-wide bg-pt-accent/20 text-pt-accent border border-pt-accent/50";
        } else {
            btn.className = "px-3 py-1 rounded-lg text-xs font-bold font-mono tracking-wide bg-white/5 text-slate-400 border border-white/10 hover:bg-white/10 transition-colors";
        }
    });

    // Timeframe toggles
    document.querySelectorAll('#chart-tf-toggles button').forEach(btn => {
        if (btn.getAttribute('data-tf') === currentInterval) {
            btn.className = "px-3 py-1 rounded-lg text-xs font-bold font-mono tracking-wide bg-pt-accent/20 text-pt-accent border border-pt-accent/50";
        } else {
            btn.className = "px-3 py-1 rounded-lg text-xs font-bold font-mono tracking-wide bg-white/5 text-slate-400 border border-white/10 hover:bg-white/10 transition-colors";
        }
    });
}

// Initial fetch & loop
fetchStocks();
fetchMarketVitals();
fetchPortfolio();
setInterval(fetchStocks, 10000);
setInterval(fetchMarketVitals, 15000);
setInterval(fetchPortfolio, 30000);

// Close modal on outside click
document.addEventListener('click', function(event) {
    const modal = document.getElementById('stockModal');
    const panel = modal.querySelector('.glass-panel');
    if (event.target === modal) {
        closeModal();
    }
});
document.addEventListener('keydown', function(e) {
    if(e.key === 'Escape') closeModal();
});

// Pause Feature
const togglePauseBtn = document.getElementById('toggle-pause-btn');
if(togglePauseBtn) {
    togglePauseBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/pause/toggle', { method: 'POST' });
            const data = await res.json();
            // We just reload the page to cleanly reflect the state via jinja 
            // layout, or we can hot-swap the elements
            window.location.reload();
        } catch(e) {
            console.error("Pause API failed", e);
        }
    });
}
