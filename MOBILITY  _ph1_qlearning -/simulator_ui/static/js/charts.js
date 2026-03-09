// charts.js — Real-Time Metric Charts for FANET Simulator

const Charts = {
    throughputChart: null,
    delayChart: null,
    queueChart: null,
    linkupChart: null,
    maxPoints: 300,

    init() {
        const baseOpts = (label, color, yLabel) => ({
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label,
                    data: [],
                    borderColor: color,
                    backgroundColor: color + '20',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        display: false,
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: '#5a6478',
                            font: { size: 9, family: "'JetBrains Mono'" },
                            maxTicksLimit: 4,
                        },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                    }
                },
                layout: { padding: { top: 18, right: 4, bottom: 2, left: 4 } },
            }
        });

        this.throughputChart = new Chart(
            document.getElementById('chart-throughput'),
            baseOpts('Throughput', '#00d4ff', 'Mbps')
        );
        this.delayChart = new Chart(
            document.getElementById('chart-delay'),
            baseOpts('Delay', '#ffab40', 'ms')
        );
        this.queueChart = new Chart(
            document.getElementById('chart-queue'),
            baseOpts('Queue', '#a855f7', 'pkts')
        );
        this.linkupChart = new Chart(
            document.getElementById('chart-linkup'),
            baseOpts('Link-Up', '#00e676', 'ratio')
        );
    },

    pushData(metrics, simTime) {
        const t = simTime.toFixed(1);
        this._push(this.throughputChart, t, metrics.tick_throughput_mbps);
        this._push(this.delayChart, t, metrics.tick_delay_ms);
        this._push(this.queueChart, t, metrics.avg_queue_len);
        this._push(this.linkupChart, t, metrics.link_up_ratio);
    },

    _push(chart, label, value) {
        chart.data.labels.push(label);
        chart.data.datasets[0].data.push(value);
        if (chart.data.labels.length > this.maxPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        chart.update('none'); // no animation
    },

    reset() {
        [this.throughputChart, this.delayChart, this.queueChart, this.linkupChart].forEach(c => {
            if (c) {
                c.data.labels = [];
                c.data.datasets[0].data = [];
                c.update('none');
            }
        });
    }
};
