const Charts = {
    charts: {},
    maxPoints: 180,

    init() {
        this.charts = {
            throughput: this._build('chart-throughput', 'Throughput', '#4ed8ff'),
            coord: this._build('chart-coord', 'Coordination', '#71e09c'),
            clusters: this._build('chart-clusters', 'Clusters', '#ffbe5c'),
            density: this._build('chart-density', 'Density', '#ff8aa5'),
        };
    },

    _build(canvasId, label, color) {
        return new Chart(document.getElementById(canvasId), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label,
                    data: [],
                    borderColor: color,
                    backgroundColor: `${color}33`,
                    pointRadius: 0,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.24,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: {
                        ticks: {
                            color: '#9db4d2',
                            font: { family: "'IBM Plex Mono'", size: 9 },
                            maxTicksLimit: 4,
                        },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                    },
                },
            },
        });
    },

    pushData(metrics, simTime) {
        const label = simTime.toFixed(1);
        this._push(this.charts.throughput, label, metrics.throughput_mbps || 0);
        this._push(this.charts.coord, label, metrics.coord_success || 0);
        this._push(this.charts.clusters, label, metrics.num_clusters || 0);
        this._push(this.charts.density, label, metrics.graph_density || 0);
    },

    _push(chart, label, value) {
        chart.data.labels.push(label);
        chart.data.datasets[0].data.push(value);
        if (chart.data.labels.length > this.maxPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        chart.update('none');
    },

    reset() {
        Object.values(this.charts).forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.update('none');
        });
    },

    resizeAll() {
        Object.values(this.charts).forEach(chart => chart.resize());
    },
};
