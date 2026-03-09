// main.js — Application Coordinator for FANET 3D Simulator

(function () {
    'use strict';

    let sceneBuilt = false;

    // ---- Initialize modules ----
    Scene.init();
    Charts.init();
    Panels.init();

    // ---- WebSocket message handlers ----
    WS.on('CONFIG', (msg) => {
        Panels.populateFromConfig(msg.data);
        // Build/rebuild scene with current bounds
        const c = msg.data;
        Scene.buildScene(
            [c.AREA_X, c.AREA_Y, c.AREA_Z],
            [c.SINK_X, c.SINK_Y, c.SINK_Z],
            c.COMM_RANGE_R
        );
        sceneBuilt = true;
        document.getElementById('sim-protocol').textContent = c.MAC_PROTOCOL || 'CSMA_CA';
    });

    WS.on('STATE', (msg) => {
        const state = msg.data;

        // Build scene on first state if not yet built
        if (!sceneBuilt) {
            Scene.buildScene(state.bounds, state.sink_pos, state.comm_range);
            sceneBuilt = true;
        }

        // Push into interpolation buffer (B2: smooth motion)
        Scene.pushState(state);

        // Update 3D scene (immediate — sets colors, link status, orientation)
        Scene.updateFromSnapshot(state);

        // Update metric badges
        const m = state.metrics;
        document.getElementById('val-throughput').textContent = m.throughput_mbps.toFixed(2) + ' Mbps';
        document.getElementById('val-delay').textContent = m.avg_delay_ms.toFixed(2) + ' ms';
        document.getElementById('val-drops').textContent = m.total_drops;
        document.getElementById('val-linkup').textContent = (m.link_up_ratio * 100).toFixed(0) + '%';

        // Update sim info bar
        document.getElementById('sim-time').textContent = 't = ' + state.sim_time.toFixed(2) + ' s';
        document.getElementById('sim-tick').textContent = 'tick: ' + state.tick;
        document.getElementById('sim-protocol').textContent = state.protocol;

        // Push to charts
        Charts.pushData(m, state.sim_time);
    });

    WS.on('ACK_CONFIG', (msg) => {
        showToast(`Config: ${msg.key} → ${msg.new} (${msg.mode})`);
    });

    WS.on('ACK', (msg) => {
        if (msg.action === 'RESET') {
            Charts.reset();
            sceneBuilt = false;
            WS.send('GET_CONFIG', {});
            showToast('Simulation reset');
        }
    });

    WS.on('EXPORT_DONE', (msg) => {
        showToast('Exported to: ' + msg.path);
    });

    // ---- Playback controls ----
    document.getElementById('btn-play').addEventListener('click', () => {
        WS.send('RUN', {});
    });

    document.getElementById('btn-pause').addEventListener('click', () => {
        WS.send('PAUSE', {});
    });

    document.getElementById('btn-step').addEventListener('click', () => {
        WS.send('STEP', {});
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        WS.send('RESET', {});
    });

    document.getElementById('btn-export').addEventListener('click', () => {
        WS.send('EXPORT', {});
    });

    // Speed slider
    const speedSlider = document.getElementById('speed-slider');
    const speedVal = document.getElementById('speed-val');
    speedSlider.addEventListener('input', () => {
        const factor = parseFloat(speedSlider.value);
        speedVal.textContent = factor.toFixed(1) + '×';
        WS.send('SET_SPEED', { factor });
    });

    // ---- Toast utility ----
    function showToast(message, duration = 2500) {
        const el = document.getElementById('toast');
        el.textContent = message;
        el.classList.remove('hidden');
        clearTimeout(el._timer);
        el._timer = setTimeout(() => el.classList.add('hidden'), duration);
    }

    // ---- Connect WebSocket ----
    WS.connect();

    console.log('[FANET 3D Simulator] Initialized');
})();
