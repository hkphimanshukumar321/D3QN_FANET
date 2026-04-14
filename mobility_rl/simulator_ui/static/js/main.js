(function () {
    'use strict';

    let latestState = null;
    let simRunning = false;

    function showToast(message, duration = 2400) {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.classList.remove('hidden');
        window.clearTimeout(toast._timer);
        toast._timer = window.setTimeout(() => toast.classList.add('hidden'), duration);
    }

    function policyCompatText(runtime) {
        if (!runtime) {
            return 'compat unknown';
        }
        if (runtime.policy_compatible) {
            return 'compatible';
        }
        return 'incompatible';
    }

    function ensureSelections(state) {
        if (!state || !state.clusters || state.clusters.length === 0) {
            Panels.selectedClusterId = null;
            Panels.selectedNodeId = null;
            Scene.setSelectedCluster(null);
            Scene.setSelectedNode(null);
            return;
        }

        const clusters = state.clusters;
        const nodes = state.nodes || [];
        const clusterExists = clusters.some(cluster => cluster.cluster_id === Panels.selectedClusterId);
        if (!clusterExists) {
            Panels.selectedClusterId = clusters[0].cluster_id;
        }

        const selectedCluster = clusters.find(cluster => cluster.cluster_id === Panels.selectedClusterId) || clusters[0];
        const nodeExists = nodes.some(node => node.id === Panels.selectedNodeId);
        const nodeInsideCluster = Array.isArray(selectedCluster.members)
            && selectedCluster.members.includes(Panels.selectedNodeId);
        if (!nodeExists || !nodeInsideCluster) {
            Panels.selectedNodeId = selectedCluster.leader_id;
        }

        Scene.setSelectedCluster(Panels.selectedClusterId);
        Scene.setSelectedNode(Panels.selectedNodeId);
    }

    function updateStatusStrip(state) {
        const tdmaCount = (state.clusters || []).filter(cluster => cluster.mac_label === 'TDMA').length;
        const csmaCount = (state.clusters || []).filter(cluster => cluster.mac_label === 'CSMA_CA').length;
        const warning = state.runtime.policy_warning || state.runtime.policy_compatibility_note || 'No warnings';

        document.getElementById('status-session').textContent = simRunning ? 'Running' : 'Paused';
        document.getElementById('status-time').textContent = `t = ${state.sim_time.toFixed(2)} s`;
        document.getElementById('status-tick').textContent = String(state.tick);
        document.getElementById('status-policy').textContent = state.runtime.policy_label;
        document.getElementById('status-policy-source').textContent = `${state.runtime.policy_source} / ${state.runtime.policy_type}`;
        document.getElementById('status-mac').textContent = `${tdmaCount} TDMA / ${csmaCount} CSMA`;
        document.getElementById('status-policy-compat').textContent = policyCompatText(state.runtime);
        document.getElementById('status-preset').textContent = state.scenario.preset_label;
        document.getElementById('status-study-block').textContent = state.scenario.study_block || state.scenario.preset_group;
        document.getElementById('status-warning').textContent = warning;
        document.getElementById('warning-card').classList.toggle('warning', !!warning && warning !== 'No warnings');
    }

    function renderState(state) {
        if (latestState && state.tick < latestState.tick) {
            Charts.reset();
            Panels.resetEventFeed();
        }

        latestState = state;
        ensureSelections(state);
        Scene.updateFromSnapshot(state);
        updateStatusStrip(state);

        Panels.renderScenario(state.scenario);
        Panels.renderRuntime(state.runtime, state.clusters || []);
        Panels.renderClusterList(state.clusters, Panels.selectedClusterId);
        Panels.renderClusterInspector(state, Panels.selectedClusterId);
        Panels.renderNodeInspector(state, Panels.selectedNodeId);
        Panels.renderGraphInspector(state);
        Panels.renderEventFeed(state.events?.feed || []);
        Panels.renderDiagnostics(state);

        Charts.pushData(state.metrics, state.sim_time);
    }

    Scene.init();
    Charts.init();
    Panels.init();
    Panels.onNotice = showToast;

    Panels.onClusterSelected = (clusterId) => {
        Panels.selectedClusterId = clusterId;
        if (latestState) {
            const cluster = latestState.clusters.find(item => item.cluster_id === clusterId);
            if (cluster) {
                Panels.selectedNodeId = cluster.leader_id;
            }
            Scene.setSelectedCluster(clusterId);
            Scene.setSelectedNode(Panels.selectedNodeId);
            Panels.renderClusterList(latestState.clusters, clusterId);
            Panels.renderClusterInspector(latestState, clusterId);
            Panels.renderNodeInspector(latestState, Panels.selectedNodeId);
            Scene.updateFromSnapshot(latestState);
        }
    };

    Panels.onNodeSelected = (nodeId) => {
        Panels.selectedNodeId = nodeId;
        if (latestState) {
            const node = latestState.nodes.find(item => item.id === nodeId);
            if (node && node.cluster_id >= 0) {
                Panels.selectedClusterId = node.cluster_id;
            }
            Scene.setSelectedCluster(Panels.selectedClusterId);
            Scene.setSelectedNode(nodeId);
            Panels.renderClusterList(latestState.clusters, Panels.selectedClusterId);
            Panels.renderClusterInspector(latestState, Panels.selectedClusterId);
            Panels.renderNodeInspector(latestState, nodeId);
            Scene.updateFromSnapshot(latestState);
        }
    };

    Scene.onNodeSelected = (nodeId) => {
        if (typeof Panels.onNodeSelected === 'function') {
            Panels.onNodeSelected(nodeId);
        }
    };

    WS.on('CONFIG', (msg) => {
        Panels.populateConfig(msg.data);
    });

    WS.on('STATE', (msg) => {
        renderState(msg.data);
    });

    WS.on('ACK_CONFIG', (msg) => {
        const value = msg.new === null
            ? 'inherit preset/default'
            : (typeof msg.new === 'object' ? JSON.stringify(msg.new) : msg.new);
        showToast(`${msg.key} -> ${value}`);
    });

    WS.on('ACK', (msg) => {
        if (msg.action === 'RESET') {
            simRunning = false;
            Charts.reset();
            Panels.resetEventFeed();
            showToast('Session reset');
        }
        if (msg.action === 'RUN') {
            simRunning = true;
        }
        if (msg.action === 'PAUSE') {
            simRunning = false;
        }
        if (latestState) {
            updateStatusStrip(latestState);
        }
    });

    WS.on('EXPORT_DONE', (msg) => {
        showToast(`Exported to ${msg.path}`);
    });

    document.getElementById('btn-play').addEventListener('click', () => {
        simRunning = true;
        if (latestState) {
            updateStatusStrip(latestState);
        }
        WS.send('RUN', {});
    });
    document.getElementById('btn-pause').addEventListener('click', () => {
        simRunning = false;
        if (latestState) {
            updateStatusStrip(latestState);
        }
        WS.send('PAUSE', {});
    });
    document.getElementById('btn-step').addEventListener('click', () => {
        simRunning = false;
        if (latestState) {
            updateStatusStrip(latestState);
        }
        WS.send('STEP', {});
    });
    document.getElementById('btn-reset').addEventListener('click', () => {
        simRunning = false;
        Charts.reset();
        Panels.resetEventFeed();
        if (latestState) {
            updateStatusStrip(latestState);
        }
        WS.send('RESET', {});
    });
    document.getElementById('btn-export').addEventListener('click', () => WS.send('EXPORT', {}));

    const speedSlider = document.getElementById('speed-slider');
    speedSlider.addEventListener('input', () => {
        const factor = parseFloat(speedSlider.value);
        document.getElementById('speed-val').textContent = `${factor.toFixed(1)}x`;
        WS.send('SET_SPEED', { factor });
    });

    WS.connect();
})();
