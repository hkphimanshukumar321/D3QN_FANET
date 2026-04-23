function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function fmtNumber(value, digits = 2) {
    const num = Number(value || 0);
    return num.toFixed(digits);
}

function fmtList(items) {
    return (items || []).map(item => escapeHtml(item)).join('<br>');
}

const Panels = {
    currentConfig: null,
    envSchemaByKey: {},
    selectedClusterId: null,
    selectedNodeId: null,
    eventFeed: [],
    analysisMode: 'live',
    onClusterSelected: null,
    onNodeSelected: null,
    onNotice: null,

    init() {
        document.getElementById('preset-select').addEventListener('change', (event) => {
            WS.send('SET_PRESET', { preset_id: event.target.value });
        });

        document.getElementById('policy-select').addEventListener('change', (event) => {
            WS.send('SET_POLICY', { policy_id: event.target.value });
        });

        document.getElementById('deterministic-toggle').addEventListener('change', (event) => {
            WS.send('SET_RUNTIME', { key: 'deterministic', value: event.target.checked });
        });

        document.querySelectorAll('[data-base-key]').forEach((element) => {
            element.addEventListener('change', () => {
                const key = element.dataset.baseKey;
                const value = element.value === '' ? null : parseFloat(element.value);
                if (value === null || Number.isNaN(value)) {
                    return;
                }
                WS.send('SET_BASE_PARAM', { key, value });
            });
        });

        document.getElementById('advanced-env-controls').addEventListener('click', (event) => {
            const applyButton = event.target.closest('[data-env-apply]');
            if (applyButton) {
                this._applyEnvControl(applyButton.dataset.envApply);
                return;
            }

            const clearButton = event.target.closest('[data-env-clear]');
            if (clearButton) {
                WS.send('SET_ENV_OPTION', { key: clearButton.dataset.envClear, value: null });
            }
        });

        document.getElementById('cluster-inspector').addEventListener('click', (event) => {
            const nodeButton = event.target.closest('[data-node-id]');
            if (!nodeButton) {
                return;
            }
            const nodeId = Number(nodeButton.dataset.nodeId);
            this.selectedNodeId = nodeId;
            if (typeof this.onNodeSelected === 'function') {
                this.onNodeSelected(nodeId);
            }
        });

        document.querySelectorAll('[data-analysis-mode]').forEach((button) => {
            button.addEventListener('click', () => this.setAnalysisMode(button.dataset.analysisMode));
        });
    },

    notify(message) {
        if (typeof this.onNotice === 'function') {
            this.onNotice(message);
        }
    },

    populateConfig(config) {
        this.currentConfig = config;
        this.envSchemaByKey = Object.fromEntries((config.ui_schema?.env_option_schema || []).map(item => [item.key, item]));
        this._populatePresetSelect(config.presets || [], config.selected_preset_id);
        this._populatePolicySelect(config.policies || [], config.selected_policy_id);
        this._populateBaseParams(config.base_params || {});
        this._populateAdvancedEnvControls(config);
        document.getElementById('deterministic-toggle').checked = !!(config.runtime && config.runtime.deterministic);
        document.getElementById('action-semantics').className = 'readout-block';
        document.getElementById('action-semantics').textContent = config.ui_schema?.notes?.action_semantics || 'Each cluster head acts independently per burst.';
    },

    _populatePresetSelect(presets, selectedId) {
        const select = document.getElementById('preset-select');
        const groups = {};
        presets.forEach((preset) => {
            if (!groups[preset.group_label]) {
                groups[preset.group_label] = [];
            }
            groups[preset.group_label].push(preset);
        });

        select.innerHTML = '';
        Object.entries(groups).forEach(([groupLabel, items]) => {
            const optGroup = document.createElement('optgroup');
            optGroup.label = groupLabel;
            items.forEach((preset) => {
                const option = document.createElement('option');
                option.value = preset.id;
                option.textContent = preset.label;
                optGroup.appendChild(option);
            });
            select.appendChild(optGroup);
        });
        select.value = selectedId;
    },

    _populatePolicySelect(policies, selectedId) {
        const select = document.getElementById('policy-select');
        select.innerHTML = '';
        policies.forEach((policy) => {
            const option = document.createElement('option');
            option.value = policy.id;
            const incompatible = policy.available === false || policy.compatible === false;
            option.textContent = incompatible
                ? `${policy.label} [incompatible]`
                : `${policy.label} [${policy.source}]`;
            option.disabled = incompatible;
            select.appendChild(option);
        });
        select.value = selectedId;
    },

    _populateBaseParams(baseParams) {
        document.querySelectorAll('[data-base-key]').forEach((element) => {
            const key = element.dataset.baseKey;
            if (baseParams[key] === undefined) {
                return;
            }
            element.value = baseParams[key];
        });
    },

    _fieldDisplayValue(schema, config) {
        const manualValues = config.env_reset_options || {};
        const effectiveValues = config.effective_env_reset_options || {};
        const rawValue = manualValues[schema.key] !== undefined ? manualValues[schema.key] : effectiveValues[schema.key];
        if (rawValue === undefined) {
            return '';
        }
        if (schema.type === 'json') {
            return JSON.stringify(rawValue, null, 2);
        }
        if (schema.type === 'boolean') {
            return String(!!rawValue);
        }
        return String(rawValue);
    },

    _fieldSource(schema, config) {
        const manualValues = config.env_reset_options || {};
        const effectiveValues = config.effective_env_reset_options || {};
        if (manualValues[schema.key] !== undefined) {
            return 'manual';
        }
        if (effectiveValues[schema.key] !== undefined) {
            return 'preset/default';
        }
        return 'inherit';
    },

    _renderEnvControl(schema, config) {
        const value = this._fieldDisplayValue(schema, config);
        const source = this._fieldSource(schema, config);
        const inputId = `env-control-${schema.key}`;
        let inputHtml = '';

        if (schema.type === 'select') {
            const inheritOption = `<option value=""${value === '' ? ' selected' : ''}>inherit preset/default</option>`;
            const options = (schema.options || []).map((option) => {
                const selected = String(option) === value ? ' selected' : '';
                return `<option value="${escapeHtml(option)}"${selected}>${escapeHtml(option)}</option>`;
            }).join('');
            inputHtml = `
                <div class="field-stack">
                    <select id="${inputId}" data-env-key="${schema.key}">
                        ${inheritOption}
                        ${options}
                    </select>
                </div>
            `;
        } else if (schema.type === 'boolean') {
            const boolValue = value === '' ? '' : (value === 'true' ? 'true' : 'false');
            inputHtml = `
                <div class="field-stack">
                    <select id="${inputId}" data-env-key="${schema.key}">
                        <option value=""${boolValue === '' ? ' selected' : ''}>inherit preset/default</option>
                        <option value="true"${boolValue === 'true' ? ' selected' : ''}>true</option>
                        <option value="false"${boolValue === 'false' ? ' selected' : ''}>false</option>
                    </select>
                </div>
            `;
        } else if (schema.type === 'json') {
            inputHtml = `
                <div class="field-stack">
                    <textarea id="${inputId}" data-env-key="${schema.key}" placeholder="${escapeHtml(schema.placeholder || '')}">${escapeHtml(value)}</textarea>
                </div>
            `;
        } else {
            const minAttr = schema.min !== undefined ? ` min="${schema.min}"` : '';
            const maxAttr = schema.max !== undefined ? ` max="${schema.max}"` : '';
            const stepAttr = schema.step !== undefined ? ` step="${schema.step}"` : '';
            inputHtml = `
                <div class="field-stack">
                    <input id="${inputId}" data-env-key="${schema.key}" type="number"${minAttr}${maxAttr}${stepAttr} value="${escapeHtml(value)}" placeholder="inherit preset/default">
                </div>
            `;
        }

        return `
            <div class="env-field">
                <div class="control-head">
                    <span>${escapeHtml(schema.label)}</span>
                    <span class="control-source">${escapeHtml(source)}</span>
                </div>
                ${inputHtml}
                <div class="control-actions">
                    <button type="button" class="action-btn primary" data-env-apply="${schema.key}">Apply</button>
                    <button type="button" class="action-btn" data-env-clear="${schema.key}">Clear</button>
                </div>
            </div>
        `;
    },

    _populateAdvancedEnvControls(config) {
        const container = document.getElementById('advanced-env-controls');
        const schema = config.ui_schema?.env_option_schema || [];
        const grouped = {};
        schema.forEach((item) => {
            if (!grouped[item.section]) {
                grouped[item.section] = [];
            }
            grouped[item.section].push(item);
        });

        container.innerHTML = Object.entries(grouped).map(([section, items]) => `
            <div class="control-group">
                <div class="control-group-title">${escapeHtml(section)}</div>
                ${items.map((item) => this._renderEnvControl(item, config)).join('')}
            </div>
        `).join('');
    },

    _applyEnvControl(key) {
        const schema = this.envSchemaByKey[key];
        if (!schema) {
            return;
        }
        const element = document.getElementById(`env-control-${key}`);
        if (!element) {
            return;
        }

        let parsedValue = null;
        const rawValue = element.value;

        if (schema.type === 'json') {
            if (!rawValue.trim()) {
                parsedValue = null;
            } else {
                try {
                    parsedValue = JSON.parse(rawValue);
                } catch (error) {
                    this.notify(`Invalid JSON for ${schema.label}`);
                    return;
                }
            }
        } else if (schema.type === 'boolean') {
            if (!rawValue) {
                parsedValue = null;
            } else {
                parsedValue = rawValue === 'true';
            }
        } else if (schema.type === 'number') {
            if (rawValue.trim() === '') {
                parsedValue = null;
            } else {
                const num = Number(rawValue);
                if (Number.isNaN(num)) {
                    this.notify(`Invalid numeric value for ${schema.label}`);
                    return;
                }
                parsedValue = num;
            }
        } else {
            parsedValue = rawValue || null;
        }

        WS.send('SET_ENV_OPTION', { key, value: parsedValue });
    },

    renderScenario(scenario) {
        const target = document.getElementById('scenario-summary');
        if (!scenario) {
            target.className = 'readout-block empty';
            target.textContent = 'Scenario context unavailable.';
            return;
        }

        const headline = `${scenario.preset_group} / ${scenario.study_block || 'base'}`;
        const description = scenario.preset_description || 'Live scenario summary';
        const lines = scenario.summary_lines && scenario.summary_lines.length ? scenario.summary_lines : ['Base environment'];
        target.className = 'readout-block';
        target.innerHTML = `<strong>${escapeHtml(scenario.preset_label)}</strong><br>${escapeHtml(headline)}<br>${escapeHtml(description)}<br><br>${fmtList(lines)}`;
    },

    renderRuntime(runtime, clusters) {
        const runtimeTarget = document.getElementById('runtime-note');
        const compatTarget = document.getElementById('policy-compat-list');
        if (!runtime) {
            runtimeTarget.className = 'readout-block empty';
            runtimeTarget.textContent = 'Runtime diagnostics unavailable.';
            compatTarget.className = 'stack-list empty-state';
            compatTarget.textContent = 'Compatibility notes unavailable.';
            return;
        }

        const tdmaCount = clusters.filter(cluster => cluster.mac_label === 'TDMA').length;
        const csmaCount = clusters.filter(cluster => cluster.mac_label === 'CSMA_CA').length;
        const selectedPolicy = (this.currentConfig?.policies || []).find((policy) => policy.id === runtime.policy_id);
        const compatibility = runtime.policy_compatible ? 'compatible' : 'incompatible';
        const note = runtime.policy_warning || runtime.policy_compatibility_note || 'No policy warnings.';

        runtimeTarget.className = 'readout-block';
        runtimeTarget.innerHTML = [
            `<strong>${escapeHtml(runtime.policy_label)}</strong>`,
            `source = ${escapeHtml(runtime.policy_source)}`,
            `compat = ${escapeHtml(compatibility)}`,
            `live MAC mix = ${tdmaCount} TDMA / ${csmaCount} CSMA`,
            escapeHtml(note),
        ].join('<br>');

        const incompatiblePolicies = (this.currentConfig?.policies || []).filter((policy) => policy.available === false || policy.compatible === false);
        if (incompatiblePolicies.length === 0) {
            compatTarget.className = 'stack-list empty-state';
            compatTarget.textContent = 'All discovered policies are usable in this UI session.';
        } else {
            compatTarget.className = 'stack-list';
            compatTarget.innerHTML = incompatiblePolicies.map((policy) => `
                <div class="policy-note warning">
                    <strong>${escapeHtml(policy.label)}</strong><br>
                    ${escapeHtml(policy.compatibility_note || 'Incompatible with current env shape.')}
                </div>
            `).join('');
        }
    },

    renderClusterList(clusters, selectedClusterId) {
        const list = document.getElementById('cluster-list');
        list.innerHTML = '';

        if (!clusters || clusters.length === 0) {
            list.innerHTML = '<div class="readout-block empty">No active clusters</div>';
            return;
        }

        clusters.forEach((cluster) => {
            const item = document.createElement('button');
            item.type = 'button';
            let extraClass = selectedClusterId === cluster.cluster_id ? ' selected' : '';
            if (cluster.failure_flag) {
                extraClass += ' failure-pulse';
            } else if (cluster.handover_flag) {
                extraClass += ' handover-pulse';
            }
            item.className = `cluster-item${extraClass}`;
            item.innerHTML = `
                <div class="cluster-item-title">
                    <span>Cluster ${cluster.cluster_id}</span>
                    <span>${escapeHtml(cluster.mac_label)}</span>
                </div>
                <div class="cluster-meta">
                    <span>leader ${cluster.leader_id}</span>
                    <span>size ${cluster.cluster_size}</span>
                    <span>rho ${fmtNumber(cluster.rho, 2)}</span>
                    <span>deg ${cluster.graph_degree}</span>
                    <span>backlog ${fmtNumber(cluster.local_backlog, 1)}</span>
                </div>
            `;
            item.addEventListener('click', () => {
                this.selectedClusterId = cluster.cluster_id;
                if (typeof this.onClusterSelected === 'function') {
                    this.onClusterSelected(cluster.cluster_id);
                }
            });
            list.appendChild(item);
        });
    },

    renderClusterInspector(state, selectedClusterId) {
        const target = document.getElementById('cluster-inspector');
        const clusters = state?.clusters || [];
        const cluster = clusters.find((item) => item.cluster_id === selectedClusterId);
        if (!cluster) {
            target.className = 'readout-block empty';
            target.textContent = 'Select a live cluster to inspect its burst action, health, and backlog.';
            return;
        }

        const health = cluster.health_inputs || {};
        const memberChips = (cluster.members || []).map((memberId) => `
            <button type="button" class="node-chip${this.selectedNodeId === memberId ? ' selected' : ''}" data-node-id="${memberId}">
                Node ${memberId}${memberId === cluster.leader_id ? ' (L)' : ''}
            </button>
        `).join('');

        target.className = '';
        target.innerHTML = `
            <div class="inspector-shell">
                <div class="inspector-title">
                    <strong>Cluster ${cluster.cluster_id}</strong>
                    <span>leader ${cluster.leader_id} | size ${cluster.cluster_size}</span>
                </div>
                <div class="metric-grid">
                    <div class="metric-cell"><span class="cell-label">Burst Action</span><span class="cell-value">${escapeHtml(cluster.mac_label)} | rho ${fmtNumber(cluster.rho, 2)}</span></div>
                    <div class="metric-cell"><span class="cell-label">T1 / T2</span><span class="cell-value">${fmtNumber(cluster.t1_time, 2)} s / ${fmtNumber(cluster.t2_time, 2)} s</span></div>
                    <div class="metric-cell"><span class="cell-label">Local / Inter Th</span><span class="cell-value">${fmtNumber(cluster.throughput_mbps, 3)} / ${fmtNumber(cluster.inter_throughput_mbps, 3)} Mbps</span></div>
                    <div class="metric-cell"><span class="cell-label">Local / Inter Delay</span><span class="cell-value">${fmtNumber(cluster.delay_ms, 2)} / ${fmtNumber(cluster.inter_delay_ms, 2)} ms</span></div>
                    <div class="metric-cell"><span class="cell-label">Local / Inter Backlog</span><span class="cell-value">${fmtNumber(cluster.local_backlog, 2)} / ${fmtNumber(cluster.inter_backlog, 2)}</span></div>
                    <div class="metric-cell"><span class="cell-label">Graph Degree</span><span class="cell-value">${cluster.graph_degree}</span></div>
                    <div class="metric-cell"><span class="cell-label">Coord Success</span><span class="cell-value">${fmtNumber(cluster.recent_coord_success, 2)}</span></div>
                    <div class="metric-cell"><span class="cell-label">Handover / Failure</span><span class="cell-value">${cluster.handover_flag ? '<span class="event-tag handover">HANDOVER</span>' : 'no'} / ${cluster.failure_flag ? '<span class="event-tag failure">FAILURE</span>' : 'no'}</span></div>
                </div>
                <div class="meta-grid">
                    <div class="meta-cell"><span class="cell-label">Health Score</span><span class="cell-value">${fmtNumber(cluster.leader_health, 3)}</span></div>
                    <div class="meta-cell"><span class="cell-label">Energy</span><span class="cell-value">${fmtNumber(health.leader_energy, 2)} (${fmtNumber(health.energy_norm, 2)})</span></div>
                    <div class="meta-cell"><span class="cell-label">Leader Speed</span><span class="cell-value">${fmtNumber(health.leader_speed, 2)} m/s</span></div>
                    <div class="meta-cell"><span class="cell-label">Leader Queue</span><span class="cell-value">${fmtNumber(health.leader_queue, 2)} (${fmtNumber(health.queue_norm, 2)})</span></div>
                    <div class="meta-cell"><span class="cell-label">Degree Norm</span><span class="cell-value">${fmtNumber(health.degree_norm, 2)}</span></div>
                    <div class="meta-cell"><span class="cell-label">Mobility Stability / Risk</span><span class="cell-value">${fmtNumber(health.mobility_stability, 2)} / ${fmtNumber(health.risk, 2)}</span></div>
                    <div class="meta-cell wide"><span class="cell-label">Members</span><div class="chip-list">${memberChips}</div></div>
                </div>
            </div>
        `;
    },

    renderNodeInspector(state, selectedNodeId) {
        const target = document.getElementById('node-inspector');
        const nodes = state?.nodes || [];
        const node = nodes.find((item) => item.id === selectedNodeId);
        if (!node) {
            target.className = 'readout-block empty';
            target.textContent = 'Select a node from the scene or from the selected cluster.';
            return;
        }

        target.className = '';
        target.innerHTML = `
            <div class="inspector-shell">
                <div class="inspector-title">
                    <strong>Node ${node.id}</strong>
                    <span>${escapeHtml(node.role_label)} | cluster ${node.cluster_id}</span>
                </div>
                <div class="metric-grid">
                    <div class="metric-cell"><span class="cell-label">Position</span><span class="cell-value">${fmtNumber(node.position[0], 1)}, ${fmtNumber(node.position[1], 1)}, ${fmtNumber(node.position[2], 1)}</span></div>
                    <div class="metric-cell"><span class="cell-label">Velocity</span><span class="cell-value">${fmtNumber(node.velocity[0], 2)}, ${fmtNumber(node.velocity[1], 2)}, ${fmtNumber(node.velocity[2], 2)}</span></div>
                    <div class="metric-cell"><span class="cell-label">Speed</span><span class="cell-value">${fmtNumber(node.speed, 2)} m/s</span></div>
                    <div class="metric-cell"><span class="cell-label">Queue</span><span class="cell-value">${fmtNumber(node.queue, 2)}</span></div>
                    <div class="metric-cell"><span class="cell-label">Energy</span><span class="cell-value">${fmtNumber(node.energy, 2)}</span></div>
                    <div class="metric-cell"><span class="cell-label">Cluster Active</span><span class="cell-value">${node.active_cluster ? 'yes' : 'no'}</span></div>
                </div>
            </div>
        `;
    },

    renderGraphInspector(state) {
        const target = document.getElementById('graph-inspector');
        if (!state || !state.graphs || !state.graphs.stats) {
            target.className = 'readout-block empty';
            target.textContent = 'Graph diagnostics unavailable.';
            return;
        }

        const stats = state.graphs.stats;
        target.className = 'readout-block';
        target.innerHTML = [
            `<strong>Observed Graph Diagnostics</strong>`,
            `true edges = ${stats.true_edge_count} | observed edges = ${stats.observed_edge_count}`,
            `missing = ${stats.missing_edge_count} | false = ${stats.false_edge_count}`,
            `true density = ${fmtNumber(stats.true_density, 2)} | observed density = ${fmtNumber(stats.observed_density, 2)}`,
            `graph mode = ${escapeHtml(stats.graph_mode)}`,
            `missing prob = ${fmtNumber(stats.graph_missing_edge_prob, 2)} | false prob = ${fmtNumber(stats.graph_false_edge_prob, 2)}`,
            `graph stale = ${stats.graph_staleness_steps} | obs stale = ${stats.obs_staleness_steps} | handover stale = ${stats.handover_info_staleness_steps}`,
            `decision overhead = ${fmtNumber(state.metrics.decision_overhead_bytes || 0, 0)} B`,
        ].join('<br>');
    },

    renderEventFeed(eventBatch) {
        const target = document.getElementById('event-feed');
        (eventBatch || []).forEach((item) => {
            if (this.eventFeed.some((row) => row.id === item.id)) {
                return;
            }
            this.eventFeed.unshift(item);
        });
        this.eventFeed = this.eventFeed.slice(0, 40);

        if (this.eventFeed.length === 0) {
            target.className = 'event-feed empty-state';
            target.textContent = 'No events yet.';
            return;
        }

        target.className = 'event-feed';
        target.innerHTML = this.eventFeed.map((event) => {
            const badge = this._eventBadge(event.kind);
            return `
            <div class="event-item ${escapeHtml(event.severity || 'info')}">
                <div class="event-head">
                    <span>${badge}${escapeHtml(event.kind || 'event')}</span>
                    <span>t=${fmtNumber(event.sim_time || 0, 2)} | tick ${event.tick}</span>
                </div>
                <div class="event-text">${escapeHtml(event.message || '')}</div>
            </div>
        `;
        }).join('');
    },

    resetEventFeed() {
        this.eventFeed = [];
        this.renderEventFeed([]);
    },

    renderDiagnostics(state) {
        const metrics = state?.metrics || {};
        const events = state?.events || { counts: {} };
        const scenario = state?.scenario || {};

        document.getElementById('diag-local-backlog').textContent = fmtNumber(metrics.avg_local_backlog || 0, 2);
        document.getElementById('diag-inter-backlog').textContent = fmtNumber(metrics.avg_inter_backlog || 0, 2);
        document.getElementById('diag-overhead').textContent = `${fmtNumber(metrics.decision_overhead_bytes || 0, 0)} B`;
        document.getElementById('diag-event-counts').textContent = `S${events.counts?.splits || 0} M${events.counts?.merges || 0} R${events.counts?.reassociations || 0} H${events.counts?.handovers || 0} F${events.counts?.failure_events || 0}`;

        const contextTarget = document.getElementById('diag-scenario-context');
        contextTarget.className = 'readout-block';
        contextTarget.innerHTML = fmtList(scenario.summary_lines || ['Base environment']);

        const impairmentTarget = document.getElementById('diag-impairments');
        impairmentTarget.className = 'readout-block';
        impairmentTarget.innerHTML = fmtList(scenario.impairment_lines || ['No active impairments']);
    },

    setAnalysisMode(mode) {
        this.analysisMode = mode;
        document.querySelectorAll('.analysis-tab').forEach((button) => {
            button.classList.toggle('active', button.dataset.analysisMode === mode);
        });
        document.getElementById('analysis-live').classList.toggle('active', mode === 'live');
        document.getElementById('analysis-diagnostics').classList.toggle('active', mode === 'diagnostics');
        document.getElementById('analysis-summary').textContent = mode === 'live'
            ? 'Rolling burst-level traces.'
            : 'Study context, impairments, and event deltas for the active burst.';
        if (mode === 'live' && typeof Charts !== 'undefined' && typeof Charts.resizeAll === 'function') {
            window.setTimeout(() => Charts.resizeAll(), 0);
        }
    },

    _eventBadge(kind) {
        const map = {
            reassociation: '<span class="event-badge">🔄</span>',
            deassociation: '<span class="event-badge">⛔</span>',
            handover: '<span class="event-badge">⚡</span>',
            failure: '<span class="event-badge">💥</span>',
            split: '<span class="event-badge">🔀</span>',
            merge: '<span class="event-badge">🔗</span>',
            leader_change: '<span class="event-badge">⚡</span>',
        };
        return map[kind] || '';
    },
};
