// panels.js — Config Panel Controls for FANET Simulator

const Panels = {
    applyMode: 'live', // 'live' or 'restart'

    init() {
        // Apply mode toggle
        const toggle = document.getElementById('apply-mode-toggle');
        const hint = document.getElementById('apply-hint');
        toggle.addEventListener('change', () => {
            this.applyMode = toggle.checked ? 'live' : 'restart';
            hint.textContent = toggle.checked
                ? 'Changes take effect at next tick boundary.'
                : 'Changes queued until you press Reset.';
        });

        // Bind all inputs
        this._bindInputs();
    },

    _bindInputs() {
        // Number inputs
        document.querySelectorAll('input[data-key][type="number"]').forEach(input => {
            input.addEventListener('change', () => {
                const key = input.dataset.key;
                let value = parseFloat(input.value);
                // Guard time: UI shows μs, config stores seconds
                if (key === 'TDMA_GUARD_TIME_S') {
                    value = value * 1e-6;
                }
                WS.send('SET_CONFIG', { key, value, mode: this.applyMode });
            });
        });

        // Select dropdowns
        document.querySelectorAll('select[data-key]').forEach(sel => {
            sel.addEventListener('change', () => {
                WS.send('SET_CONFIG', { key: sel.dataset.key, value: sel.value, mode: this.applyMode });
            });
        });

        // Checkbox toggles
        document.querySelectorAll('input[data-key][type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', () => {
                WS.send('SET_CONFIG', { key: cb.dataset.key, value: cb.checked, mode: this.applyMode });
            });
        });
    },

    populateFromConfig(config) {
        // Number inputs
        document.querySelectorAll('input[data-key][type="number"]').forEach(input => {
            const key = input.dataset.key;
            if (config[key] !== undefined) {
                let val = config[key];
                // Guard time: config stores seconds, UI shows μs
                if (key === 'TDMA_GUARD_TIME_S') {
                    val = val * 1e6;
                }
                input.value = val;
            }
        });

        // Selects
        document.querySelectorAll('select[data-key]').forEach(sel => {
            const key = sel.dataset.key;
            if (config[key] !== undefined) {
                sel.value = config[key];
            }
        });

        // Checkboxes
        document.querySelectorAll('input[data-key][type="checkbox"]').forEach(cb => {
            const key = cb.dataset.key;
            if (config[key] !== undefined) {
                cb.checked = !!config[key];
            }
        });
    }
};
