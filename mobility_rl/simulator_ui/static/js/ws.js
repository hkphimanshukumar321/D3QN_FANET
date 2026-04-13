const WS = {
    socket: null,
    handlers: {},
    reconnectDelayMs: 2000,

    get url() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.hostname}:8765`;
    },

    connect() {
        this.socket = new WebSocket(this.url);

        this.socket.onopen = () => {
            const status = document.getElementById('status-ws');
            if (status) {
                status.textContent = 'connected';
            }
            this.send('GET_CONFIG', {});
        };

        this.socket.onclose = () => {
            const status = document.getElementById('status-ws');
            if (status) {
                status.textContent = 'offline';
            }
            window.setTimeout(() => this.connect(), this.reconnectDelayMs);
        };

        this.socket.onerror = (error) => {
            console.warn('[WS] error', error);
        };

        this.socket.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                const handlers = this.handlers[msg.type] || [];
                handlers.forEach(handler => handler(msg));
            } catch (error) {
                console.warn('[WS] parse error', error);
            }
        };
    },

    send(type, payload) {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
            return;
        }
        this.socket.send(JSON.stringify({ type, ...payload }));
    },

    on(type, handler) {
        if (!this.handlers[type]) {
            this.handlers[type] = [];
        }
        this.handlers[type].push(handler);
    },
};
