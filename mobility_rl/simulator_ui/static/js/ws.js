// ws.js — WebSocket Client for FANET 3D Simulator

const WS = {
    socket: null,
    url: 'ws://localhost:8765',
    handlers: {},
    reconnectDelay: 2000,

    connect() {
        this.socket = new WebSocket(this.url);

        this.socket.onopen = () => {
            console.log('[WS] Connected');
            document.getElementById('ws-status').className = 'ws-dot connected';
            this.send('GET_CONFIG', {});
        };

        this.socket.onclose = () => {
            console.log('[WS] Disconnected');
            document.getElementById('ws-status').className = 'ws-dot disconnected';
            setTimeout(() => this.connect(), this.reconnectDelay);
        };

        this.socket.onerror = (err) => {
            console.warn('[WS] Error', err);
        };

        this.socket.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                const type = msg.type;
                if (this.handlers[type]) {
                    this.handlers[type].forEach(fn => fn(msg));
                }
            } catch (e) {
                console.warn('[WS] Parse error', e);
            }
        };
    },

    send(type, payload) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({ type, ...payload }));
        }
    },

    on(type, handler) {
        if (!this.handlers[type]) this.handlers[type] = [];
        this.handlers[type].push(handler);
    },

    get isConnected() {
        return this.socket && this.socket.readyState === WebSocket.OPEN;
    }
};
