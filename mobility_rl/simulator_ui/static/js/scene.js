const Scene = {
    renderer: null,
    scene: null,
    camera: null,
    controls: null,
    raycaster: null,
    pointer: null,
    bounds: null,
    scaleFactor: 1,
    nodeMeshes: [],
    membershipGroup: null,
    trueEdgeGroup: null,
    observedEdgeGroup: null,
    eventGroup: null,
    resizeObserver: null,
    boxHelper: null,
    gridHelper: null,
    selectedClusterId: null,
    selectedNodeId: null,
    onNodeSelected: null,
    driftOffset: 0,
    /* track events for visual indicators */
    _eventRings: [],
    _flashNodes: {},
    palette: [
        0x0066CC, 0xE65100, 0xD4A017, 0x2E8B57, 0x7B1FA2,
        0xC62828, 0x00838F, 0xAD1457, 0x558B2F, 0x1565C0,
    ],

    init() {
        const canvas = document.getElementById('three-canvas');
        const container = document.getElementById('viewport');

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setClearColor(0xF0F2F5, 1);

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(52, container.clientWidth / container.clientHeight, 0.1, 2200);
        this.camera.position.set(10, 10, 10);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;

        this.raycaster = new THREE.Raycaster();
        this.pointer = new THREE.Vector2();

        const hemi = new THREE.HemisphereLight(0xffffff, 0xD0D8E0, 0.95);
        this.scene.add(hemi);

        const key = new THREE.DirectionalLight(0xffffff, 0.72);
        key.position.set(8, 14, 10);
        this.scene.add(key);

        const fill = new THREE.AmbientLight(0xffffff, 0.25);
        this.scene.add(fill);

        this.membershipGroup = new THREE.Group();
        this.trueEdgeGroup = new THREE.Group();
        this.observedEdgeGroup = new THREE.Group();
        this.eventGroup = new THREE.Group();
        this.scene.add(this.membershipGroup);
        this.scene.add(this.trueEdgeGroup);
        this.scene.add(this.observedEdgeGroup);
        this.scene.add(this.eventGroup);

        this.renderer.domElement.addEventListener('click', (event) => this._handleClick(event));
        window.addEventListener('resize', () => this.onResize());
        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.onResize());
            this.resizeObserver.observe(container);
        }
        window.requestAnimationFrame(() => this.onResize());
        this.animate();
    },

    buildScene(bounds) {
        this.bounds = bounds.slice();
        const maxDim = Math.max(bounds[0], bounds[1], bounds[2], 1);
        this.scaleFactor = 12 / maxDim;

        if (this.boxHelper) {
            this.scene.remove(this.boxHelper);
        }
        if (this.gridHelper) {
            this.scene.remove(this.gridHelper);
        }

        const sx = bounds[0] * this.scaleFactor;
        const sy = bounds[1] * this.scaleFactor;
        const sz = bounds[2] * this.scaleFactor;

        const box = new THREE.BoxGeometry(sx, sz, sy);
        const edges = new THREE.EdgesGeometry(box);
        this.boxHelper = new THREE.LineSegments(
            edges,
            new THREE.LineBasicMaterial({ color: 0x9AADBE, transparent: true, opacity: 0.55 }),
        );
        this.boxHelper.position.set(sx / 2, sz / 2, sy / 2);
        this.scene.add(this.boxHelper);

        const gridSize = Math.max(sx, sy) * 5;
        this.gridHelper = new THREE.GridHelper(gridSize, 16 * 5, 0xC0CDD8, 0xDDE5ED);
        this.gridHelper.position.set(sx / 2, 0, sy / 2);
        this.scene.add(this.gridHelper);
        this.gridStep = gridSize / (16 * 5);

        this.camera.position.set(sx * 1.5, Math.max(sz * 3.5, 8), sy * 2.2);
        this.controls.target.set(sx / 2, sz / 2, sy / 2);
        this.controls.update();
    },

    setSelectedCluster(clusterId) {
        this.selectedClusterId = clusterId;
    },

    setSelectedNode(nodeId) {
        this.selectedNodeId = nodeId;
    },

    _ensureNodes(count) {
        while (this.nodeMeshes.length < count) {
            const geometry = new THREE.SphereGeometry(0.12, 18, 18);
            const material = new THREE.MeshStandardMaterial({
                color: 0x0066CC,
                roughness: 0.35,
                metalness: 0.05,
                emissive: 0x000000,
            });
            const mesh = new THREE.Mesh(geometry, material);
            this.scene.add(mesh);
            this.nodeMeshes.push(mesh);
        }
        this.nodeMeshes.forEach((mesh, index) => {
            mesh.visible = index < count;
        });
    },

    _worldToScene(position) {
        return [
            position[0] * this.scaleFactor,
            position[2] * this.scaleFactor,
            position[1] * this.scaleFactor,
        ];
    },

    _clusterColor(clusterId) {
        if (clusterId === null || clusterId === undefined || clusterId < 0) {
            return 0xBBCCDD;
        }
        return this.palette[clusterId % this.palette.length];
    },

    _clearGroup(group) {
        while (group.children.length) {
            const child = group.children.pop();
            if (child) {
                group.remove(child);
            }
        }
    },

    updateFromSnapshot(state) {
        if (!state || !Array.isArray(state.nodes) || state.nodes.length === 0) {
            return;
        }

        const bounds = Array.isArray(state.bounds) && state.bounds.length === 3
            ? state.bounds
            : [100, 100, 60];

        if (
            !this.bounds
            || this.bounds[0] !== bounds[0]
            || this.bounds[1] !== bounds[1]
            || this.bounds[2] !== bounds[2]
        ) {
            this.buildScene(bounds);
        }

        /* capture drift offset from engine */
        this.driftOffset = state.drift_offset || 0;
        const driftScaled = this.driftOffset * this.scaleFactor;

        this._ensureNodes(state.nodes.length);

        state.nodes.forEach((node, index) => {
            const mesh = this.nodeMeshes[index];
            const pos = node.position.slice();
            /* apply horizontal drift — whole swarm moves forward (no wrap) */
            if (this.driftOffset) {
                pos[0] += this.driftOffset;
            }
            const [x, y, z] = this._worldToScene(pos);
            const color = this._clusterColor(node.cluster_id);
            const clusterHighlight = this.selectedClusterId !== null && node.cluster_id === this.selectedClusterId;
            const nodeHighlight = this.selectedNodeId !== null && node.id === this.selectedNodeId;
            const isLeader = node.leader;
            const baseScale = isLeader ? 1.55 : 1.0;
            mesh.position.set(x, y, z);
            mesh.scale.setScalar(nodeHighlight ? baseScale * 1.25 : baseScale);
            mesh.material.color.setHex(color);
            mesh.material.opacity = node.active_cluster ? 1.0 : 0.28;
            mesh.material.transparent = !node.active_cluster;

            /* flash nodes that just reassociated / deassociated */
            const flash = this._flashNodes[node.id];
            if (flash && flash.ttl > 0) {
                mesh.material.emissive.setHex(flash.color);
                mesh.material.emissiveIntensity = 0.6 * (flash.ttl / flash.maxTtl);
                flash.ttl--;
            } else {
                mesh.material.emissive.setHex(
                    nodeHighlight ? 0x333333 : (clusterHighlight || isLeader ? color : 0x000000)
                );
                mesh.material.emissiveIntensity = nodeHighlight ? 0.45 : (clusterHighlight ? 0.30 : (isLeader ? 0.18 : 0.0));
            }
            mesh.userData = { nodeId: node.id };
        });

        /* slide bounding box, grid, and camera orbit target to follow drift */
        if (this.bounds && driftScaled) {
            const sx = this.bounds[0] * this.scaleFactor;
            const sy = this.bounds[1] * this.scaleFactor;
            const sz = this.bounds[2] * this.scaleFactor;
            
            if (this.boxHelper) {
                this.boxHelper.position.set(sx / 2 + driftScaled, sz / 2, sy / 2);
            }
            if (this.gridHelper && this.gridStep) {
                const offset = driftScaled % this.gridStep;
                this.gridHelper.position.set(sx / 2 + driftScaled - offset, 0, sy / 2);
            }
            this.controls.target.set(sx / 2 + driftScaled, sz / 2, sy / 2);
            
            const deltaDrift = driftScaled - (this._lastDriftScaled || 0);
            if (deltaDrift > 0 && deltaDrift < sx) {
                this.camera.position.x += deltaDrift;
            }
            this._lastDriftScaled = driftScaled;
            
            this.controls.update();
        }

        /* process event indicators */
        this._processEventIndicators(state);

        this._rebuildEdges(state);
        this._tickEventRings();
    },

    _processEventIndicators(state) {
        const events = state.events || {};

        /* reassociation flash (blue) */
        (events.reassociated_nodes || []).forEach((item) => {
            this._flashNodes[item.node_id] = { color: 0x0066CC, ttl: 6, maxTtl: 6 };
        });

        /* deassociation flash (red) */
        (events.deassociated_nodes || []).forEach((item) => {
            this._flashNodes[item.node_id] = { color: 0xDC3545, ttl: 8, maxTtl: 8 };
        });

        /* leader change — expanding ring at old leader position */
        (events.leader_changes || []).forEach((item) => {
            const oldNode = (state.nodes || []).find((n) => n.id === item.old_leader);
            if (oldNode) {
                const pos = oldNode.position.slice();
                if (this.driftOffset) {
                    pos[0] += this.driftOffset;
                }
                const [x, y, z] = this._worldToScene(pos);
                this._spawnEventRing(x, y, z, 0xE67E22);
            }
            /* flash old leader orange */
            this._flashNodes[item.old_leader] = { color: 0xE67E22, ttl: 10, maxTtl: 10 };
        });

        /* failure events — red ring */
        (events.failure_nodes || []).forEach((item) => {
            const failNode = (state.nodes || []).find((n) => n.id === item.node_id);
            if (failNode) {
                const pos = failNode.position.slice();
                if (this.driftOffset) {
                    pos[0] += this.driftOffset;
                }
                const [x, y, z] = this._worldToScene(pos);
                this._spawnEventRing(x, y, z, 0xDC3545);
            }
            this._flashNodes[item.node_id] = { color: 0xDC3545, ttl: 12, maxTtl: 12 };
        });
    },

    _spawnEventRing(x, y, z, color) {
        const geometry = new THREE.RingGeometry(0.05, 0.08, 32);
        const material = new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.85,
            side: THREE.DoubleSide,
        });
        const ring = new THREE.Mesh(geometry, material);
        ring.position.set(x, y, z);
        ring.lookAt(this.camera.position);
        this.eventGroup.add(ring);
        this._eventRings.push({ mesh: ring, ttl: 18, maxTtl: 18 });
    },

    _tickEventRings() {
        for (let i = this._eventRings.length - 1; i >= 0; i--) {
            const ring = this._eventRings[i];
            ring.ttl--;
            const progress = 1 - ring.ttl / ring.maxTtl;
            const scale = 1 + progress * 6;
            ring.mesh.scale.setScalar(scale);
            ring.mesh.material.opacity = Math.max(0, 0.85 * (ring.ttl / ring.maxTtl));
            ring.mesh.lookAt(this.camera.position);
            if (ring.ttl <= 0) {
                this.eventGroup.remove(ring.mesh);
                this._eventRings.splice(i, 1);
            }
        }
    },

    _rebuildEdges(state) {
        this._clearGroup(this.membershipGroup);
        this._clearGroup(this.trueEdgeGroup);
        this._clearGroup(this.observedEdgeGroup);

        const bounds = state.bounds || [100, 100, 60];
        const nodeById = {};
        state.nodes.forEach((node) => {
            const pos = node.position.slice();
            /* apply drift — straight forward shift (no wrap) */
            if (this.driftOffset) {
                pos[0] += this.driftOffset;
            }
            nodeById[node.id] = this._worldToScene(pos);
        });

        (state.clusters || []).forEach((cluster) => {
            const leaderPos = nodeById[cluster.leader_id];
            if (!leaderPos) {
                return;
            }
            (cluster.members || []).forEach((memberId) => {
                if (memberId === cluster.leader_id) {
                    return;
                }
                const memberPos = nodeById[memberId];
                if (!memberPos) {
                    return;
                }
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(...memberPos),
                    new THREE.Vector3(...leaderPos),
                ]);
                const material = new THREE.LineBasicMaterial({
                    color: this._clusterColor(cluster.cluster_id),
                    transparent: true,
                    opacity: this.selectedClusterId !== null && cluster.cluster_id !== this.selectedClusterId ? 0.1 : 0.38,
                });
                this.membershipGroup.add(new THREE.Line(geometry, material));
            });
        });

        const leaderByCluster = {};
        (state.clusters || []).forEach((cluster) => {
            leaderByCluster[cluster.cluster_id] = nodeById[cluster.leader_id];
        });

        ((state.graphs && state.graphs.true_edges) || []).forEach((edge) => {
            const source = leaderByCluster[edge.source];
            const target = leaderByCluster[edge.target];
            if (!source || !target) {
                return;
            }
            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(...source),
                new THREE.Vector3(...target),
            ]);
            this.trueEdgeGroup.add(
                new THREE.Line(
                    geometry,
                    new THREE.LineBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.45 }),
                ),
            );
        });

        ((state.graphs && state.graphs.observed_edges) || []).forEach((edge) => {
            const source = leaderByCluster[edge.source];
            const target = leaderByCluster[edge.target];
            if (!source || !target) {
                return;
            }
            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(...source),
                new THREE.Vector3(...target),
            ]);
            const line = new THREE.Line(
                geometry,
                new THREE.LineDashedMaterial({
                    color: 0xE67E22,
                    transparent: true,
                    opacity: 0.55,
                    dashSize: 0.12,
                    gapSize: 0.1,
                }),
            );
            line.computeLineDistances();
            this.observedEdgeGroup.add(line);
        });
    },

    _handleClick(event) {
        if (!this.camera || !this.renderer) {
            return;
        }

        const rect = this.renderer.domElement.getBoundingClientRect();
        this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.pointer, this.camera);

        const visibleMeshes = this.nodeMeshes.filter((mesh) => mesh.visible);
        const hits = this.raycaster.intersectObjects(visibleMeshes, false);
        if (!hits.length) {
            return;
        }

        const nodeId = hits[0].object.userData?.nodeId;
        if (nodeId === undefined || typeof this.onNodeSelected !== 'function') {
            return;
        }
        this.onNodeSelected(Number(nodeId));
    },

    onResize() {
        if (!this.renderer || !this.camera) {
            return;
        }
        const container = document.getElementById('viewport');
        const width = Math.max(container.clientWidth || 0, 1);
        const height = Math.max(container.clientHeight || 0, 1);
        this.camera.aspect = width / Math.max(height, 1);
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    },

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    },
};
