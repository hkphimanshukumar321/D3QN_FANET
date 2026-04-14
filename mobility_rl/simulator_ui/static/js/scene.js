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
    resizeObserver: null,
    boxHelper: null,
    gridHelper: null,
    selectedClusterId: null,
    selectedNodeId: null,
    onNodeSelected: null,
    palette: [
        0x4ed8ff, 0xff8b5c, 0xffcc66, 0x71e09c, 0xc78cff,
        0xf56f9c, 0x87ceeb, 0xff6b6b, 0x9ee07a, 0x8fb3ff,
    ],

    init() {
        const canvas = document.getElementById('three-canvas');
        const container = document.getElementById('viewport');

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setClearColor(0x040911, 1);

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(52, container.clientWidth / container.clientHeight, 0.1, 2200);
        this.camera.position.set(10, 10, 10);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;

        this.raycaster = new THREE.Raycaster();
        this.pointer = new THREE.Vector2();

        const hemi = new THREE.HemisphereLight(0x90dcff, 0x05101d, 0.88);
        this.scene.add(hemi);

        const key = new THREE.DirectionalLight(0xffffff, 0.72);
        key.position.set(8, 14, 10);
        this.scene.add(key);

        this.membershipGroup = new THREE.Group();
        this.trueEdgeGroup = new THREE.Group();
        this.observedEdgeGroup = new THREE.Group();
        this.scene.add(this.membershipGroup);
        this.scene.add(this.trueEdgeGroup);
        this.scene.add(this.observedEdgeGroup);

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
            new THREE.LineBasicMaterial({ color: 0x2d4e76, transparent: true, opacity: 0.45 }),
        );
        this.boxHelper.position.set(sx / 2, sz / 2, sy / 2);
        this.scene.add(this.boxHelper);

        this.gridHelper = new THREE.GridHelper(Math.max(sx, sy), 16, 0x284766, 0x16304a);
        this.gridHelper.position.set(sx / 2, 0, sy / 2);
        this.scene.add(this.gridHelper);

        this.camera.position.set(sx * 1.12, Math.max(sz * 2.05, 4), sy * 1.15);
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
            const geometry = new THREE.SphereGeometry(0.1, 18, 18);
            const material = new THREE.MeshStandardMaterial({
                color: 0x4ed8ff,
                roughness: 0.28,
                metalness: 0.1,
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
            return 0x6a7d91;
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

        this._ensureNodes(state.nodes.length);

        state.nodes.forEach((node, index) => {
            const mesh = this.nodeMeshes[index];
            const [x, y, z] = this._worldToScene(node.position);
            const color = this._clusterColor(node.cluster_id);
            const clusterHighlight = this.selectedClusterId !== null && node.cluster_id === this.selectedClusterId;
            const nodeHighlight = this.selectedNodeId !== null && node.id === this.selectedNodeId;
            const baseScale = node.leader ? 1.45 : 1.0;
            mesh.position.set(x, y, z);
            mesh.scale.setScalar(nodeHighlight ? baseScale * 1.25 : baseScale);
            mesh.material.color.setHex(color);
            mesh.material.opacity = node.active_cluster ? 1.0 : 0.22;
            mesh.material.transparent = !node.active_cluster;
            mesh.material.emissive.setHex(nodeHighlight ? 0xffffff : (clusterHighlight || node.leader ? color : 0x000000));
            mesh.material.emissiveIntensity = nodeHighlight ? 0.82 : (clusterHighlight ? 0.48 : (node.leader ? 0.26 : 0.0));
            mesh.userData = { nodeId: node.id };
        });

        this._rebuildEdges(state);
    },

    _rebuildEdges(state) {
        this._clearGroup(this.membershipGroup);
        this._clearGroup(this.trueEdgeGroup);
        this._clearGroup(this.observedEdgeGroup);

        const nodeById = {};
        state.nodes.forEach((node) => {
            nodeById[node.id] = this._worldToScene(node.position);
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
                    opacity: this.selectedClusterId !== null && cluster.cluster_id !== this.selectedClusterId ? 0.1 : 0.32,
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
                    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.42 }),
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
                    color: 0xffc96c,
                    transparent: true,
                    opacity: 0.5,
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
