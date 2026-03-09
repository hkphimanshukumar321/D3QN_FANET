// scene.js — Three.js 3D Scene for FANET Simulator
// B1: Procedural drone geometry (body + arms + rotors)
// B2: Smooth interpolation between simulation snapshots

const Scene = {
    renderer: null,
    scene: null,
    camera: null,
    controls: null,
    // Scene objects
    cubeWireframe: null,
    gridHelper: null,
    sinkMesh: null,
    uavGroups: [],    // THREE.Group per UAV (body + arms + rotors)
    linkLines: [],
    rotorMeshes: [],  // flat array of all rotor meshes for animation
    // State
    currentN: 0,
    scaleF: 1,

    // B2: Interpolation state
    prevState: null,
    currState: null,
    stateTimestamp: 0,
    expectedTickMs: 100,  // updated from server
    interpEnabled: true,

    init() {
        const canvas = document.getElementById('three-canvas');
        const container = document.getElementById('viewport');

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setClearColor(0x060612, 1);
        this.renderer.shadowMap.enabled = true;

        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x060612, 0.06);

        // Camera
        this.camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 1000);
        this.camera.position.set(12, 10, 12);
        this.camera.lookAt(0, 0, 0);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 2;
        this.controls.maxDistance = 50;

        // Lighting
        const ambient = new THREE.AmbientLight(0x5566aa, 0.5);
        this.scene.add(ambient);
        const directional = new THREE.DirectionalLight(0xffffff, 0.9);
        directional.position.set(10, 15, 10);
        this.scene.add(directional);
        const hemi = new THREE.HemisphereLight(0x4488ff, 0x002244, 0.3);
        this.scene.add(hemi);

        // Axes
        const axes = new THREE.AxesHelper(0.5);
        this.scene.add(axes);

        window.addEventListener('resize', () => this.onResize());
        this.animate();
    },

    // ── Build scene bounds ──
    buildScene(bounds, sinkPos, commRange) {
        if (this.cubeWireframe) this.scene.remove(this.cubeWireframe);
        if (this.gridHelper) this.scene.remove(this.gridHelper);
        if (this.sinkMesh) this.scene.remove(this.sinkMesh);

        const maxDim = Math.max(bounds[0], bounds[1], bounds[2]);
        this.scaleF = 10.0 / maxDim;

        const sx = bounds[0] * this.scaleF;
        const sy = bounds[1] * this.scaleF;
        const sz = bounds[2] * this.scaleF;

        // Cube wireframe
        const cubeGeo = new THREE.BoxGeometry(sx, sz, sy);
        const cubeEdges = new THREE.EdgesGeometry(cubeGeo);
        const cubeMat = new THREE.LineBasicMaterial({ color: 0x3344aa, opacity: 0.35, transparent: true });
        this.cubeWireframe = new THREE.LineSegments(cubeEdges, cubeMat);
        this.cubeWireframe.position.set(sx / 2, sz / 2, sy / 2);
        this.scene.add(this.cubeWireframe);

        // Grid
        const gridSize = Math.max(sx, sy);
        this.gridHelper = new THREE.GridHelper(gridSize, 20, 0x222244, 0x111133);
        this.gridHelper.position.set(sx / 2, 0, sy / 2);
        this.scene.add(this.gridHelper);

        // Sink — antenna tower
        const sinkGroup = new THREE.Group();
        const poleMat = new THREE.MeshPhongMaterial({ color: 0xff2255, emissive: 0x881133, emissiveIntensity: 0.4 });
        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.05, 0.6, 8), poleMat);
        pole.position.y = 0.3;
        sinkGroup.add(pole);
        const dish = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 6, 0, Math.PI), poleMat);
        dish.position.y = 0.6;
        dish.rotation.x = -Math.PI / 4;
        sinkGroup.add(dish);
        // Pulsing ring
        const ringGeo = new THREE.RingGeometry(0.12, 0.15, 24);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0xff3366, transparent: true, opacity: 0.4, side: THREE.DoubleSide });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.y = 0.6;
        ring.rotation.x = -Math.PI / 2;
        sinkGroup.add(ring);

        this.sinkMesh = sinkGroup;
        this._setSinkPos(sinkPos);
        this.scene.add(this.sinkMesh);

        // Camera
        this.camera.position.set(sx * 1.2, sz * 2, sy * 1.2);
        this.controls.target.set(sx / 2, sz / 2, sy / 2);
        this.controls.update();
    },

    _setSinkPos(pos) {
        if (!this.sinkMesh) return;
        this.sinkMesh.position.set(
            pos[0] * this.scaleF,
            pos[2] * this.scaleF,
            pos[1] * this.scaleF
        );
    },

    // ── Create procedural drone ──
    _createDrone() {
        const group = new THREE.Group();
        const bodyMat = new THREE.MeshPhongMaterial({
            color: 0x00d4ff, emissive: 0x003355, emissiveIntensity: 0.3,
            transparent: true, opacity: 0.9,
        });
        const armMat = new THREE.MeshPhongMaterial({
            color: 0x888888, emissive: 0x222222, emissiveIntensity: 0.2,
        });
        const rotorMat = new THREE.MeshBasicMaterial({
            color: 0x66ddff, transparent: true, opacity: 0.5,
        });

        // Body - flat rounded box
        const body = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.025, 0.08), bodyMat);
        body.name = 'body';
        group.add(body);

        // LED light on top
        const ledGeo = new THREE.SphereGeometry(0.012, 6, 4);
        const ledMat = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
        const led = new THREE.Mesh(ledGeo, ledMat);
        led.position.y = 0.015;
        led.name = 'led';
        group.add(led);

        // 4 arms + rotors
        const armLen = 0.07;
        const armGeo = new THREE.BoxGeometry(armLen, 0.008, 0.008);
        const rotorGeo = new THREE.CircleGeometry(0.03, 8);
        const rotors = [];

        const armOffsets = [
            { x:  1, z:  1, angle: Math.PI / 4 },
            { x:  1, z: -1, angle: -Math.PI / 4 },
            { x: -1, z:  1, angle: -Math.PI / 4 },
            { x: -1, z: -1, angle: Math.PI / 4 },
        ];

        armOffsets.forEach(off => {
            const arm = new THREE.Mesh(armGeo, armMat);
            arm.position.set(off.x * armLen / 2, 0, off.z * armLen / 2);
            arm.rotation.y = off.angle;
            group.add(arm);

            const rotor = new THREE.Mesh(rotorGeo, rotorMat.clone());
            rotor.position.set(off.x * armLen, 0.012, off.z * armLen);
            rotor.rotation.x = -Math.PI / 2;
            group.add(rotor);
            rotors.push(rotor);
        });

        group.userData.rotors = rotors;
        group.userData.bodyMat = bodyMat;
        group.userData.ledMat = ledMat;
        return group;
    },

    // ── Rebuild UAVs ──
    _rebuildUAVs(N) {
        this.uavGroups.forEach(g => this.scene.remove(g));
        this.linkLines.forEach(l => this.scene.remove(l));
        this.uavGroups = [];
        this.linkLines = [];
        this.rotorMeshes = [];

        for (let i = 0; i < N; i++) {
            const drone = this._createDrone();
            this.scene.add(drone);
            this.uavGroups.push(drone);
            this.rotorMeshes.push(...drone.userData.rotors);

            // Link line
            const lineGeo = new THREE.BufferGeometry();
            const positions = new Float32Array(6);
            lineGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const lineMat = new THREE.LineBasicMaterial({
                color: 0x00aa66, transparent: true, opacity: 0.2,
            });
            const line = new THREE.Line(lineGeo, lineMat);
            this.scene.add(line);
            this.linkLines.push(line);
        }
    },

    // ── Push new state (for interpolation) ──
    pushState(state) {
        this.prevState = this.currState;
        this.currState = state;
        this.stateTimestamp = performance.now();
    },

    // ── Update from snapshot (immediate, no interp) ──
    updateFromSnapshot(state) {
        const N = state.N;
        const positions = state.positions;
        const linkUp = state.link_up;
        const qLens = state.q_lens;
        const sf = this.scaleF;

        if (N !== this.currentN) {
            this._rebuildUAVs(N);
            this.currentN = N;
        }

        this._setSinkPos(state.sink_pos);

        for (let i = 0; i < N; i++) {
            const group = this.uavGroups[i];
            const line = this.linkLines[i];
            if (!group || !line) continue;

            const px = positions[i][0] * sf;
            const py = positions[i][2] * sf;
            const pz = positions[i][1] * sf;

            group.position.set(px, py, pz);

            // Orientation from velocity
            if (state.velocities && state.velocities[i]) {
                const vx = state.velocities[i][0];
                const vy = state.velocities[i][1];
                const vz = state.velocities[i][2];
                const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);
                if (speed > 0.5) {
                    // Yaw: look in direction of horizontal velocity
                    const targetX = px + vx * sf * 0.5;
                    const targetZ = pz + vy * sf * 0.5;
                    const targetY = py + vz * sf * 0.5;
                    group.lookAt(targetX, targetY, targetZ);
                    // Tilt proportional to speed
                    group.rotation.z = Math.min(speed * 0.02, 0.3);
                }
            }

            // Color by link status
            const linked = linkUp[i] === 1;
            const bodyMat = group.userData.bodyMat;
            const ledMat = group.userData.ledMat;
            bodyMat.color.setHex(linked ? 0x00d4ff : 0xff4444);
            bodyMat.emissive.setHex(linked ? 0x003355 : 0x441111);
            ledMat.color.setHex(linked ? 0x00ff88 : 0xff2222);

            // Scale by queue (subtle)
            const qScale = 1.0 + (qLens[i] / 100) * 0.4;
            group.scale.setScalar(Math.min(qScale, 1.8));

            // Link line
            const sinkScene = this.sinkMesh.position;
            const linePos = line.geometry.attributes.position;
            linePos.setXYZ(0, px, py, pz);
            linePos.setXYZ(1, sinkScene.x, sinkScene.y, sinkScene.z);
            linePos.needsUpdate = true;
            line.material.color.setHex(linked ? 0x00aa66 : 0x662222);
            line.material.opacity = linked ? 0.2 : 0.05;
        }
    },

    // ── Interpolated render (B2) ──
    _renderInterpolated() {
        if (!this.prevState || !this.currState || !this.interpEnabled) return;

        const now = performance.now();
        const elapsed = now - this.stateTimestamp;
        let alpha = Math.min(elapsed / this.expectedTickMs, 1.0);

        const prev = this.prevState;
        const curr = this.currState;
        const N = curr.N;
        const sf = this.scaleF;

        if (N !== this.currentN) return;

        for (let i = 0; i < N; i++) {
            const group = this.uavGroups[i];
            if (!group || !prev.positions[i] || !curr.positions[i]) continue;

            // Lerp position
            const px = ((1 - alpha) * prev.positions[i][0] + alpha * curr.positions[i][0]) * sf;
            const py = ((1 - alpha) * prev.positions[i][2] + alpha * curr.positions[i][2]) * sf;
            const pz = ((1 - alpha) * prev.positions[i][1] + alpha * curr.positions[i][1]) * sf;
            group.position.set(px, py, pz);

            // Interpolate link line
            const line = this.linkLines[i];
            if (line) {
                const sinkScene = this.sinkMesh.position;
                const linePos = line.geometry.attributes.position;
                linePos.setXYZ(0, px, py, pz);
                linePos.setXYZ(1, sinkScene.x, sinkScene.y, sinkScene.z);
                linePos.needsUpdate = true;
            }
        }
    },

    onResize() {
        const container = document.getElementById('viewport');
        const w = container.clientWidth;
        const h = container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    },

    animate() {
        requestAnimationFrame(() => this.animate());

        // Animate rotors
        const t = performance.now() * 0.015;
        this.rotorMeshes.forEach((r, i) => {
            r.rotation.z = t + i * 0.7;
        });

        // Smooth interpolation between snapshots
        if (this.interpEnabled && this.prevState && this.currState) {
            this._renderInterpolated();
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
};
