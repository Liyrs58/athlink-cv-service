"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

/* ------------------------------------------------------------------ */
/*  Formation phases — 11 players per team, smooth tactical shifts     */
/* ------------------------------------------------------------------ */
const TEAM_A_PHASES = [
  // 4-3-3
  [[-0.5,0,-4.2],[-2.5,0,-2.5],[2.5,0,-2.5],[-0.8,0,-2.5],[0.8,0,-2.5],[-3,0,-0.5],[0,0,-1],[3,0,-0.5],[-2,0,1.5],[0,0,2.5],[2,0,1.5]],
  // Attacking
  [[-0.5,0,-4],[-2.8,0,-1.8],[2.8,0,-1.8],[-0.8,0,-2],[0.8,0,-2],[-3.5,0,0.5],[0.2,0,0],[3.5,0,0.5],[-2.5,0,2.5],[0,0,3.5],[2.5,0,2.5]],
  // Defensive
  [[-0.5,0,-4.5],[-2.2,0,-3.2],[2.2,0,-3.2],[-0.6,0,-3],[0.6,0,-3],[-2.5,0,-1.5],[0,0,-2],[2.5,0,-1.5],[-1.8,0,0],[0,0,0.5],[1.8,0,0]],
].map(phase => phase.map(p => p as [number, number, number]));

const TEAM_B_PHASES = [
  // 4-4-2
  [[0.5,0,4.2],[-2.5,0,2.5],[2.5,0,2.5],[-0.8,0,2.5],[0.8,0,2.5],[-3,0,0.8],[3,0,0.8],[-1.2,0,0.3],[1.2,0,0.3],[-1,0,-1.5],[1,0,-1.5]],
  // Defensive
  [[0.5,0,4.5],[-2.2,0,3],[2.2,0,3],[-0.6,0,3],[0.6,0,3],[-2.5,0,1.5],[2.5,0,1.5],[-1,0,1],[1,0,1],[-0.8,0,-0.5],[0.8,0,-0.5]],
  // Attacking
  [[0.5,0,4],[-2.8,0,2],[2.8,0,2],[-1,0,2],[1,0,2],[-3.5,0,-0.2],[3.5,0,-0.2],[-1.5,0,-0.8],[1.5,0,-0.8],[-1.2,0,-2.5],[1.2,0,-2.5]],
].map(phase => phase.map(p => p as [number, number, number]));

const BALL_TARGETS = [
  [0, 0.11, 0], [-1.5, 0.11, -1], [2, 0.11, 0.5], [-0.5, 0.11, 1.5],
  [1.2, 0.11, -0.8], [-2, 0.11, 0.3], [0.8, 0.11, 2], [0, 0.11, -1.5],
];

const PHASE_DURATION = 5;

/* ------------------------------------------------------------------ */
/*  Realistic player model with proper human proportions               */
/* ------------------------------------------------------------------ */
function createRealisticPlayer(jerseyColor: number, shortsColor: number, isGoalkeeper: boolean): THREE.Group {
  const group = new THREE.Group();

  const skinColor = 0xc68642;
  const skinMat = new THREE.MeshStandardMaterial({ color: skinColor, roughness: 0.7, metalness: 0 });
  const jerseyMat = new THREE.MeshStandardMaterial({ color: jerseyColor, roughness: 0.55, metalness: 0.02 });
  const shortsMat = new THREE.MeshStandardMaterial({ color: shortsColor, roughness: 0.5, metalness: 0.02 });
  const sockMat = new THREE.MeshStandardMaterial({ color: isGoalkeeper ? 0x333333 : 0xffffff, roughness: 0.5 });
  const bootMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.3, metalness: 0.4 });
  const hairMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.85 });

  // Boots
  for (const side of [-1, 1]) {
    const bootGroup = new THREE.Group();
    const soleGeo = new THREE.BoxGeometry(0.09, 0.02, 0.18);
    soleGeo.translate(0, 0.01, 0.01);
    bootGroup.add(new THREE.Mesh(soleGeo, bootMat));
    const upperGeo = new THREE.CylinderGeometry(0.04, 0.045, 0.06, 10);
    upperGeo.translate(0, 0.05, 0);
    bootGroup.add(new THREE.Mesh(upperGeo, bootMat));
    const toeGeo = new THREE.SphereGeometry(0.04, 8, 6, 0, Math.PI * 2, 0, Math.PI / 2);
    toeGeo.rotateX(-Math.PI / 2);
    toeGeo.translate(0, 0.025, 0.08);
    bootGroup.add(new THREE.Mesh(toeGeo, bootMat));
    bootGroup.position.set(side * 0.08, 0, 0);
    group.add(bootGroup);
  }

  // Socks + shins
  for (const side of [-1, 1]) {
    const sockGeo = new THREE.CylinderGeometry(0.038, 0.035, 0.22, 10);
    const sock = new THREE.Mesh(sockGeo, sockMat);
    sock.position.set(side * 0.08, 0.19, 0);
    sock.castShadow = true;
    group.add(sock);
  }

  // Knees + thighs (skin)
  for (const side of [-1, 1]) {
    const kneeGeo = new THREE.SphereGeometry(0.04, 10, 10);
    const knee = new THREE.Mesh(kneeGeo, skinMat);
    knee.position.set(side * 0.08, 0.33, 0);
    group.add(knee);
    const thighGeo = new THREE.CylinderGeometry(0.055, 0.045, 0.12, 10);
    const thigh = new THREE.Mesh(thighGeo, skinMat);
    thigh.position.set(side * 0.08, 0.41, 0);
    thigh.castShadow = true;
    group.add(thigh);
  }

  // Shorts
  const shortsGeo = new THREE.CylinderGeometry(0.13, 0.15, 0.14, 14);
  const shorts = new THREE.Mesh(shortsGeo, shortsMat);
  shorts.position.set(0, 0.53, 0);
  shorts.castShadow = true;
  group.add(shorts);

  // Torso
  const torsoGeo = new THREE.CylinderGeometry(0.11, 0.13, 0.28, 14);
  const torso = new THREE.Mesh(torsoGeo, jerseyMat);
  torso.position.set(0, 0.74, 0);
  torso.castShadow = true;
  group.add(torso);

  // Shoulders
  const shoulderGeo = new THREE.CylinderGeometry(0.145, 0.12, 0.06, 14);
  const shoulder = new THREE.Mesh(shoulderGeo, jerseyMat);
  shoulder.position.set(0, 0.92, 0);
  group.add(shoulder);

  // Arms
  for (const side of [-1, 1]) {
    // Sleeve
    const sleeveGeo = new THREE.CylinderGeometry(0.038, 0.045, 0.1, 10);
    const sleeve = new THREE.Mesh(sleeveGeo, jerseyMat);
    sleeve.position.set(side * 0.16, 0.86, 0);
    sleeve.rotation.z = side * -0.25;
    sleeve.castShadow = true;
    group.add(sleeve);
    // Forearm (skin)
    const armGeo = new THREE.CylinderGeometry(0.028, 0.035, 0.18, 8);
    const arm = new THREE.Mesh(armGeo, skinMat);
    arm.position.set(side * 0.2, 0.72, 0.01);
    arm.rotation.z = side * -0.12;
    arm.castShadow = true;
    group.add(arm);
    // Hand
    const handGeo = new THREE.SphereGeometry(0.025, 8, 8);
    const hand = new THREE.Mesh(handGeo, skinMat);
    hand.position.set(side * 0.22, 0.62, 0.02);
    group.add(hand);
  }

  // Neck
  const neckGeo = new THREE.CylinderGeometry(0.04, 0.05, 0.05, 10);
  const neck = new THREE.Mesh(neckGeo, skinMat);
  neck.position.set(0, 0.98, 0);
  group.add(neck);

  // Head
  const headGeo = new THREE.SphereGeometry(0.085, 20, 20);
  headGeo.scale(1, 1.1, 1);
  const head = new THREE.Mesh(headGeo, skinMat);
  head.position.set(0, 1.1, 0);
  head.castShadow = true;
  group.add(head);

  // Hair
  const hairGeo = new THREE.SphereGeometry(0.09, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2.2);
  hairGeo.scale(1.05, 0.9, 1.05);
  const hair = new THREE.Mesh(hairGeo, hairMat);
  hair.position.set(0, 1.12, -0.005);
  group.add(hair);

  // Eyes (tiny dots)
  const eyeMat = new THREE.MeshBasicMaterial({ color: 0x111111 });
  for (const side of [-1, 1]) {
    const eyeGeo = new THREE.SphereGeometry(0.008, 6, 6);
    const eye = new THREE.Mesh(eyeGeo, eyeMat);
    eye.position.set(side * 0.03, 1.12, 0.08);
    group.add(eye);
  }

  group.traverse(c => { if (c instanceof THREE.Mesh) { c.castShadow = true; c.receiveShadow = true; } });
  return group;
}

/* ------------------------------------------------------------------ */
/*  Goal net geometry                                                  */
/* ------------------------------------------------------------------ */
function createGoalNet(scene: THREE.Scene, x: number, facing: number) {
  const postMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.6 });
  const netMat = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.15, side: THREE.DoubleSide, wireframe: true });

  const postRadius = 0.05;
  const goalW = 2.4, goalH = 0.8, goalD = 0.6;

  // Posts
  for (const side of [-1, 1]) {
    const postGeo = new THREE.CylinderGeometry(postRadius, postRadius, goalH, 8);
    const post = new THREE.Mesh(postGeo, postMat);
    post.position.set(x, goalH / 2, side * goalW / 2);
    post.castShadow = true;
    scene.add(post);
  }
  // Crossbar
  const barGeo = new THREE.CylinderGeometry(postRadius, postRadius, goalW, 8);
  const bar = new THREE.Mesh(barGeo, postMat);
  bar.rotation.x = Math.PI / 2;
  bar.position.set(x, goalH, 0);
  bar.castShadow = true;
  scene.add(bar);

  // Net (box mesh)
  const netBackGeo = new THREE.PlaneGeometry(goalW, goalH, 12, 6);
  const netBack = new THREE.Mesh(netBackGeo, netMat);
  netBack.position.set(x + facing * goalD, goalH / 2, 0);
  scene.add(netBack);

  // Net sides
  for (const side of [-1, 1]) {
    const sideGeo = new THREE.PlaneGeometry(goalD, goalH, 4, 6);
    const sideMesh = new THREE.Mesh(sideGeo, netMat);
    sideMesh.rotation.y = Math.PI / 2;
    sideMesh.position.set(x + facing * goalD / 2, goalH / 2, side * goalW / 2);
    scene.add(sideMesh);
  }
  // Net top
  const topGeo = new THREE.PlaneGeometry(goalW, goalD, 12, 3);
  const topMesh = new THREE.Mesh(topGeo, netMat);
  topMesh.rotation.x = Math.PI / 2;
  topMesh.position.set(x + facing * goalD / 2, goalH, 0);
  scene.add(topMesh);
}

/* ------------------------------------------------------------------ */
/*  Corner flags                                                       */
/* ------------------------------------------------------------------ */
function createCornerFlag(scene: THREE.Scene, x: number, z: number) {
  const poleMat = new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.3 });
  const flagMat = new THREE.MeshStandardMaterial({ color: 0xff4444, roughness: 0.6, side: THREE.DoubleSide });
  const poleGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.6, 6);
  const pole = new THREE.Mesh(poleGeo, poleMat);
  pole.position.set(x, 0.3, z);
  scene.add(pole);
  const flagGeo = new THREE.PlaneGeometry(0.12, 0.08);
  const flag = new THREE.Mesh(flagGeo, flagMat);
  flag.position.set(x + 0.06, 0.56, z);
  scene.add(flag);
}

/* ------------------------------------------------------------------ */
/*  Pitch line builder                                                 */
/* ------------------------------------------------------------------ */
function buildPitchLines(scene: THREE.Scene) {
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 });
  const addLine = (pts: THREE.Vector3[]) => {
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, lineMat);
    line.position.y = 0.006;
    scene.add(line);
  };
  const W = 7, H = 5;

  // Outline
  addLine([new THREE.Vector3(-W, 0, -H), new THREE.Vector3(W, 0, -H), new THREE.Vector3(W, 0, H), new THREE.Vector3(-W, 0, H), new THREE.Vector3(-W, 0, -H)]);
  // Halfway
  addLine([new THREE.Vector3(0, 0, -H), new THREE.Vector3(0, 0, H)]);
  // Centre circle
  const cc: THREE.Vector3[] = [];
  for (let i = 0; i <= 64; i++) { const a = (i / 64) * Math.PI * 2; cc.push(new THREE.Vector3(Math.cos(a) * 1.5, 0, Math.sin(a) * 1.5)); }
  addLine(cc);
  // Centre spot
  const spotGeo = new THREE.CircleGeometry(0.06, 16);
  const spotMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 });
  const spot = new THREE.Mesh(spotGeo, spotMat);
  spot.rotation.x = -Math.PI / 2;
  spot.position.y = 0.006;
  scene.add(spot);
  // Penalty boxes
  const pbW = 2.2, pbH = 2.8;
  addLine([new THREE.Vector3(-W, 0, -pbW), new THREE.Vector3(-W + pbH, 0, -pbW), new THREE.Vector3(-W + pbH, 0, pbW), new THREE.Vector3(-W, 0, pbW)]);
  addLine([new THREE.Vector3(W, 0, -pbW), new THREE.Vector3(W - pbH, 0, -pbW), new THREE.Vector3(W - pbH, 0, pbW), new THREE.Vector3(W, 0, pbW)]);
  // Goal boxes
  const gbW = 1.0, gbH = 1.2;
  addLine([new THREE.Vector3(-W, 0, -gbW), new THREE.Vector3(-W + gbH, 0, -gbW), new THREE.Vector3(-W + gbH, 0, gbW), new THREE.Vector3(-W, 0, gbW)]);
  addLine([new THREE.Vector3(W, 0, -gbW), new THREE.Vector3(W - gbH, 0, -gbW), new THREE.Vector3(W - gbH, 0, gbW), new THREE.Vector3(W, 0, gbW)]);
  // Penalty spots
  for (const xp of [-W + 2.2, W - 2.2]) {
    const psGeo = new THREE.CircleGeometry(0.04, 12);
    const ps = new THREE.Mesh(psGeo, spotMat);
    ps.rotation.x = -Math.PI / 2;
    ps.position.set(xp, 0.006, 0);
    scene.add(ps);
  }
  // Penalty arcs
  for (const [xp, startA, endA] of [[-W + 2.2, -0.9, 0.9], [W - 2.2, Math.PI - 0.9, Math.PI + 0.9]] as [number, number, number][]) {
    const arc: THREE.Vector3[] = [];
    for (let i = 0; i <= 32; i++) { const a = startA + (i / 32) * (endA - startA); arc.push(new THREE.Vector3(xp + Math.cos(a) * 1.5, 0, Math.sin(a) * 1.5)); }
    addLine(arc);
  }
  // Corner arcs
  const cR = 0.3;
  for (const [cx, cz, sa] of [[-W, -H, 0], [W, -H, Math.PI / 2], [W, H, Math.PI], [-W, H, Math.PI * 1.5]] as [number, number, number][]) {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= 12; i++) { const a = sa + (i / 12) * (Math.PI / 2); pts.push(new THREE.Vector3(cx + Math.cos(a) * cR, 0, cz + Math.sin(a) * cR)); }
    addLine(pts);
  }
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */
export default function FootballPitch3D() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    // Atmospheric fog
    scene.fog = new THREE.FogExp2(0x0a1a0a, 0.035);

    const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 10, 9);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.querySelector("canvas")?.remove();
    container.prepend(renderer.domElement);
    renderer.domElement.style.display = "block";
    renderer.domElement.style.borderRadius = "12px";

    // Lighting — stadium floodlight feel
    scene.add(new THREE.AmbientLight(0x334433, 0.8));

    const mainLight = new THREE.DirectionalLight(0xffeedd, 1.2);
    mainLight.position.set(5, 12, 4);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.set(2048, 2048);
    mainLight.shadow.camera.near = 0.5;
    mainLight.shadow.camera.far = 30;
    mainLight.shadow.camera.left = -12;
    mainLight.shadow.camera.right = 12;
    mainLight.shadow.camera.top = 8;
    mainLight.shadow.camera.bottom = -8;
    mainLight.shadow.bias = -0.001;
    scene.add(mainLight);

    // Secondary fill
    const fill = new THREE.DirectionalLight(0xaaccff, 0.4);
    fill.position.set(-6, 8, -3);
    scene.add(fill);

    // Floodlight spots (4 corners)
    for (const [x, z] of [[-8, -6], [8, -6], [-8, 6], [8, 6]]) {
      const spot = new THREE.SpotLight(0xffffee, 0.5, 25, Math.PI / 5, 0.6, 1);
      spot.position.set(x, 10, z);
      spot.target.position.set(0, 0, 0);
      scene.add(spot);
      scene.add(spot.target);
    }

    // Pitch — realistic striped grass
    const stripeCount = 20;
    const pitchW = 14, pitchH = 10;
    const stripeW = pitchW / stripeCount;
    for (let i = 0; i < stripeCount; i++) {
      const geo = new THREE.PlaneGeometry(stripeW, pitchH);
      const shade = i % 2 === 0 ? 0x1a6b1a : 0x1e7b1e;
      const mat = new THREE.MeshStandardMaterial({ color: shade, roughness: 0.85, metalness: 0 });
      const stripe = new THREE.Mesh(geo, mat);
      stripe.rotation.x = -Math.PI / 2;
      stripe.position.set(-pitchW / 2 + stripeW / 2 + i * stripeW, 0, 0);
      stripe.receiveShadow = true;
      scene.add(stripe);
    }

    // Surroundings — dark area around pitch
    const surroundGeo = new THREE.PlaneGeometry(30, 22);
    const surroundMat = new THREE.MeshStandardMaterial({ color: 0x0a1a0a, roughness: 1 });
    const surround = new THREE.Mesh(surroundGeo, surroundMat);
    surround.rotation.x = -Math.PI / 2;
    surround.position.y = -0.01;
    surround.receiveShadow = true;
    scene.add(surround);

    // Lines
    buildPitchLines(scene);

    // Goals
    createGoalNet(scene, -7, -1);
    createGoalNet(scene, 7, 1);

    // Corner flags
    for (const [x, z] of [[-7, -5], [7, -5], [-7, 5], [7, 5]]) {
      createCornerFlag(scene, x, z);
    }

    // Players — 11 per team
    const teamAPlayers: THREE.Group[] = [];
    const teamBPlayers: THREE.Group[] = [];

    for (let i = 0; i < 11; i++) {
      const isGK = i === 0;
      const pA = createRealisticPlayer(isGK ? 0xffcc00 : 0x00d4aa, isGK ? 0x333333 : 0x008866, isGK);
      const posA = TEAM_A_PHASES[0][i];
      pA.position.set(posA[0], 0, posA[2]);
      scene.add(pA);
      teamAPlayers.push(pA);

      const pB = createRealisticPlayer(isGK ? 0x22cc22 : 0xeeeeee, isGK ? 0x222222 : 0x333333, isGK);
      const posB = TEAM_B_PHASES[0][i];
      pB.position.set(posB[0], 0, posB[2]);
      scene.add(pB);
      teamBPlayers.push(pB);
    }

    // Ball
    const ballGeo = new THREE.SphereGeometry(0.11, 24, 24);
    const ballMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.1 });
    const ball = new THREE.Mesh(ballGeo, ballMat);
    ball.position.set(0, 0.11, 0);
    ball.castShadow = true;
    scene.add(ball);

    // Ball trail (fading spheres)
    const trailSpheres: THREE.Mesh[] = [];
    const trailMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.15 });
    for (let i = 0; i < 6; i++) {
      const ts = new THREE.Mesh(new THREE.SphereGeometry(0.04 - i * 0.005, 8, 8), trailMat.clone());
      (ts.material as THREE.MeshBasicMaterial).opacity = 0.15 - i * 0.02;
      ts.visible = false;
      scene.add(ts);
      trailSpheres.push(ts);
    }

    // ========== ANIMATION ==========
    let frameId: number;
    const clock = new THREE.Clock();
    const ballHistory: THREE.Vector3[] = [];

    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      // Phase interpolation
      const phaseFloat = (elapsed / PHASE_DURATION) % TEAM_A_PHASES.length;
      const phaseFrom = Math.floor(phaseFloat) % TEAM_A_PHASES.length;
      const phaseTo = (phaseFrom + 1) % TEAM_A_PHASES.length;
      const t = phaseFloat - Math.floor(phaseFloat);
      const smoothT = t * t * (3 - 2 * t); // smoothstep

      for (let i = 0; i < 11; i++) {
        const aFrom = TEAM_A_PHASES[phaseFrom][i];
        const aTo = TEAM_A_PHASES[phaseTo][i];
        teamAPlayers[i].position.x = THREE.MathUtils.lerp(aFrom[0], aTo[0], smoothT);
        teamAPlayers[i].position.z = THREE.MathUtils.lerp(aFrom[2], aTo[2], smoothT);
        // Players face movement direction
        const dx = aTo[0] - aFrom[0];
        const dz = aTo[2] - aFrom[2];
        if (Math.abs(dx) + Math.abs(dz) > 0.01) {
          teamAPlayers[i].rotation.y = Math.atan2(dx, dz);
        }
        // Subtle running bob
        teamAPlayers[i].position.y = Math.abs(Math.sin(elapsed * 6 + i)) * 0.01;

        const bFrom = TEAM_B_PHASES[phaseFrom][i];
        const bTo = TEAM_B_PHASES[phaseTo][i];
        teamBPlayers[i].position.x = THREE.MathUtils.lerp(bFrom[0], bTo[0], smoothT);
        teamBPlayers[i].position.z = THREE.MathUtils.lerp(bFrom[2], bTo[2], smoothT);
        const bdx = bTo[0] - bFrom[0];
        const bdz = bTo[2] - bFrom[2];
        if (Math.abs(bdx) + Math.abs(bdz) > 0.01) {
          teamBPlayers[i].rotation.y = Math.atan2(bdx, bdz);
        }
        teamBPlayers[i].position.y = Math.abs(Math.sin(elapsed * 6 + i + 5)) * 0.01;
      }

      // Ball — smooth between targets with arc
      const ballFloat = (elapsed / 2) % BALL_TARGETS.length;
      const bF = Math.floor(ballFloat) % BALL_TARGETS.length;
      const bT = (bF + 1) % BALL_TARGETS.length;
      const bt = ballFloat - Math.floor(ballFloat);
      const smoothBt = bt * bt * (3 - 2 * bt);
      const arcY = 0.11 + 0.25 * Math.sin(bt * Math.PI);
      ball.position.x = THREE.MathUtils.lerp(BALL_TARGETS[bF][0], BALL_TARGETS[bT][0], smoothBt);
      ball.position.y = arcY;
      ball.position.z = THREE.MathUtils.lerp(BALL_TARGETS[bF][2], BALL_TARGETS[bT][2], smoothBt);
      ball.rotation.x += 0.08;
      ball.rotation.z += 0.04;

      // Ball trail
      ballHistory.unshift(ball.position.clone());
      if (ballHistory.length > 6) ballHistory.pop();
      trailSpheres.forEach((ts, i) => {
        if (i < ballHistory.length) {
          ts.visible = true;
          ts.position.copy(ballHistory[i]);
        }
      });

      // Camera: slow orbit
      const camAngle = elapsed * 0.15;
      const camRadius = 13;
      camera.position.x = Math.sin(camAngle) * camRadius * 0.3;
      camera.position.z = 9 + Math.cos(camAngle) * 2;
      camera.position.y = 10 + Math.sin(elapsed * 0.2) * 0.5;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full aspect-[4/3] max-w-[560px] mx-auto rounded-xl overflow-hidden"
    >
      {/* Vignette */}
      <div
        className="absolute inset-0 pointer-events-none z-10 rounded-xl"
        style={{ background: "radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.6) 100%)" }}
      />
      {/* Scanlines */}
      <div
        className="absolute inset-0 pointer-events-none z-10 rounded-xl"
        style={{ background: "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.02) 3px, rgba(0,0,0,0.02) 4px)" }}
      />
      {/* Broadcast overlay */}
      <div className="absolute top-3 left-3 z-20 pointer-events-none">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <span className="text-[10px] text-white/60 font-mono tracking-wider">LIVE TACTICAL VIEW</span>
        </div>
      </div>
      <div className="absolute top-3 right-3 z-20 pointer-events-none">
        <span className="text-[10px] text-white/40 font-mono">ATHLINK CV</span>
      </div>
      {/* Inner glow */}
      <div className="absolute inset-0 pointer-events-none z-10 rounded-xl" style={{ boxShadow: "inset 0 0 60px rgba(0,212,170,0.12)" }} />
    </div>
  );
}
