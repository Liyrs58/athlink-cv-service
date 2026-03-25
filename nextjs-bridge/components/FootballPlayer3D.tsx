'use client'

import { useEffect, useMemo, useRef, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { useGLTF, useAnimations, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

/*
 * Ch38 texture pixel analysis (sampled from actual GLB texture):
 *
 * Kit  (white shirt/shorts): brightness > 0.75, warmth (r-b) ≈ 0.008-0.02
 * Skin (face/neck/hands):    warmth > 0.08,     brightness  ≈ 0.35-0.75
 * Dark (boots/seams):        brightness < 0.08
 * Other (socks/midtones):    brightness 0.46-0.71, warmth < 0.08
 *
 * Material structure:
 *   Ch38_body → Mesh, Mesh.001, Mesh.002, Mesh.003, Mesh.004 (full body atlas)
 *   Ch38_hair → Mesh.005, Mesh.006 (hair texture, nearly all dark)
 */

function makeBodyMaterial(
  originalMap: THREE.Texture | null,
  kitHex: string,
  skinHex: string,
  hairHex: string,
): THREE.MeshStandardMaterial {
  const uKit  = { value: new THREE.Color(kitHex) }
  const uSkin = { value: new THREE.Color(skinHex) }
  const uHair = { value: new THREE.Color(hairHex) }

  const mat = new THREE.MeshStandardMaterial({ map: originalMap, roughness: 0.85, metalness: 0 })
  ;(mat as any)._u = { uKit, uSkin, uHair }

  // Force a unique cache key so Three.js never reuses another program
  mat.customProgramCacheKey = () => 'ch38body'

  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uKit  = uKit
    shader.uniforms.uSkin = uSkin
    shader.uniforms.uHair = uHair

    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <common>',
      `#include <common>
uniform vec3 uKit;
uniform vec3 uSkin;
uniform vec3 uHair;`,
    )

    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <map_fragment>',
      `#include <map_fragment>
{
  vec3 c = diffuseColor.rgb;
  float br = (c.r + c.g + c.b) / 3.0;
  float wm = c.r - c.b;   // warmth: high = skin-toned, near-zero = neutral/grey

  // Kit: neutral-bright (shirt, shorts, socks)
  float isKit = 0.0;
  if (br > 0.72 && wm < 0.06) {
    isKit = smoothstep(0.72, 0.82, br) * (1.0 - smoothstep(0.0, 0.06, abs(wm)));
  }

  // Skin: warm-toned mid-brightness (face, neck, hands, forearms)
  float isSkin = 0.0;
  if (wm > 0.07 && br > 0.32 && br < 0.78) {
    isSkin = smoothstep(0.07, 0.14, wm)
           * smoothstep(0.30, 0.40, br)
           * (1.0 - smoothstep(0.70, 0.80, br));
  }

  // Boots / very dark areas (fixed near-black)
  float isBoots = 0.0;
  if (br < 0.10) {
    isBoots = smoothstep(0.10, 0.04, br);
  }

  // Kit and skin are mutually exclusive — skin wins
  isKit = isKit * (1.0 - isSkin);

  vec3 result = c;
  result = mix(result, uSkin, isSkin * 0.90);
  result = mix(result, uKit,  isKit  * 0.92);
  result = mix(result, vec3(0.05, 0.05, 0.06), isBoots);

  diffuseColor.rgb = result;
}`,
    )
  }

  return mat
}

function makeHairMaterial(
  originalMap: THREE.Texture | null,
  hairHex: string,
): THREE.MeshStandardMaterial {
  const uHair = { value: new THREE.Color(hairHex) }
  const mat = new THREE.MeshStandardMaterial({ map: originalMap, roughness: 0.9, metalness: 0 })
  ;(mat as any)._uHair = uHair

  mat.customProgramCacheKey = () => 'ch38hair'

  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uHair = uHair
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <common>',
      `#include <common>
uniform vec3 uHair;`,
    )
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <map_fragment>',
      `#include <map_fragment>
{
  // Hair texture is near-black — tint any pixel that has alpha
  float br = (diffuseColor.r + diffuseColor.g + diffuseColor.b) / 3.0;
  // Darken the target hair colour to preserve shading
  float shade = br * 4.0; // amplify: dark tex stays dark, lighter areas get more colour
  shade = clamp(shade, 0.0, 1.0);
  diffuseColor.rgb = uHair * shade;
}`,
    )
  }

  return mat
}

function applyMaterials(
  scene: THREE.Object3D,
  kitColour: string,
  skinColour: string,
  hairColour: string,
) {
  scene.traverse((child) => {
    if (!(child instanceof THREE.Mesh || child instanceof THREE.SkinnedMesh)) return
    child.frustumCulled = false
    child.castShadow = true

    const raw = child.material as THREE.MeshStandardMaterial
    if ((raw as any)._u || (raw as any)._uHair) return // already replaced

    const matName = (raw?.name || '').toLowerCase()
    const meshName = child.name.toLowerCase()
    const name = matName || meshName

    if (name.includes('hair') || name.includes('eyelash')) {
      child.material = makeHairMaterial(raw.map, hairColour)
    } else if (
      name.includes('body') || name.includes('shirt') || name.includes('shorts') ||
      name.includes('socks') || name.includes('shoes') || name.includes('mesh')
    ) {
      child.material = makeBodyMaterial(raw.map, kitColour, skinColour, hairColour)
    }
  })
}

function updateUniforms(
  scene: THREE.Object3D,
  kitColour: string,
  skinColour: string,
  hairColour: string,
) {
  scene.traverse((child) => {
    if (!(child instanceof THREE.Mesh || child instanceof THREE.SkinnedMesh)) return
    const mat = child.material as any
    if (mat?._u) {
      mat._u.uKit.value.set(kitColour)
      mat._u.uSkin.value.set(skinColour)
      mat._u.uHair.value.set(hairColour)
    }
    if (mat?._uHair) {
      mat._uHair.value.set(hairColour)
    }
  })
}

/* ── PlayerModel ─────────────────────────────────────────── */
function PlayerModel({
  kitColour,
  skinColour,
  hairColour,
  onScene,
}: {
  kitColour: string
  skinColour: string
  hairColour: string
  onScene: (scene: THREE.Object3D) => void
}) {
  const { scene: idleScene, animations: idleAnims } = useGLTF('/models/idle.glb')
  const { animations: shovedAnims } = useGLTF('/models/shoved.glb')

  // Merge both animation clips into one useAnimations call on the idle scene
  const allAnims = useMemo(
    () => [...idleAnims, ...shovedAnims],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [idleAnims, shovedAnims],
  )
  const { actions, mixer } = useAnimations(allAnims, idleScene)

  // Expose scene to parent for football bone tracking
  useEffect(() => { onScene(idleScene) }, [idleScene, onScene])

  // Install materials once on scene load
  useEffect(() => {
    applyMaterials(idleScene, kitColour, skinColour, hairColour)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idleScene])

  // Live-update uniforms when colour pickers change
  useEffect(() => {
    updateUniforms(idleScene, kitColour, skinColour, hairColour)
  }, [idleScene, kitColour, skinColour, hairColour])

  // Animation sequencing: idle loop → shoved (one-shot) → back to idle
  useEffect(() => {
    const idleAction  = actions['Armature|mixamo.com|Layer0']
    const shovedAction = actions['mixamo.com']

    if (!idleAction) return

    idleAction.reset().setLoop(THREE.LoopRepeat, Infinity).play()

    if (!shovedAction) return

    shovedAction.setLoop(THREE.LoopOnce, 1)
    shovedAction.clampWhenFinished = true

    // Every ~8 seconds, crossfade to shoved then back to idle
    let cancelled = false
    const IDLE_HOLD = 8000     // ms before triggering shoved
    const FADE_IN   = 0.4      // s crossfade idle → shoved
    const FADE_OUT  = 0.4      // s crossfade shoved → idle
    const SHOVED_DURATION = 4500 // ms (4.5s)

    function triggerShoved() {
      if (cancelled) return
      shovedAction!.reset().play()
      idleAction!.crossFadeTo(shovedAction!, FADE_IN, true)

      // After shoved finishes, fade back to idle
      setTimeout(() => {
        if (cancelled) return
        idleAction!.reset().play()
        shovedAction!.crossFadeTo(idleAction!, FADE_OUT, true)

        // Schedule next shoved cycle
        setTimeout(triggerShoved, IDLE_HOLD)
      }, SHOVED_DURATION)
    }

    const firstTimer = setTimeout(triggerShoved, IDLE_HOLD)

    return () => {
      cancelled = true
      clearTimeout(firstTimer)
      idleAction.stop()
      shovedAction.stop()
    }
  }, [actions, mixer])

  return <primitive object={idleScene} scale={1.3} position={[0, -1.15, 0]} />
}

/* ── Football ────────────────────────────────────────────── */
function Football({ playerScene }: { playerScene: THREE.Object3D | null }) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const rightFootBone = useRef<THREE.Object3D | null>(null)

  const texture = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 512; canvas.height = 512
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, 512, 512)
    const patches = [
      { x: 256, y: 256 }, { x: 256, y: 130 }, { x: 362, y: 188 },
      { x: 362, y: 324 }, { x: 256, y: 382 }, { x: 150, y: 324 }, { x: 150, y: 188 },
    ]
    patches.forEach(({ x, y }) => {
      ctx.fillStyle = '#111'
      ctx.beginPath()
      for (let i = 0; i < 5; i++) {
        const a = (i * 2 * Math.PI / 5) - Math.PI / 2
        i === 0 ? ctx.moveTo(x + 40 * Math.cos(a), y + 40 * Math.sin(a))
                : ctx.lineTo(x + 40 * Math.cos(a), y + 40 * Math.sin(a))
      }
      ctx.closePath(); ctx.fill()
    })
    return new THREE.CanvasTexture(canvas)
  }, [])

  useEffect(() => {
    if (!playerScene) return
    rightFootBone.current = null
    playerScene.traverse((obj) => {
      if (!(obj instanceof THREE.Bone)) return
      const n = obj.name.toLowerCase()
      if (n.includes('rightfoot') || n.endsWith('rightfoot')) {
        rightFootBone.current = obj
      }
    })
  }, [playerScene])

  useFrame((state) => {
    if (!meshRef.current) return
    if (rightFootBone.current) {
      const pos = new THREE.Vector3()
      rightFootBone.current.getWorldPosition(pos)
      meshRef.current.position.set(pos.x + 0.05, pos.y - 0.08, pos.z + 0.1)
    } else {
      const t = state.clock.elapsedTime
      meshRef.current.position.set(
        0.15 + Math.sin(t * 2) * 0.05, -1.05, 0.2 + Math.cos(t * 2) * 0.03,
      )
    }
    meshRef.current.rotation.x += 0.03
    meshRef.current.rotation.z -= 0.01
  })

  useEffect(() => { return () => { texture.dispose() } }, [texture])

  return (
    <mesh ref={meshRef} castShadow>
      <sphereGeometry args={[0.1, 32, 32]} />
      <meshStandardMaterial map={texture} roughness={0.5} metalness={0.0} />
    </mesh>
  )
}

/* ── Canvas ──────────────────────────────────────────────── */
export interface FootballPlayer3DProps {
  kitColour: string
  skinColour: string
  hairColour: string
  shortsColour?: string
}

export default function FootballPlayer3D({
  kitColour = '#00d4aa',
  skinColour = '#c68642',
  hairColour = '#1a1a1a',
}: FootballPlayer3DProps) {
  const playerSceneRef = useRef<THREE.Object3D | null>(null)

  return (
    <Canvas
      camera={{ position: [0, 0.4, 3.0], fov: 48 }}
      style={{ width: '100%', height: '100%', background: 'transparent' }}
      gl={{ antialias: true, alpha: true }}
      shadows
    >
      <ambientLight intensity={0.7} />
      <directionalLight position={[2, 5, 3]} intensity={1.4} castShadow />
      <directionalLight position={[-2, 3, -1]} intensity={0.5} color="#b0d0ff" />
      <pointLight position={[0, 2, 2]} intensity={0.6} color="#fff5e0" />

      <Suspense fallback={null}>
        <PlayerModel
          kitColour={kitColour}
          skinColour={skinColour}
          hairColour={hairColour}
          onScene={(s) => { playerSceneRef.current = s }}
        />
        <Football playerScene={playerSceneRef.current} />
      </Suspense>

      <OrbitControls
        enablePan={false}
        minDistance={1.8}
        maxDistance={5}
        target={[0, 0.2, 0]}
        minPolarAngle={Math.PI * 0.1}
        maxPolarAngle={Math.PI * 0.8}
      />

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.15, 0]}>
        <circleGeometry args={[0.9, 32]} />
        <meshStandardMaterial color="#00d4aa" opacity={0.1} transparent />
      </mesh>
    </Canvas>
  )
}

useGLTF.preload('/models/idle.glb')
useGLTF.preload('/models/shoved.glb')
