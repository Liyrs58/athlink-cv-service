import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";

export const PostFX: React.FC = () => (
  <EffectComposer>
    <Bloom intensity={0.6} luminanceThreshold={0.9} mipmapBlur />
    <Vignette eskil={false} offset={0.15} darkness={0.65} />
  </EffectComposer>
);
