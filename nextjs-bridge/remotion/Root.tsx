import { Composition } from "remotion";
import { PitchReplay, pitchReplaySchema } from "./compositions/PitchReplay";

// The composition runs at the source video's fps so 1 composition frame =
// 1 video frame. The render invocation passes --props '{"videoFps": N,
// "exportUrl": "...", "jobId": "..."}' and calculateMetadata picks up
// fps + durationInFrames from the loaded export.
const DEFAULT_FPS = 25;
const DEFAULT_DURATION_SECONDS = 10;

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="PitchReplay"
        component={PitchReplay}
        durationInFrames={DEFAULT_FPS * DEFAULT_DURATION_SECONDS}
        fps={DEFAULT_FPS}
        width={1920}
        height={1080}
        schema={pitchReplaySchema}
        defaultProps={{"jobId":"villa_psg_10s_v2","exportUrl":"http://localhost:8001/api/v1/export/villa_psg_10s_v2","videoFps":25}}
        calculateMetadata={async ({ props }) => {
          // If an exportUrl is supplied, read its videoMeta to lock fps/duration
          // to the source video. Without this, a 2-minute clip at 25 fps would
          // render at the default 10 s / 25 fps.
          if (!props.exportUrl) {
            return { props };
          }
          try {
            const res = await fetch(props.exportUrl);
            if (!res.ok) return { props };
            const raw = (await res.json()) as {
              videoMeta?: { fps?: number; frameCount?: number };
            };
            const fps = raw.videoMeta?.fps || DEFAULT_FPS;
            const frameCount =
              raw.videoMeta?.frameCount || DEFAULT_FPS * DEFAULT_DURATION_SECONDS;
            return {
              fps,
              durationInFrames: frameCount,
              props: { ...props, videoFps: fps },
            };
          } catch {
            return { props };
          }
        }}
      />
    </>
  );
};
