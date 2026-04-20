export const PITCH_LENGTH = 105; // metres, along X
export const PITCH_WIDTH = 68; // metres, along Z
export const PLAYER_HEIGHT = 1.85; // metres
export const PLAYER_RADIUS = 0.38; // metres
export const BALL_RADIUS = 0.11; // metres (FIFA size 5)
export const CHALK_COLOR = "#eaf2ec";
export const PITCH_GREEN_LIGHT = "#0b2e1c";
export const PITCH_GREEN_DARK = "#071c10";
export const SKY_COLOR = "#03060a";

/** Convert pitch coords (0..105, 0..68) to centered world coords. */
export function pitchToWorld(px: number, py: number): [number, number] {
  /**
   * Python Pitch Coords (0 to 105, 0 to 68) 
   * to Three.js World Coords (Centered at 0,0)
   * We negate Y because Three.js -Z is "Forward"
   */
  return [
    px - PITCH_LENGTH / 2, 
    -(py - PITCH_WIDTH / 2)
  ];
}
