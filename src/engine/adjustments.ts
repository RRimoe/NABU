// src/engine/adjustments.ts
//
// GPU pipeline for non-destructive adjustments.
//
// Every slider is evaluated in a fragment shader rather than in a JavaScript
// pixel loop: on a 24-megapixel photo a per-pixel loop takes seconds, which
// makes a draggable slider useless. On the GPU the whole stack costs one draw
// call, so dragging stays live no matter the resolution.
//
// The result is cached per (image, settings): re-rendering the canvas for an
// unrelated reason must not re-run the pipeline.

import {
  type Adjustments,
  type Curves,
  type FocusEffect,
  areCurvesNeutral,
  isFocusNeutral,
  isNeutral,
  NEUTRAL_CURVES,
} from '../types/document';
import { buildCurveLut, LUT_SIZE } from './curves';

const VERTEX_SHADER = `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
  // A single triangle covering the viewport: cheaper than a quad and avoids
  // the diagonal seam some drivers show between two triangles.
  v_uv = (a_position + 1.0) * 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

/** Separable Gaussian blur, used on its own and as a reference for clarity. */
const BLUR_SHADER = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColour;

uniform sampler2D u_image;
uniform vec2 u_direction;   // (1,0) horizontal, (0,1) vertical
uniform vec2 u_texelSize;
uniform float u_radius;

void main() {
  if (u_radius < 0.5) {
    outColour = texture(u_image, v_uv);
    return;
  }

  // Sigma chosen so the visible extent matches the requested radius.
  float sigma = max(u_radius / 2.0, 0.5);
  float twoSigmaSq = 2.0 * sigma * sigma;

  vec4 sum = texture(u_image, v_uv);
  float weightSum = 1.0;

  // Fixed loop bound: WebGL requires a constant, and 24 taps per direction is
  // plenty for the radii the interface allows.
  for (int i = 1; i <= 24; i++) {
    float offset = float(i);
    if (offset > u_radius) break;

    float weight = exp(-(offset * offset) / twoSigmaSq);
    vec2 delta = u_direction * u_texelSize * offset;

    sum += texture(u_image, v_uv + delta) * weight;
    sum += texture(u_image, v_uv - delta) * weight;
    weightSum += weight * 2.0;
  }

  outColour = sum / weightSum;
}`;

const ADJUST_SHADER = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColour;

uniform sampler2D u_image;      // original pixels
uniform sampler2D u_blurred;    // blurred copy, for blur, clarity and focus
uniform sampler2D u_bloomTex;   // separately blurred copy, for the glow
uniform sampler2D u_lut;        // 256x1 tone-curve table, one axis per channel
uniform sampler2D u_focusMask;  // white where the picture stays sharp
uniform float u_hasLut;         // 0 when every curve is still the diagonal
uniform highp sampler3D u_lut3d; // colour lookup table from a .cube file
uniform float u_lut3dAmount;    // 0 when there is none
uniform float u_lut3dSize;
uniform vec2 u_texelSize;

// Selective focus. 0 off, 1 mask, 2 linear band, 3 ellipse, 4 depth plane.
uniform float u_focusMode;
uniform float u_focusPlane;     // the distance that stays sharp, depth mode
uniform float u_focusWeight;    // blur mix inside the soft zone, 0..1
uniform float u_focusFeather;
uniform float u_focusInvert;
uniform vec2 u_focusCentre;
uniform float u_focusSize;
uniform float u_focusAspect;
uniform float u_focusAngle;     // radians
uniform float u_frameAspect;    // image width / height

uniform float u_exposure;
uniform float u_contrast;
uniform float u_highlights;
uniform float u_shadows;
uniform float u_whites;
uniform float u_blacks;
uniform float u_temperature;
uniform float u_tint;
uniform float u_vibrance;
uniform float u_saturation;
uniform float u_hue;
uniform float u_clarity;
uniform float u_dehaze;
uniform float u_mono;
uniform float u_sharpness;
uniform float u_blurAmount;
uniform float u_vignette;
uniform float u_grain;
uniform float u_bloom;
uniform float u_aberration;
uniform float u_seed;

// Colour mixer, one entry per band: red, orange, yellow, green, aqua, blue,
// purple, magenta. Hue in degrees of rotation, saturation and luminance as
// fractions.
uniform float u_mixHue[8];
uniform float u_mixSat[8];
uniform float u_mixLum[8];
uniform float u_hasMix;

// Colour grading: the colour added to each tonal range, already scaled by its
// saturation and with its own brightness taken out.
uniform vec3 u_gradeShadow;
uniform vec3 u_gradeMid;
uniform vec3 u_gradeHighlight;
uniform float u_gradeBalance;
uniform float u_hasGrade;

/** Centre of each mixer band on the hue circle, in degrees. */
const float BAND_CENTRES[8] = float[8](0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0);

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

/** Brightness a pixel has to pass before it is allowed to glow. */
const float BLOOM_FLOOR = 0.62;

float luminance(vec3 c) { return dot(c, LUMA); }

vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toSrgb(vec3 c)   { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }

/** Cheap deterministic noise, so grain does not shimmer between frames. */
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 applyHue(vec3 colour, float degrees) {
  if (abs(degrees) < 0.001) return colour;
  float angle = radians(degrees);
  float c = cos(angle);
  float s = sin(angle);
  // Rotation about the grey axis in RGB space.
  mat3 m = mat3(
    0.299 + 0.701 * c + 0.168 * s, 0.587 - 0.587 * c + 0.330 * s, 0.114 - 0.114 * c - 0.497 * s,
    0.299 - 0.299 * c - 0.328 * s, 0.587 + 0.413 * c + 0.035 * s, 0.114 - 0.114 * c + 0.292 * s,
    0.299 - 0.300 * c + 1.250 * s, 0.587 - 0.588 * c - 1.050 * s, 0.114 + 0.886 * c - 0.203 * s
  );
  return clamp(colour * m, 0.0, 1.0);
}

vec3 rgbToHsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsvToRgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

/** Shortest distance between two hues, in degrees. */
float hueDistance(float a, float b) {
  float d = abs(a - b);
  return min(d, 360.0 - d);
}

/**
 * The colour mixer. Each pixel takes a blend of the bands its hue sits
 * between, weighted by how close it is to each centre, so moving one band
 * never leaves a hard edge where two colours meet. Greys have no hue, so the
 * effect fades out with saturation: a white wall does not turn blue because
 * the blues were pushed.
 */
vec3 applyMixer(vec3 colour) {
  vec3 hsv = rgbToHsv(colour);
  float hue = hsv.x * 360.0;

  float shift = 0.0;
  float sat = 0.0;
  float lum = 0.0;
  float total = 0.0;
  for (int i = 0; i < 8; i++) {
    int next = i == 7 ? 0 : i + 1;
    int prev = i == 0 ? 7 : i - 1;
    // Each band reaches as far as its neighbours' centres.
    float reach = 0.5 * (hueDistance(BAND_CENTRES[i], BAND_CENTRES[next]) +
                         hueDistance(BAND_CENTRES[i], BAND_CENTRES[prev]));
    float w = max(0.0, 1.0 - hueDistance(hue, BAND_CENTRES[i]) / reach);
    w = w * w * (3.0 - 2.0 * w);
    shift += u_mixHue[i] * w;
    sat += u_mixSat[i] * w;
    lum += u_mixLum[i] * w;
    total += w;
  }
  if (total < 0.0001) return colour;
  shift /= total;
  sat /= total;
  lum /= total;

  float presence = smoothstep(0.02, 0.25, hsv.y);
  hsv.x = fract((hue + shift * presence) / 360.0 + 1.0);
  hsv.y = clamp(hsv.y * (1.0 + sat * presence), 0.0, 1.0);
  vec3 result = hsvToRgb(hsv);
  // Luminance moves the colour towards black or white, scaled the same way.
  float amount = lum * presence;
  result = amount >= 0.0 ? mix(result, vec3(1.0), amount * 0.5)
                         : mix(result, vec3(0.0), -amount * 0.5);
  return clamp(result, 0.0, 1.0);
}

/**
 * Colour grading: shadows, midtones and highlights each take their own tint.
 * The tints carry no brightness of their own, so a grade changes the mood of
 * a picture without making it lighter or darker.
 */
vec3 applyGrade(vec3 colour) {
  float lum = luminance(colour);
  // Balance slides the crossover point between shadows and highlights.
  float pivot = clamp(0.5 - u_gradeBalance * 0.3, 0.15, 0.85);
  float shadow = 1.0 - smoothstep(0.0, pivot, lum);
  float highlight = smoothstep(pivot, 1.0, lum);
  float mid = clamp(1.0 - shadow - highlight, 0.0, 1.0);
  vec3 tint = u_gradeShadow * shadow + u_gradeMid * mid + u_gradeHighlight * highlight;
  return clamp(colour + tint * 0.35, 0.0, 1.0);
}

/**
 * How much this pixel should be blurred by the focus effect: 0 stays sharp,
 * 1 takes the blurred copy in full.
 *
 * The three modes share one number so the rest of the shader never has to know
 * whether the sharp zone came from a segmented object or from a band drawn
 * across the frame.
 */
float focusBlend() {
  if (u_focusMode < 0.5) return 0.0;

  float outside;

  if (u_focusMode < 1.5) {
    // A cutout mask: white marks what stays sharp.
    outside = 1.0 - texture(u_focusMask, v_uv).r;
  } else if (u_focusMode > 3.5) {
    // A depth map, read as a lens reads it: one distance is in focus and
    // everything falls off on both sides of it. That is what separates this
    // from a cutout: the background does not blur uniformly, it blurs more
    // the further away it is, and a foreground branch blurs too.
    float distance = abs(texture(u_focusMask, v_uv).r - u_focusPlane);
    outside = smoothstep(u_focusSize, u_focusSize + max(0.002, u_focusFeather), distance);
  } else {
    // Both axes are put into the same unit (fractions of the frame's height)
    // before any distance is measured. Working in raw texture coordinates
    // would stretch the band and the ellipse with the picture's shape, so a
    // tilt-shift set at 15 degrees would come out at some other angle on a
    // panorama than on a square.
    //
    // Keep this file's GLSL pure ASCII: a shader that fails to compile takes
    // every adjustment down with it, quietly.
    vec2 p = v_uv - u_focusCentre;
    p.x *= u_frameAspect;

    if (u_focusMode < 2.5) {
      // Distance from the centre line of a band, measured across it.
      float c = cos(u_focusAngle);
      float s = sin(u_focusAngle);
      float across = abs(p.x * s + p.y * c);
      outside = smoothstep(u_focusSize, u_focusSize + max(0.002, u_focusFeather), across);
    } else {
      p.x /= max(0.05, u_focusAspect);
      outside = smoothstep(u_focusSize, u_focusSize + max(0.002, u_focusFeather), length(p));
    }
  }

  if (u_focusInvert > 0.5) outside = 1.0 - outside;
  return clamp(outside, 0.0, 1.0) * u_focusWeight;
}

void main() {
  vec4 texel = texture(u_image, v_uv);
  vec4 blurred = texture(u_blurred, v_uv);

  // --- Chromatic aberration, before anything else -------------------------
  // The fringing belongs to the lens, so it happens to the light on its way in
  // rather than to the finished grade. Red is pulled towards the centre and
  // blue away from it, which is the direction real glass disperses them.
  vec3 base = texel.rgb;
  if (u_aberration > 0.001) {
    vec2 outward = v_uv - 0.5;
    float shift = u_aberration * 0.012;
    base = vec3(
      texture(u_image, v_uv - outward * shift).r,
      base.g,
      texture(u_image, v_uv + outward * shift).b
    );
  }

  // Blur mixes towards the blurred copy so the slider reads as an amount. The
  // focus effect adds to it per pixel: outside the sharp zone, more of the
  // blurred copy shows through.
  float blurMix = clamp(u_blurAmount + focusBlend(), 0.0, 1.0);
  vec3 colour = mix(base, blurred.rgb, blurMix);
  float alpha = texel.a;

  // --- Exposure, in linear light where doubling really is one stop --------
  if (abs(u_exposure) > 0.001) {
    colour = toSrgb(toLinear(colour) * pow(2.0, u_exposure));
  }

  // --- Tone regions -------------------------------------------------------
  float lum = luminance(colour);

  if (abs(u_highlights) > 0.001) {
    // Weight peaks in the bright end and fades to nothing by mid grey.
    float w = smoothstep(0.5, 1.0, lum);
    colour = clamp(colour + u_highlights * w * 0.5, 0.0, 1.0);
  }
  if (abs(u_shadows) > 0.001) {
    float w = 1.0 - smoothstep(0.0, 0.5, lum);
    colour = clamp(colour + u_shadows * w * 0.5, 0.0, 1.0);
  }
  if (abs(u_whites) > 0.001) {
    float w = smoothstep(0.6, 1.0, lum);
    colour = clamp(colour + u_whites * w * 0.35, 0.0, 1.0);
  }
  if (abs(u_blacks) > 0.001) {
    float w = 1.0 - smoothstep(0.0, 0.4, lum);
    colour = clamp(colour + u_blacks * w * 0.35, 0.0, 1.0);
  }

  // --- Contrast: S-curve pivoting on mid grey -----------------------------
  if (abs(u_contrast) > 0.001) {
    float amount = 1.0 + u_contrast;
    colour = clamp((colour - 0.5) * amount + 0.5, 0.0, 1.0);
  }

  // --- White balance ------------------------------------------------------
  if (abs(u_temperature) > 0.001 || abs(u_tint) > 0.001) {
    colour.r = clamp(colour.r + u_temperature * 0.25, 0.0, 1.0);
    colour.b = clamp(colour.b - u_temperature * 0.25, 0.0, 1.0);
    colour.g = clamp(colour.g + u_tint * 0.2, 0.0, 1.0);
    colour.r = clamp(colour.r - u_tint * 0.06, 0.0, 1.0);
    colour.b = clamp(colour.b - u_tint * 0.06, 0.0, 1.0);
  }

  // --- Clarity: local contrast on the luminance channel -------------------
  if (abs(u_clarity) > 0.001) {
    float detail = luminance(colour) - luminance(blurred.rgb);
    colour = clamp(colour + detail * u_clarity * 2.0, 0.0, 1.0);
  }

  // --- Dehaze: a dark-channel estimate of the haze, lifted or added --------
  // Haze is the light that every channel shares over a wide area, so the
  // darkest channel of the blurred copy is a fair map of how much of it lies
  // over each point. Positive values take that veil off and stretch what is
  // left; negative ones lay a pale veil on instead.
  if (abs(u_dehaze) > 0.001) {
    float haze = min(blurred.r, min(blurred.g, blurred.b));
    if (u_dehaze > 0.0) {
      float veil = haze * u_dehaze * 0.7;
      colour = clamp((colour - veil) / max(1.0 - veil, 0.2), 0.0, 1.0);
    } else {
      colour = mix(colour, vec3(0.82), -u_dehaze * 0.45 * (0.4 + 0.6 * haze));
    }
  }

  // --- Sharpness: 3x3 unsharp mask ---------------------------------------
  if (u_sharpness > 0.001) {
    vec3 neighbours =
      texture(u_image, v_uv + vec2(u_texelSize.x, 0.0)).rgb +
      texture(u_image, v_uv - vec2(u_texelSize.x, 0.0)).rgb +
      texture(u_image, v_uv + vec2(0.0, u_texelSize.y)).rgb +
      texture(u_image, v_uv - vec2(0.0, u_texelSize.y)).rgb;
    vec3 edge = texel.rgb - neighbours * 0.25;
    colour = clamp(colour + edge * u_sharpness * 1.5, 0.0, 1.0);
  }

  // --- Saturation and vibrance -------------------------------------------
  if (abs(u_vibrance) > 0.001) {
    // Vibrance protects colours that are already saturated, and skin tones.
    float mx = max(colour.r, max(colour.g, colour.b));
    float mn = min(colour.r, min(colour.g, colour.b));
    float sat = mx - mn;
    float weight = 1.0 - sat;
    float grey = luminance(colour);
    colour = clamp(mix(vec3(grey), colour, 1.0 + u_vibrance * weight), 0.0, 1.0);
  }
  if (abs(u_saturation) > 0.001) {
    float grey = luminance(colour);
    colour = clamp(mix(vec3(grey), colour, 1.0 + u_saturation), 0.0, 1.0);
  }

  colour = applyHue(colour, u_hue);

  // --- Colour mixer and grading ------------------------------------------
  // After the global colour, so a mixer band acts on the colours the user is
  // looking at; before the curves, which finish the picture.
  if (u_hasMix > 0.5) colour = applyMixer(colour);
  // Black and white comes after the mixer and before the grade: the mixer's
  // luminance is the black-and-white mix, and a grade can still tone the
  // result into sepia or selenium.
  if (u_mono > 0.001) colour = mix(colour, vec3(luminance(colour)), u_mono);
  if (u_hasGrade > 0.5) colour = applyGrade(colour);

  // --- Tone curves --------------------------------------------------------
  // Last of the colour stages, so the curve grades the finished tones rather
  // than a half-corrected image: the order every photo editor uses.
  if (u_hasLut > 0.5) {
    colour = vec3(
      texture(u_lut, vec2(colour.r, 0.5)).r,
      texture(u_lut, vec2(colour.g, 0.5)).g,
      texture(u_lut, vec2(colour.b, 0.5)).b
    );
  }

  // --- 3D LUT -------------------------------------------------------------
  // After the curves: a table is usually a finished grade, made to be the
  // last colour step. Coordinates are moved half a cell in so the first and
  // last entries land exactly on black and white.
  if (u_lut3dAmount > 0.001) {
    float n = u_lut3dSize;
    vec3 coord = clamp(colour, 0.0, 1.0) * ((n - 1.0) / n) + 0.5 / n;
    colour = mix(colour, texture(u_lut3d, coord).rgb, u_lut3dAmount);
  }

  // --- Bloom --------------------------------------------------------------
  // Only what is genuinely bright is allowed to glow, and it is added rather
  // than mixed: light spilling past an edge makes the picture brighter there,
  // it does not replace what was underneath.
  if (u_bloom > 0.001) {
    vec3 spill = max(texture(u_bloomTex, v_uv).rgb - BLOOM_FLOOR, vec3(0.0)) / (1.0 - BLOOM_FLOOR);
    colour = clamp(colour + spill * u_bloom, 0.0, 1.0);
  }

  // --- Vignette -----------------------------------------------------------
  if (abs(u_vignette) > 0.001) {
    vec2 centred = v_uv - 0.5;
    float dist = length(centred) * 1.41421;
    float falloff = smoothstep(0.3, 1.0, dist);
    colour = clamp(colour - falloff * u_vignette, 0.0, 1.0);
  }

  // --- Grain --------------------------------------------------------------
  if (u_grain > 0.001) {
    float noise = hash(v_uv * 1024.0 + u_seed) - 0.5;
    colour = clamp(colour + noise * u_grain * 0.35, 0.0, 1.0);
  }

  outColour = vec4(colour, alpha);
}`;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Could not create shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compilation failed: ${log}`);
  }
  return shader;
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error('Could not create program');
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link failed: ${log}`);
  }
  return program;
}

/** Normalises slider values into the units the shader expects. */
function uniformsFor(a: Adjustments) {
  return {
    // Exposure is in stops, so the slider maps to roughly +/- 2 stops.
    u_exposure: (a.exposure / 100) * 2,
    u_contrast: a.contrast / 100,
    u_highlights: a.highlights / 100,
    u_shadows: a.shadows / 100,
    u_whites: a.whites / 100,
    u_blacks: a.blacks / 100,
    u_temperature: a.temperature / 100,
    u_tint: a.tint / 100,
    u_vibrance: a.vibrance / 100,
    u_saturation: a.saturation / 100,
    u_hue: a.hue,
    u_clarity: a.clarity / 100,
    u_dehaze: (a.dehaze ?? 0) / 100,
    u_mono: Math.max(0, a.mono ?? 0) / 100,
    u_sharpness: Math.max(0, a.sharpness) / 100,
    u_blurAmount: Math.min(1, a.blur / 20),
    u_vignette: a.vignette / 100,
    u_grain: Math.max(0, a.grain) / 100,
    u_bloom: Math.max(0, a.bloom) / 100,
    u_aberration: Math.max(0, a.aberration) / 100,
  };
}

/** The mixer bands, in the order the shader's arrays hold them. */
export const MIX_BANDS = [
  'Red',
  'Orange',
  'Yellow',
  'Green',
  'Aqua',
  'Blue',
  'Purple',
  'Magenta',
] as const;

/** Largest hue rotation a mixer band can apply, in degrees. */
const MIX_MAX_SHIFT = 30;

/**
 * A grading tint as an RGB offset: the hue at full strength, with its own
 * luminance removed so adding it shifts colour but not brightness.
 */
function gradeTint(hue: number, saturation: number): Float32Array {
  const h = ((((hue ?? 0) % 360) + 360) % 360) / 60;
  const x = 1 - Math.abs((h % 2) - 1);
  const [r, g, b] =
    h < 1 ? [1, x, 0] : h < 2 ? [x, 1, 0] : h < 3 ? [0, 1, x] : h < 4 ? [0, x, 1] : h < 5 ? [x, 0, 1] : [1, 0, x];
  const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const amount = Math.max(0, saturation ?? 0) / 100;
  return new Float32Array([(r - luma) * amount, (g - luma) * amount, (b - luma) * amount]);
}

/** Mixer and grading values in the shapes the shader takes them. */
export function colourUniforms(a: Adjustments) {
  const mixHue = new Float32Array(8);
  const mixSat = new Float32Array(8);
  const mixLum = new Float32Array(8);
  MIX_BANDS.forEach((band, i) => {
    mixHue[i] = ((a[`mix${band}Hue`] ?? 0) / 100) * MIX_MAX_SHIFT;
    mixSat[i] = (a[`mix${band}Sat`] ?? 0) / 100;
    mixLum[i] = (a[`mix${band}Lum`] ?? 0) / 100;
  });
  const hasMix = [...mixHue, ...mixSat, ...mixLum].some((value) => value !== 0);
  const hasGrade =
    (a.gradeShadowSat ?? 0) > 0 || (a.gradeMidSat ?? 0) > 0 || (a.gradeHighlightSat ?? 0) > 0;
  return {
    mixHue,
    mixSat,
    mixLum,
    hasMix,
    shadow: gradeTint(a.gradeShadowHue, a.gradeShadowSat),
    mid: gradeTint(a.gradeMidHue, a.gradeMidSat),
    highlight: gradeTint(a.gradeHighlightHue, a.gradeHighlightSat),
    balance: (a.gradeBalance ?? 0) / 100,
    hasGrade,
  };
}

/** Radius, in pixels, of the blurred reference clarity compares against. */
const CLARITY_RADIUS = 12;

/** Radius of the area haze is estimated over: wide, since haze is. */
const DEHAZE_RADIUS = 24;

/** Spread of the bloom, in thousandths of the longest side. */
const BLOOM_RADIUS = 14;

/**
 * Maps the focus settings onto their uniforms.
 *
 * The blur strength is separated from the geometry: the shader mixes towards a
 * copy that has already been blurred by the right radius, so this only has to
 * say *how much* of it shows at full effect.
 */
function focusUniforms(focus?: FocusEffect) {
  if (!focus) {
    return {
      u_focusMode: 0,
      u_focusWeight: 0,
      u_focusFeather: 0,
      u_focusInvert: 0,
      u_focusSize: 0,
      u_focusAspect: 1,
      u_focusAngle: 0,
      u_focusPlane: 0.5,
    };
  }

  const modes: Record<FocusEffect['mode'], number> = {
    off: 0,
    mask: 1,
    linear: 2,
    radial: 3,
    depth: 4,
  };

  return {
    u_focusMode: modes[focus.mode],
    // A soft zone always takes the blurred copy in full; the radius is what
    // makes it gentle or extreme, which is how a real aperture behaves.
    u_focusWeight: 1,
    u_focusFeather: Math.max(0.002, focus.feather),
    u_focusInvert: focus.invert ? 1 : 0,
    u_focusSize: Math.max(0, focus.size),
    u_focusAspect: Math.max(0.05, focus.aspect),
    u_focusAngle: (focus.angle * Math.PI) / 180,
    u_focusPlane: Math.min(1, Math.max(0, focus.plane)),
  };
}

/**
 * Applies an adjustment stack to an image and returns the result as a canvas
 * that can be drawn straight into a 2D context.
 *
 * A single instance is reused for the whole application: creating a WebGL
 * context per image would exhaust the browser's context limit within a handful
 * of layers.
 */
export class AdjustmentProcessor {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext | null;
  private blurProgram: WebGLProgram | null = null;
  private adjustProgram: WebGLProgram | null = null;
  private quad: WebGLBuffer | null = null;
  /**
   * The last blurred copy, kept alive between calls.
   *
   * One slot, because the case worth optimising is a repeated call about the
   * same image — a slider or a guide being dragged. Two images alternating
   * simply miss, which is what happened every time before.
   */
  private blurCache: { key: string; texture: WebGLTexture } | null = null;
  /** The last source image uploaded, kept on the GPU for the same reason. */
  private sourceCache: { key: string; texture: WebGLTexture } | null = null;
  private failed = false;

  constructor() {
    this.canvas = window.document.createElement('canvas');
    this.gl = this.canvas.getContext('webgl2', {
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });

    if (!this.gl) {
      // Without WebGL2 the adjustments are simply skipped: the editor still
      // works, it just cannot alter tone or colour.
      console.warn('[Nabu] WebGL2 unavailable; adjustments are disabled.');
      this.failed = true;
      return;
    }

    try {
      this.blurProgram = link(this.gl, VERTEX_SHADER, BLUR_SHADER);
      this.adjustProgram = link(this.gl, VERTEX_SHADER, ADJUST_SHADER);

      this.quad = this.gl.createBuffer();
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quad);
      this.gl.bufferData(
        this.gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 3, -1, -1, 3]),
        this.gl.STATIC_DRAW
      );
    } catch (err) {
      console.warn('[Nabu] Adjustment shaders failed to build:', err);
      this.failed = true;
    }
  }

  public get available(): boolean {
    return !this.failed;
  }

  /**
   * Uploads the curve table.
   *
   * Linear filtering rather than nearest: the table has one entry per 8-bit
   * value, and interpolating between two of them costs nothing while keeping a
   * steep curve free of visible banding.
   */
  private makeLutTexture(curves: Curves): WebGLTexture {
    const gl = this.gl!;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8,
      LUT_SIZE,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      buildCurveLut(curves)
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  private makeTexture(width: number, height: number): WebGLTexture {
    const gl = this.gl!;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  private drawQuad(program: WebGLProgram): void {
    const gl = this.gl!;
    const location = gl.getAttribLocation(program, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Runs the separable blur over `source`, returning a new texture. */
  private blur(
    source: WebGLTexture,
    width: number,
    height: number,
    radius: number,
    framebuffer: WebGLFramebuffer
  ): WebGLTexture {
    const gl = this.gl!;
    const program = this.blurProgram!;
    gl.useProgram(program);
    gl.uniform2f(gl.getUniformLocation(program, 'u_texelSize'), 1 / width, 1 / height);
    gl.uniform1f(gl.getUniformLocation(program, 'u_radius'), radius);
    gl.uniform1i(gl.getUniformLocation(program, 'u_image'), 0);

    // Two explicit passes: source -> horizontal -> vertical. Written out rather
    // than looped with a swap, because getting the ping-pong wrong silently
    // returns the half-blurred intermediate instead of the finished one.
    const renderPass = (
      from: WebGLTexture,
      into: WebGLTexture,
      dx: number,
      dy: number
    ): void => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, into, 0);
      gl.viewport(0, 0, width, height);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, from);
      gl.uniform2f(gl.getUniformLocation(program, 'u_direction'), dx, dy);
      this.drawQuad(program);
    };

    const horizontal = this.makeTexture(width, height);
    const vertical = this.makeTexture(width, height);

    renderPass(source, horizontal, 1, 0);
    renderPass(horizontal, vertical, 0, 1);

    gl.deleteTexture(horizontal);
    return vertical;
  }

  /**
   * Returns a canvas holding `image` with `adjustments` applied, or null when
   * nothing needs doing so the caller can draw the original directly.
   */
  public process(
    image: TexImageSource,
    width: number,
    height: number,
    adjustments: Adjustments,
    curves?: Curves,
    focus?: FocusEffect,
    /** Decoded focus mask; the caller owns the decoding so this stays sync. */
    focusMask?: TexImageSource | null,
    /**
     * Identifies the source pixels, so the blurred copy can be reused when
     * only the settings changed. Omit and nothing is cached.
     */
    cacheKey?: string,
    /** A colour lookup table, as shaderLut() returns it. */
    lut?: { size: number; rgb: Uint8Array; amount: number; key: string },
    /**
     * Size of `image` relative to the photo it stands for. A preview runs on a
     * reduced copy, and radii given in photo pixels (blur, clarity, dehaze)
     * shrink with it, or the preview would look blurrier than the result.
     */
    pixelScale = 1
  ): HTMLCanvasElement | null {
    const flatCurves = areCurvesNeutral(curves);
    // A mask that has not finished decoding yet is treated as no focus at all:
    // showing the whole picture blurred for one frame would look like a bug.
    const activeFocus =
      focus && !isFocusNeutral(focus) && (focus.mode !== 'mask' || focusMask)
        ? focus
        : undefined;

    if (this.failed || (isNeutral(adjustments) && flatCurves && !activeFocus && !lut)) return null;

    const gl = this.gl!;
    // Assigning a canvas size reallocates its buffer even when it is the same.
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;

    const framebuffer = gl.createFramebuffer()!;
    const created: WebGLTexture[] = [];

    try {
      // Upload the source image, unless it is already on the GPU. A slider
      // drag changes only the settings: sending tens of megabytes of the same
      // pixels to the graphics card on every step was most of its cost.
      const sourceKey = cacheKey ? `${cacheKey}|${width}x${height}` : null;
      let sourceTexture: WebGLTexture;
      if (sourceKey && this.sourceCache?.key === sourceKey) {
        sourceTexture = this.sourceCache.texture;
      } else {
        sourceTexture = this.makeTexture(width, height);
        gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
        if (sourceKey) {
          if (this.sourceCache) gl.deleteTexture(this.sourceCache.texture);
          this.sourceCache = { key: sourceKey, texture: sourceTexture };
        } else {
          created.push(sourceTexture);
        }
      }

      // One blurred copy serves the blur slider, clarity's reference and the
      // focus effect, so the widest request wins. Scaling the focus radius by
      // the image size keeps a tilt-shift looking the same on a thumbnail as
      // on the full picture — otherwise every preview would lie.
      const focusRadius = activeFocus
        ? (activeFocus.amount / 1000) * Math.max(width, height)
        : 0;
      const needsClarity = adjustments.clarity !== 0;
      const needsDehaze = Boolean(adjustments.dehaze);
      const radius = Math.max(
        adjustments.blur * pixelScale,
        focusRadius,
        needsClarity ? CLARITY_RADIUS * pixelScale : 0,
        needsDehaze ? DEHAZE_RADIUS * pixelScale : 0
      );

      // The blurred copy is the expensive part of the pass — two separable
      // passes of up to twenty-four taps each — and it depends on nothing but
      // the image and the radius. Dragging a tilt-shift across a photograph
      // changes only *where* the band is, so rebuilding the blur on every frame
      // was paying the whole cost of the effect to move a line.
      let blurredTexture = sourceTexture;
      if (radius > 0) {
        const key = `${cacheKey ?? ''}|${width}x${height}|${radius.toFixed(2)}`;
        if (this.blurCache && this.blurCache.key === key) {
          blurredTexture = this.blurCache.texture;
        } else {
          if (this.blurCache) gl.deleteTexture(this.blurCache.texture);
          this.blurCache = null;
          const built = this.blur(sourceTexture, width, height, radius, framebuffer);
          // Deliberately *not* pushed onto `created`: it outlives this call.
          // Without a cache key there is nothing to match on later, so it is
          // freed here like anything else.
          if (cacheKey) this.blurCache = { key, texture: built };
          else created.push(built);
          blurredTexture = built;
        }
      }

      // Bloom gets a copy of its own: sharing the blur above would make the
      // glow as wide as whatever a tilt-shift happened to ask for.
      let bloomTexture = blurredTexture;
      if (adjustments.bloom > 0) {
        bloomTexture = this.blur(
          sourceTexture,
          width,
          height,
          (BLOOM_RADIUS / 1000) * Math.max(width, height),
          framebuffer
        );
        created.push(bloomTexture);
      }

      // Final pass renders to the canvas itself.
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);

      const program = this.adjustProgram!;
      gl.useProgram(program);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
      gl.uniform1i(gl.getUniformLocation(program, 'u_image'), 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, blurredTexture);
      gl.uniform1i(gl.getUniformLocation(program, 'u_blurred'), 1);

      // The unit is selected *before* the table is built: creating a texture
      // binds it to whichever unit is active, and doing that afterwards would
      // quietly replace the blurred copy on unit 1.
      //
      // A sampler left unbound reads as texture unit 0 on some drivers, which
      // would run the photo itself through the curve lookup: bind a table
      // either way and let the flag decide whether it is used.
      gl.activeTexture(gl.TEXTURE2);
      const lutTexture = this.makeLutTexture(flatCurves ? NEUTRAL_CURVES : curves!);
      created.push(lutTexture);
      gl.uniform1i(gl.getUniformLocation(program, 'u_lut'), 2);
      gl.uniform1f(gl.getUniformLocation(program, 'u_hasLut'), flatCurves ? 0 : 1);

      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, bloomTexture);
      gl.uniform1i(gl.getUniformLocation(program, 'u_bloomTex'), 3);

      // The mask is uploaded with the same vertical flip as the photo, so a
      // white pixel in the mask lands on the object it was cut from.
      gl.activeTexture(gl.TEXTURE4);
      const maskTexture = this.makeTexture(1, 1);
      created.push(maskTexture);
      gl.bindTexture(gl.TEXTURE_2D, maskTexture);
      if (focusMask) {
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, focusMask);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
      } else {
        // Opaque white: no mask means nothing is ever pushed out of focus.
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA8,
          1,
          1,
          0,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          new Uint8Array([255, 255, 255, 255])
        );
      }
      gl.uniform1i(gl.getUniformLocation(program, 'u_focusMask'), 4);

      // Unit 5 always holds a 3D texture, a 1x1x1 one when there is no table:
      // a sampler3D left without one is an error on some drivers.
      gl.activeTexture(gl.TEXTURE5);
      const cube = gl.createTexture()!;
      created.push(cube);
      gl.bindTexture(gl.TEXTURE_3D, cube);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      if (lut) {
        gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB8, lut.size, lut.size, lut.size, 0, gl.RGB, gl.UNSIGNED_BYTE, lut.rgb);
      } else {
        gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB8, 1, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array(3));
      }
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
      gl.uniform1i(gl.getUniformLocation(program, 'u_lut3d'), 5);
      gl.uniform1f(gl.getUniformLocation(program, 'u_lut3dAmount'), lut ? lut.amount : 0);
      gl.uniform1f(gl.getUniformLocation(program, 'u_lut3dSize'), lut ? lut.size : 1);

      for (const [name, value] of Object.entries(focusUniforms(activeFocus))) {
        gl.uniform1f(gl.getUniformLocation(program, name), value);
      }
      gl.uniform2f(
        gl.getUniformLocation(program, 'u_focusCentre'),
        activeFocus?.centreX ?? 0.5,
        // Texture space runs bottom-up; the interface talks top-down.
        1 - (activeFocus?.centreY ?? 0.5)
      );
      gl.uniform1f(
        gl.getUniformLocation(program, 'u_frameAspect'),
        height > 0 ? width / height : 1
      );

      gl.uniform2f(gl.getUniformLocation(program, 'u_texelSize'), 1 / width, 1 / height);
      gl.uniform1f(gl.getUniformLocation(program, 'u_seed'), 0.0);

      for (const [name, value] of Object.entries(uniformsFor(adjustments))) {
        gl.uniform1f(gl.getUniformLocation(program, name), value);
      }

      const colour = colourUniforms(adjustments);
      gl.uniform1fv(gl.getUniformLocation(program, 'u_mixHue'), colour.mixHue);
      gl.uniform1fv(gl.getUniformLocation(program, 'u_mixSat'), colour.mixSat);
      gl.uniform1fv(gl.getUniformLocation(program, 'u_mixLum'), colour.mixLum);
      gl.uniform1f(gl.getUniformLocation(program, 'u_hasMix'), colour.hasMix ? 1 : 0);
      gl.uniform3fv(gl.getUniformLocation(program, 'u_gradeShadow'), colour.shadow);
      gl.uniform3fv(gl.getUniformLocation(program, 'u_gradeMid'), colour.mid);
      gl.uniform3fv(gl.getUniformLocation(program, 'u_gradeHighlight'), colour.highlight);
      gl.uniform1f(gl.getUniformLocation(program, 'u_gradeBalance'), colour.balance);
      gl.uniform1f(gl.getUniformLocation(program, 'u_hasGrade'), colour.hasGrade ? 1 : 0);

      this.drawQuad(program);

      // Copy out, because the shared canvas is reused by the next call.
      const output = window.document.createElement('canvas');
      output.width = width;
      output.height = height;
      output.getContext('2d')?.drawImage(this.canvas, 0, 0);
      return output;
    } catch (err) {
      console.warn('[Nabu] Adjustment pass failed:', err);
      return null;
    } finally {
      for (const texture of created) gl.deleteTexture(texture);
      gl.deleteFramebuffer(framebuffer);
    }
  }
}

let shared: AdjustmentProcessor | null = null;

/**
 * The one processor everything goes through.
 *
 * A browser allows only a handful of live WebGL contexts, and the filter panel
 * alone would want one per thumbnail. Since `process` is synchronous and hands
 * back a copy of its result, a single context serves the viewport, the export
 * and every preview without them ever seeing each other's pixels.
 */
export function sharedProcessor(): AdjustmentProcessor {
  if (!shared) shared = new AdjustmentProcessor();
  return shared;
}

/** Stable key for caching a processed result. */
export function adjustmentKey(adjustments?: Adjustments): string {
  if (!adjustments) return '';
  return (Object.keys(adjustments) as Array<keyof Adjustments>)
    .sort()
    .map((k) => `${k}:${adjustments[k]}`)
    .join('|');
}
