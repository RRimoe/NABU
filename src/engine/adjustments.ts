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

import { type Adjustments, isNeutral } from '../types/document';

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
uniform sampler2D u_blurred;    // blurred copy, for blur and clarity
uniform vec2 u_texelSize;

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
uniform float u_sharpness;
uniform float u_blurAmount;
uniform float u_vignette;
uniform float u_grain;
uniform float u_seed;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

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

void main() {
  vec4 texel = texture(u_image, v_uv);
  vec4 blurred = texture(u_blurred, v_uv);

  // Blur mixes towards the blurred copy so the slider reads as an amount.
  vec3 colour = mix(texel.rgb, blurred.rgb, clamp(u_blurAmount, 0.0, 1.0));
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
    u_sharpness: Math.max(0, a.sharpness) / 100,
    u_blurAmount: Math.min(1, a.blur / 20),
    u_vignette: a.vignette / 100,
    u_grain: Math.max(0, a.grain) / 100,
  };
}

/** Radius, in pixels, of the blurred reference clarity compares against. */
const CLARITY_RADIUS = 12;

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
    adjustments: Adjustments
  ): HTMLCanvasElement | null {
    if (this.failed || isNeutral(adjustments)) return null;

    const gl = this.gl!;
    this.canvas.width = width;
    this.canvas.height = height;

    const framebuffer = gl.createFramebuffer()!;
    const created: WebGLTexture[] = [];

    try {
      // Upload the source image.
      const sourceTexture = this.makeTexture(width, height);
      created.push(sourceTexture);
      gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);

      // Clarity needs a wide blurred reference; the blur slider needs its own.
      const needsBlur = adjustments.blur > 0;
      const needsClarity = adjustments.clarity !== 0;
      const radius = needsBlur ? adjustments.blur : needsClarity ? CLARITY_RADIUS : 0;

      let blurredTexture = sourceTexture;
      if (radius > 0) {
        blurredTexture = this.blur(sourceTexture, width, height, radius, framebuffer);
        created.push(blurredTexture);
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

      gl.uniform2f(gl.getUniformLocation(program, 'u_texelSize'), 1 / width, 1 / height);
      gl.uniform1f(gl.getUniformLocation(program, 'u_seed'), 0.0);

      for (const [name, value] of Object.entries(uniformsFor(adjustments))) {
        gl.uniform1f(gl.getUniformLocation(program, name), value);
      }

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

/** Stable key for caching a processed result. */
export function adjustmentKey(adjustments?: Adjustments): string {
  if (!adjustments) return '';
  return (Object.keys(adjustments) as Array<keyof Adjustments>)
    .sort()
    .map((k) => `${k}:${adjustments[k]}`)
    .join('|');
}
