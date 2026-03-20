/**
 * Formatting utilities for scientific values.
 */

/**
 * Format a number in scientific notation when appropriate.
 */
export function formatScientific(value: number, precision: number = 3): string {
  if (value === 0) return '0';
  const absVal = Math.abs(value);
  if (absVal >= 0.01 && absVal < 10000) {
    return value.toPrecision(precision);
  }
  return value.toExponential(precision - 1);
}

/**
 * Format a time value to human-readable string.
 */
export function formatTime(seconds: number): string {
  if (seconds < 1e-3) {
    return `${(seconds * 1e6).toFixed(1)} us`;
  }
  if (seconds < 1) {
    return `${(seconds * 1e3).toFixed(1)} ms`;
  }
  if (seconds < 60) {
    return `${seconds.toFixed(2)} s`;
  }
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${minutes}m ${secs.toFixed(1)}s`;
}

/**
 * Format a length value with appropriate unit prefix.
 */
export function formatLength(meters: number): string {
  if (meters < 1e-6) {
    return `${(meters * 1e9).toFixed(1)} nm`;
  }
  if (meters < 1e-3) {
    return `${(meters * 1e6).toFixed(1)} um`;
  }
  if (meters < 1) {
    return `${(meters * 1e3).toFixed(1)} mm`;
  }
  return `${meters.toFixed(3)} m`;
}

/**
 * Generate a unique ID for log entries, etc.
 */
export function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
}

/**
 * Clamp a number to a given range.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Map a value from one range to another.
 */
export function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number,
): number {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

/**
 * Linearly interpolate between two colors (hex strings).
 */
export function lerpColor(color1: string, color2: string, t: number): string {
  const c1 = parseInt(color1.slice(1), 16);
  const c2 = parseInt(color2.slice(1), 16);
  const r1 = (c1 >> 16) & 0xff;
  const g1 = (c1 >> 8) & 0xff;
  const b1 = c1 & 0xff;
  const r2 = (c2 >> 16) & 0xff;
  const g2 = (c2 >> 8) & 0xff;
  const b2 = c2 & 0xff;
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
}

/**
 * Generate a diverging color (blue-white-red) for voltage heatmaps.
 */
export function voltageToColor(vmem: number, vmin: number, vmax: number): string {
  const t = clamp((vmem - vmin) / (vmax - vmin), 0, 1);
  if (t < 0.5) {
    // Blue (#2166ac) to White (#f7f7f7)
    return lerpColor('#2166ac', '#f7f7f7', t * 2);
  }
  // White (#f7f7f7) to Red (#b2182b)
  return lerpColor('#f7f7f7', '#b2182b', (t - 0.5) * 2);
}

/**
 * Generate a sequential color (viridis-like) for concentration heatmaps.
 */
export function concentrationToColor(value: number, vmin: number, vmax: number): string {
  const t = clamp((value - vmin) / (vmax - vmin), 0, 1);
  // Simplified viridis: dark purple -> teal -> yellow
  if (t < 0.5) {
    return lerpColor('#440154', '#21918c', t * 2);
  }
  return lerpColor('#21918c', '#fde725', (t - 0.5) * 2);
}
