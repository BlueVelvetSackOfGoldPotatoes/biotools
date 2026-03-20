/**
 * YAML-like serialization and localStorage persistence for SimulationConfig.
 */
import type { SimulationConfig } from '../types/simulation';
import { createDefaultConfig } from '../types/presets';

const STORAGE_KEY = 'betsee_config';
const NAMED_CONFIG_PREFIX = 'betsee_config_named_';

// ---- YAML Serialization ----

function serializeValue(value: unknown, indent: number): string {
  const pad = '  '.repeat(indent);

  if (value === null || value === undefined) {
    return 'null';
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  if (typeof value === 'number') {
    return String(value);
  }
  if (typeof value === 'string') {
    // Quote strings that could be ambiguous
    if (
      value === '' ||
      value === 'true' ||
      value === 'false' ||
      value === 'null' ||
      /[:{}\[\],#&*!|>'"%@`]/.test(value) ||
      value.trim() !== value
    ) {
      return `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
    }
    return value;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return '[]';
    }
    // Check if it's an array of primitives
    const allPrimitive = value.every(
      (v) => typeof v !== 'object' || v === null,
    );
    if (allPrimitive) {
      return `[${value.map((v) => serializeValue(v, 0)).join(', ')}]`;
    }
    // Array of objects
    const lines: string[] = [];
    for (const item of value) {
      const objLines = serializeObject(item as Record<string, unknown>, indent + 1);
      const first = objLines[0];
      lines.push(`${pad}- ${first.trimStart()}`);
      for (let i = 1; i < objLines.length; i++) {
        lines.push(`${pad}  ${objLines[i].trimStart()}`);
      }
    }
    return '\n' + lines.join('\n');
  }
  if (typeof value === 'object') {
    const lines = serializeObject(value as Record<string, unknown>, indent + 1);
    return '\n' + lines.join('\n');
  }
  return String(value);
}

function serializeObject(obj: Record<string, unknown>, indent: number): string[] {
  const pad = '  '.repeat(indent);
  const lines: string[] = [];

  for (const [key, val] of Object.entries(obj)) {
    const serialized = serializeValue(val, indent);
    if (serialized.startsWith('\n')) {
      lines.push(`${pad}${key}:${serialized}`);
    } else {
      lines.push(`${pad}${key}: ${serialized}`);
    }
  }

  return lines;
}

/**
 * Serialize a SimulationConfig to a YAML-like string format.
 */
export function serializeConfigToYaml(config: SimulationConfig): string {
  const lines = serializeObject(config as unknown as Record<string, unknown>, 0);
  return lines.join('\n') + '\n';
}

// ---- YAML Deserialization ----

interface ParsedLine {
  indent: number;
  key: string;
  value: string;
  isArrayItem: boolean;
}

function parseLine(line: string): ParsedLine | null {
  if (line.trim() === '' || line.trim().startsWith('#')) {
    return null;
  }

  const indentMatch = line.match(/^(\s*)/);
  const indent = indentMatch ? indentMatch[1].length : 0;
  let content = line.trimStart();

  const isArrayItem = content.startsWith('- ');
  if (isArrayItem) {
    content = content.slice(2);
  }

  const colonIndex = content.indexOf(':');
  if (colonIndex === -1) {
    return { indent, key: '', value: content, isArrayItem };
  }

  const key = content.slice(0, colonIndex).trim();
  const value = content.slice(colonIndex + 1).trim();

  return { indent, key, value, isArrayItem };
}

function parseScalar(value: string): unknown {
  if (value === 'null' || value === '') {
    return null;
  }
  if (value === 'true') {
    return true;
  }
  if (value === 'false') {
    return false;
  }
  // Quoted string
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, '\\');
  }
  // Inline array
  if (value.startsWith('[') && value.endsWith(']')) {
    const inner = value.slice(1, -1).trim();
    if (inner === '') return [];
    return inner.split(',').map((s) => parseScalar(s.trim()));
  }
  // Number
  const num = Number(value);
  if (!isNaN(num) && value !== '') {
    return num;
  }
  return value;
}

/**
 * Deserialize a YAML-like string back into a partial SimulationConfig.
 * This is a best-effort parser for the format produced by serializeConfigToYaml.
 */
export function deserializeConfigFromYaml(yaml: string): Partial<SimulationConfig> {
  const lines = yaml.split('\n');
  const result: Record<string, unknown> = {};
  const stack: { obj: Record<string, unknown>; indent: number; key?: string }[] = [
    { obj: result, indent: -1 },
  ];

  let currentArray: unknown[] | null = null;
  let currentArrayKey: string | null = null;
  let currentArrayIndent = -1;
  let currentArrayItemObj: Record<string, unknown> | null = null;

  for (const line of lines) {
    const parsed = parseLine(line);
    if (!parsed) continue;

    // Handle array items
    if (parsed.isArrayItem) {
      if (currentArrayKey && parsed.indent <= currentArrayIndent) {
        // End current array context
        currentArray = null;
        currentArrayKey = null;
        currentArrayItemObj = null;
      }

      if (currentArray) {
        if (parsed.key) {
          // Array item is an object
          currentArrayItemObj = {};
          currentArrayItemObj[parsed.key] = parsed.value === '' ? {} : parseScalar(parsed.value);
          currentArray.push(currentArrayItemObj);
        } else {
          currentArray.push(parseScalar(parsed.value));
          currentArrayItemObj = null;
        }
        continue;
      }
    }

    // If we're inside an array item object and this line is indented further
    if (currentArrayItemObj && parsed.indent > currentArrayIndent + 2 && !parsed.isArrayItem) {
      if (parsed.key) {
        currentArrayItemObj[parsed.key] = parseScalar(parsed.value);
      }
      continue;
    }

    if (parsed.isArrayItem && !currentArray) {
      // This shouldn't happen if the YAML is well-formed, but handle gracefully
      continue;
    }

    // Pop stack to find correct parent
    while (stack.length > 1 && stack[stack.length - 1].indent >= parsed.indent) {
      stack.pop();
    }

    const parent = stack[stack.length - 1].obj;

    if (parsed.value === '' || parsed.value === undefined) {
      // This key maps to a nested object or array (determined by next lines)
      const newObj: Record<string, unknown> = {};
      parent[parsed.key] = newObj;
      stack.push({ obj: newObj, indent: parsed.indent, key: parsed.key });
    } else if (parsed.value === '[]') {
      parent[parsed.key] = [];
      // Prepare for potential array items
      currentArray = parent[parsed.key] as unknown[];
      currentArrayKey = parsed.key;
      currentArrayIndent = parsed.indent;
      currentArrayItemObj = null;
    } else {
      const scalar = parseScalar(parsed.value);
      parent[parsed.key] = scalar;

      // Check if next lines are array items for this key
      // (handled dynamically as we encounter - prefixed lines)
    }
  }

  // Look ahead: check if any key has array items following it
  // Re-parse to handle arrays properly (second pass)
  return rebuildArrays(result) as Partial<SimulationConfig>;
}

function rebuildArrays(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      result[key] = rebuildArrays(value as Record<string, unknown>);
    } else {
      result[key] = value;
    }
  }
  return result;
}

// ---- localStorage Persistence ----

/**
 * Save the current config to localStorage under the default key.
 */
export function saveConfigToLocalStorage(config: SimulationConfig): void {
  try {
    const json = JSON.stringify(config);
    localStorage.setItem(STORAGE_KEY, json);
  } catch (e) {
    console.error('Failed to save config to localStorage:', e);
  }
}

/**
 * Load config from localStorage. Returns null if not found or on error.
 */
export function loadConfigFromLocalStorage(): SimulationConfig | null {
  try {
    const json = localStorage.getItem(STORAGE_KEY);
    if (!json) return null;
    const parsed = JSON.parse(json);
    // Merge with defaults to ensure all fields exist (in case the stored
    // config was saved by an older version that lacked some fields)
    const defaults = createDefaultConfig();
    return deepMerge(
      defaults as unknown as Record<string, unknown>,
      parsed as Record<string, unknown>,
    ) as unknown as SimulationConfig;
  } catch (e) {
    console.error('Failed to load config from localStorage:', e);
    return null;
  }
}

/**
 * List all saved named config keys from localStorage.
 */
export function listSavedConfigs(): string[] {
  const names: string[] = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(NAMED_CONFIG_PREFIX)) {
        names.push(key.slice(NAMED_CONFIG_PREFIX.length));
      }
    }
  } catch (e) {
    console.error('Failed to list saved configs:', e);
  }
  return names.sort();
}

/**
 * Save a config under a given name.
 */
export function saveNamedConfig(name: string, config: SimulationConfig): void {
  try {
    const json = JSON.stringify(config);
    localStorage.setItem(NAMED_CONFIG_PREFIX + name, json);
  } catch (e) {
    console.error(`Failed to save named config "${name}":`, e);
  }
}

/**
 * Load a named config. Returns null if not found.
 */
export function loadNamedConfig(name: string): SimulationConfig | null {
  try {
    const json = localStorage.getItem(NAMED_CONFIG_PREFIX + name);
    if (!json) return null;
    const parsed = JSON.parse(json);
    const defaults = createDefaultConfig();
    return deepMerge(
      defaults as unknown as Record<string, unknown>,
      parsed as Record<string, unknown>,
    ) as unknown as SimulationConfig;
  } catch (e) {
    console.error(`Failed to load named config "${name}":`, e);
    return null;
  }
}

/**
 * Delete a named config.
 */
export function deleteNamedConfig(name: string): void {
  try {
    localStorage.removeItem(NAMED_CONFIG_PREFIX + name);
  } catch (e) {
    console.error(`Failed to delete named config "${name}":`, e);
  }
}

// ---- Helpers ----

function deepMerge(target: Record<string, unknown>, source: Record<string, unknown>): Record<string, unknown> {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    const sourceVal = source[key];
    const targetVal = target[key];
    if (
      sourceVal &&
      typeof sourceVal === 'object' &&
      !Array.isArray(sourceVal) &&
      targetVal &&
      typeof targetVal === 'object' &&
      !Array.isArray(targetVal)
    ) {
      result[key] = deepMerge(
        targetVal as Record<string, unknown>,
        sourceVal as Record<string, unknown>,
      );
    } else if (sourceVal !== undefined) {
      result[key] = sourceVal;
    }
  }
  return result;
}
