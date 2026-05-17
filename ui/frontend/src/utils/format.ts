export function formatNumber(value: number | null | undefined, fractionDigits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: fractionDigits });
  }
  return value.toFixed(fractionDigits);
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let abs = Math.abs(value);
  let unitIdx = 0;
  while (abs >= 1024 && unitIdx < units.length - 1) {
    abs /= 1024;
    unitIdx += 1;
  }
  const sign = value < 0 ? '-' : '';
  return `${sign}${abs.toFixed(unitIdx === 0 ? 0 : 2)} ${units[unitIdx]}`;
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return '—';
  if (Math.abs(ms) < 1) return `${ms.toFixed(3)} ms`;
  if (Math.abs(ms) < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatTimestamp(seconds: number | null | undefined): string {
  if (!seconds) return '—';
  const date = new Date(seconds * 1000);
  return date.toLocaleString();
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
