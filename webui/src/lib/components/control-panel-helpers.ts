export function parseModels(raw: string): string[] {
  return raw
    .split(/[\s,]+/)
    .map((value) => value.trim())
    .filter(Boolean);
}

export function parseLimit(raw: string | number): number | null {
  const str = String(raw).trim();
  if (!str) {
    return null;
  }
  const parsed = Number.parseInt(str, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

export function formatLimit(limit: number | null | undefined): string {
  return limit == null ? "" : String(limit);
}
