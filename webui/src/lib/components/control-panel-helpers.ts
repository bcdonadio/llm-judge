import type { ModelInfo } from "@/lib/types";

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

export function displayModelName(model: ModelInfo): string {
  return model.name ?? model.id;
}

export function deriveJudgeOptionsRendered(
  options: ModelInfo[],
  fallbackId: string,
): ModelInfo[] {
  return options.length > 0 ? options : [{ id: fallbackId, name: fallbackId }];
}

export function coerceJudgeModelId(
  judgeModel: string | null | undefined,
): string {
  return judgeModel ?? "";
}

export function modelKey(model: ModelInfo, index: number): string {
  return model.id || `model-${index}`;
}

export function judgeOptionKey(option: ModelInfo, index: number): string {
  if (option.id) {
    return `judge-${option.id}`;
  }
  if (option.name) {
    return `judge-name-${option.name}`;
  }
  return `judge-fallback-${index}`;
}

export function modelCheckboxValue(model: ModelInfo): string {
  return model.id || displayModelName(model);
}
