export interface RunColor {
  base: string;
  soft: string;
}

const PALETTE: RunColor[] = [
  { base: "#3b5bdb", soft: "#5c77e0" },
  { base: "#c2410c", soft: "#d4602f" },
  { base: "#0f766e", soft: "#2d9089" },
  { base: "#7c3aed", soft: "#9461f0" },
];

const MUTED: RunColor = { base: "#9c9c92", soft: "#b0b0a6" };

export function runColorByIndex(i: number): RunColor {
  return PALETTE[i % PALETTE.length];
}

export function buildRunColorMap(gridExpIds: string[]): Map<string, RunColor> {
  const m = new Map<string, RunColor>();
  gridExpIds.forEach((id, i) => m.set(id, runColorByIndex(i)));
  return m;
}

export function colorFor(map: Map<string, RunColor>, expId: string): RunColor {
  return map.get(expId) ?? MUTED;
}
