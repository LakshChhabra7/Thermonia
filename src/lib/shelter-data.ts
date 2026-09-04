import { fit, predict, type FittedModel } from "./regression";

export const FEATURES = [
  { key: "windSpeed", label: "Peak wind speed", unit: "km/h", min: 0, max: 260, step: 1, demo: 120 },
  { key: "minTemp", label: "Lowest temperature", unit: "°C", min: -70, max: 45, step: 1, demo: -28 },
  { key: "snowLoad", label: "Snow load", unit: "kg/m²", min: 0, max: 600, step: 5, demo: 180 },
  { key: "humidity", label: "Average humidity", unit: "%", min: 0, max: 100, step: 1, demo: 65 },
  { key: "rainfall", label: "Yearly rainfall", unit: "mm", min: 0, max: 4000, step: 10, demo: 700 },
  { key: "quakeRisk", label: "Earthquake risk", unit: "0–5", min: 0, max: 5, step: 0.1, demo: 2 },
  { key: "occupants", label: "People sheltered", unit: "", min: 1, max: 40, step: 1, demo: 6 },
] as const;

export type FeatureKey = (typeof FEATURES)[number]["key"];

export const TARGETS = [
  { key: "wallThickness", label: "Wall thickness", unit: "mm", digits: 0 },
  { key: "insulationR", label: "Insulation value", unit: "m²K/W", digits: 2 },
  { key: "foundationDepth", label: "Foundation depth", unit: "cm", digits: 0 },
  { key: "roofPitch", label: "Roof pitch", unit: "°", digits: 1 },
  { key: "anchorLoad", label: "Anchor strength", unit: "kN", digits: 1 },
] as const;

export type TargetKey = (typeof TARGETS)[number]["key"];

export interface DataRow {
  features: number[];
  targets: number[];
}

/** Deterministic pseudo-random generator so the sample set is identical everywhere. */
function makeRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

/**
 * Reference training set of 240 engineered shelter builds across polar, desert,
 * cyclone and high-altitude sites. Replace it by loading your own CSV in the app.
 */
export function buildSampleDataset(): DataRow[] {
  const rand = makeRandom(20260904);
  const rows: DataRow[] = [];
  const noise = (scale: number) => (rand() - 0.5) * 2 * scale;

  for (let i = 0; i < 240; i++) {
    const windSpeed = 30 + rand() * 220;
    const minTemp = -60 + rand() * 100;
    const snowLoad = minTemp < 5 ? rand() * 550 : rand() * 60;
    const humidity = 10 + rand() * 88;
    const rainfall = 20 + rand() * 3600 * (humidity / 100);
    const quakeRisk = rand() * 5;
    const occupants = 1 + Math.floor(rand() * 30);

    const wallThickness =
      70 + windSpeed * 0.45 + Math.max(0, -minTemp) * 2.6 + snowLoad * 0.08 + occupants * 1.4 + noise(9);
    const insulationR =
      0.9 + Math.max(0, -minTemp) * 0.085 + snowLoad * 0.0016 - Math.max(0, minTemp) * 0.012 + noise(0.12);
    const foundationDepth =
      40 + windSpeed * 0.16 + quakeRisk * 11 + Math.max(0, -minTemp) * 0.9 + rainfall * 0.004 + noise(4);
    const roofPitch = 8 + snowLoad * 0.05 + rainfall * 0.0035 - windSpeed * 0.02 + noise(1.5);
    const anchorLoad =
      2 + windSpeed * 0.135 + quakeRisk * 1.7 + occupants * 0.22 + snowLoad * 0.006 + noise(0.8);

    rows.push({
      features: [windSpeed, minTemp, snowLoad, humidity, rainfall, quakeRisk, occupants],
      targets: [wallThickness, insulationR, foundationDepth, roofPitch, anchorLoad],
    });
  }
  return rows;
}

export function trainModel(rows: DataRow[]): FittedModel {
  return fit(
    rows.map((r) => r.features),
    rows.map((r) => r.targets),
    FEATURES.map((f) => f.label),
    TARGETS.map((t) => t.label),
  );
}

export interface MaterialPick {
  part: string;
  choice: string;
  why: string;
}

export interface ShelterSpec {
  values: Record<TargetKey, number>;
  materials: MaterialPick[];
  severity: { label: string; score: number };
}

export function buildSpec(model: FittedModel, input: Record<FeatureKey, number>): ShelterSpec {
  const raw = predict(
    model,
    FEATURES.map((f) => input[f.key]),
  );

  const values = {} as Record<TargetKey, number>;
  TARGETS.forEach((t, i) => {
    values[t.key] = Math.max(0, raw[i] ?? 0);
  });

  const { windSpeed, minTemp, humidity, rainfall, snowLoad, quakeRisk } = input;
  const cold = minTemp < -5;
  const veryCold = minTemp < -25;
  const wet = humidity > 70 || rainfall > 1500;
  const materials: MaterialPick[] = [];

  materials.push({
    part: "Wall structure",
    choice: veryCold
      ? "Insulated structural panels: plywood skins over polyurethane core"
      : windSpeed > 160
        ? "Reinforced concrete blockwork with steel bond beams"
        : cold
          ? "Timber frame with dense mineral wool infill"
          : "Compressed earth blocks with lime render",
    why: `${values.wallThickness.toFixed(0)} mm total thickness for ${windSpeed.toFixed(0)} km/h winds at ${minTemp.toFixed(0)} °C.`,
  });

  materials.push({
    part: "Insulation",
    choice:
      values.insulationR > 5
        ? "Vacuum-panel core plus 200 mm polyurethane, taped joints"
        : values.insulationR > 3
          ? "150 mm polyisocyanurate board, continuous layer"
          : "100 mm mineral wool batts",
    why: `Target insulation value ${values.insulationR.toFixed(2)} m²K/W.`,
  });

  materials.push({
    part: "Roof",
    choice:
      snowLoad > 250
        ? "Steel truss frame with standing-seam metal deck"
        : rainfall > 1800
          ? "Corrugated fibre-cement sheet on treated purlins"
          : "Aluminium sandwich panel roof",
    why: `Pitch ${values.roofPitch.toFixed(1)}° to shed ${snowLoad.toFixed(0)} kg/m² snow and ${rainfall.toFixed(0)} mm rain.`,
  });

  materials.push({
    part: "Foundation",
    choice:
      quakeRisk > 3
        ? "Reinforced raft slab on base isolators"
        : veryCold
          ? "Adjustable steel screw piles below frost line"
          : "Continuous strip footing in reinforced concrete",
    why: `Bear down ${values.foundationDepth.toFixed(0)} cm; earthquake risk ${quakeRisk.toFixed(1)}/5.`,
  });

  materials.push({
    part: "Anchoring",
    choice:
      values.anchorLoad > 22
        ? "Grouted rock anchors with galvanised strap ties at 600 mm"
        : values.anchorLoad > 12
          ? "Helical ground anchors with steel hold-down straps"
          : "Expansion bolts into footing with hurricane clips",
    why: `Each anchor rated for ${values.anchorLoad.toFixed(1)} kN uplift.`,
  });

  materials.push({
    part: "Weather skin",
    choice: wet
      ? "Ventilated rainscreen cladding with breather membrane"
      : windSpeed > 180
        ? "Abrasion-resistant steel cladding, screw-fixed on all edges"
        : "Reflective coated aluminium cladding",
    why: wet ? "High moisture load needs a drained, ventilated cavity." : "Dry, high-wind exposure favours a sealed hard skin.",
  });

  const score = Math.min(
    100,
    Math.round(windSpeed * 0.28 + Math.max(0, -minTemp) * 0.7 + snowLoad * 0.05 + quakeRisk * 4),
  );
  const label = score > 75 ? "Extreme" : score > 50 ? "Severe" : score > 28 ? "Demanding" : "Moderate";

  return { values, materials, severity: { label, score } };
}

/** Parses a CSV whose headers match the feature and target keys or labels. */
export function parseCsv(text: string): { rows: DataRow[]; error?: string } {
  const lines = text.trim().split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 3) return { rows: [], error: "Need a header row and at least two data rows." };

  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const header = lines[0]!.split(",").map((h) => norm(h));

  const indexFor = (key: string, label: string) => {
    const i = header.indexOf(norm(key));
    return i >= 0 ? i : header.indexOf(norm(label));
  };

  const fIdx = FEATURES.map((f) => indexFor(f.key, f.label));
  const tIdx = TARGETS.map((t) => indexFor(t.key, t.label));
  const missing = [
    ...FEATURES.filter((f, i) => fIdx[i]! < 0).map((f) => f.label),
    ...TARGETS.filter((t, i) => tIdx[i]! < 0).map((t) => t.label),
  ];
  if (missing.length) return { rows: [], error: `Missing columns: ${missing.join(", ")}` };

  const rows: DataRow[] = [];
  for (const line of lines.slice(1)) {
    const cells = line.split(",");
    const features = fIdx.map((i) => Number(cells[i]));
    const targets = tIdx.map((i) => Number(cells[i]));
    if ([...features, ...targets].some((v) => !Number.isFinite(v))) continue;
    rows.push({ features, targets });
  }
  if (rows.length < 2) return { rows: [], error: "Could not read enough numeric rows." };
  return { rows };
}

export function csvTemplate(): string {
  return [
    [...FEATURES.map((f) => f.key), ...TARGETS.map((t) => t.key)].join(","),
    ...buildSampleDataset()
      .slice(0, 5)
      .map((r) => [...r.features, ...r.targets].map((v) => v.toFixed(2)).join(",")),
  ].join("\n");
}
