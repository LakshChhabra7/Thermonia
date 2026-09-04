import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";

import heroImage from "@/assets/shelter-hero.jpg";
import {
  FEATURES,
  TARGETS,
  buildSampleDataset,
  buildSpec,
  csvTemplate,
  parseCsv,
  trainModel,
  type DataRow,
  type FeatureKey,
} from "@/lib/shelter-data";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Shelter Spec Engine — Materials for Harsh Conditions" },
      {
        name: "description",
        content:
          "Enter wind speed, temperature, snow load and site conditions to get regression-based shelter material, wall thickness and anchoring specifications.",
      },
      { property: "og:title", content: "Shelter Spec Engine — Materials for Harsh Conditions" },
      {
        property: "og:description",
        content:
          "A multiple regression model trained on engineered shelter builds turns site conditions into a buildable material spec.",
      },
    ],
  }),
  component: Home,
});

function Home() {
  const [dataset, setDataset] = useState<DataRow[]>(() => buildSampleDataset());
  const [datasetName, setDatasetName] = useState("Reference set (240 builds)");
  const [csvError, setCsvError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const [inputs, setInputs] = useState<Record<FeatureKey, number>>(() => {
    const initial = {} as Record<FeatureKey, number>;
    FEATURES.forEach((f) => {
      initial[f.key] = f.demo;
    });
    return initial;
  });

  const model = useMemo(() => trainModel(dataset), [dataset]);
  const spec = useMemo(() => buildSpec(model, inputs), [model, inputs]);

  const avgR2 = model.r2.reduce((a, b) => a + b, 0) / model.r2.length;

  const handleFile = async (file: File) => {
    const text = await file.text();
    const { rows, error } = parseCsv(text);
    if (error) {
      setCsvError(error);
      return;
    }
    setCsvError(null);
    setDataset(rows);
    setDatasetName(`${file.name} (${rows.length} builds)`);
  };

  const downloadTemplate = () => {
    const blob = new Blob([csvTemplate()], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "shelter-training-template.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="min-h-screen">
      <header className="relative overflow-hidden border-b border-border">
        <img
          src={heroImage}
          width={1600}
          height={912}
          alt="Reinforced shelter anchored on a snow-blasted mountain ridge at dusk"
          className="absolute inset-0 h-full w-full object-cover opacity-45"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background via-background/70 to-background/30" />
        <div className="relative mx-auto max-w-6xl px-6 py-20 md:py-28">
          <p className="label-eyebrow">Multiple regression · shelter engineering</p>
          <h1 className="mt-4 max-w-3xl text-4xl leading-tight font-extrabold md:text-6xl">
            Turn brutal site conditions into a buildable shelter spec
          </h1>
          <p className="mt-5 max-w-2xl text-base text-muted-foreground md:text-lg">
            The model learns from a set of engineered builds, then predicts wall thickness,
            insulation, foundations and anchoring for the numbers you enter — so people inside can
            actually rest.
          </p>
          <div className="mt-8 flex flex-wrap gap-3 font-mono text-xs text-muted-foreground">
            <span className="panel px-3 py-2">{model.sampleCount} training builds</span>
            <span className="panel px-3 py-2">{FEATURES.length} site inputs</span>
            <span className="panel px-3 py-2">Model fit R² {avgR2.toFixed(3)}</span>
          </div>
        </div>
      </header>

      <section className="mx-auto grid max-w-6xl gap-6 px-6 py-12 lg:grid-cols-[380px_1fr]">
        <div className="panel h-fit p-6">
          <h2 className="text-xl font-bold">Site conditions</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Adjust the values for your location. Results update as you type.
          </p>

          <div className="mt-6 space-y-4">
            {FEATURES.map((f) => (
              <label key={f.key} className="block">
                <span className="flex items-baseline justify-between">
                  <span className="text-sm font-medium">{f.label}</span>
                  <span className="font-mono text-xs text-muted-foreground">{f.unit}</span>
                </span>
                <input
                  className="field-input mt-1.5"
                  type="number"
                  value={inputs[f.key]}
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  onChange={(e) =>
                    setInputs((prev) => ({ ...prev, [f.key]: Number(e.target.value) }))
                  }
                />
                <input
                  className="mt-2 w-full accent-primary"
                  type="range"
                  value={inputs[f.key]}
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  onChange={(e) =>
                    setInputs((prev) => ({ ...prev, [f.key]: Number(e.target.value) }))
                  }
                />
              </label>
            ))}
          </div>
        </div>

        <div className="space-y-6">
          <div className="panel p-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="label-eyebrow">Condition severity</p>
                <p className="mt-1 text-2xl font-bold">{spec.severity.label}</p>
              </div>
              <p className="font-mono text-3xl text-primary">{spec.severity.score}/100</p>
            </div>
            <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-surface-raised">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${spec.severity.score}%` }}
              />
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {TARGETS.map((t) => (
              <div key={t.key} className="panel p-5">
                <p className="label-eyebrow">{t.label}</p>
                <p className="mt-2 font-mono text-2xl font-medium text-foreground">
                  {spec.values[t.key].toFixed(t.digits)}
                  <span className="ml-1 text-sm text-muted-foreground">{t.unit}</span>
                </p>
              </div>
            ))}
          </div>

          <div className="panel p-6">
            <h2 className="text-xl font-bold">Recommended materials</h2>
            <ul className="mt-5 space-y-4">
              {spec.materials.map((m) => (
                <li key={m.part} className="border-l-2 border-primary pl-4">
                  <p className="label-eyebrow">{m.part}</p>
                  <p className="mt-1 font-medium">{m.choice}</p>
                  <p className="mt-1 text-sm text-muted-foreground">{m.why}</p>
                </li>
              ))}
            </ul>
          </div>

          <div className="panel p-6">
            <h2 className="text-xl font-bold">Training data</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Currently using <span className="font-mono text-foreground">{datasetName}</span>. Load
              your own CSV to retrain instantly — column names must match the template.
            </p>
            <div className="mt-4 flex flex-wrap gap-3">
              <button className="btn-primary" onClick={() => fileRef.current?.click()}>
                Upload CSV
              </button>
              <button className="btn-ghost" onClick={downloadTemplate}>
                Download template
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void handleFile(file);
                }}
              />
            </div>
            {csvError && <p className="mt-3 text-sm text-destructive">{csvError}</p>}

            <div className="mt-6 overflow-x-auto">
              <table className="w-full font-mono text-xs">
                <thead>
                  <tr className="text-left text-muted-foreground">
                    <th className="pb-2 pr-4 font-medium">Predicted quantity</th>
                    <th className="pb-2 pr-4 font-medium">R²</th>
                  </tr>
                </thead>
                <tbody>
                  {TARGETS.map((t, i) => (
                    <tr key={t.key} className="border-t border-border">
                      <td className="py-2 pr-4">{t.label}</td>
                      <td className="py-2 pr-4 text-primary">{(model.r2[i] ?? 0).toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-border px-6 py-8 text-center text-sm text-muted-foreground">
        Predictions are engineering guidance from a fitted regression model — have a qualified
        engineer sign off before building.
      </footer>
    </main>
  );
}
