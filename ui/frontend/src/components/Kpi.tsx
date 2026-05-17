interface KpiProps {
  label: string;
  value: string;
  unit?: string;
}

export function Kpi({ label, value, unit }: KpiProps) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className="value">
        {value}
        {unit ? <span className="unit">{unit}</span> : null}
      </div>
    </div>
  );
}

export function KpiGrid({ children }: { children: React.ReactNode }) {
  return <div className="kpi-grid">{children}</div>;
}
