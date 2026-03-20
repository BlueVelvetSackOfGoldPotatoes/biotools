import { Children, useState } from "react";
import { formatSigned } from "./core";
export function StatCard({ label, value, note, tone = "default" }) {
  return (
    <div className={`cellengine-stat-card cellengine-tone-${tone}`}>
      <div className="cellengine-stat-label">{label}</div>
      <div className="cellengine-stat-value">{value}</div>
      <div className="cellengine-stat-note">{note}</div>
    </div>
  );
}

export function HelpTip({ text }) {
  return (
    <span className="help-tip" tabIndex={0} role="note" aria-label={text} title={text} data-help={text}>
      ?
    </span>
  );
}

export function SectionTitle({ title, help }) {
  return (
    <div className="help-heading">
      <h3>{title}</h3>
      {help ? <HelpTip text={help} /> : null}
    </div>
  );
}

export function FieldLabel({ label, help }) {
  return (
    <span className="help-label">
      <span>{label}</span>
      {help ? <HelpTip text={help} /> : null}
    </span>
  );
}

export function ActionButton({ label, help, className = "", children, ...buttonProps }) {
  return (
    <span className="cellengine-action-with-help">
      <button
        {...buttonProps}
        className={className}
        title={help || ""}
      >
        {children || label}
      </button>
      {help ? <HelpTip text={help} /> : null}
    </span>
  );
}

export function ActionLink({ label, help, className = "", children, ...linkProps }) {
  return (
    <span className="cellengine-action-with-help">
      <a
        {...linkProps}
        className={className}
        title={help || ""}
      >
        {children || label}
      </a>
      {help ? <HelpTip text={help} /> : null}
    </span>
  );
}

export function ActionGroup({ title, children, className = "" }) {
  const items = Children.toArray(children).filter(Boolean);
  if (!items.length) {
    return null;
  }
  return (
    <div className={`cellengine-action-group ${className}`.trim()}>
      <div className="cellengine-action-group-title">{title}</div>
      <div className="cellengine-action-group-body">
        {items}
      </div>
    </div>
  );
}

export function ActionGroups({ className = "", children }) {
  const groups = Children.toArray(children).filter(Boolean);
  if (!groups.length) {
    return null;
  }
  return <div className={`cellengine-action-groups ${className}`.trim()}>{groups}</div>;
}

export function DiscoveryDropdownSection({ title, help, summaryNote = "", defaultOpen = false, children }) {
  return (
    <details className="cellengine-discovery-dropdown" {...(defaultOpen ? { open: true } : {})}>
      <summary className="cellengine-discovery-dropdown-head">
        <span className="cellengine-discovery-dropdown-title">
          <strong>{title}</strong>
          {help ? <HelpTip text={help} /> : null}
        </span>
        {summaryNote ? <span className="cellengine-discovery-dropdown-note">{summaryNote}</span> : null}
      </summary>
      <div className="cellengine-discovery-dropdown-body">
        {children}
      </div>
    </details>
  );
}

export function DiscoverableNumberField({
  label,
  help,
  value,
  onChange,
  discoverable,
  onToggleDiscoverable,
  min,
  max,
  step = 1
}) {
  return (
    <div className="cellengine-discoverable-field">
      <label>
        <FieldLabel label={label} help={help} />
        <input
          className="task-input"
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={onChange}
        />
      </label>
      <label className="cellengine-toggle cellengine-toggle-compact">
        <input type="checkbox" checked={discoverable} onChange={onToggleDiscoverable} />
        <span>discoverable</span>
      </label>
    </div>
  );
}

export function DiscoverableSelectField({
  label,
  help,
  value,
  onChange,
  discoverable,
  onToggleDiscoverable,
  options
}) {
  return (
    <div className="cellengine-discoverable-field">
      <label>
        <FieldLabel label={label} help={help} />
        <select className="task-input" value={value} onChange={onChange}>
          {options.map((option) => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      </label>
      <label className="cellengine-toggle cellengine-toggle-compact">
        <input type="checkbox" checked={discoverable} onChange={onToggleDiscoverable} />
        <span>discoverable</span>
      </label>
    </div>
  );
}

export function ForceBar({ label, value, maxValue, colorClass }) {
  const magnitude = typeof value === "number" && Number.isFinite(value) ? Math.abs(value) : 0;
  const max = Math.max(maxValue || 1, 1e-6);
  const pct = Math.min(100, (magnitude / max) * 100);
  return (
    <div className="cellengine-force-bar">
      <div className="cellengine-force-head">
        <span>{label}</span>
        <strong>{formatSigned(value, 2)}</strong>
      </div>
      <div className="cellengine-force-track">
        <div className={`cellengine-force-fill ${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
