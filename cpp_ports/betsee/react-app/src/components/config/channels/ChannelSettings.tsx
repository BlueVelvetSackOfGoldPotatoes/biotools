import React from 'react';
import type { ChannelConfig } from '../../../types/simulation';
import { ConfigSection, ConfigField, ConfigFieldGroup, ToggleField } from '../shared/ConfigWidgets';
import { formatScientific } from '../../../utils/format';

interface ChannelSettingsProps {
  channels: ChannelConfig[];
  onChange: (channels: ChannelConfig[]) => void;
}

export const ChannelSettings: React.FC<ChannelSettingsProps> = ({ channels, onChange }) => {
  const updateChannel = (index: number, field: keyof ChannelConfig, value: unknown) => {
    const updated = channels.map((ch, i) =>
      i === index ? { ...ch, [field]: value } : ch,
    );
    onChange(updated);
  };

  return (
    <div>
      <h2 style={styles.pageTitle}>Ion Channel Configuration</h2>
      <p style={styles.pageDesc}>
        Configure voltage-gated and leak ion channels. Each channel type can be
        enabled/disabled independently and has configurable kinetic parameters.
      </p>

      {channels.map((channel, index) => (
        <ConfigSection
          key={channel.type}
          title={channel.name}
          description={`Type: ${channel.type} | Class: ${channel.channel_class}`}
          collapsible
          defaultCollapsed={!channel.enabled}
        >
          <ToggleField
            label="Enabled"
            checked={channel.enabled}
            onChange={(v) => updateChannel(index, 'enabled', v)}
            tooltip="Enable or disable this channel type across the tissue."
          />

          <ConfigFieldGroup columns={2}>
            <ConfigField
              label="Maximum Dm"
              unit="m^2/s"
              tooltip="Maximum membrane diffusion coefficient for this channel."
            >
              <input
                type="number"
                value={channel.max_Dm}
                onChange={(e) => updateChannel(index, 'max_Dm', parseFloat(e.target.value))}
                step={1e-17}
                style={styles.input}
                disabled={!channel.enabled}
              />
              <span style={styles.sci}>{formatScientific(channel.max_Dm)}</span>
            </ConfigField>
            <ConfigField
              label="Channel Class"
              tooltip="Specific channel isoform/class."
            >
              <input
                type="text"
                value={channel.channel_class}
                onChange={(e) => updateChannel(index, 'channel_class', e.target.value)}
                style={styles.input}
                disabled={!channel.enabled}
              />
            </ConfigField>
          </ConfigFieldGroup>

          {/* Gating parameters - only for voltage-gated channels */}
          {channel.V_half !== 0 && (
            <>
              <h4 style={styles.subhead}>Gating Kinetics</h4>
              <ConfigFieldGroup columns={2}>
                <ConfigField
                  label="V_half"
                  unit="mV"
                  tooltip="Half-activation voltage. The membrane potential at which 50% of channels are open."
                >
                  <input
                    type="number"
                    value={channel.V_half}
                    onChange={(e) => updateChannel(index, 'V_half', parseFloat(e.target.value))}
                    step={1}
                    style={styles.input}
                    disabled={!channel.enabled}
                  />
                </ConfigField>
                <ConfigField
                  label="V_slope"
                  unit="mV"
                  tooltip="Slope factor of the activation curve. Determines steepness of voltage dependence."
                >
                  <input
                    type="number"
                    value={channel.V_slope}
                    onChange={(e) => updateChannel(index, 'V_slope', parseFloat(e.target.value))}
                    step={0.5}
                    style={styles.input}
                    disabled={!channel.enabled}
                  />
                </ConfigField>
                <ConfigField
                  label="V_tau"
                  unit="mV"
                  tooltip="Voltage at which the time constant reaches its maximum."
                >
                  <input
                    type="number"
                    value={channel.V_tau}
                    onChange={(e) => updateChannel(index, 'V_tau', parseFloat(e.target.value))}
                    step={1}
                    style={styles.input}
                    disabled={!channel.enabled}
                  />
                </ConfigField>
                <ConfigField
                  label="tau_max"
                  unit="s"
                  tooltip="Maximum gating time constant."
                >
                  <input
                    type="number"
                    value={channel.tau_max}
                    onChange={(e) => updateChannel(index, 'tau_max', parseFloat(e.target.value))}
                    step={0.001}
                    style={styles.input}
                    disabled={!channel.enabled}
                  />
                  <span style={styles.sci}>{formatScientific(channel.tau_max)} s</span>
                </ConfigField>
              </ConfigFieldGroup>

              {/* Boltzmann activation curve visualization */}
              {channel.enabled && (
                <div style={styles.curveBox}>
                  <h5 style={styles.curveTitle}>Activation Curve (Boltzmann)</h5>
                  <ActivationCurve vHalf={channel.V_half} vSlope={channel.V_slope} />
                </div>
              )}
            </>
          )}

          <ConfigField
            label="Apply to Tissues"
            tooltip="Tissue profiles where this channel is expressed. 'all' means every tissue."
          >
            <input
              type="text"
              value={channel.apply_to.join(', ')}
              onChange={(e) =>
                updateChannel(index, 'apply_to', e.target.value.split(',').map((s) => s.trim()))
              }
              style={styles.input}
              disabled={!channel.enabled}
            />
          </ConfigField>
        </ConfigSection>
      ))}
    </div>
  );
};

/** SVG visualization of a Boltzmann activation curve. */
const ActivationCurve: React.FC<{ vHalf: number; vSlope: number }> = ({ vHalf, vSlope }) => {
  const width = 280;
  const height = 120;
  const padding = { top: 10, right: 10, bottom: 25, left: 35 };

  const vMin = vHalf - 60;
  const vMax = vHalf + 60;

  const points: string[] = [];
  for (let v = vMin; v <= vMax; v += 1) {
    const x = padding.left + ((v - vMin) / (vMax - vMin)) * (width - padding.left - padding.right);
    const activation = 1 / (1 + Math.exp(-(v - vHalf) / Math.abs(vSlope)));
    const y = padding.top + (1 - activation) * (height - padding.top - padding.bottom);
    points.push(`${x},${y}`);
  }

  return (
    <svg width={width} height={height} style={{ backgroundColor: 'var(--color-bg-primary)', borderRadius: 'var(--radius-sm)' }}>
      {/* Axes */}
      <line
        x1={padding.left} y1={height - padding.bottom}
        x2={width - padding.right} y2={height - padding.bottom}
        stroke="var(--color-border-light)" strokeWidth={1}
      />
      <line
        x1={padding.left} y1={padding.top}
        x2={padding.left} y2={height - padding.bottom}
        stroke="var(--color-border-light)" strokeWidth={1}
      />
      {/* Labels */}
      <text x={width / 2} y={height - 3} textAnchor="middle" fill="var(--color-text-muted)" fontSize={9}>
        Vm (mV)
      </text>
      <text x={5} y={height / 2} textAnchor="middle" fill="var(--color-text-muted)" fontSize={9}
        transform={`rotate(-90, 8, ${height / 2})`}>
        Open
      </text>
      {/* V_half marker */}
      <line
        x1={padding.left + 0.5 * (width - padding.left - padding.right)}
        y1={padding.top}
        x2={padding.left + 0.5 * (width - padding.left - padding.right)}
        y2={height - padding.bottom}
        stroke="var(--color-warning)" strokeWidth={1} strokeDasharray="3,3" opacity={0.5}
      />
      {/* Curve */}
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth={2}
      />
    </svg>
  );
};

const styles: Record<string, React.CSSProperties> = {
  pageTitle: {
    fontSize: 'var(--font-size-xl)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
    marginBottom: 'var(--spacing-sm)',
  },
  pageDesc: {
    fontSize: 'var(--font-size-base)',
    color: 'var(--color-text-secondary)',
    marginBottom: 'var(--spacing-xl)',
    maxWidth: '680px',
    lineHeight: 1.6,
  },
  input: {
    width: '100%',
    fontFamily: 'var(--font-family-mono)',
  },
  sci: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-family-mono)',
    display: 'block',
    marginTop: '2px',
  },
  subhead: {
    fontSize: 'var(--font-size-sm)',
    fontWeight: 600,
    color: 'var(--color-text-secondary)',
    marginTop: 'var(--spacing-lg)',
    marginBottom: 'var(--spacing-md)',
  },
  curveBox: {
    marginTop: 'var(--spacing-md)',
    marginBottom: 'var(--spacing-md)',
  },
  curveTitle: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    marginBottom: 'var(--spacing-xs)',
    fontWeight: 500,
  },
};
