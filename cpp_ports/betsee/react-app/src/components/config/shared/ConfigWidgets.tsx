import React, { useState } from 'react';

/**
 * Reusable configuration UI widgets matching the BETSEE design language.
 */

interface ConfigSectionProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

export const ConfigSection: React.FC<ConfigSectionProps> = ({
  title,
  description,
  children,
  collapsible = false,
  defaultCollapsed = false,
}) => {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <section style={sectionStyles.section}>
      <div
        style={{
          ...sectionStyles.header,
          cursor: collapsible ? 'pointer' : 'default',
        }}
        onClick={collapsible ? () => setCollapsed(!collapsed) : undefined}
      >
        <div>
          <h3 style={sectionStyles.title}>
            {collapsible && (
              <span style={sectionStyles.collapseIcon}>
                {collapsed ? '\u25B6' : '\u25BC'}
              </span>
            )}
            {title}
          </h3>
          {description && <p style={sectionStyles.description}>{description}</p>}
        </div>
      </div>
      {!collapsed && <div style={sectionStyles.body}>{children}</div>}
    </section>
  );
};

const sectionStyles: Record<string, React.CSSProperties> = {
  section: {
    backgroundColor: 'var(--color-bg-secondary)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-md)',
    marginBottom: 'var(--spacing-lg)',
    overflow: 'hidden',
  },
  header: {
    padding: 'var(--spacing-md) var(--spacing-lg)',
    borderBottom: '1px solid var(--color-border)',
  },
  title: {
    fontSize: 'var(--font-size-md)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
    margin: 0,
  },
  description: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-muted)',
    marginTop: 'var(--spacing-xs)',
    marginBottom: 0,
  },
  collapseIcon: {
    marginRight: 'var(--spacing-sm)',
    fontSize: 'var(--font-size-xs)',
  },
  body: {
    padding: 'var(--spacing-lg)',
  },
};

interface ConfigFieldProps {
  label: string;
  unit?: string;
  tooltip?: string;
  children: React.ReactNode;
  inline?: boolean;
  error?: string;
  min?: number;
  max?: number;
}

export const ConfigField: React.FC<ConfigFieldProps> = ({
  label,
  unit,
  tooltip,
  children,
  inline = false,
  error,
}) => {
  return (
    <div style={inline ? fieldStyles.fieldInline : fieldStyles.field} title={tooltip}>
      <label style={fieldStyles.label}>
        {label}
        {unit && <span style={fieldStyles.unit}> ({unit})</span>}
      </label>
      <div style={fieldStyles.control}>
        {children}
        {error && (
          <div style={fieldStyles.error}>{error}</div>
        )}
      </div>
    </div>
  );
};

const fieldStyles: Record<string, React.CSSProperties> = {
  field: {
    marginBottom: 'var(--spacing-md)',
  },
  fieldInline: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-md)',
    marginBottom: 'var(--spacing-md)',
  },
  label: {
    display: 'block',
    fontSize: 'var(--font-size-sm)',
    fontWeight: 500,
    color: 'var(--color-text-secondary)',
    marginBottom: 'var(--spacing-xs)',
  },
  unit: {
    color: 'var(--color-text-muted)',
    fontWeight: 400,
    fontSize: 'var(--font-size-xs)',
  },
  control: {
    maxWidth: '320px',
  },
  error: {
    color: '#e53935',
    fontSize: 'var(--font-size-xs)',
    marginTop: '2px',
    fontWeight: 500,
  },
};

interface ConfigFieldGroupProps {
  children: React.ReactNode;
  columns?: number;
}

export const ConfigFieldGroup: React.FC<ConfigFieldGroupProps> = ({
  children,
  columns = 3,
}) => {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: 'var(--spacing-lg)',
      }}
    >
      {children}
    </div>
  );
};

interface ToggleFieldProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  tooltip?: string;
}

export const ToggleField: React.FC<ToggleFieldProps> = ({
  label,
  checked,
  onChange,
  tooltip,
}) => {
  return (
    <div style={toggleStyles.container} title={tooltip}>
      <label style={toggleStyles.label}>
        <div
          style={{
            ...toggleStyles.track,
            backgroundColor: checked ? 'var(--color-accent)' : 'var(--color-bg-elevated)',
          }}
          onClick={() => onChange(!checked)}
        >
          <div
            style={{
              ...toggleStyles.thumb,
              transform: checked ? 'translateX(16px)' : 'translateX(0)',
            }}
          />
        </div>
        <span style={toggleStyles.text}>{label}</span>
      </label>
    </div>
  );
};

const toggleStyles: Record<string, React.CSSProperties> = {
  container: {
    marginBottom: 'var(--spacing-md)',
  },
  label: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-sm)',
    cursor: 'pointer',
  },
  track: {
    width: '36px',
    height: '20px',
    borderRadius: '10px',
    position: 'relative' as const,
    transition: 'background-color 0.15s ease',
    cursor: 'pointer',
    flexShrink: 0,
  },
  thumb: {
    width: '16px',
    height: '16px',
    borderRadius: '50%',
    backgroundColor: '#fff',
    position: 'absolute' as const,
    top: '2px',
    left: '2px',
    transition: 'transform 0.15s ease',
    boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
  },
  text: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-secondary)',
  },
};
