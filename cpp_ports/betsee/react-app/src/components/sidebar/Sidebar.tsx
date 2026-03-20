import React from 'react';
import type { ViewId } from '../../App';

interface SidebarProps {
  activeView: ViewId;
  onNavigate: (view: ViewId) => void;
}

interface NavSection {
  title: string;
  items: { id: ViewId; label: string; icon: string }[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    title: 'Configuration',
    items: [
      { id: 'general', label: 'General / Time', icon: 'C' },
      { id: 'space', label: 'Space / Cells', icon: 'S' },
      { id: 'ions', label: 'Ion Species', icon: 'I' },
      { id: 'channels', label: 'Channels', icon: 'H' },
      { id: 'tissue', label: 'Tissue Profiles', icon: 'T' },
      { id: 'interventions', label: 'Interventions', icon: 'X' },
      { id: 'network', label: 'Network / GRN', icon: 'N' },
      { id: 'physics', label: 'Physics', icon: 'P' },
      { id: 'exports', label: 'Exports', icon: 'E' },
    ],
  },
  {
    title: 'Simulation',
    items: [
      { id: 'simulation', label: 'Run Control', icon: 'R' },
      { id: 'results', label: 'Results Viewer', icon: 'V' },
      { id: 'log', label: 'Console / Log', icon: 'L' },
    ],
  },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onNavigate }) => {
  return (
    <nav style={styles.sidebar}>
      {NAV_SECTIONS.map((section) => (
        <div key={section.title} style={styles.section}>
          <div style={styles.sectionTitle}>{section.title}</div>
          {section.items.map((item) => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                style={{
                  ...styles.navItem,
                  ...(isActive ? styles.navItemActive : {}),
                }}
                title={item.label}
              >
                <span
                  style={{
                    ...styles.navIcon,
                    ...(isActive ? styles.navIconActive : {}),
                  }}
                >
                  {item.icon}
                </span>
                <span style={styles.navLabel}>{item.label}</span>
              </button>
            );
          })}
        </div>
      ))}

      <div style={styles.footer}>
        <div style={styles.footerText}>BETSEE v1.0</div>
        <div style={styles.footerText}>BETSE Bioelectric Simulator</div>
      </div>
    </nav>
  );
};

const styles: Record<string, React.CSSProperties> = {
  sidebar: {
    width: 'var(--sidebar-width)',
    minWidth: 'var(--sidebar-width)',
    backgroundColor: 'var(--color-bg-secondary)',
    borderRight: '1px solid var(--color-border)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'auto',
    flexShrink: 0,
  },
  section: {
    padding: 'var(--spacing-md) 0',
  },
  sectionTitle: {
    fontSize: 'var(--font-size-xs)',
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    color: 'var(--color-text-muted)',
    padding: 'var(--spacing-xs) var(--spacing-lg)',
    marginBottom: 'var(--spacing-xs)',
  },
  navItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-sm)',
    width: '100%',
    padding: 'var(--spacing-sm) var(--spacing-lg)',
    border: 'none',
    background: 'none',
    color: 'var(--color-text-secondary)',
    fontSize: 'var(--font-size-base)',
    cursor: 'pointer',
    textAlign: 'left' as const,
    transition: 'background-color 0.12s ease, color 0.12s ease',
    borderLeft: '3px solid transparent',
  },
  navItemActive: {
    backgroundColor: 'var(--color-bg-tertiary)',
    color: 'var(--color-text-primary)',
    borderLeftColor: 'var(--color-accent)',
  },
  navIcon: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '22px',
    height: '22px',
    borderRadius: 'var(--radius-sm)',
    backgroundColor: 'var(--color-bg-elevated)',
    fontSize: 'var(--font-size-xs)',
    fontWeight: 700,
    color: 'var(--color-text-muted)',
    flexShrink: 0,
  },
  navIconActive: {
    backgroundColor: 'var(--color-accent)',
    color: '#fff',
  },
  navLabel: {
    whiteSpace: 'nowrap' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  footer: {
    marginTop: 'auto',
    padding: 'var(--spacing-lg)',
    borderTop: '1px solid var(--color-border)',
  },
  footerText: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    lineHeight: 1.6,
  },
};
