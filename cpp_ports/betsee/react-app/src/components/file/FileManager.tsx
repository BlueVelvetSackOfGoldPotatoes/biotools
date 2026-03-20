import React, { useState } from 'react';
import type { SimulationConfig } from '../../types/simulation';
import { PRESETS } from '../../types/presets';

interface FileManagerProps {
  config: SimulationConfig;
  onNew: () => void;
  onOpen: (filename: string) => void;
  onSave: () => void;
  onLoadPreset: (index: number) => void;
  onClose: () => void;
}

type FileTab = 'file' | 'presets';

export const FileManager: React.FC<FileManagerProps> = ({
  config,
  onNew,
  onOpen,
  onSave,
  onLoadPreset,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<FileTab>('file');
  const [openFilename, setOpenFilename] = useState('');

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.modalHeader}>
          <h2 style={styles.modalTitle}>File Manager</h2>
          <button style={styles.closeBtn} onClick={onClose}>
            {'\u2715'}
          </button>
        </div>

        {/* Tabs */}
        <div style={styles.tabBar}>
          <button
            style={{ ...styles.tab, ...(activeTab === 'file' ? styles.tabActive : {}) }}
            onClick={() => setActiveTab('file')}
          >
            File Operations
          </button>
          <button
            style={{ ...styles.tab, ...(activeTab === 'presets' ? styles.tabActive : {}) }}
            onClick={() => setActiveTab('presets')}
          >
            Parameter Presets
          </button>
        </div>

        <div style={styles.modalBody}>
          {activeTab === 'file' && (
            <div>
              {/* Current file info */}
              <div style={styles.infoBox}>
                <div style={styles.infoRow}>
                  <span style={styles.infoLabel}>Current File:</span>
                  <span style={styles.infoValue}>
                    {config.filename || '(unsaved)'}
                  </span>
                </div>
                <div style={styles.infoRow}>
                  <span style={styles.infoLabel}>Status:</span>
                  <span style={{
                    ...styles.infoValue,
                    color: config.isDirty ? 'var(--color-warning)' : 'var(--color-success)',
                  }}>
                    {config.isDirty ? 'Modified (unsaved changes)' : 'Saved'}
                  </span>
                </div>
              </div>

              {/* Actions */}
              <div style={styles.actionGrid}>
                <button
                  style={styles.actionBtn}
                  onClick={() => { onNew(); onClose(); }}
                >
                  <div style={styles.actionIcon}>+</div>
                  <div style={styles.actionText}>
                    <strong>New Configuration</strong>
                    <span>Create a new simulation with default settings</span>
                  </div>
                </button>

                <div style={styles.openSection}>
                  <div style={styles.actionIcon}>O</div>
                  <div style={styles.openForm}>
                    <strong style={{ color: 'var(--color-text-primary)', fontSize: 'var(--font-size-sm)' }}>
                      Open Configuration
                    </strong>
                    <div style={styles.openInputRow}>
                      <input
                        type="text"
                        value={openFilename}
                        onChange={(e) => setOpenFilename(e.target.value)}
                        placeholder="path/to/simulation.yaml"
                        style={styles.input}
                      />
                      <button
                        style={styles.openBtn}
                        onClick={() => {
                          if (openFilename) {
                            onOpen(openFilename);
                            onClose();
                          }
                        }}
                        disabled={!openFilename}
                      >
                        Open
                      </button>
                    </div>
                  </div>
                </div>

                <button
                  style={styles.actionBtn}
                  onClick={() => { onSave(); onClose(); }}
                >
                  <div style={styles.actionIcon}>S</div>
                  <div style={styles.actionText}>
                    <strong>Save Configuration</strong>
                    <span>Save current settings{config.filename ? ` to ${config.filename}` : ''}</span>
                  </div>
                </button>
              </div>
            </div>
          )}

          {activeTab === 'presets' && (
            <div>
              <p style={styles.presetDesc}>
                Load a parameter preset to quickly configure the simulation for
                common scenarios. This will overwrite current settings.
              </p>
              <div style={styles.presetGrid}>
                {PRESETS.map((preset, index) => (
                  <button
                    key={index}
                    style={styles.presetCard}
                    onClick={() => { onLoadPreset(index); onClose(); }}
                  >
                    <strong style={styles.presetName}>{preset.name}</strong>
                    <p style={styles.presetCardDesc}>{preset.description}</p>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modal: {
    width: '700px',
    maxHeight: '80vh',
    backgroundColor: 'var(--color-bg-secondary)',
    borderRadius: 'var(--radius-lg)',
    border: '1px solid var(--color-border)',
    boxShadow: 'var(--shadow-lg)',
    display: 'flex',
    flexDirection: 'column' as const,
    overflow: 'hidden',
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 'var(--spacing-lg) var(--spacing-xl)',
    borderBottom: '1px solid var(--color-border)',
  },
  modalTitle: {
    fontSize: 'var(--font-size-lg)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
    margin: 0,
  },
  closeBtn: {
    width: '28px',
    height: '28px',
    borderRadius: 'var(--radius-sm)',
    backgroundColor: 'var(--color-bg-elevated)',
    border: '1px solid var(--color-border)',
    color: 'var(--color-text-secondary)',
    fontSize: 'var(--font-size-sm)',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabBar: {
    display: 'flex',
    gap: '2px',
    padding: 'var(--spacing-sm) var(--spacing-xl)',
    backgroundColor: 'var(--color-bg-tertiary)',
    borderBottom: '1px solid var(--color-border)',
  },
  tab: {
    padding: 'var(--spacing-sm) var(--spacing-lg)',
    borderRadius: 'var(--radius-sm)',
    fontSize: 'var(--font-size-sm)',
    fontWeight: 500,
    color: 'var(--color-text-secondary)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
  },
  tabActive: {
    backgroundColor: 'var(--color-bg-elevated)',
    color: 'var(--color-text-primary)',
  },
  modalBody: {
    padding: 'var(--spacing-xl)',
    overflow: 'auto',
    flex: 1,
  },
  infoBox: {
    backgroundColor: 'var(--color-bg-tertiary)',
    borderRadius: 'var(--radius-md)',
    padding: 'var(--spacing-md) var(--spacing-lg)',
    marginBottom: 'var(--spacing-xl)',
    border: '1px solid var(--color-border)',
  },
  infoRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 'var(--spacing-xs) 0',
  },
  infoLabel: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-muted)',
  },
  infoValue: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-primary)',
    fontFamily: 'var(--font-family-mono)',
  },
  actionGrid: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 'var(--spacing-md)',
  },
  actionBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-lg)',
    padding: 'var(--spacing-lg)',
    backgroundColor: 'var(--color-bg-tertiary)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-md)',
    cursor: 'pointer',
    textAlign: 'left' as const,
    transition: 'background-color 0.15s ease',
    color: 'var(--color-text-primary)',
  },
  openSection: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 'var(--spacing-lg)',
    padding: 'var(--spacing-lg)',
    backgroundColor: 'var(--color-bg-tertiary)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-md)',
  },
  openForm: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 'var(--spacing-sm)',
  },
  openInputRow: {
    display: 'flex',
    gap: 'var(--spacing-sm)',
  },
  input: {
    flex: 1,
    fontFamily: 'var(--font-family-mono)',
  },
  openBtn: {
    padding: 'var(--spacing-xs) var(--spacing-lg)',
    backgroundColor: 'var(--color-accent)',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    color: '#fff',
    fontSize: 'var(--font-size-sm)',
    fontWeight: 600,
    cursor: 'pointer',
  },
  actionIcon: {
    width: '36px',
    height: '36px',
    borderRadius: 'var(--radius-md)',
    backgroundColor: 'var(--color-bg-elevated)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 'var(--font-size-md)',
    fontWeight: 700,
    color: 'var(--color-accent)',
    flexShrink: 0,
  },
  actionText: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '2px',
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-secondary)',
  },
  presetDesc: {
    fontSize: 'var(--font-size-base)',
    color: 'var(--color-text-secondary)',
    marginBottom: 'var(--spacing-lg)',
    lineHeight: 1.6,
  },
  presetGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 'var(--spacing-md)',
  },
  presetCard: {
    padding: 'var(--spacing-lg)',
    backgroundColor: 'var(--color-bg-tertiary)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-md)',
    cursor: 'pointer',
    textAlign: 'left' as const,
    transition: 'border-color 0.15s ease',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 'var(--spacing-xs)',
    color: 'var(--color-text-primary)',
  },
  presetName: {
    fontSize: 'var(--font-size-md)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
  },
  presetCardDesc: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    lineHeight: 1.5,
    margin: 0,
  },
};
