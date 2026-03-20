import React, { useRef, useEffect } from 'react';
import type { LogEntry } from '../../types/simulation';

interface LogViewerProps {
  logs: LogEntry[];
  onClear: () => void;
}

const LEVEL_COLORS: Record<LogEntry['level'], string> = {
  DEBUG: 'var(--color-text-muted)',
  INFO: 'var(--color-info)',
  WARNING: 'var(--color-warning)',
  ERROR: 'var(--color-error)',
  CRITICAL: '#ff4444',
};

const LEVEL_BG: Record<LogEntry['level'], string> = {
  DEBUG: 'transparent',
  INFO: 'transparent',
  WARNING: 'rgba(224, 160, 48, 0.08)',
  ERROR: 'rgba(211, 80, 80, 0.08)',
  CRITICAL: 'rgba(255, 68, 68, 0.12)',
};

export const LogViewer: React.FC<LogViewerProps> = ({ logs, onClear }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new log entries
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const formatTimestamp = (date: Date): string => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>Console / Log Output</h2>
        <div style={styles.headerActions}>
          <span style={styles.logCount}>{logs.length} entries</span>
          <button style={styles.clearBtn} onClick={onClear}>
            Clear Log
          </button>
        </div>
      </div>

      <div style={styles.logContainer} ref={scrollRef}>
        {logs.length === 0 ? (
          <div style={styles.emptyLog}>
            Log output will appear here during simulation execution.
          </div>
        ) : (
          logs.map((entry) => (
            <div
              key={entry.id}
              style={{
                ...styles.logEntry,
                backgroundColor: LEVEL_BG[entry.level],
              }}
            >
              <span style={styles.timestamp}>
                {formatTimestamp(entry.timestamp)}
              </span>
              <span
                style={{
                  ...styles.level,
                  color: LEVEL_COLORS[entry.level],
                }}
              >
                [{entry.level.padEnd(8)}]
              </span>
              <span style={styles.source}>{entry.source}</span>
              <span style={styles.message}>{entry.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    height: 'calc(100vh - var(--header-height) - var(--spacing-xl) * 2)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 'var(--spacing-md)',
    flexShrink: 0,
  },
  title: {
    fontSize: 'var(--font-size-xl)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-md)',
  },
  logCount: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-muted)',
  },
  clearBtn: {
    padding: 'var(--spacing-xs) var(--spacing-md)',
    backgroundColor: 'var(--color-bg-elevated)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--color-text-secondary)',
    fontSize: 'var(--font-size-sm)',
    cursor: 'pointer',
  },
  logContainer: {
    flex: 1,
    overflow: 'auto',
    backgroundColor: 'var(--color-bg-primary)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-md)',
    fontFamily: 'var(--font-family-mono)',
    fontSize: 'var(--font-size-sm)',
    lineHeight: 1.7,
  },
  emptyLog: {
    padding: 'var(--spacing-xl)',
    color: 'var(--color-text-muted)',
    textAlign: 'center' as const,
    fontFamily: 'var(--font-family)',
  },
  logEntry: {
    display: 'flex',
    gap: 'var(--spacing-sm)',
    padding: '2px var(--spacing-md)',
    borderBottom: '1px solid rgba(58, 63, 74, 0.3)',
    alignItems: 'baseline',
  },
  timestamp: {
    color: 'var(--color-text-muted)',
    flexShrink: 0,
    fontSize: 'var(--font-size-xs)',
  },
  level: {
    fontWeight: 600,
    flexShrink: 0,
    fontSize: 'var(--font-size-xs)',
  },
  source: {
    color: 'var(--color-text-accent)',
    flexShrink: 0,
    fontSize: 'var(--font-size-xs)',
  },
  message: {
    color: 'var(--color-text-primary)',
    wordBreak: 'break-word' as const,
    fontSize: 'var(--font-size-sm)',
  },
};
