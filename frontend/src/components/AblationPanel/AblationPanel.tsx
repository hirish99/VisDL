import { useState } from 'react';
import { useGraphStore } from '../../store/graphStore';
import { useExecutionStore } from '../../store/executionStore';
import { executeGraph, saveGraph, loadGraph } from '../../api/client';

export function AblationPanel() {
  const nodes = useGraphStore((s) => s.nodes);
  const toggleDisabled = useGraphStore((s) => s.toggleNodeDisabled);
  const toSchema = useGraphStore((s) => s.toGraphSchema);
  const loadFromSchema = useGraphStore((s) => s.loadFromSchema);
  const sessionId = useExecutionStore((s) => s.sessionId);
  const addRun = useExecutionStore((s) => s.addAblationRun);
  const clearRuns = useExecutionStore((s) => s.clearAblationRuns);
  const isRunning = useExecutionStore((s) => s.isRunning);
  const setRunning = useExecutionStore((s) => s.setRunning);
  const clearProgress = useExecutionStore((s) => s.clearProgress);

  const [configName, setConfigName] = useState('');
  const [savedConfigs, setSavedConfigs] = useState<{ id: string; name: string }[]>([]);

  const ablatable = nodes.filter((n) =>
    ['Layers'].includes(n.data.definition.category),
  );

  const handleSaveConfig = async () => {
    const name = configName || `config_${Date.now()}`;
    const schema = toSchema();
    schema.name = name;
    const res = await saveGraph(schema, '', name);
    setSavedConfigs((prev) => [...prev, { id: res.id, name }]);
    setConfigName('');
  };

  const handleLoadConfig = async (id: string) => {
    const data = await loadGraph(id);
    loadFromSchema(data.graph);
  };

  const handleRunAblation = async (runName: string) => {
    if (isRunning) return;
    setRunning(true);
    clearProgress();
    const schema = toSchema();
    try {
      const res = await executeGraph(schema, sessionId);
      if (res.status === 'success') {
        // Extract metrics from results
        const metricsNode = Object.entries(res.results).find(
          ([, outputs]) => outputs[0] && typeof outputs[0] === 'object' &&
            (outputs[0] as any)?.train_loss !== undefined,
        );
        if (metricsNode) {
          const metrics = metricsNode[1][0] as any;
          const history = (metrics.epochs as number[]).map((epoch: number, i: number) => ({
            epoch,
            train_loss: metrics.train_loss[i],
            val_loss: metrics.val_loss?.[i] ?? null,
          }));
          addRun({
            id: `run_${Date.now()}`,
            name: runName,
            history,
            finalTrainLoss: metrics.final_train_loss,
            finalValLoss: metrics.final_val_loss,
          });
        }
      }
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={{
      width: 260,
      background: '#12121a',
      borderLeft: '1px solid #2a2a3e',
      overflow: 'auto',
      fontFamily: 'system-ui, sans-serif',
      padding: 12,
    }}>
      <div style={{
        color: '#808090', fontSize: 11, fontWeight: 600,
        textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12,
      }}>
        Ablation
      </div>

      {/* Toggle layer nodes */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ color: '#666', fontSize: 10, marginBottom: 4 }}>Layer Toggles</div>
        {ablatable.map((node) => (
          <label
            key={node.id}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              color: node.data.disabled ? '#555' : '#a0a0b0',
              fontSize: 11, marginBottom: 4, cursor: 'pointer',
            }}
          >
            <input
              type="checkbox"
              checked={!node.data.disabled}
              onChange={() => toggleDisabled(node.id)}
            />
            {node.data.definition.display_name}
            {node.data.disabled && <span style={{ color: '#ef4444', fontSize: 9 }}>OFF</span>}
          </label>
        ))}
        {ablatable.length === 0 && (
          <div style={{ color: '#444', fontSize: 10 }}>No ablatable nodes</div>
        )}
      </div>

      {/* Save config */}
      <div style={{ marginBottom: 12, borderTop: '1px solid #2a2a3e', paddingTop: 8 }}>
        <div style={{ color: '#666', fontSize: 10, marginBottom: 4 }}>Save Config</div>
        <div style={{ display: 'flex', gap: 4 }}>
          <input
            type="text"
            value={configName}
            onChange={(e) => setConfigName(e.target.value)}
            placeholder="Config name..."
            style={{
              flex: 1, background: '#1e1e2e', border: '1px solid #2a2a3e',
              borderRadius: 4, color: '#c0c0d0', padding: '4px 8px', fontSize: 10,
            }}
          />
          <button onClick={handleSaveConfig} style={btnStyle}>Save</button>
        </div>
      </div>

      {/* Saved configs */}
      {savedConfigs.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: '#666', fontSize: 10, marginBottom: 4 }}>Saved Configs</div>
          {savedConfigs.map((cfg) => (
            <div key={cfg.id} style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
              <span style={{ color: '#a0a0b0', fontSize: 10, flex: 1 }}>{cfg.name}</span>
              <button onClick={() => handleLoadConfig(cfg.id)} style={btnSmStyle}>Load</button>
              <button onClick={() => handleRunAblation(cfg.name)} style={btnSmStyle}>Run</button>
            </div>
          ))}
        </div>
      )}

      {/* Quick ablation run */}
      <div style={{ borderTop: '1px solid #2a2a3e', paddingTop: 8 }}>
        <button
          onClick={() => handleRunAblation(`run_${Date.now()}`)}
          disabled={isRunning}
          style={{ ...btnStyle, width: '100%', opacity: isRunning ? 0.5 : 1 }}
        >
          {isRunning ? 'Running...' : 'Run Current Config'}
        </button>
        <button
          onClick={clearRuns}
          style={{ ...btnStyle, width: '100%', marginTop: 4, background: '#2a2a3e' }}
        >
          Clear Comparison
        </button>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  background: '#3b82f6',
  color: '#fff',
  border: 'none',
  borderRadius: 4,
  padding: '4px 10px',
  fontSize: 10,
  cursor: 'pointer',
};

const btnSmStyle: React.CSSProperties = {
  background: '#2a2a3e',
  color: '#a0a0b0',
  border: 'none',
  borderRadius: 3,
  padding: '2px 6px',
  fontSize: 9,
  cursor: 'pointer',
};
