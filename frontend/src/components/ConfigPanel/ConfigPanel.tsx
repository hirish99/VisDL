import { useCallback, useState } from 'react';
import { useConfigStore } from '../../store/configStore';
import { useGraphStore, type NodeData } from '../../store/graphStore';
import { uploadCSV } from '../../api/client';
import type { Node } from '@xyflow/react';

export function ConfigPanel() {
  const config = useConfigStore();
  const selectedId = useGraphStore((s) => s.selectedNodeId);
  const nodes = useGraphStore((s) => s.nodes);
  const updateParam = useGraphStore((s) => s.updateNodeParam);
  const toggleDisabled = useGraphStore((s) => s.toggleNodeDisabled);

  const selectedNode: Node<NodeData> | undefined = nodes.find((n) => n.id === selectedId);

  return (
    <div style={panelStyle}>
      <DataSection />
      <TrainingSection />
      <TestDataSection />
      <ExportSection />
      {selectedNode && (
        <SelectedNodeSection
          node={selectedNode}
          updateParam={updateParam}
          toggleDisabled={toggleDisabled}
        />
      )}
    </div>
  );
}

// --- Data Section ---
function DataSection() {
  const config = useConfigStore();
  const [uploading, setUploading] = useState(false);
  const [filename, setFilename] = useState('');

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const res = await uploadCSV(file);
      config.setField('file_id', res.file_id);
      config.setField('input_columns', '');
      config.setField('target_columns', '');
      config.setAvailableColumns(res.columns);
      setFilename(res.filename);
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  }, [config]);

  const selectedInputs = config.input_columns.split(',').filter(Boolean);
  const selectedTargets = config.target_columns.split(',').filter(Boolean);

  const toggleColumn = (col: string, field: 'input_columns' | 'target_columns', current: string[]) => {
    const next = current.includes(col)
      ? current.filter((c) => c !== col)
      : [...current, col];
    config.setField(field, next.join(','));
  };

  return (
    <div style={sectionStyle}>
      <div style={sectionHeaderStyle}>Data</div>
      <div style={sectionBodyStyle}>
        <label style={labelStyle}>CSV File</label>
        <input
          type="file"
          accept=".csv"
          onChange={handleUpload}
          style={{ fontSize: 10, color: '#ccc', width: '100%', marginBottom: 4 }}
        />
        {uploading && <div style={{ color: '#22c55e', fontSize: 10 }}>Uploading...</div>}
        {filename && <div style={{ color: '#808090', fontSize: 10, marginBottom: 4 }}>{filename}</div>}

        {config.availableColumns.length > 0 && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
              <label style={{ ...labelStyle, marginBottom: 0, marginTop: 0, flex: 1 }}>Input Columns</label>
              <button onClick={() => config.setField('input_columns', config.availableColumns.join(','))} style={bulkBtnStyle}>All</button>
              <button onClick={() => config.setField('input_columns', '')} style={bulkBtnStyle}>None</button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
              {config.availableColumns.map((col) => (
                <button
                  key={`in_${col}`}
                  onClick={() => toggleColumn(col, 'input_columns', selectedInputs)}
                  style={{
                    background: selectedInputs.includes(col) ? '#22c55e' : '#1a1a2e',
                    color: selectedInputs.includes(col) ? '#fff' : '#808090',
                    border: `1px solid ${selectedInputs.includes(col) ? '#22c55e' : '#2a2a3e'}`,
                    borderRadius: 3, padding: '2px 6px', fontSize: 9, cursor: 'pointer',
                  }}
                >
                  {col}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
              <label style={{ ...labelStyle, marginBottom: 0, marginTop: 0, flex: 1 }}>Target Columns</label>
              <button onClick={() => config.setField('target_columns', config.availableColumns.join(','))} style={bulkBtnStyle}>All</button>
              <button onClick={() => config.setField('target_columns', '')} style={bulkBtnStyle}>None</button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
              {config.availableColumns.map((col) => (
                <button
                  key={`tgt_${col}`}
                  onClick={() => toggleColumn(col, 'target_columns', selectedTargets)}
                  style={{
                    background: selectedTargets.includes(col) ? '#ef4444' : '#1a1a2e',
                    color: selectedTargets.includes(col) ? '#fff' : '#808090',
                    border: `1px solid ${selectedTargets.includes(col) ? '#ef4444' : '#2a2a3e'}`,
                    borderRadius: 3, padding: '2px 6px', fontSize: 9, cursor: 'pointer',
                  }}
                >
                  {col}
                </button>
              ))}
            </div>
          </>
        )}

        <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
          <div style={{ flex: 1 }}>
            <label style={labelStyle}>Val Split</label>
            <input
              type="number"
              value={config.val_ratio}
              min={0.01} max={0.99} step={0.05}
              onChange={(e) => config.setField('val_ratio', parseFloat(e.target.value) || 0.2)}
              style={inputStyle}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={labelStyle}>Batch Size</label>
            <input
              type="number"
              value={config.batch_size}
              min={1} step={1}
              onChange={(e) => config.setField('batch_size', parseInt(e.target.value) || 32)}
              style={inputStyle}
            />
          </div>
        </div>

        <label style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={config.shuffle}
            onChange={(e) => config.setField('shuffle', e.target.checked)}
          />
          Shuffle
        </label>
      </div>
    </div>
  );
}

// --- Training Section ---
function TrainingSection() {
  const config = useConfigStore();

  return (
    <div style={sectionStyle}>
      <div style={sectionHeaderStyle}>Training</div>
      <div style={sectionBodyStyle}>
        <label style={labelStyle}>Loss Function</label>
        <select
          value={config.loss_fn}
          onChange={(e) => config.setField('loss_fn', e.target.value)}
          style={inputStyle}
        >
          <option value="MSELoss">MSE Loss</option>
          <option value="CrossEntropyLoss">Cross Entropy</option>
          <option value="L1Loss">L1 Loss (MAE)</option>
        </select>

        <label style={labelStyle}>Optimizer</label>
        <select
          value={config.optimizer}
          onChange={(e) => config.setField('optimizer', e.target.value)}
          style={inputStyle}
        >
          <option value="Adam">Adam</option>
          <option value="SGD">SGD</option>
          <option value="AdamW">AdamW</option>
        </select>

        <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
          <div style={{ flex: 1 }}>
            <label style={labelStyle}>Learning Rate</label>
            <input
              type="number"
              value={config.lr}
              min={1e-8} max={10} step={0.001}
              onChange={(e) => config.setField('lr', parseFloat(e.target.value) || 0.001)}
              style={inputStyle}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={labelStyle}>Epochs</label>
            <input
              type="number"
              value={config.epochs}
              min={1} max={10000} step={1}
              onChange={(e) => config.setField('epochs', parseInt(e.target.value) || 10)}
              style={inputStyle}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Test Data Section (collapsible) ---
function TestDataSection() {
  const config = useConfigStore();
  const [open, setOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [filename, setFilename] = useState('');

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const res = await uploadCSV(file);
      config.setField('test_file_id', res.file_id);
      config.setField('test_input_columns', '');
      config.setField('test_target_columns', '');
      config.setTestAvailableColumns(res.columns);
      setFilename(res.filename);
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  }, [config]);

  const selectedInputs = (config.test_input_columns || '').split(',').filter(Boolean);
  const selectedTargets = (config.test_target_columns || '').split(',').filter(Boolean);

  const toggleColumn = (col: string, field: 'test_input_columns' | 'test_target_columns', current: string[]) => {
    const next = current.includes(col)
      ? current.filter((c) => c !== col)
      : [...current, col];
    config.setField(field, next.join(','));
  };

  return (
    <div style={sectionStyle}>
      <div
        style={{ ...sectionHeaderStyle, cursor: 'pointer', userSelect: 'none' }}
        onClick={() => setOpen(!open)}
      >
        Test Data {open ? '\u25B4' : '\u25BE'}
        <span style={{ fontSize: 9, color: '#555', marginLeft: 4 }}>(optional)</span>
      </div>
      {open && (
        <div style={sectionBodyStyle}>
          <label style={labelStyle}>Test CSV</label>
          <input
            type="file"
            accept=".csv"
            onChange={handleUpload}
            style={{ fontSize: 10, color: '#ccc', width: '100%', marginBottom: 4 }}
          />
          {uploading && <div style={{ color: '#22c55e', fontSize: 10 }}>Uploading...</div>}
          {filename && <div style={{ color: '#808090', fontSize: 10, marginBottom: 4 }}>{filename}</div>}

          {config.testAvailableColumns.length > 0 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <label style={{ ...labelStyle, marginBottom: 0, marginTop: 0, flex: 1 }}>Input Columns</label>
                <button onClick={() => config.setField('test_input_columns', config.testAvailableColumns.join(','))} style={bulkBtnStyle}>All</button>
                <button onClick={() => config.setField('test_input_columns', '')} style={bulkBtnStyle}>None</button>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
                {config.testAvailableColumns.map((col) => (
                  <button
                    key={`tin_${col}`}
                    onClick={() => toggleColumn(col, 'test_input_columns', selectedInputs)}
                    style={{
                      background: selectedInputs.includes(col) ? '#22c55e' : '#1a1a2e',
                      color: selectedInputs.includes(col) ? '#fff' : '#808090',
                      border: `1px solid ${selectedInputs.includes(col) ? '#22c55e' : '#2a2a3e'}`,
                      borderRadius: 3, padding: '2px 6px', fontSize: 9, cursor: 'pointer',
                    }}
                  >
                    {col}
                  </button>
                ))}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <label style={{ ...labelStyle, marginBottom: 0, marginTop: 0, flex: 1 }}>Target Columns</label>
                <button onClick={() => config.setField('test_target_columns', config.testAvailableColumns.join(','))} style={bulkBtnStyle}>All</button>
                <button onClick={() => config.setField('test_target_columns', '')} style={bulkBtnStyle}>None</button>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
                {config.testAvailableColumns.map((col) => (
                  <button
                    key={`ttgt_${col}`}
                    onClick={() => toggleColumn(col, 'test_target_columns', selectedTargets)}
                    style={{
                      background: selectedTargets.includes(col) ? '#ef4444' : '#1a1a2e',
                      color: selectedTargets.includes(col) ? '#fff' : '#808090',
                      border: `1px solid ${selectedTargets.includes(col) ? '#ef4444' : '#2a2a3e'}`,
                      borderRadius: 3, padding: '2px 6px', fontSize: 9, cursor: 'pointer',
                    }}
                  >
                    {col}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// --- Export Section ---
function ExportSection() {
  const config = useConfigStore();

  return (
    <div style={sectionStyle}>
      <div style={sectionHeaderStyle}>Export</div>
      <div style={sectionBodyStyle}>
        <label style={labelStyle}>Model Name</label>
        <input
          type="text"
          value={config.export_name}
          onChange={(e) => config.setField('export_name', e.target.value)}
          placeholder="auto-generated if empty"
          style={inputStyle}
        />
      </div>
    </div>
  );
}

// --- Selected Node Section ---
function SelectedNodeSection({
  node,
  updateParam,
  toggleDisabled,
}: {
  node: Node<NodeData>;
  updateParam: (nodeId: string, key: string, value: unknown) => void;
  toggleDisabled: (nodeId: string) => void;
}) {
  const { definition, params, disabled } = node.data;
  const properties = Object.entries(definition.inputs).filter(([, s]) => !s.is_handle);

  return (
    <div style={{ ...sectionStyle, borderTop: '2px solid #6366f1' }}>
      <div style={{ ...sectionHeaderStyle, color: '#6366f1' }}>
        {definition.display_name}
        <span style={{ fontSize: 9, color: '#555', marginLeft: 6 }}>{node.id}</span>
      </div>
      <div style={sectionBodyStyle}>
        <label style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={disabled}
            onChange={() => toggleDisabled(node.id)}
          />
          Disabled (ablation)
        </label>

        {properties.map(([key, spec]) => {
          const value = params[key] ?? spec.default ?? '';
          return (
            <div key={key} style={{ marginBottom: 4 }}>
              <label style={labelStyle}>
                {key}
                {spec.required && <span style={{ color: '#ef4444' }}> *</span>}
              </label>
              {spec.choices ? (
                <select
                  value={String(value)}
                  onChange={(e) => updateParam(node.id, key, e.target.value)}
                  style={inputStyle}
                >
                  {spec.choices.map((c: unknown) => (
                    <option key={String(c)} value={String(c)}>{String(c)}</option>
                  ))}
                </select>
              ) : spec.dtype === 'BOOL' ? (
                <input
                  type="checkbox"
                  checked={Boolean(value)}
                  onChange={(e) => updateParam(node.id, key, e.target.checked)}
                />
              ) : spec.dtype === 'INT' || spec.dtype === 'FLOAT' ? (
                <input
                  type="number"
                  value={value === null || value === undefined ? '' : Number(value)}
                  min={spec.min_val ?? undefined}
                  max={spec.max_val ?? undefined}
                  step={spec.dtype === 'FLOAT' ? 0.001 : 1}
                  onChange={(e) => {
                    if (e.target.value === '') {
                      updateParam(node.id, key, null);
                    } else {
                      const v = spec.dtype === 'INT'
                        ? parseInt(e.target.value)
                        : parseFloat(e.target.value);
                      updateParam(node.id, key, isNaN(v) ? null : v);
                    }
                  }}
                  style={inputStyle}
                />
              ) : (
                <input
                  type="text"
                  value={String(value)}
                  onChange={(e) => updateParam(node.id, key, e.target.value)}
                  style={inputStyle}
                />
              )}
            </div>
          );
        })}

        {properties.length === 0 && (
          <div style={{ color: '#555', fontSize: 11, textAlign: 'center', padding: 8 }}>
            No configurable properties
          </div>
        )}
      </div>
    </div>
  );
}

// --- Styles ---
const panelStyle: React.CSSProperties = {
  width: 280,
  background: '#12121a',
  borderLeft: '1px solid #2a2a3e',
  overflow: 'auto',
  fontFamily: 'system-ui, sans-serif',
};

const sectionStyle: React.CSSProperties = {
  borderBottom: '1px solid #2a2a3e',
};

const sectionHeaderStyle: React.CSSProperties = {
  padding: '8px 12px',
  color: '#a0a0b0',
  fontSize: 11,
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  background: '#0e0e16',
};

const sectionBodyStyle: React.CSSProperties = {
  padding: '8px 12px',
};

const labelStyle: React.CSSProperties = {
  color: '#808090',
  fontSize: 10,
  display: 'block',
  marginBottom: 2,
  marginTop: 4,
};

const bulkBtnStyle: React.CSSProperties = {
  background: 'transparent',
  border: '1px solid #2a2a3e',
  borderRadius: 3,
  color: '#808090',
  fontSize: 9,
  padding: '1px 6px',
  cursor: 'pointer',
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: '#1e1e2e',
  border: '1px solid #2a2a3e',
  borderRadius: 4,
  color: '#c0c0d0',
  padding: '4px 8px',
  fontSize: 11,
  outline: 'none',
  marginBottom: 4,
};
