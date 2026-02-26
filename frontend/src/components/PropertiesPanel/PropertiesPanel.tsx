import { useGraphStore, type NodeData } from '../../store/graphStore';
import type { Node } from '@xyflow/react';

export function PropertiesPanel() {
  const selectedId = useGraphStore((s) => s.selectedNodeId);
  const nodes = useGraphStore((s) => s.nodes);
  const updateParam = useGraphStore((s) => s.updateNodeParam);
  const toggleDisabled = useGraphStore((s) => s.toggleNodeDisabled);

  const node: Node<NodeData> | undefined = nodes.find((n) => n.id === selectedId);
  if (!node) {
    return (
      <div style={panelStyle}>
        <div style={headerStyle}>Properties</div>
        <div style={{ color: '#555', fontSize: 12, padding: '20px 12px', textAlign: 'center' }}>
          Select a node to edit
        </div>
      </div>
    );
  }

  const { definition, params, disabled } = node.data;

  // Only show non-handle inputs (properties)
  const properties = Object.entries(definition.inputs).filter(([, s]) => !s.is_handle);

  return (
    <div style={panelStyle}>
      <div style={headerStyle}>
        {definition.display_name}
        <span style={{ fontSize: 9, color: '#666', marginLeft: 6 }}>{node.id}</span>
      </div>

      {/* Disabled toggle */}
      <div style={{ padding: '8px 12px', borderBottom: '1px solid #2a2a3e' }}>
        <label style={{ color: '#a0a0b0', fontSize: 11, display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={disabled}
            onChange={() => toggleDisabled(node.id)}
          />
          Disabled (ablation)
        </label>
      </div>

      {/* Properties */}
      {properties.map(([key, spec]) => {
        const value = params[key] ?? spec.default ?? '';
        return (
          <div key={key} style={{ padding: '6px 12px' }}>
            <label style={{ color: '#a0a0b0', fontSize: 10, display: 'block', marginBottom: 2 }}>
              {key}
              {spec.required && <span style={{ color: '#ef4444' }}> *</span>}
              <span style={{ color: '#555', marginLeft: 4 }}>({spec.dtype})</span>
            </label>

            {spec.choices ? (
              <select
                value={String(value)}
                onChange={(e) => updateParam(node.id, key, e.target.value)}
                style={inputStyle}
              >
                {spec.choices.map((c: any) => (
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
                  const v = spec.dtype === 'INT'
                    ? parseInt(e.target.value) || 0
                    : parseFloat(e.target.value) || 0;
                  updateParam(node.id, key, v);
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
        <div style={{ color: '#555', fontSize: 11, padding: '12px', textAlign: 'center' }}>
          No configurable properties
        </div>
      )}
    </div>
  );
}

const panelStyle: React.CSSProperties = {
  width: 260,
  background: '#12121a',
  borderLeft: '1px solid #2a2a3e',
  overflow: 'auto',
  fontFamily: 'system-ui, sans-serif',
};

const headerStyle: React.CSSProperties = {
  padding: '10px 12px',
  color: '#e0e0f0',
  fontSize: 13,
  fontWeight: 600,
  borderBottom: '1px solid #2a2a3e',
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
};
