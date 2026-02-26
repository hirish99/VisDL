import { memo, useCallback, useState } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { NodeData } from '../../store/graphStore';
import { useGraphStore } from '../../store/graphStore';
import { uploadCSV } from '../../api/client';

function DataNodeComponent({ id, data, selected }: NodeProps & { data: NodeData }) {
  const updateParam = useGraphStore((s) => s.updateNodeParam);
  const setSelected = useGraphStore((s) => s.setSelectedNode);
  const [uploading, setUploading] = useState(false);
  const [columns, setColumns] = useState<string[]>([]);

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const res = await uploadCSV(file);
      updateParam(id, 'file_id', res.file_id);
      setColumns(res.columns);
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  }, [id, updateParam]);

  const outputs = data.definition.outputs;

  return (
    <div
      onClick={() => setSelected(id)}
      style={{
        background: '#1e1e2e',
        border: `2px solid ${selected ? '#60a5fa' : '#22c55e'}`,
        borderRadius: 8,
        minWidth: 200,
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      <div style={{
        background: '#22c55e', padding: '6px 12px',
        borderRadius: '6px 6px 0 0', color: '#fff', fontSize: 12, fontWeight: 600,
      }}>
        {data.definition.display_name}
      </div>
      <div style={{ padding: '8px 12px' }}>
        <input
          type="file"
          accept=".csv"
          onChange={handleUpload}
          style={{ fontSize: 10, color: '#ccc', width: '100%', marginBottom: 4 }}
        />
        {uploading && <div style={{ color: '#22c55e', fontSize: 10 }}>Uploading...</div>}
        {Boolean(data.params.file_id) && (
          <div style={{ color: '#a0a0b0', fontSize: 10, marginBottom: 4 }}>
            File loaded {columns.length > 0 && `(${columns.length} cols)`}
          </div>
        )}
        {columns.length > 0 && (
          <div style={{ color: '#666', fontSize: 9, marginBottom: 4 }}>
            {columns.join(', ')}
          </div>
        )}
        {outputs.map((output, i) => (
          <div key={output.name} style={{ position: 'relative', textAlign: 'right' }}>
            <span style={{ color: '#a0a0b0', fontSize: 10, marginRight: 8 }}>
              {output.name}
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={`output_${i}`}
              style={{ background: '#f472b6', width: 10, height: 10 }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export const DataNode = memo(DataNodeComponent);
