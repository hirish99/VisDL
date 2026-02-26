import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import type { NodeData } from '../../store/graphStore';
import { useGraphStore } from '../../store/graphStore';
import { useExecutionStore } from '../../store/executionStore';

function MetricsNodeComponent({ id, data, selected }: NodeProps & { data: NodeData }) {
  const setSelected = useGraphStore((s) => s.setSelectedNode);
  const progress = useExecutionStore((s) => s.progress);

  const handleInputs = Object.entries(data.definition.inputs).filter(([, s]) => s.is_handle);

  const chartData = progress
    .filter((p) => p.type === 'training_progress')
    .map((p) => ({
      epoch: p.epoch,
      train: p.train_loss,
      val: p.val_loss,
    }));

  return (
    <div
      onClick={() => setSelected(id)}
      style={{
        background: '#1e1e2e',
        border: `2px solid ${selected ? '#60a5fa' : '#06b6d4'}`,
        borderRadius: 8,
        minWidth: 240,
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      <div style={{
        background: '#06b6d4', padding: '6px 12px',
        borderRadius: '6px 6px 0 0', color: '#fff', fontSize: 12, fontWeight: 600,
      }}>
        Metrics
      </div>
      <div style={{ padding: '8px 12px' }}>
        {handleInputs.map(([name]) => (
          <div key={name} style={{ position: 'relative', marginBottom: 4 }}>
            <Handle
              type="target"
              position={Position.Left}
              id={`input_${name}`}
              style={{ background: '#60a5fa', width: 10, height: 10 }}
            />
            <span style={{ color: '#a0a0b0', fontSize: 10, marginLeft: 8 }}>{name}</span>
          </div>
        ))}

        {chartData.length > 0 && (
          <div style={{ width: 210, height: 80, marginTop: 4 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <XAxis dataKey="epoch" hide />
                <YAxis hide />
                <Line type="monotone" dataKey="train" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="val" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {chartData.length === 0 && (
          <div style={{ color: '#555', fontSize: 10, textAlign: 'center', padding: 8 }}>
            No data yet
          </div>
        )}
      </div>
    </div>
  );
}

export const MetricsNode = memo(MetricsNodeComponent);
