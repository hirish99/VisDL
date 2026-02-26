import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { NodeData } from '../../store/graphStore';
import { useGraphStore } from '../../store/graphStore';
import { useExecutionStore } from '../../store/executionStore';

function TrainingNodeComponent({ id, data, selected }: NodeProps & { data: NodeData }) {
  const setSelected = useGraphStore((s) => s.setSelectedNode);
  const progress = useExecutionStore((s) => s.progress);
  const isRunning = useExecutionStore((s) => s.isRunning);

  const latest = progress.length > 0 ? progress[progress.length - 1] : null;
  const pct = latest ? (latest.epoch / latest.total_epochs) * 100 : 0;

  const handleInputs = Object.entries(data.definition.inputs).filter(([, s]) => s.is_handle);
  const outputs = data.definition.outputs;

  return (
    <div
      onClick={() => setSelected(id)}
      style={{
        background: '#1e1e2e',
        border: `2px solid ${selected ? '#60a5fa' : '#3b82f6'}`,
        borderRadius: 8,
        minWidth: 200,
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      <div style={{
        background: '#3b82f6', padding: '6px 12px',
        borderRadius: '6px 6px 0 0', color: '#fff', fontSize: 12, fontWeight: 600,
      }}>
        Training Loop
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

        {/* Progress bar */}
        {isRunning && (
          <div style={{ marginTop: 6 }}>
            <div style={{
              background: '#2a2a3e', borderRadius: 4, height: 8, overflow: 'hidden',
            }}>
              <div style={{
                background: '#3b82f6', height: '100%', width: `${pct}%`,
                transition: 'width 0.3s',
              }} />
            </div>
            {latest && (
              <div style={{ color: '#a0a0b0', fontSize: 9, marginTop: 2 }}>
                Epoch {latest.epoch}/{latest.total_epochs} | Loss: {latest.train_loss.toFixed(4)}
              </div>
            )}
          </div>
        )}

        <div style={{ color: '#808090', fontSize: 10, marginTop: 4 }}>
          epochs: {String(data.params.epochs ?? 10)}
        </div>

        {outputs.map((output, i) => (
          <div key={output.name} style={{ position: 'relative', textAlign: 'right', marginTop: 4 }}>
            <span style={{ color: '#a0a0b0', fontSize: 10, marginRight: 8 }}>{output.name}</span>
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

export const TrainingNode = memo(TrainingNodeComponent);
