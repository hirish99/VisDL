import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { useExecutionStore } from '../../store/executionStore';

export function TrainingDashboard() {
  const progress = useExecutionStore((s) => s.progress);
  const isRunning = useExecutionStore((s) => s.isRunning);
  const errors = useExecutionStore((s) => s.errors);
  const ablationRuns = useExecutionStore((s) => s.ablationRuns);

  const chartData = progress
    .filter((p) => p.type === 'training_progress')
    .map((p) => ({
      epoch: p.epoch,
      'Train Loss': p.train_loss,
      'Val Loss': p.val_loss,
    }));

  // Merge ablation runs into comparison data
  const comparisonData: Record<string, unknown>[] = [];
  if (ablationRuns.length > 0) {
    const maxEpochs = Math.max(...ablationRuns.map((r) => r.history.length));
    for (let i = 0; i < maxEpochs; i++) {
      const point: Record<string, unknown> = { epoch: i + 1 };
      for (const run of ablationRuns) {
        if (i < run.history.length) {
          point[`${run.name} (train)`] = run.history[i].train_loss;
          if (run.history[i].val_loss !== null) {
            point[`${run.name} (val)`] = run.history[i].val_loss;
          }
        }
      }
      comparisonData.push(point);
    }
  }

  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#06b6d4'];

  return (
    <div style={{
      background: '#12121a',
      borderTop: '1px solid #2a2a3e',
      padding: 12,
      fontFamily: 'system-ui, sans-serif',
      maxHeight: 300,
      overflow: 'auto',
    }}>
      <div style={{
        color: '#808090', fontSize: 11, fontWeight: 600,
        textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span>Training Dashboard</span>
        {isRunning && <span style={{ color: '#3b82f6', fontSize: 10 }}>Running...</span>}
      </div>

      {errors.length > 0 && (
        <div style={{ color: '#ef4444', fontSize: 11, marginBottom: 8 }}>
          {errors.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}

      {/* Current run chart */}
      {chartData.length > 0 && (
        <div style={{ height: 180 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
              <XAxis dataKey="epoch" stroke="#555" fontSize={10} />
              <YAxis stroke="#555" fontSize={10} />
              <Tooltip
                contentStyle={{ background: '#1e1e2e', border: '1px solid #2a2a3e', fontSize: 11 }}
              />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="Train Loss" stroke="#3b82f6" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="Val Loss" stroke="#f59e0b" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Ablation comparison chart */}
      {comparisonData.length > 0 && (
        <>
          <div style={{ color: '#808090', fontSize: 10, fontWeight: 600, marginTop: 12, marginBottom: 4 }}>
            ABLATION COMPARISON
          </div>
          <div style={{ height: 180 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                <XAxis dataKey="epoch" stroke="#555" fontSize={10} />
                <YAxis stroke="#555" fontSize={10} />
                <Tooltip
                  contentStyle={{ background: '#1e1e2e', border: '1px solid #2a2a3e', fontSize: 11 }}
                />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                {ablationRuns.flatMap((run, ri) => [
                  <Line
                    key={`${run.name}-train`}
                    type="monotone"
                    dataKey={`${run.name} (train)`}
                    stroke={colors[ri % colors.length]}
                    dot={false}
                    strokeWidth={2}
                  />,
                  <Line
                    key={`${run.name}-val`}
                    type="monotone"
                    dataKey={`${run.name} (val)`}
                    stroke={colors[ri % colors.length]}
                    dot={false}
                    strokeWidth={1}
                    strokeDasharray="5 5"
                  />,
                ])}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Metrics table */}
          <table style={{ width: '100%', marginTop: 8, fontSize: 10, color: '#a0a0b0', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #2a2a3e' }}>
                <th style={{ textAlign: 'left', padding: '4px 8px' }}>Run</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>Final Train Loss</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>Final Val Loss</th>
              </tr>
            </thead>
            <tbody>
              {ablationRuns.map((run) => (
                <tr key={run.id} style={{ borderBottom: '1px solid #1e1e2e' }}>
                  <td style={{ padding: '4px 8px' }}>{run.name}</td>
                  <td style={{ textAlign: 'right', padding: '4px 8px' }}>
                    {run.finalTrainLoss?.toFixed(6) ?? '—'}
                  </td>
                  <td style={{ textAlign: 'right', padding: '4px 8px' }}>
                    {run.finalValLoss?.toFixed(6) ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {chartData.length === 0 && comparisonData.length === 0 && (
        <div style={{ color: '#444', fontSize: 11, textAlign: 'center', padding: 20 }}>
          Execute a graph to see training progress
        </div>
      )}
    </div>
  );
}
