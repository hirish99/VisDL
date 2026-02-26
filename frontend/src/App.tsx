import { useState } from 'react';
import { Canvas } from './components/Canvas/Canvas';
import { NodePalette } from './components/NodePalette/NodePalette';
import { PropertiesPanel } from './components/PropertiesPanel/PropertiesPanel';
import { TrainingDashboard } from './components/TrainingDashboard/TrainingDashboard';
import { AblationPanel } from './components/AblationPanel/AblationPanel';
import { Toolbar } from './components/Toolbar/Toolbar';
import { useNodeRegistry } from './hooks/useNodeRegistry';
import { useWebSocket } from './hooks/useWebSocket';

export default function App() {
  const definitions = useNodeRegistry();
  useWebSocket();

  const [showAblation, setShowAblation] = useState(false);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      background: '#0a0a12',
      color: '#e0e0f0',
    }}>
      <Toolbar />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <NodePalette definitions={definitions} />
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
          <Canvas />
          <TrainingDashboard />
        </div>
        <PropertiesPanel />
        {showAblation && <AblationPanel />}
      </div>
      {/* Ablation toggle */}
      <button
        onClick={() => setShowAblation(!showAblation)}
        style={{
          position: 'fixed',
          bottom: 12,
          right: 12,
          background: '#6366f1',
          color: '#fff',
          border: 'none',
          borderRadius: 20,
          padding: '6px 14px',
          fontSize: 11,
          cursor: 'pointer',
          fontWeight: 600,
          zIndex: 100,
        }}
      >
        {showAblation ? 'Hide Ablation' : 'Ablation'}
      </button>
    </div>
  );
}
