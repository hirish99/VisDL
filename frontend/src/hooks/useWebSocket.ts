import { useEffect, useRef, useCallback } from 'react';
import { useExecutionStore } from '../store/executionStore';
import type { TrainingProgress } from '../types/graph';

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const sessionId = useExecutionStore((s) => s.sessionId);
  const addProgress = useExecutionStore((s) => s.addProgress);
  const setRunning = useExecutionStore((s) => s.setRunning);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/training/${sessionId}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'training_progress') {
        addProgress(data as TrainingProgress);
      } else if (data.type === 'execution_complete') {
        setRunning(false);
      } else if (data.type === 'execution_error') {
        setRunning(false);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected, reconnecting...');
      reconnectTimer.current = setTimeout(connect, 2000);
    };

    wsRef.current = ws;
  }, [sessionId, addProgress, setRunning]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return wsRef;
}
