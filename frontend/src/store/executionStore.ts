import { create } from 'zustand';
import type { TrainingProgress } from '../types/graph';

export interface AblationRun {
  id: string;
  name: string;
  history: { epoch: number; train_loss: number; val_loss: number | null }[];
  finalTrainLoss: number | null;
  finalValLoss: number | null;
}

interface ExecutionState {
  isRunning: boolean;
  sessionId: string;
  executionId: string | null;
  progress: TrainingProgress[];
  results: Record<string, unknown[]>;
  errors: string[];
  ablationRuns: AblationRun[];

  setRunning: (running: boolean) => void;
  setSessionId: (id: string) => void;
  setExecutionId: (id: string | null) => void;
  addProgress: (p: TrainingProgress) => void;
  setResults: (r: Record<string, unknown[]>) => void;
  setErrors: (e: string[]) => void;
  clearProgress: () => void;
  addAblationRun: (run: AblationRun) => void;
  clearAblationRuns: () => void;
}

export const useExecutionStore = create<ExecutionState>((set) => ({
  isRunning: false,
  sessionId: `session_${Date.now()}`,
  executionId: null,
  progress: [],
  results: {},
  errors: [],
  ablationRuns: [],

  setRunning: (running) => set({ isRunning: running }),
  setSessionId: (id) => set({ sessionId: id }),
  setExecutionId: (id) => set({ executionId: id }),
  addProgress: (p) => set((s) => ({ progress: [...s.progress, p] })),
  setResults: (r) => set({ results: r }),
  setErrors: (e) => set({ errors: e }),
  clearProgress: () => set({ progress: [], errors: [] }),
  addAblationRun: (run) => set((s) => ({ ablationRuns: [...s.ablationRuns, run] })),
  clearAblationRuns: () => set({ ablationRuns: [] }),
}));
