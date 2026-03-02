import { create } from "zustand";

type Density = "comfortable" | "compact";

interface UiState {
  density: Density;
  followTail: boolean;
  paused: boolean;
  selectedLogId: string | null;
  selectedStep: number;
  setDensity: (density: Density) => void;
  setFollowTail: (followTail: boolean) => void;
  setPaused: (paused: boolean) => void;
  selectLog: (id: string | null) => void;
  selectStep: (step: number) => void;
}

export const useUiStore = create<UiState>((set) => ({
  density: "comfortable",
  followTail: true,
  paused: false,
  selectedLogId: null,
  selectedStep: 1,
  setDensity: (density) => set({ density }),
  setFollowTail: (followTail) => set({ followTail }),
  setPaused: (paused) => set({ paused }),
  selectLog: (selectedLogId) => set({ selectedLogId }),
  selectStep: (selectedStep) => set({ selectedStep }),
}));
