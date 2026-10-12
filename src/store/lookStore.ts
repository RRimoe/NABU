// src/store/lookStore.ts
import { create } from 'zustand';
import type { Adjustments } from '../types/document';

/** A look the user saved from their own slider positions. */
export interface UserLook {
  id: string;
  name: string;
  adjustments: Partial<Adjustments>;
}

/** Prefix that keeps saved looks apart from the built-in catalogue. */
export const USER_LOOK_PREFIX = 'user:';

const STORAGE_KEY = 'nabu.looks';

const read = (): UserLook[] => {
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '[]');
    return Array.isArray(saved)
      ? saved.filter(
          (look): look is UserLook =>
            typeof look?.id === 'string' &&
            typeof look?.name === 'string' &&
            typeof look?.adjustments === 'object'
        )
      : [];
  } catch {
    return [];
  }
};

const write = (looks: UserLook[]) => {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(looks));
  } catch {
    // Storage unavailable: the looks last for this session, which beats failing.
  }
};

interface LookState {
  looks: UserLook[];
  save: (name: string, adjustments: Partial<Adjustments>) => UserLook;
  remove: (id: string) => void;
}

/**
 * Looks the user made, kept on this computer. A layer using one carries the
 * values too (see LayerFilter.adjustments), so deleting a look or opening the
 * project elsewhere never changes a photo that already uses it.
 */
export const useLookStore = create<LookState>((set, get) => ({
  looks: read(),
  save: (name, adjustments) => {
    const look: UserLook = {
      id: `${USER_LOOK_PREFIX}${Date.now().toString(36)}`,
      name: name.trim() || 'Look',
      adjustments: { ...adjustments },
    };
    const looks = [...get().looks, look];
    write(looks);
    set({ looks });
    return look;
  },
  remove: (id) => {
    const looks = get().looks.filter((look) => look.id !== id);
    write(looks);
    set({ looks });
  },
}));
