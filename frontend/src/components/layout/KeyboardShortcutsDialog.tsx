import { Keyboard, X } from "lucide-react";
import React from "react";
import { useUiStore } from "../../store/uiStore";

export const KeyboardShortcutsDialog: React.FC = () => {
  const isOpen = useUiStore((s) => s.shortcutsDialogOpen);
  const setOpen = useUiStore((s) => s.setShortcutsDialogOpen);

  if (!isOpen) return null;

  const shortcuts = [
    { key: "Space", desc: "Play / Pause simulation" },
    { key: "→", desc: "Step 1 tick" },
    { key: "Shift + →", desc: "Step 10 ticks" },
    { key: "R", desc: "Reset simulation" },
    { key: "F", desc: "Inject fire alarm" },
    { key: "B", desc: "Break car (fault injection)" },
    { key: "1 – 5", desc: "Switch navigation pages" },
    { key: "P", desc: "Toggle presentation / story mode" },
    { key: "T", desc: "Toggle dark / light theme" },
    { key: "?", desc: "Open this cheat-sheet" },
    { key: "Esc", desc: "Close dialog / exit presentation mode" },
  ];

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard Shortcuts"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4"
    >
      <div className="w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-2xl">
        <div className="mb-4 flex items-center justify-between border-b border-border pb-3">
          <div className="flex items-center gap-2">
            <Keyboard className="h-5 w-5 text-primary" />
            <h2 className="text-base font-semibold">Keyboard Shortcuts</h2>
          </div>
          <button
            onClick={() => setOpen(false)}
            aria-label="Close shortcuts dialog"
            className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-2">
          {shortcuts.map((s, idx) => (
            <div key={idx} className="flex items-center justify-between py-1 text-sm">
              <span className="text-muted-foreground">{s.desc}</span>
              <kbd className="rounded border border-border bg-muted px-2 py-0.5 font-mono text-xs font-semibold text-foreground">
                {s.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={() => setOpen(false)}
            className="rounded-lg bg-primary px-4 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};
