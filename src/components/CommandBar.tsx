// src/components/CommandBar.tsx
import { useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { interpret, type Intent } from '../ai/intents';
import { useT } from '../i18n';

interface Props {
  onRun: (intent: Intent) => void;
  busy: boolean;
  /** With no photo open there is nothing to change. */
  disabled?: boolean;
}

/** Example prompts, held as keys so they follow the active language. */
const EXAMPLE_KEYS = [
  'command.example.removeBackground',
  'command.example.enhance',
  'command.example.changeSky',
  'command.example.cinematic',
];

/**
 * Natural-language command line.
 *
 * Interpretation is keyword-based, not semantic: without a local language model
 * it cannot understand an arbitrary sentence. When a request does not match an
 * available operation it says so, instead of pretending to have understood.
 */
export const CommandBar: React.FC<Props> = ({ onRun, busy, disabled = false }) => {
  const [text, setText] = useState('');
  const [rejectedKey, setRejectedKey] = useState<string | null>(null);
  const t = useT();

  const submit = (value: string) => {
    const phrase = value.trim();
    if (!phrase) return;

    const intent = interpret(phrase);
    if (intent.kind === 'unknown') {
      setRejectedKey(intent.reasonKey);
      return;
    }
    setRejectedKey(null);
    setText('');
    onRun(intent);
  };

  return (
    <div className="nabu-command">
      <form
        className="command-input-row"
        onSubmit={(e) => {
          e.preventDefault();
          submit(text);
        }}
      >
        <input
          className="command-input"
          placeholder={t('command.placeholder')}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setRejectedKey(null);
          }}
          disabled={busy || disabled}
        />
        <button
          className="command-send"
          type="submit"
          disabled={busy || disabled || !text.trim()}
        >
          <ArrowRight size={18} />
        </button>
      </form>

      {rejectedKey ? (
        <div className="command-rejected">{t(rejectedKey)}</div>
      ) : (
        <div className="command-examples">
          <span className="command-examples-label">{t('command.examples')}</span>
          {EXAMPLE_KEYS.map((key) => (
            <button key={key} className="command-chip" onClick={() => submit(t(key))}>
              {t(key)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
