// src/components/CommandBar.tsx
import { useEffect, useState } from 'react';
import { ArrowRight, Sparkles } from 'lucide-react';
import { route, routerAvailable } from '../ai/router';
import { useT } from '../i18n';

interface Props {
  busy: boolean;
  /** With no photo open there is nothing to change. */
  disabled?: boolean;
}

/** Example prompts, held as keys so they follow the active language. */
const EXAMPLE_KEYS = [
  'command.example.removeBackground',
  'command.example.enhance',
  'command.example.brighter',
  'command.example.warmer',
];

/**
 * Natural-language command line.
 *
 * When the language model is installed it understands free phrasing in any
 * language; otherwise a keyword matcher handles the common requests. Which one
 * answered is shown, because "it understood me" and "it recognised a phrase I
 * happened to get right" are very different promises.
 */
export const CommandBar: React.FC<Props> = ({ busy, disabled = false }) => {
  const [text, setText] = useState('');
  const [notice, setNotice] = useState<{ key: string; params?: Record<string, string | number> } | null>(null);
  const [smart, setSmart] = useState(false);
  const t = useT();

  useEffect(() => {
    void routerAvailable().then(setSmart);
  }, []);

  const submit = async (value: string) => {
    const phrase = value.trim();
    if (!phrase) return;

    setNotice(null);
    const outcome = await route(phrase);
    if (outcome.handled) setText('');
    setNotice({ key: outcome.messageKey, params: outcome.params });
  };

  return (
    <div className="nabu-command">
      <form
        className="command-input-row"
        onSubmit={(e) => {
          e.preventDefault();
          void submit(text);
        }}
      >
        {smart && (
          <span className="command-smart" title={t('command.smart.hint')}>
            <Sparkles size={14} />
          </span>
        )}

        <input
          className="command-input"
          placeholder={t('command.placeholder')}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setNotice(null);
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

      {notice ? (
        <div className="command-rejected">{t(notice.key, notice.params)}</div>
      ) : (
        <div className="command-examples">
          <span className="command-examples-label">{t('command.examples')}</span>
          {EXAMPLE_KEYS.map((key) => (
            <button
              key={key}
              className="command-chip"
              onClick={() => void submit(t(key))}
              disabled={busy || disabled}
            >
              {t(key)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
