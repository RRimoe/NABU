# Translating Nabu

Every language is a single flat JSON file. You do not need to touch any other
file, and you do not need to know TypeScript.

## Two ways to add a language

**As a contribution.** Copy `en.json` to `<code>.json` (an
[ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) code
such as `fr`, `de`, `pt-BR`), translate the values, and open a pull request.
The build picks the file up automatically — no code change is required.

**Without building anything.** Put the same file into the `languages` folder
inside Nabu's data directory and restart the application:

| Platform | Folder |
| --- | --- |
| Windows | `%APPDATA%\com.editor.nabu\languages\` |
| macOS | `~/Library/Application Support/com.editor.nabu/languages/` |
| Linux | `~/.local/share/com.editor.nabu/languages/` |

A file placed there overrides a built-in language with the same code, which is
handy for fixing a wording without waiting for a release.

## Rules

- **Only translate the values**, never the keys on the left.
- **Keep the placeholders.** `{size}`, `{model}`, `{name}` are replaced at
  runtime. Move them wherever the sentence needs them, but do not rename or
  drop them — `check-locales` fails if you do.
- **Partial is fine.** Any key you leave out falls back to English, so you can
  ship a translation before it is complete.
- `$meta.name` is the language name shown to users; write it in that language
  (`Français`, not `French`).

## Checking your work

```
npm run check:locales
```

It reports missing keys as warnings and empty strings or broken placeholders as
errors.
