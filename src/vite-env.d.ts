/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL of the Nabu service that runs, for the web version, what the
   * desktop app runs in Rust. Unset: those features say they are unavailable.
   */
  readonly VITE_NABU_SERVICE_URL?: string;
}
