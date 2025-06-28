from solidlsp.language_servers.clojure_lsp.clojure_lsp import verify_clojure_cli


def _test_clojure_cli() -> str | bool:
    try:
        verify_clojure_cli()
        return False
    except (FileNotFoundError, RuntimeError) as e:
        return str(e)


CLOJURE_CLI_FAIL = _test_clojure_cli()
