#!/usr/bin/env python3
"""SessionStart hook: injects Akita Way operational context."""

import json


def main() -> None:
    message = (
        "Akita Way ativo: disciplina > intuicao. "
        "Planeje arquitetura e dominio antes de implementar. "
        "Para features, exija TDD (testes primeiro) e so aceite implementacao apos testes. "
        "Atualize CLAUDE.md quando arquitetura, responsabilidades ou regras de negocio mudarem. "
        "Evite operacoes destrutivas sem confirmacao explicita."
    )
    print(json.dumps({"systemMessage": message}, ensure_ascii=True))


if __name__ == "__main__":
    main()
