#!/bin/sh
# Pornește pagina web pe Raspberry Pi.  Pagina: http://<ip-ul-pi>:8000  (ip: hostname -I)
cd "$(dirname "$0")"
exec python3 -m dofbot.web "$@"
