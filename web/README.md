# Two-site local setup

The repository contains two intentionally separate static sites:

- `web/research/` is the public-facing minimal question-and-evidence UI. It
  calls the research API only.
- `web/admin/` is the private corpus-owner upload/status UI. It calls only the
  ingestion API and requires `INGESTION_ADMIN_TOKEN`.

Serve each directory independently during local development:

```bash
python3 -m http.server 8080 --directory web/research
python3 -m http.server 8081 --directory web/admin
```

The default API bases are `http://localhost:8000` and `http://localhost:8001`.
Change the `data-api-base` attribute in the corresponding HTML when deploying
the sites separately. Configure exact allowed origins in `.env`; do not use a
wildcard origin for the private admin site.
