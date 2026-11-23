# Security Policy

## Supported release

Only the latest signed client delivery tag is supported. Research tags and
uncommitted worktrees are not production releases.

## Reporting

Report suspected vulnerabilities privately to the project owner and the
client's nominated security contact. Do not include order data, credentials,
private model prompts, or customer identifiers in ordinary issue trackers.

## Deployment requirements

- Production mode must not start without an API service credential.
- Terminate TLS and user authentication at the client's approved gateway.
- Use a dedicated non-administrator service account.
- Restrict write access to the configured runtime and log directories.
- Keep `.env`, certificates, API keys, backups, and database files out of Git.
- Treat uploaded orders, manifests, model context, and plans as commercially
  sensitive records.
- Rotate credentials and test restoration of encrypted backups regularly.

## Dependency response

The CI pipeline audits locked dependencies. A critical exploitable finding is
a release blocker unless a documented risk acceptance and compensating control
are approved by the client.

### Temporary ChromaDB exception

`chromadb==1.5.9` is affected by CVE-2026-45829 / PYSEC-2026-311 and no patched
PyPI release exists as of 2026-08-18. The vulnerable path is the Chroma HTTP
server collection-creation endpoint accepting a remote model repository with
`trust_remote_code`. This project uses only the embedded local persistent
client against the bundled, read-only experiment index; it does not start,
publish, proxy, or depend on a Chroma HTTP server and does not accept an
embedding model reference from API/UI input.

Consequently, production deployment is prohibited from running `chroma run`,
exposing a Chroma port, enabling `trust_remote_code`, or replacing the bundled
index/model from an untrusted source. CI explicitly identifies this single
temporary exception while continuing to fail on every other known finding.
Upgrade and remove the exception immediately when an upstream patched release
is available. Any change to the local-only architecture invalidates this risk
acceptance and blocks release.
