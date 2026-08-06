# Security policy

JANNUS processes patient imaging inside hospital networks. Security reports are taken
seriously and handled privately.

## Reporting a vulnerability

**Do not open a public issue for a security or privacy problem.**

Report privately via GitHub's [private vulnerability
reporting](https://github.com/ckirby04/JANNUS/security/advisories/new), or email the
maintainer listed in [CITATION.cff](CITATION.cff).

Please include what you can: affected version, reproduction steps, and impact. Expect an
acknowledgement within a few working days.

**Never include patient data, DICOM files, or real case identifiers in a report.** A
synthetic reproduction is always sufficient.

## Scope

In scope: the `jannus` package, the CLI, the optional REST API, and the published
configuration and deployment guidance.

Out of scope: vulnerabilities in third-party dependencies (report those upstream, though
we welcome a heads-up), and the research scripts under `scripts/`, which are not part of
the distributed package and are not intended for deployment.

## Deployment guidance

The core validation workflow (`validate-data`, `predict`, `evaluate`) makes no network
connections and needs no credentials.

If you deploy the optional REST API:

- Leave `AUTH_REQUIRED=true` (the default). Do not disable it on a networked host.
- Leave `CORS_ORIGINS` empty unless you have a specific browser origin to allow. Never
  set it to `*`.
- Bind to localhost and place a TLS-terminating reverse proxy in front. JANNUS does not
  provide TLS; without it, PHI transits in cleartext.
- Disable FastAPI's `/docs` and `/openapi.json` on any networked deployment.
- Put the audit database on encrypted storage.

## Known considerations

- **Checkpoint loading uses `torch.load` with `weights_only=False`.** Deserialising a
  `.pth` this way executes arbitrary code via pickle. Checkpoint paths come from your
  configuration, never from network input, so this is a supply-chain concern rather than
  a remotely-triggerable one. Verify every checkpoint against `weights.lock.json` before
  loading it — `jannus doctor` does this — and obtain weights only from the maintainer.
- **The BM25 index is a pickle** (`jannus[rag]` only), with the same profile.
- **DICOM-SEG output is fully identified by design.** See
  [PHI_AND_DEIDENTIFICATION.md](docs/PHI_AND_DEIDENTIFICATION.md).

## Fixed in 1.50

The following were live defects in 1.40 and earlier. Sites running an earlier version
should upgrade.

| Issue | Severity |
|---|---|
| Arbitrary file write via unsanitised DICOM upload filename (RCE as root in-container) | Critical |
| Authentication disabled by default (`AUTH_REQUIRED=false`) | Critical |
| `check_permission` silently permitted anonymous callers, making every admin guard a no-op | Critical |
| `/predict/{job_id}/mask` served patient masks with no authentication at all | High |
| Anonymous requests bypassed rate limiting entirely | High |
| CORS defaulted to `*` with credentials allowed | High |
| Temp directories containing patient segmentations were never cleaned up | Medium |
| DICOM private tags (a PHI reservoir) were loaded unnecessarily | Medium |
| Container ran as root | Low |

See [CHANGELOG.md](CHANGELOG.md) for details.
