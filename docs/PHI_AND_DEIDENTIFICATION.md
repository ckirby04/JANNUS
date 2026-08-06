# PHI handling and de-identification

JANNUS runs inside hospitals on identifiable imaging. This document states plainly what
the software does and does not do about that, so your privacy office can assess it
accurately.

**Summary: JANNUS does not de-identify your data. De-identification is your
responsibility and must happen before JANNUS sees the imaging.**

---

## What the core workflow does

`jannus validate-data`, `jannus predict` and `jannus evaluate`:

- Make **no outbound network connections**. Your imaging never leaves your network.
- Write outputs only under the `--output` directory you specify.
- Pseudonymise case identifiers in all logs and reports with a salted SHA-256 digest.
- Redact identifier-shaped text (long digit runs, DICOM UIDs, date-like strings, API
  keys) from log output via a filter, so a call site cannot leak one by forgetting.

The Markdown report, `results.json`, `provenance.json` and the validation report are
designed to be safe to email to a coordinating site. Case identifiers appear in them only
as `case-<hash>` tokens.

---

## What JANNUS does *not* do

- **It does not de-identify DICOM.** There is no PS3.15 confidentiality profile
  implementation, no tag scrubbing, and no `PatientID` consistency check.
- **It does not de-identify NIfTI.** Converting DICOM to NIfTI drops the DICOM patient
  module as a side effect of the format — that is incidental, not a control. NIfTI headers
  can still carry a description field, and **the file and directory names you choose are
  preserved verbatim**.
- **It does not defacing or skull-strip.** A T1 volume with the face intact is
  re-identifiable by surface rendering. If your protocol requires defacing, do it
  upstream.

> **Correction to earlier documentation.** `docs/risk_analysis.md` in v1.40 listed
> mitigations M-013 ("DICOM handler validates PatientID consistency"), M-021 ("No PHI
> stored in system logs — only anonymized case IDs"), M-022 ("Temporary file cleanup")
> and M-023 ("HTTPS/TLS encryption for all API communication") as implemented. M-013 and
> M-023 were not implemented at all; M-021 and M-022 were partially implemented. v1.50
> implements the log pseudonymisation and the temp-file cleanup, and the risk file has
> been corrected. If an earlier version of that document accompanied an IRB or FDA
> submission, the discrepancy should be disclosed.

---

## Case identifiers

**The case directory name becomes the case identifier.** If you name directories by MRN,
your MRN is the identifier, and it will appear in:

- output mask filenames (`{case_id}_seg.nii.gz`)
- the API database, if you run the optional service
- the PDF report header, if you generate one

JANNUS pseudonymises it in logs and returned reports, but the correct control is not to
put an identifier in a folder name. **Use a study ID and keep the linking key
separately**, under your existing PHI controls.

### The log salt

Set `JANNUS_LOG_SALT` to a site-specific secret so pseudonyms cannot be correlated across
sites:

```bash
export JANNUS_LOG_SALT="$(openssl rand -hex 32)"
```

Keep it stable — changing it renames every case in future reports. Store it with your
linking key, not in the repository.

---

## The optional components

These are **not** part of the validation workflow and are not installed by default.

### The REST API (`jannus[api]`)

If you deploy it, note that:

- Authentication is required by default (`AUTH_REQUIRED=true`). Do not disable it on a
  networked host.
- It binds localhost by default. Exposing it requires a reverse proxy that terminates
  **TLS** — JANNUS does not provide TLS, and PHI would otherwise transit in cleartext.
- The audit database stores case identifiers. Place it on encrypted storage covered by
  your PHI controls (`DATABASE_PATH`).
- FastAPI's interactive docs at `/docs` are unauthenticated and expose the endpoint map.
  Disable them on any networked deployment.

### DICOM-SEG export

**Every DICOM-SEG object JANNUS emits is fully identified.** A valid, PACS-linkable SEG
must carry the source series' Patient and Study modules, so `PatientName`, `PatientID`,
`PatientBirthDate`, `StudyInstanceUID` and `AccessionNumber` are copied from the source
images. That is correct DICOM behaviour, not a defect — but it means SEG output is PHI and
must be handled as such. There is no de-identified export mode.

### Literature retrieval (`jannus[rag]`)

**This extra can transmit case context to a third party.** Report generation on the
`--use_openai` / `--use_claude` path sends the case identifier and primary cancer type —
an identifier bound to a diagnosis — to OpenAI or Anthropic.

Do not enable it inside a hospital network without a Business Associate Agreement with the
provider and privacy-office review. The local report path (`generate_report_local`) makes
no such call and is what the API server uses.

Also note: ChromaDB enables anonymised usage telemetry by default, and Gradio pings
`api.gradio.app` unless analytics are disabled. Neither transmits PHI, but both are
unannounced outbound connections from your network.

---

## Recommended site checklist

- [ ] Imaging de-identified per institutional policy before JANNUS sees it.
- [ ] Case directories named by study ID, not MRN or accession number.
- [ ] Linking key stored separately, under existing PHI controls.
- [ ] `JANNUS_LOG_SALT` set to a site-specific secret.
- [ ] Output directory on storage covered by your PHI controls.
- [ ] Defacing applied if your protocol requires it.
- [ ] `jannus[rag]` **not** installed, unless separately reviewed and approved.
- [ ] Reports reviewed once before return — they are designed to be PHI-free, but a
      one-time check costs nothing.

---

## Reporting a privacy issue

Please report suspected PHI leakage privately rather than in a public issue — see
[SECURITY.md](../SECURITY.md).
