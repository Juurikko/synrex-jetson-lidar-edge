# Historical verification and editorial archive

This folder organizes supporting public information outside the code-first root
of the repository. **This is presentation
reorganization only, not a new qualified source release.**

- [Full historical qualification receipts](evidence/) — original JSON/CSV content,
  preserving all successes, failures and NOT_RUN states.
- [RC2/RC3/RC4 editorial work and owner-review notes](editorial/) — historical
  storyboards, draft LinkedIn material and media review documents.
- [Historical reports moved from the repository root](history/) — R0/R1 staging
  and publication reports, the historical Jetson benchmark, validation and
  reproducibility notes, rights audit and the retained earlier README.
- [Immutable, evidence-complete Streaming R2 RC4 source tag](https://github.com/Juurikko/synrex-jetson-lidar-edge/tree/streaming-r2-rc4)
- [Public Streaming R2 RC4 assets, video and audit](https://github.com/Juurikko/synrex-jetson-lidar-edge/releases/tag/streaming-r2-rc4)

The `evidence/` entry at repository root is a relative Git symlink targeting
`archive/evidence`, for existing Linux tooling and tests. For web browsing,
link directly to `archive/evidence/`.

Files have not been erased from Git history. The frozen public tag and release
archive retain their original paths and checksums. The stable runtime source,
protocol, numerical tests, kernel implementations and trained model identities
are unchanged in this maintained branch.
