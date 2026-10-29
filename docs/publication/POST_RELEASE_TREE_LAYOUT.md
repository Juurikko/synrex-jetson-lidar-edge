# Maintained source tree versus verified RC4 release tag

This post-release maintenance commit reorganizes the public GitHub browsing tree.
The qualified release is the immutable `streaming-r2-rc4` tag at
`619e838863a8fde013eb50c2e17e0a604d30ea1d`; its source archive,
assets and documented SHA256 values have **not** been modified.

For code, begin at `native/`, `src/`, `tests/`, `tools/` and the
public quickstart. The large receipt collection moved to `archive/evidence/`,
with a relative Linux symlink `evidence -> archive/evidence` to preserve legacy
paths for tooling. Historical LinkedIn drafts, internal review notes and
storyboards live in `archive/editorial/`.

The original RC4 release source-equivalence audit applies to the immutable
release tag. This maintained branch carries editorial path changes; **do not
represent the maintenance commit as a byte-identical 497-file reconstruction
of the private RC4 source**. Code under `native/`, `src/`, `tests/`,
`tools/` remains at the verified released blob identities.

The historical tag's README included earlier candidate language. The maintained
branch README corrects that status without rewriting the immutable tag.
