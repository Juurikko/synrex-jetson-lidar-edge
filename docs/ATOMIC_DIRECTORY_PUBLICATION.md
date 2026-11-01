<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
<!-- Author and project lead: Tuomas Pietilä. AI-assisted engineering, including OpenAI Codex. -->

# Atomic package and generic-export directory publication

Package-v2/v3 packing, taxonomy repacking and generic single-file export publish
only completed, validated directories. They first attempt Linux
`renameat2(RENAME_NOREPLACE)`. On supporting filesystems the output remains an
ordinary directory. Existing output names are never replaced.

Only `EINVAL`, `ENOSYS` and `EOPNOTSUPP` from that native operation permit an
alternative: `symlinkat` atomically creates the requested output name, with a
relative target naming the hidden, completed stage in the same parent directory.
This supports filesystems that reject the rename flag but support Linux directory
symlinks. No check-then-rename sequence, overwrite, copy into a visible directory,
or cross-filesystem move is used. All other native errors fail closed. A failed
symlink operation also fails closed, including when symlinks are unsupported.
Every existing name, including an empty directory or dangling link, collides.

The helper checks that the stage is a real directory owned by the current user,
has a distinct hidden basename and shares the destination's canonical parent. It
pins the parent descriptor for the native rename and symlink operations, and
rechecks stage inode/device/type after an unsupported native operation. Callers
own the private stage and verify its contents before publication. This is not a
defence against another process with authority to mutate the same backing
directory or its contents. Filesystem permission enforcement and durability
semantics still depend on the host filesystem.

The output symlink **owns its hidden backing stage** after successful creation.
Never delete that stage as temporary debris. Keep both entries when moving,
archiving or backing up the containing tree, or use a copy tool that deliberately
dereferences the output. Moving only the symlink breaks it. Removing only the
output symlink leaves the backing directory allocated. To remove a published
bundle, identify and remove its owned same-parent backing directory as well as
the symlink; never follow an arbitrary untrusted link for cleanup. Unpublished
stages are removed on ordinary exceptions; a process crash may leave an orphan,
which requires ownership and job-state reconciliation before removal.

Files and the stage are fsynced before publication; the parent is fsynced after.
Failure of that final fsync reports an error but preserves the published output
and its backing storage. The output may already be visible in that error case;
inspect it before retrying. No power-loss guarantee beyond the filesystem's
fsync implementation is claimed.

Package admission now permits a root directory alias. The `Package` constructor
resolves its root exactly once to a canonical real directory before manifest or
member reads, and retains that canonical path for subsequent artifact reads and
source rechecks. Retargeting the caller's root alias cannot redirect an admitted
package or its owned engine capture. Final-component symlinks for manifest,
model, license and taxonomy members remain rejected by their existing checks.
Dangling root aliases and aliases to files are rejected. Canonical paths do not
protect against subsequent replacement of the real backing directory itself.

`publish_directory_noreplace` returns the actual `DirectoryPublication` method;
`directory_publication_name` produces `renameat2_noreplace` or
`relative_directory_symlink` for external receipts. A successfully published
root's symlink status and relative target are also directly observable. Package
manifests and export bundle schemas remain unchanged: publication representation
does not alter their validated content identity.

The CPU `atomic_directory_publication` test exercises the production helper
directly with narrow injected native errors. It checks the three permitted
fallback errors, denied errors, directory/file/live-link/dangling-link collisions,
concurrent publishers, complete bytes read by a separate process, stage
retention, and relocation with the containing tree. Its `native_method` reports
what the test filesystem actually supports. `immutable_engine_artifact` checks
canonical root aliases, retargeting and unchanged member-link rejection.
These tests do not qualify CUDA, TensorRT or trained-model quality.

`generic_directory_publication_filesystem` invokes the CPU export fixture with
the explicit `--publication-filesystem` option. It verifies actual v2/v3 packing,
repacking, package readback and PDRF6/7/8 preserved exports, including injected
parent-fsync errors after publication for every caller. This mode reports the
separate inotify-based late-mutation test as not run; the complete default export
fixture remains mandatory on a filesystem supporting that test. The full engine
capture fixture also requires FIFO creation. A filesystem lacking either
capability does not acquire those qualifications from the publication mode.

This behavior applies to the package and generic-export paths above; the legacy
streaming exporter retains its existing native rename-only contract. Historical
porting guidance requiring native no-replace rename support for every export
predates this scoped alternative.
