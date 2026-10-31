<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Captured model bytes before device initialization

`synrex::EngineArtifact` admits a package from its directory pathname using the
production CPU package reader, then retains its model bytes in private owned
storage. Its constructor accepts no caller-built `Package` or `Contract`. The
captured manifest bytes, manifest digest, parsed manifest, contract and optional
taxonomy remain privately owned and are exposed only through const views.

The model capture opens a regular file without following its final symlink, with
nonblocking open so a FIFO cannot stall admission. It accepts 1 through
2,147,483,648 bytes, reads exactly the initial size, and checks device, inode,
mode, link count, size, modification time and change time before and after the
read, including the final pathname identity. SHA-256 is computed from the owned
bytes and checked against the privately admitted manifest. A pathname or inode
write after construction cannot change those bytes. Callers must keep the
artifact alive for the lifetime of any retained byte view. This is a memory
capture, not a filesystem lock or a trusted-model signature.

The model bytes remain in host memory for the lifetime of the owning artifact.
Persistent sessions previously released their temporary serialized engine after
deserialization; they now retain this allocation. Session identity reports
`retained_model_bytes` and `artifact_bytes_ownership`. Future RSS and GPU
qualification must include this host-memory cost; no memory or performance
improvement is claimed. The maximum admitted artifact alone can retain 2 GiB.

ONNX packages remain admissible for CPU metadata/descriptor preparation and
engine building. `model_bytes()` returns that captured model; `engine_bytes()`
rejects an ONNX package. An execution backend must check this distinction before
querying or creating device resources. The existing legacy streaming guard is
unchanged: package-v3 is rejected by `PackageUse::legacy_stream`.

The TensorRT file inference and build entrypoints retain their legacy mutable
`Package` signatures. They freshly capture the directory, require the supplied
manifest bytes/digest, parsed manifest, contract specification and taxonomy to
match that admission, then use the privately captured contract and bytes. They
do not deserialize by reopening `Package::artifact()` after a device query. The
persistent legacy session captures and checks the engine in a CPU preparation
member declared before CUDA-owning members, including event/stream constructors.

The generic startup descriptor must derive its identity from the same privately
owned capture retained for eventual execution. Retaining an artifact alone does
not grant GPU permission, initialize a session, or complete the new generic
transport/session integration. Its backend must eventually deserialize that
retained artifact's `engine_bytes()` after the generic startup gate permits it.

The CPU regression uses opaque synthetic bytes; it exercises pathname
replacement, writes to the same inode, source removal, retained manifest and
taxonomy consistency, ONNX/engine distinctions, legacy admission and rejection of
symlinks, FIFOs, empty/oversized artifacts and digest mismatches. A throwing
`target_info()` sentinel records zero device queries. These tests do not establish
TensorRT deserialization, inference parity, quality or generic file/stream
interchangeability. Real GPU execution remains `NOT_RUN` for this change.
