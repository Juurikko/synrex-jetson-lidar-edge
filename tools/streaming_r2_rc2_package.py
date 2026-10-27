# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Seal a complete LOCAL owner-review archive; never push, publish or access devices."""
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from streaming_r2_rc2 import BASE,OUT,ROOT,sha,save

TAG='streaming-r2-rc2-qualified'

def main():
    qa=json.loads((OUT/'evidence/VIDEO_QA_RC2.json').read_text())
    assert qa['status']=='PASS'
    playback=json.loads((OUT/'evidence/REALTIME_PLAYBACK_RC2.json').read_text())
    assert playback['status']=='PASS' and playback['video_sha256']==qa['video_sha256']
    assert subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)==''
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(['git','rev-parse',TAG+'^{commit}'],cwd=ROOT,text=True).strip()==commit
    source=OUT/'source'
    if source.exists():shutil.rmtree(source)
    source.mkdir()
    payload=subprocess.check_output(['git','archive',commit],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload)) as t:t.extractall(source,filter='data')
    bundle=OUT/'source.bundle'
    if bundle.exists():bundle.unlink()
    subprocess.run(['git','bundle','create',str(bundle),TAG],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    for name in ['STORYBOARD_RC2.md','RC1_TO_RC2_CHANGES.md','LICENSE_SCOPE_RC2.json']:
        shutil.copy2(ROOT/'docs/streaming/rc2'/name,OUT/'docs'/name)
    (OUT/'README.md').write_text('''# SynRex Streaming R2 RC2 — local owner review

49 s • 1080×1350 • 30 fps • H.264 • silent autoplay.

- [Final video](media/synrex-streaming-r2-rc2.mp4)
- [Preview](media/preview.mp4) and [thumbnail](media/thumbnail.png)
- [Decoded contact sheet](media/contact-sheet.jpg)
- [QA receipt](evidence/VIDEO_QA_RC2.json) and [visual provenance](evidence/VISUAL_PROVENANCE_RC2.json)
- [Storyboard/reproduction](docs/STORYBOARD_RC2.md) and [changes](docs/RC1_TO_RC2_CHANGES.md)
- [Final gates](evidence/FINAL_REPORT_RC2.json)

The complete, unchanged accepted RC1 payload is in `inputs/streaming-r2-rc1/`, including
model notices, input data, actual model predictions, native events, original video,
benchmark receipts and historical failures. The RC1 historical JSON p99 274.478 ms
FAIL remains a diagnostic failure and is excluded from throughput/latency claims.
RC2 changes only media tooling, documentation and evidence. No neural training,
optimization, new benchmark, live physical sensor or Jetson AGX/Thor qualification.

Original native source/tools: scoped MIT; presentation: CC BY 4.0; underlying IGN
FRACTAL data: Etalab 2.0 (source updated 2025-04-05). Model, architecture, dependencies
and fonts retain separate notices. See `docs/LICENSE_SCOPE_RC2.json`, the complete
baseline notices, and `source/docs/streaming/STREAMING_R2_RIGHTS.md`. This archive is
not under a blanket MIT licence. No external renderer/encoder/font binary is bundled.

`MANIFEST_RC2.json` lists payload hashes/sizes. `SHA256SUMS` also hashes that manifest;
checksums exclude themselves and the `owner-review/` container/sidecar directory to
avoid recursive self hashes. The archive is verified by reading every compressed
member payload and comparing to the manifest. Its final archive SHA256 and integrity
receipt are in `owner-review/` beside the archive, outside its own payload.

The local candidate source commit is recorded in the final report and source bundle.
Owner-review PASS means a complete local package, not owner publication approval.
PUBLICATION=NOT_RUN. No public push, release or LinkedIn post. Stop here for owner review.
''')
    gates={
      'VIDEO_RENDER':dict(status='PASS',scope='49s 1080x1350/30fps H.264/yuv420p, 1470 decoded frames'),
      'VISUAL_DEPTH':dict(status='PASS',scope='Reviewed close point geometry; perspective near/far motion and depth witnesses'),
      'CAMERA_MOTION':dict(status='PASS',scope='Every frame has a camera; continuous low tracked views and short orbits; no static geometry holds'),
      'MAP_UPDATE_VISUALIZATION':dict(status='PASS',scope='400654 verified actual events; 24458 new cells, 376196 reobservations; accelerated overview and labeled cropped detail repeats'),
      'TECHNICAL_PROVENANCE':dict(status='PASS',scope='648 baseline archive payloads, full native-event reconstruction, model/sidecar bindings and derived-event arrays verified'),
      'VIDEO_QA':dict(status='PASS',scope='All 1470 frames independently decoded, headless full-rate playback, mobile and decoded visual review'),
      'RC1_PRESERVED':dict(status='PASS',scope='Original archive/MP4 hashes and every pre-existing tracked blob match eb4cab0; accepted model/runtime/benchmark bytes unchanged'),
      'RC2_OWNER_REVIEW':dict(status='PASS',scope='Complete LOCAL candidate; archive command verifies every compressed payload before successful exit; final archive checksum/integrity receipt in owner-review/'),
      'HISTORICAL_JSON_CAPTURE_P99':dict(status='FAIL',p99_ms=274.477963,scope='Preserved historical diagnostic; excluded from accepted performance claims'),
      'NEW_MODEL_TRAINING':dict(status='NOT_RUN'), 'NEW_PERFORMANCE_QUALIFICATION':dict(status='NOT_RUN'),
      'PHYSICAL_SENSOR':dict(status='NOT_RUN'),'AGX_THOR_ACCESS':dict(status='NOT_RUN'),
      'OWNER_PUBLICATION_APPROVAL':dict(status='NOT_RUN'),'PUBLICATION':dict(status='NOT_RUN')}
    save(OUT/'evidence/FINAL_REPORT_RC2.json',dict(schema='synrex.streaming_r2.rc2.final.v1',status='PASS',status_scope='Local media owner-review candidate only',
        source_commit=commit,local_candidate_tag=TAG,baseline_source_seal='eb4cab0',rc1_candidate_source='e33679c',
        video_sha256=qa['video_sha256'],source_bundle_sha256=sha(bundle),gates=gates,
        accepted_measurements=json.loads((ROOT/'evidence/streaming_r2_release_candidate.json').read_text())['accepted_measurements_preserved'],
        publication='NOT_RUN'))
    payloads={}
    for p in sorted(OUT.rglob('*')):
        rel=p.relative_to(OUT)
        if not p.is_file() or rel.parts[0]=='owner-review' or str(rel) in ['MANIFEST_RC2.json','SHA256SUMS']:continue
        assert not p.is_symlink()
        payloads[str(rel)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    manifest=OUT/'MANIFEST_RC2.json'
    save(manifest,dict(schema='synrex.streaming_r2.rc2.manifest.v1',source_commit=commit,files=payloads,
       exclusions=['MANIFEST_RC2.json self','SHA256SUMS self','owner-review/ container and external checksum receipts'],publication='NOT_RUN'))
    sums=dict(payloads);sums['MANIFEST_RC2.json']=dict(sha256=sha(manifest),bytes=manifest.stat().st_size)
    (OUT/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {p}\n" for p,r in sorted(sums.items())))
    sums['SHA256SUMS']=dict(sha256=sha(OUT/'SHA256SUMS'),bytes=(OUT/'SHA256SUMS').stat().st_size)
    dest=OUT/'owner-review';dest.mkdir(exist_ok=True)
    archive=dest/'synrex-streaming-r2-rc2-owner-review.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as t:
        for rel in sorted(sums):t.add(OUT/rel,arcname='streaming-r2-rc2/'+rel,recursive=False)
    checked={}
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            assert member.isfile()
            rel=str(Path(member.name).relative_to('streaming-r2-rc2'))
            assert rel in sums and rel not in checked
            stream=t.extractfile(member);h=hashlib.file_digest(stream,'sha256').hexdigest()
            assert h==sums[rel]['sha256'] and member.size==sums[rel]['bytes'],rel
            checked[rel]=h
    assert set(checked)==set(sums)
    digest=sha(archive)
    (dest/'SHA256SUMS').write_text(f'{digest}  {archive.name}\n')
    save(dest/'ARCHIVE_INTEGRITY_RC2.json',dict(status='PASS',archive=archive.name,sha256=digest,bytes=archive.stat().st_size,
        compressed_payloads_verified=len(checked),manifest_sha256=sha(manifest),video_sha256=qa['video_sha256'],
        source_commit=commit,tag=TAG,method='Read and SHA256 every actual compressed member; exact member set and byte sizes verified',publication='NOT_RUN'))
    print(json.dumps(dict(status='PASS',archive=str(archive),sha256=digest,payloads=len(checked)),indent=2))

if __name__=='__main__':main()
