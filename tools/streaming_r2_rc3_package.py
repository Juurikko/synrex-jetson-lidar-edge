# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Seal a complete local RC3 review archive; qualify local tag only after verification."""
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from streaming_r2_rc3 import OUT,ROOT,sha,save
from streaming_r2_rc3_qa import audit

TAG='streaming-r2-rc3-qualified'

def main():
    qa=json.loads((OUT/'evidence/VIDEO_QA_RC3.json').read_text());assert qa['status']=='PASS'
    assert sha(OUT/'media/synrex-streaming-r2-rc3.mp4')==qa['video_sha256']
    for name in ['BENCHMARK_WORDING_RC3','SOURCE_MAP_COPC_RC3','PRESERVATION_RC3','VISUAL_REVIEW_RC3']:
        gate=json.loads((OUT/f'evidence/{name}.json').read_text());assert gate['status']=='PASS',name
        if 'video_sha256' in gate:assert gate['video_sha256']==qa['video_sha256'],name
    proof=json.loads((OUT/'evidence/PROOF_REVIEW_RC3.json').read_text());assert proof['status']=='PASS'
    assert proof['renderer_sha256']==sha(ROOT/'tools/streaming_r2_rc3.py')
    assert subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)==''
    audit()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=OUT/'source'
    if source.exists():shutil.rmtree(source)
    source.mkdir()
    payload=subprocess.check_output(['git','archive',commit],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload)) as t:t.extractall(source,filter='data')
    bundle=OUT/'source.bundle'
    if bundle.exists():bundle.unlink()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    for p in (ROOT/'docs/streaming/rc3').glob('*'):shutil.copy2(p,OUT/'docs'/p.name)
    (OUT/'README.md').write_text('''# SynRex Native LiDAR Streaming R2 RC3 — local review

49 s · 1080×1350 · 30 fps · H.264/yuv420p · silent.

- Final: `media/synrex-streaming-r2-rc3.mp4`
- Complete reduced preview: `media/preview.mp4`; `media/thumbnail.png`
- Decoded film: `media/contact-sheet.jpg`, `media/decoded-contact-*.jpg`
- RC2 diagnosis: `docs/RC2_SHOT_DIAGNOSIS.md`, `media/rc2-contact-sheet.jpg`
- Before/after: `media/rc2-to-rc3-comparison.jpg`
- Accepted camera/ending proof: `media/camera-proof.mp4`, `media/ending-proof.mp4`, decoded proof contact sheets
- Exact cameras: `evidence/CAMERA_FRAMES_RC3.json`; table: `docs/CAMERA_TABLE_RC3.md`
- Final gates: `evidence/FINAL_REPORT_RC3.json`; full decode: `evidence/VIDEO_QA_RC3.json`
- Sources: `evidence/VISUAL_PROVENANCE_RC3.json`; native/COPC checks: `evidence/SOURCE_MAP_COPC_RC3.json`
- Reproduction and rights: `docs/STORYBOARD_RC3.md`; `docs/RC2_TO_RC3_CHANGES.md`
- Exact committed media source: `source/` and `source.bundle`; run/environment receipts in `evidence/`.

All original RC1 payload files (data, model, native output, model notices, accepted
benchmarks and historical failures) remain intact in `inputs/streaming-r2-rc1/`.
Original RC2 archive/video/tag and original RC1 archive were preserved and hash checked.
RC3 changes only media tooling, documentation and evidence. Diagnostic JSON p99
274.477963 ms FAIL remains in evidence and is excluded from accepted latency claims.
No new inference, training, performance qualification, physical sensor/Orin/Thor or
airborne inference qualification is claimed. QGIS/COPC source geometry, own sidecar
labels and project CRS are checked; new QGIS GUI playback is NOT_RUN. Forest COPC
and film recording use distinct preserved actual output artifacts, documented explicitly.

Local review PASS reflects agent visual judgement plus technical gates. It is not
human owner approval or calibrated-display/device playback qualification. Map chapter
uses genuine source classifications with small exact-delta annotations, not a detailed
reconstruction; repeated capture and event-count retiming are disclosed.

Scoped original tools: MIT; presentation: CC BY 4.0; IGN FRACTAL information: Etalab 2.0.
Models, architecture, dependencies and fonts retain their separate notices. See baseline
rights documents and `source/docs/streaming/STREAMING_R2_RIGHTS.md`. No blanket MIT
licence or external encoder/font binaries are bundled. Public R4 source does not make
R2 streaming public. PUBLICATION=NOT_RUN.

`MANIFEST_RC3.json` records each payload's SHA256 and size. `SHA256SUMS` additionally
hashes that manifest. The manifest/checksum self references and `owner-review/` are
excluded to avoid recursive hashing. Every compressed tar member is read and checked
against the exact expected member set, size and digest. Archive integrity and the local
qualified-tag receipt are external sidecars in `owner-review/`. The tag is created only
after the archive passes this verification. No public push/release/social post/device contact.
''')
    statuses=['RC2_PRESERVED','CAMERA_COMPOSITION','CAMERA_MOTION','3D_PARALLAX','PIXEL_BLOCK_ENDING_REMOVED','MAP_VISUAL_PROVENANCE','BENCHMARK_WORDING','VIDEO_QA','RC3_ARCHIVE_VERIFICATION','OWNER_REVIEW_READY']
    gates={name:'PASS' for name in statuses};gates['PUBLICATION']='NOT_RUN'
    save(OUT/'evidence/FINAL_REPORT_RC3.json',dict(schema='synrex.streaming_r2.rc3.final.v1',status='PASS',status_scope='Local media owner-review candidate only; archive verification is enforced before successful command exit and qualified tag creation',source_commit=commit,local_candidate_tag=TAG,baseline_tag='streaming-r2-rc2-qualified',baseline_commit='8586a6d742379310f2543fb5f7725c5fb528190f',video_sha256=qa['video_sha256'],source_bundle_sha256=sha(bundle),gates=gates,accepted_measurements=json.loads((ROOT/'evidence/streaming_r2_release_candidate.json').read_text())['accepted_measurements_preserved'],historical_json_capture_p99=dict(status='FAIL',ms=274.477963,scope='Diagnostic only; excluded from accepted latency claims'),limitations=dict(human_owner_acceptance='NOT_RUN',new_qgis_gui_playback='NOT_RUN',calibrated_mobile_display_playback='NOT_RUN',new_training='NOT_RUN',new_performance_qualification='NOT_RUN',physical_sensor='NOT_RUN',orin_thor_access='NOT_RUN',airborne_inference='NOT_RUN'),publication='NOT_RUN'))
    payloads={}
    for p in sorted(OUT.rglob('*')):
        rel=p.relative_to(OUT)
        if not p.is_file() or rel.parts[0]=='owner-review' or str(rel) in ['MANIFEST_RC3.json','SHA256SUMS']:continue
        assert not p.is_symlink()
        payloads[str(rel)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    manifest=OUT/'MANIFEST_RC3.json'
    save(manifest,dict(schema='synrex.streaming_r2.rc3.manifest.v1',source_commit=commit,files=payloads,exclusions=['Manifest self','SHA256SUMS self','owner-review/ containers and external seal receipts'],publication='NOT_RUN'))
    sums=dict(payloads);sums['MANIFEST_RC3.json']=dict(sha256=sha(manifest),bytes=manifest.stat().st_size)
    (OUT/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {p}\n" for p,r in sorted(sums.items())))
    sums['SHA256SUMS']=dict(sha256=sha(OUT/'SHA256SUMS'),bytes=(OUT/'SHA256SUMS').stat().st_size)
    dest=OUT/'owner-review';dest.mkdir(exist_ok=True);archive=dest/'synrex-streaming-r2-rc3-owner-review.tar.gz'
    print(f'Packaging {len(sums)} payloads',flush=True)
    with tarfile.open(archive,'w:gz',compresslevel=3) as t:
        for rel in sorted(sums):t.add(OUT/rel,arcname='streaming-r2-rc3/'+rel,recursive=False)
    checked=set()
    with tarfile.open(archive,'r:gz') as t:
        for m in t:
            assert m.isfile();rel=str(Path(m.name).relative_to('streaming-r2-rc3'))
            assert rel in sums and rel not in checked
            assert hashlib.file_digest(t.extractfile(m),'sha256').hexdigest()==sums[rel]['sha256'] and m.size==sums[rel]['bytes'],rel
            checked.add(rel)
    assert checked==set(sums);digest=sha(archive)
    (dest/'SHA256SUMS').write_text(f'{digest}  {archive.name}\n')
    save(dest/'ARCHIVE_INTEGRITY_RC3.json',dict(status='PASS',archive=archive.name,sha256=digest,bytes=archive.stat().st_size,compressed_payloads_verified=len(checked),manifest_sha256=sha(manifest),video_sha256=qa['video_sha256'],source_commit=commit,tag=TAG,method='Read every actual compressed payload; exact member set, byte sizes and SHA256 checked',publication='NOT_RUN'))
    assert subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)==''
    exists=subprocess.run(['git','rev-parse','--verify',f'refs/tags/{TAG}'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if exists.returncode==0:
        assert subprocess.check_output(['git','rev-parse',TAG+'^{commit}'],cwd=ROOT,text=True).strip()==commit
    else:
        subprocess.run(['git','tag','-a',TAG,commit,'-m',f'RC3 local media qualified: 49s, 1080x1350/30fps, oblique camera and decoded editorial gates. Archive SHA256 {digest}. PUBLICATION=NOT_RUN.'],cwd=ROOT,check=True)
    save(dest/'LOCAL_QUALIFICATION_RC3.json',dict(status='PASS',tag=TAG,tag_object=subprocess.check_output(['git','rev-parse',TAG],cwd=ROOT,text=True).strip(),commit=commit,archive_sha256=digest,qualified_after_archive_verification=True,clean_tree=True,publication='NOT_RUN'))
    print(json.dumps(dict(status='PASS',archive=str(archive),sha256=digest,payloads=len(checked),tag=TAG),indent=2),flush=True)

if __name__=='__main__':main()
