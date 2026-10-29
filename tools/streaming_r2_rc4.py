# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""RC4 subtraction edit. No native runtime, model, prediction or prior asset writes.

Use the qualified RC3 offline media venv. Stages: freeze, render, qa, package.
Package requires a separately recorded visual review bound to the final video hash.
"""
import argparse
import hashlib
import io
import json
import math
import shutil
import subprocess
import tarfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/streaming-r2-review/streaming-r2-rc4'
RC3 = ROOT.parent / 'streaming-r2-rc3-worktree/build/streaming-r2-review/streaming-r2-rc3'
REF = ROOT.parent / 'streaming-r2-rc4-reference'
VIDEO = OUT / 'media/synrex-streaming-r2-rc4.mp4'
FF = imageio_ffmpeg.get_ffmpeg_exe()
TAG = 'streaming-r2-rc4-final-cut-qualified'
BASELINE = '8e78417e732968d06ff0b9e63f45c394fe03837d'
RC3_SHA = 'acce2565328a8763b4ce8c44a697ff891be407144a65b1bf5e596bf89298e40a'
EDL = [(0, 840), (1200, 1470)]  # half-open source frame intervals, 30 fps


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def run(args, log):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (OUT / 'evidence' / log).write_bytes(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-2000:]
    return result


def freeze():
    assert git('rev-parse', 'streaming-r2-rc3-qualified^{commit}') == BASELINE
    assert git('remote', 'get-url', '--push', 'origin') == 'DISABLED'
    manifest = json.loads((RC3 / 'MANIFEST_RC3.json').read_text())
    for rel, row in manifest['files'].items():
        p = RC3 / rel
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], rel
    for line in (RC3 / 'SHA256SUMS').read_text().splitlines():
        digest, rel = line.split('  ', 1)
        assert sha(RC3 / rel) == digest, rel
    protected = {str(p): sha(p) for parent in [RC3, REF] for p in sorted(parent.rglob('*')) if p.is_file()}
    # Recheck original baseline identities carried by qualified RC3.
    inherited = json.loads((RC3 / 'evidence/PRESERVATION_RC3.json').read_text())
    for path, digest in inherited['protected_sha256'].items():
        assert sha(path) == digest, path
        protected[path] = digest
    assert sha(RC3 / 'media/synrex-streaming-r2-rc3.mp4') == RC3_SHA
    assert sha(REF / 'media/synrex-streaming-r2-rc3.mp4') == RC3_SHA
    receipt = OUT / 'evidence/BASELINE_FREEZE_RC4.json'
    value = dict(status='PASS', baseline_commit=BASELINE, rc3_manifest_files_checked=len(manifest['files']),
                 protected_sha256=protected, tags=git('show-ref', '--tags'),
                 public_main_refs=git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/remotes', 'refs/heads/main'),
                 push_url=git('remote', 'get-url', '--push', 'origin'))
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        assert previous == value
    else:
        save(receipt, value)
    print(f'Preserved {len(protected)} files; {len(manifest["files"])} original RC3 manifest entries verified', flush=True)


def preserved():
    frozen = json.loads((OUT / 'evidence/BASELINE_FREEZE_RC4.json').read_text())
    for path, digest in frozen['protected_sha256'].items():
        assert sha(path) == digest, path
    current_tags = '\n'.join(line for line in git('show-ref', '--tags').splitlines() if not line.endswith('/' + TAG))
    assert current_tags == frozen['tags']
    assert git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/remotes', 'refs/heads/main') == frozen['public_main_refs']
    assert git('remote', 'get-url', '--push', 'origin') == 'DISABLED'
    # No baseline tracked file may change; all RC4 contributions are additions.
    diff = git('diff', '--name-status', BASELINE).splitlines()
    assert all(line.startswith('A\t') for line in diff), diff
    return dict(status='PASS', protected_files=len(frozen['protected_sha256']),
                baseline_commit=BASELINE, all_baseline_tracked_bytes_unchanged=True,
                previous_tags_and_public_main_unchanged=True, original_payloads_models_and_measurements='SHA256 verified',
                push_url='DISABLED', publication='NOT_RUN')


def render():
    assert sha(RC3 / 'media/synrex-streaming-r2-rc3.mp4') == RC3_SHA
    graph = '[0:v]split=2[a][b];[a]trim=start_frame=0:end_frame=840,setpts=PTS-STARTPTS[x];[b]trim=start_frame=1200:end_frame=1470,setpts=PTS-STARTPTS[y];[x][y]concat=n=2:v=1:a=0[out]'
    run([FF, '-v', 'error', '-y', '-i', str(RC3 / 'media/synrex-streaming-r2-rc3.mp4'),
         '-filter_complex', graph, '-map', '[out]', '-an', '-c:v', 'libx264', '-preset', 'slow',
         '-crf', '16', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-r', '30', '-movflags', '+faststart', str(VIDEO)], 'render.log')
    run([FF, '-v', 'error', '-y', '-i', str(VIDEO), '-vf', 'scale=432:540', '-c:v', 'libx264',
         '-crf', '21', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart',
         str(OUT / 'media/preview.mp4')], 'preview-render.log')
    save(OUT / 'evidence/RENDER_RC4.json', dict(status='PASS', method='Two frame-accurate trim ranges and hard concat; single H.264 generation from qualified RC3',
         filter_complex=graph, source_sha256=RC3_SHA, video_sha256=sha(VIDEO), source_frames=EDL,
         output_frames=1110, seconds=37, audio='Source silent; output silent', encoder=FF,
         encoder_sha256=sha(FF), tool_sha256=sha(__file__), added_visual_effects=[], geometry_or_prediction_changes=False))


def sheet(paths, dest, cols=4, width=320):
    height = round(width * 1.25)
    im = Image.new('RGB', (cols * width, math.ceil(len(paths) / cols) * (height + 25)), '#07101c')
    d = ImageDraw.Draw(im)
    for n, path in enumerate(paths):
        x, y = n % cols * width, n // cols * (height + 25)
        im.paste(Image.open(path).resize((width, height), Image.Resampling.LANCZOS), (x, y + 25))
        d.text((x + 5, y + 5), path.stem, fill='white')
    im.save(dest, quality=94)


def frames(path, label, expected, transitions):
    dest = OUT / 'media' / (label + '-decoded')
    dest.mkdir(exist_ok=True)
    samples = set(range(0, expected, 30)) | {expected - 1}
    for cut in transitions:
        samples.update(cut + offset for offset in [-15, -6, -2, -1, 0, 1, 2, 6, 15] if 0 <= cut + offset < expected)
    proc = subprocess.Popen([FF, '-v', 'error', '-xerror', '-i', str(path), '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                            stdout=subprocess.PIPE, stderr=(OUT / 'evidence' / (label + '-full-decode.log')).open('wb'))
    metrics, hashes, paths = [], [], []
    prev = None
    n = 0
    size = 1080 * 1350 * 3
    while True:
        data = bytearray()
        while len(data) < size:
            block = proc.stdout.read(size - len(data))
            if not block:
                break
            data.extend(block)
        if not data:
            break
        assert len(data) == size
        arr = np.frombuffer(data, np.uint8).reshape(1350, 1080, 3)
        # BT.709 luminance proxy on regular 3px grid, same grid for both films.
        grid = arr[::3, ::3].astype(np.float32)
        lum = grid @ np.array([.2126, .7152, .0722], np.float32)
        roi = grid[115:360, 13:347]
        bright = (roi.min(axis=2) > 170) | ((roi[:, :, 0] < 150) & (roi[:, :, 1] > 170) & (roi[:, :, 2] > 150))
        row = dict(frame=n, seconds=n/30, mean_luminance=float(lum.mean()), bright_cyan_roi_fraction=float(bright.mean()))
        if prev is not None:
            diff = lum - prev
            row.update(mean_luminance_change=float(diff.mean()), luminance_mae=float(np.abs(diff).mean()),
                       fraction_luminance_change_over_20=float((np.abs(diff)>20).mean()))
        metrics.append(row)
        hashes.append(hashlib.sha256(data).hexdigest())
        prev = lum
        if n in samples:
            p = dest / f'{label}-{n/30:06.2f}s-f{n:04d}.jpg'
            img = Image.fromarray(arr)
            img.save(p, quality=94)
            paths.append(p)
            if label == 'rc4' and n == 0:
                img.save(OUT / 'media/thumbnail.png')
        n += 1
    assert proc.wait() == 0 and n == expected, (label, n)
    assert len(set(hashes)) == n, 'Unexpected exact repeated frame'
    save(OUT / 'evidence' / (label.upper() + '_FRAME_LUMINANCE_RC4.json'), metrics)
    for cut in transitions:
        selected = [p for p in paths if abs(int(p.stem.split('-f')[-1]) - cut) <= 15]
        sheet(selected, OUT / 'media' / f'{label}-transition-f{cut:04d}.jpg', 3, 300)
    return metrics, paths


def qa():
    rc3cuts = [150,270,390,510,840,1020,1200,1272,1350]
    rc4cuts = [150,270,390,510,840,912,990]
    a, ap = frames(RC3 / 'media/synrex-streaming-r2-rc3.mp4', 'rc3', 1470, rc3cuts)
    b, bp = frames(VIDEO, 'rc4', 1110, rc4cuts)
    for label, paths, times in [('rc3', ap, [0,3,5,8,9,12,13,16,17,22,27,28,30,33,34,37,39,40,43,45,48]),
                                ('rc4', bp, [0,3,5,8,9,12,13,16,17,22,27,28,30,31,32,33,35,36])]:
        selected = [min(paths, key=lambda p: abs(int(p.stem.split('-f')[-1])/30-t)) for t in times]
        sheet(selected, OUT / 'media' / ('contact-sheet.jpg' if label=='rc4' else 'rc3-contact-sheet.jpg'))
    mobile_times = [0,5,9,13,17,22,27,28,30,31,32,33,36]
    mobile = []
    for t in mobile_times:
        src = min(bp, key=lambda p: abs(int(p.stem.split('-f')[-1])/30-t))
        p = OUT / 'media' / f'mobile-{t:02d}s.png'
        Image.open(src).resize((432,540), Image.Resampling.LANCZOS).save(p)
        mobile.append(p)
    sheet(mobile, OUT / 'media/mobile-readability.jpg', 3, 432)
    # Measure re-encode fidelity on EVERY retained frame via FFmpeg SSIM.
    graph = '[0:v]split=2[a][b];[a]trim=start_frame=0:end_frame=840,setpts=PTS-STARTPTS[x];[b]trim=start_frame=1200:end_frame=1470,setpts=PTS-STARTPTS[y];[x][y]concat=n=2:v=1:a=0[ref];[ref][1:v]ssim=stats_file=ssim-frames.log'
    # Relative log path resolved to this isolated output evidence directory.
    result = subprocess.run([FF,'-hide_banner','-i',str(RC3/'media/synrex-streaming-r2-rc3.mp4'),'-i',str(VIDEO),'-lavfi',graph,'-an','-f','null','-'],cwd=OUT/'evidence',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (OUT/'evidence/ssim-summary.log').write_bytes(result.stdout+result.stderr)
    assert result.returncode == 0
    vals = [float(line.split('All:')[1].split()[0]) for line in (OUT/'evidence/ssim-frames.log').read_text().splitlines()]
    assert len(vals) == 1110 and min(vals) > .97, min(vals)
    run([FF,'-v','error','-xerror','-i',str(OUT/'media/preview.mp4'),'-f','null','-'],'preview-decode.log')
    info=run([FF,'-hide_banner','-i',str(VIDEO),'-map','0:v:0','-c','copy','-f','null','-'],'container-info.log').stderr.decode()
    assert all(s in info for s in ['h264','1080x1350','30 fps','yuv420p','00:00:37.00']) and 'Audio:' not in info
    cams=json.loads((RC3/'evidence/CAMERA_FRAMES_RC3.json').read_text())
    retained=[dict(cams[k], output_frame=n, output_time=n/30, rc3_source_frame=k) for n,k in enumerate(list(range(840))+list(range(1200,1470)))]
    assert len(retained)==1110 and not any('map_event_stop' in c for c in retained)
    save(OUT/'evidence/CAMERA_FRAMES_RC4.json',retained)
    def stats(rows,cuts):
        continuous=[r for r in rows[1:] if r['frame'] not in cuts]
        unusual=[r for r in rows[1:] if abs(r['mean_luminance_change'])>2 or r['fraction_luminance_change_over_20']>.1]
        return dict(max_abs_continuous_mean_change=max(abs(r['mean_luminance_change']) for r in continuous),
                    max_continuous_fraction_over_20=max(r['fraction_luminance_change_over_20'] for r in continuous),
                    unusual_pairs=unusual, cut_frames=cuts, thresholds='Screening only: >2/255 mean change or >10% pixels changing >20/255; NOT a perceptual flicker certification')
    save(OUT/'evidence/PRESERVATION_RC4.json',preserved())
    save(OUT/'evidence/VIDEO_QA_RC4.json',dict(status='AWAITING_VISUAL_REVIEW',video_sha256=sha(VIDEO),
        decoded_rc3_frames=len(a),decoded_rc4_frames=len(b),width=1080,height=1350,fps=30,seconds=37,codec='H.264 High',pixel_format='yuv420p',
        all_native_frames_decode='PASS',preview_decode='PASS',audio='Silent original and final',
        retained_frame_ssim_min=min(vals),retained_frame_ssim_mean=sum(vals)/len(vals),
        luminance_rc3=stats(a,rc3cuts),luminance_rc4=stats(b,rc4cuts),
        map_replay_markers_in_retained_camera_metadata=0,retained_source_ranges=EDL,removed_source_frames=[840,1200],
        mobile_review='AWAITING_VISUAL_REVIEW',publication='NOT_RUN'))
    original=json.loads((RC3/'evidence/VISUAL_PROVENANCE_RC3.json').read_text())
    save(OUT/'evidence/VISUAL_PROVENANCE_RC4.json',dict(status='PASS',schema='synrex.streaming_r2.rc4.provenance.v1',
        video_sha256=sha(VIDEO),source_video_sha256=RC3_SHA,source_commit=BASELINE,
        rc3_provenance_sha256=sha(RC3/'evidence/VISUAL_PROVENANCE_RC3.json'),
        retained_rc3_frames=EDL,removed_rc3_frames=[840,1200],map_shots_in_final=0,
        map_annotation_overlays_in_final=0,new_rendered_geometry=False,ground_truth_used_as_prediction=False,
        palette='Original seven-class RC3 palette unchanged; same actual frozen native U-Net output',
        comparison='Approved synchronized height-shaded source XYZ versus actual prediction divider preserved',
        original_source_provenance=original,
        scope='Original provenance is historical context; RC3 map timing/crop fields do NOT describe RC4, which contains no map-event shot.',
        model_and_source_integrity='All original RC3 manifest payloads including models, source geometry, accepted states and prediction sidecars independently rehashed',
        publication='NOT_RUN'))
    print('Both films fully decoded; transition and mobile sheets ready for visual review',flush=True)


def package():
    qa=json.loads((OUT/'evidence/VIDEO_QA_RC4.json').read_text())
    assert qa['status']=='PASS' and qa['video_sha256']==sha(VIDEO)
    review=json.loads((OUT/'evidence/VISUAL_REVIEW_RC4.json').read_text())
    assert review['status']=='PASS' and review['video_sha256']==sha(VIDEO)
    for name in ['VISUAL_PROVENANCE_RC4','CLAIM_AUDIT_RC4','LINKEDIN_DRAFTS_RC4']:
        assert json.loads((OUT/f'evidence/{name}.json').read_text())['status']=='PASS',name
    assert not git('status','--porcelain')
    save(OUT/'evidence/PRESERVATION_RC4.json',preserved())
    commit=git('rev-parse','HEAD')
    source=OUT/'source';source.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit],cwd=ROOT))) as archive:
        archive.extractall(source,filter='data')
    bundle=OUT/'source.bundle'
    if bundle.exists():bundle.unlink()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=ROOT,check=True)
    run(['git','bundle','verify',str(bundle)],'source-bundle-verify.log')
    # Full independent payloads: no symlinks or dependency on the original worktree.
    shutil.copytree(RC3/'inputs',OUT/'inputs',dirs_exist_ok=True)
    for parent in ['docs','linkedin']:
        for p in (ROOT/parent).glob('*RC4*'):
            shutil.copy2(p,OUT/parent/p.name)
    gates={key:'PASS' for key in ['RC3_PRESERVED','FLASHING_EFFECT_REMOVED','MAP_REPLAY_RESET_REMOVED','VIDEO_RENDER','VIDEO_QA','VISUAL_PROVENANCE','LINKEDIN_DRAFTS','CLAIM_AUDIT','ARCHIVE_INTEGRITY','OWNER_REVIEW_READY']}
    gates['PUBLICATION']='NOT_RUN'
    save(OUT/'evidence/FINAL_REPORT_RC4.json',dict(status='PASS',schema='synrex.streaming_r2.rc4.final.v1',
        status_scope='Local candidate only; archive gate contingent on full readback before qualified tag creation',
        source_commit=commit,baseline_commit=BASELINE,local_candidate_tag=TAG,video_sha256=sha(VIDEO),
        source_bundle_sha256=sha(bundle),seconds=37,frames=1110,gates=gates,
        accepted_measurements=json.loads((RC3/'evidence/FINAL_REPORT_RC3.json').read_text())['accepted_measurements'],
        historical_json_p99_fail_ms=274.477963,owner_human_approval='NOT_RUN',
        limits='No new GPU campaign, sensor, Jetson, flight, QGIS GUI or calibrated mobile device qualification; agent editorial review and full native decode only',publication='NOT_RUN'))
    payloads={}
    for p in sorted(OUT.rglob('*')):
        rel=p.relative_to(OUT)
        if not p.is_file() or rel.parts[0]=='owner-review' or str(rel) in ['MANIFEST_RC4.json','SHA256SUMS']:continue
        assert not p.is_symlink()
        payloads[str(rel)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    manifest=OUT/'MANIFEST_RC4.json'
    save(manifest,dict(source_commit=commit,files=payloads,exclusions=['Manifest self','SHA256SUMS self','owner-review containers and external verification receipts'],publication='NOT_RUN'))
    payloads['MANIFEST_RC4.json']=dict(sha256=sha(manifest),bytes=manifest.stat().st_size)
    (OUT/'SHA256SUMS').write_text(''.join(f"{row['sha256']}  {rel}\n" for rel,row in sorted(payloads.items())))
    payloads['SHA256SUMS']=dict(sha256=sha(OUT/'SHA256SUMS'),bytes=(OUT/'SHA256SUMS').stat().st_size)
    dest=OUT/'owner-review';dest.mkdir(exist_ok=True)
    archive=dest/'synrex-streaming-r2-rc4-owner-review.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for rel in sorted(payloads):tar.add(OUT/rel,arcname='streaming-r2-rc4/'+rel,recursive=False)
    # Reopen the compressed artifact; check exact member set and every actual byte.
    checked=set()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            rel=str(Path(member.name).relative_to('streaming-r2-rc4'))
            assert member.isfile() and rel in payloads and rel not in checked
            row=payloads[rel]
            assert member.size==row['bytes'] and hashlib.file_digest(tar.extractfile(member),'sha256').hexdigest()==row['sha256'],rel
            checked.add(rel)
    assert checked==set(payloads)
    digest=sha(archive)
    (dest/'SHA256SUMS').write_text(f'{digest}  {archive.name}\n')
    save(dest/'ARCHIVE_INTEGRITY_RC4.json',dict(status='PASS',archive_sha256=digest,video_sha256=sha(VIDEO),
        manifest_sha256=sha(manifest),bytes=archive.stat().st_size,payloads_verified=len(checked),
        method='Independent reopen, decompress, exact member set, size and SHA256 of every member',source_commit=commit,publication='NOT_RUN'))
    preserved()
    assert not git('status','--porcelain')
    subprocess.run(['git','tag','-a',TAG,commit,'-m',f'Local RC4 qualified after full frame/editorial/provenance/claim/archive gates. 37 s subtraction cut. Archive SHA256 {digest}. PUBLICATION=NOT_RUN.'],cwd=ROOT,check=True)
    save(dest/'LOCAL_QUALIFICATION_RC4.json',dict(status='PASS',tag=TAG,tag_object=git('rev-parse',TAG),
        commit=commit,archive_sha256=digest,clean_tree=not bool(git('status','--porcelain')),qualified_after_archive_verification=True,publication='NOT_RUN'))
    print(json.dumps(dict(status='PASS',video_sha256=sha(VIDEO),archive=str(archive),archive_sha256=digest,tag=TAG,commit=commit),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['freeze','render','qa','package'],required=True)
    globals()[parser.parse_args().stage]()
