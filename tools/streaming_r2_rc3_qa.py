# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""RC3 full decode, camera vectors, independent COPC/event parity and preservation."""
import json
import math
import subprocess
import time
from pathlib import Path
import laspy
import numpy as np
from PIL import Image,ImageDraw
from streaming_r2_rc3 import OUT,BASE,ROOT,FFMPEG,Film,basis,sha,save,SHOTS

VIDEO=OUT/'media/synrex-streaming-r2-rc3.mp4'

def run(cmd,log):
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (OUT/'evidence'/log).write_bytes(r.stdout+r.stderr);assert r.returncode==0,r.stderr[-1000:]
    return r

def sheet(paths,dest,cols=4,width=360):
    h=round(width*1.25);im=Image.new('RGB',(cols*width,math.ceil(len(paths)/cols)*(h+25)),'#07101c');d=ImageDraw.Draw(im)
    for i,p in enumerate(paths):
        x=i%cols*width;y=i//cols*(h+25);im.paste(Image.open(p).resize((width,h),Image.Resampling.LANCZOS),(x,y+25));d.text((x+8,y+6),p.stem,fill='white')
    im.save(dest,quality=94)

def audit():
    original=ROOT.parent/'streaming-r2-rc2-worktree/build/streaming-r2-review/streaming-r2-rc2'
    reference=ROOT.parent/'streaming-r2-rc3-reference'
    protected={}
    for p,expected in [(original/'owner-review/synrex-streaming-r2-rc2-owner-review.tar.gz','56a3a0b8d66444daf8dca42202208c2ce795487cac784bd65e9dbd4a846fdd9a'),(reference/'media/synrex-streaming-r2-rc2.mp4','3ac6c0bb98b4f7bc993b347051e2933ec4875dc0c34163de7bce19eda922534d'),(ROOT.parent/'streaming-r0/build/streaming-r2-review/streaming-r2-rc1.tar.gz','a73aa7ce4b4bfd27377bf2d691e213bba70881ca5fefe817c3f460214b650805')]:
        assert sha(p)==expected;protected[str(p)]=expected
    for p in reference.rglob('*'):
        if p.is_file():protected[str(p)]=sha(p)
    tags=dict(line.split(' ',1)[::-1] for line in subprocess.check_output(['git','show-ref','--tags'],cwd=ROOT,text=True).splitlines() if not line.endswith('/streaming-r2-rc3-qualified'))
    tagseal=OUT/'evidence/ORIGINAL_TAGS_RC3.json'
    if tagseal.exists():assert json.loads(tagseal.read_text())==tags
    else:save(tagseal,tags)
    changed=subprocess.check_output(['git','diff','streaming-r2-rc2-qualified','--name-only'],cwd=ROOT,text=True).splitlines()
    assert all(p.startswith(('tools/streaming_r2_rc3','docs/streaming/rc3/','evidence/streaming_r2_rc3')) for p in changed),changed
    save(OUT/'evidence/PRESERVATION_RC3.json',dict(status='PASS',protected_sha256=protected,baseline_tag='streaming-r2-rc2-qualified',baseline_commit=subprocess.check_output(['git','rev-parse','streaming-r2-rc2-qualified^{commit}'],cwd=ROOT,text=True).strip(),changed_tracked_files=changed,scope='Original RC1/RC2 archives hash checked. All pre-existing tracked RC2 bytes unchanged; only RC3 media tooling/docs/evidence may be added.'))
    cams=json.loads((original/'evidence/CAMERA_FRAMES_RC2.json').read_text());rows=[]
    for a,b in [(0,4),(4,8),(8,11),(11,14),(14,25),(25,28),(28,33),(33,38),(38,45),(45,49)]:
        cs=cams[a*30:b*30];v=np.array([c['target_local'] for c in cs])-np.array([c['eye_local'] for c in cs]);p=np.degrees(np.arctan2(-v[:,2],np.linalg.norm(v[:,:2],axis=1)))
        rows.append(dict(time=[a,b],scene=cs[0]['scene'],start_eye=cs[0]['eye_local'],start_target=cs[0]['target_local'],end_eye=cs[-1]['eye_local'],end_target=cs[-1]['target_local'],pitch_min=float(p.min()),pitch_max=float(p.max()),below_20_frames=int((p<20).sum())))
    save(OUT/'evidence/RC2_CAMERA_AUDIT.json',rows)
    times=[0,2,3.9,4,6,7.9,8,9.5,10.9,11,12.5,13.9,14,19.5,24.9,25,26.5,27.9,28,30.5,32.9,33,35.5,37.9]+list(range(39,49))+[48.9]
    paths=[]
    for t in times:
        p=OUT/f'media/rc2-decoded/{t:g}.png';p.parent.mkdir(exist_ok=True)
        if not p.exists():run([FFMPEG,'-v','error','-y','-ss',str(t),'-i',str(reference/'media/synrex-streaming-r2-rc2.mp4'),'-frames:v','1',str(p)],'rc2-extract.log')
        paths.append(p)
    sheet(paths,OUT/'media/rc2-contact-sheet.jpg',5,300)
    lines=['# RC2 shot diagnosis','', 'Actual RC2 H.264 was decoded at all listed timestamps, including every second from 39–48 and 48.9 s. The contact sheet was visually inspected. Historical automated PASS did not establish editorial quality.','', 'Pitch is atan2(eye.z − target.z, horizontal distance), positive downward. All vectors are illustrative local metres, not sensor poses. Exact endpoints appear in evidence/RC2_CAMERA_AUDIT.json.','', '| Seconds | Scene | Start eye → target | End eye → target | Actual downward pitch |','|---|---|---|---|---|']
    for c in rows:
        lines.append(f"| {c['time']} | {c['scene']} | {c['start_eye']} → {c['start_target']} | {c['end_eye']} → {c['end_target']} | {c['pitch_min']:.2f}–{c['pitch_max']:.2f}° |")
    lines += ['', '## Decoded visual findings', '', '0–8 s: near-horizontal church views expose the walls but compress roofs and ground into narrow strips. 8–11 s: tree crowns merge into a forward-facing wall. 11–14 s: slightly upward-looking crane view leaves a large empty centre and flattens ground. 14–25 s: synchronized reveal is authentic but repeats the low pitch. 25–38 s: coarse cubes obscure canopy structure and form an oversized green block wall. 38–45 s: large metric card hides much of the same flattened urban view. 45–49 s: actual closing frames are point splats, not cubes, but the camera is only about 1.25–1.59° down and retains the flattened look. The owner-described ending defect is removed across the late sequence, not relocated.', '', '## Exact block-effect cause', '', '`tools/streaming_r2_rc2.py:voxel_raster` constructs eight corners at ±0.42 m around every actual 1 m map cell centre, projects six faces at focal 850 px, fills depth-buffered triangles and shades cube faces. `Film.draw` dispatches to that function whenever `mapstate is not None`. `Film.frame` supplies mapstate only at 25–38 s, including low 27 m-radius detail orbits. Large projected cube faces plus dense occupancy create the observed wall/pixel motif. It is not H.264 corruption or a pixel-dissolve transition. No block effect is visible in the decoded final ten seconds (39–49); their separate defects are low camera pitch and overlaid metric-card occlusion.', '', '## Inspected evidence', '', '`media/rc2-contact-sheet.jpg`; actual decoded PNGs in `media/rc2-decoded/`; reference MP4 SHA256 `3ac6c0bb98b4f7bc993b347051e2933ec4875dc0c34163de7bce19eda922534d`. RC2 camera metadata was independently recomputed from its exact eye/target vectors.']
    (OUT/'docs/RC2_SHOT_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')

def cameras(film,cams):
    reports=[]
    for a,b,scene in SHOTS:
        cs=cams[a*30:b*30];eyes=np.array([c['eye_local'] for c in cs]);targets=np.array([c['target_local'] for c in cs]);v=targets-eyes
        p=np.degrees(np.arctan2(-v[:,2],np.linalg.norm(v[:,:2],axis=1)));step=np.linalg.norm(np.diff(eyes,axis=0),axis=1)
        assert np.all((p>=30)&(p<=45)) and step.min()>0 and step.max()<.25
        assert np.allclose(p,[c['pitch_down_degrees'] for c in cs])
        xyz=film.scenes[scene][0]
        if a in (28,34):xyz=xyz[np.linalg.norm(xyz[:,:2]-np.array(cs[15]['map_crop_center_xy']),axis=1)<18]
        def project(c):
            e=np.array(c['eye_local']);r,u,f=basis(e,np.array(c['target_local']));d=xyz-e;z=d@f;fl=c['focal_pixels']
            return np.column_stack([540+fl*(d@r)/z,785-fl*(d@u)/z]),z
        pa,za=project(cs[15]);pb,zb=project(cs[30]);valid=(za>1.5)&(zb>1.5)&(pa[:,0]>40)&(pa[:,0]<1040)&(pa[:,1]>340)&(pa[:,1]<1080)
        depth=za[valid];motion=(pb-pa)[valid];near=motion[depth<np.quantile(depth,.25)];far=motion[depth>np.quantile(depth,.75)]
        diff=float(np.linalg.norm(np.median(near,axis=0)-np.median(far,axis=0)))
        assert diff>.1,(a,diff)
        reports.append(dict(time=[a,b],scene=scene,pitch_range=[float(p.min()),float(p.max())],start_eye=cs[0]['eye_local'],start_target=cs[0]['target_local'],end_eye=cs[-1]['eye_local'],end_target=cs[-1]['target_local'],translation_m=float(np.linalg.norm(eyes[-1]-eyes[0])),camera_step_m_range=[float(step.min()),float(step.max())],target_translation_m=float(np.linalg.norm(targets[-1]-targets[0])),depth_p05_p95_m=np.quantile(depth,[.05,.95]).tolist(),parallax_near_far_difference_px_per_half_second=diff,near_motion_median_px=float(np.median(np.linalg.norm(near,axis=1))),far_motion_median_px=float(np.median(np.linalg.norm(far,axis=1))),visible_source_points=int(valid.sum()),scope='Source points inside geometric frustum; occlusion not filtered. Parallax from exact perspective projection, not optical-flow claim.'))
    save(OUT/'evidence/CAMERA_AND_PARALLAX_RC3.json',reports)
    (OUT/'docs/CAMERA_TABLE_RC3.md').write_text('# RC3 actual camera vectors\n\nPositive pitch is downward, computed from camera-target vectors. Every one of 1470 frames is recorded in CAMERA_FRAMES_RC3.json.\n\n| Seconds | Scene | Pitch range | Eye start → end | Target start → end | Parallax near/far difference / 0.5s |\n|---|---|---|---|---|---|\n'+''.join(f"| {r['time']} | {r['scene']} | {r['pitch_range']} | {r['start_eye']} → {r['end_eye']} | {r['start_target']} → {r['end_target']} | {r['parallax_near_far_difference_px_per_half_second']:.3f}px |\n" for r in reports))
    return reports

def source_checks(film):
    results=[]
    for scene,stem in [('urban','TEST-0438_6393-002595911'),('tower','TEST-0741_6244-006630086'),('forest','TEST-0944_6564-005900112')]:
        q=BASE/('qgis' if scene=='forest' else 'qgis-r4-foundation');copc=q/f'views/{stem}.copc.laz';c=laspy.read(copc)
        ids=np.asarray(c.SynRexSourceIndex).astype(int);order=np.argsort(ids);assert np.array_equal(ids[order],np.arange(len(ids)))
        xyz=np.column_stack([c.x,c.y,c.z])[order];expected,labels,origin=film.scenes[scene]
        err=float(np.max(np.abs(xyz-(expected+origin))));assert err<=.00101
        copc_ids=np.asarray(c.SynRexSemanticID)[order]
        side=q/f'data/semantics/{stem}-semantics.json'
        side_labels=np.array(json.loads(side.read_text())['semantic_ids'],np.uint8)
        assert np.array_equal(copc_ids,side_labels)
        video_difference=int(np.count_nonzero(copc_ids!=labels))
        if scene!='forest':assert video_difference==0
        project=json.loads((q/'qgis-project.json').read_text());entry=next(e for e in project['files'] if e['source']==stem+'.laz');assert entry['crs']=='EPSG:2154'
        results.append(dict(scene=scene,copc_sha256=sha(copc),sidecar_sha256=sha(side),points=len(ids),max_xyz_error_m=err,copc_to_own_semantic_sidecar_parity='PASS',copc_to_video_prediction_differences=video_difference,prediction_scope='Forest COPC is a separate preserved native export; video uses independently reverified short JSON recording predictions. They are not interchangeable prediction arrays.' if scene=='forest' else 'Same frozen R4 prediction sidecar in COPC and film.',project_crs='EPSG:2154',header_crs=str(c.header.parse_crs()),explicit_viewing_crs_assignment=entry['explicit_viewing_crs_assignment'],qgis_project_sha256=sha(q/'qgis-project.json')))
    cache=film.events;state={};keys={};ordinal=0
    for line in (BASE/'recording/events.jsonl').open():
        e=json.loads(line)
        if e['schema']!='synrex.semantic_delta.v1':continue
        k=tuple(e['key']);new=k not in keys
        if new:keys[k]=len(keys)
        assert cache['index'][ordinal]==keys[k] and cache['labels'][ordinal]==e['semantic_id'] and cache['new'][ordinal]==new
        assert np.array_equal(cache['keys'][keys[k]],k)
        assert cache['changed'][ordinal]==(not new and state[k]!=e['semantic_id'])
        state[k]=e['semantic_id'];ordinal+=1
    assert ordinal==400654
    save(OUT/'evidence/SOURCE_MAP_COPC_RC3.json',dict(status='PASS',scenes=results,map_events_reverified=ordinal,new_cells=int(cache['new'].sum()),semantic_changes=int(cache['changed'].sum()),qgis_scene_correctness='PASS: independent COPC source-ID/XYZ/predicted-label parity and preserved project CRS; new QGIS GUI playback NOT_RUN',new_qgis_gui='NOT_RUN'))

def decode():
    dest=OUT/'media/decoded';dest.mkdir(exist_ok=True);samples=sorted(set([0,1,15,45,149,150,269,270,389,390,509,510,839,840,1019,1020,1199,1200,1271,1272,1349,1350,1469]+list(range(0,1470,30))))
    p=subprocess.Popen([FFMPEG,'-v','error','-xerror','-i',str(VIDEO),'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=(OUT/'evidence/full-decode.log').open('wb'))
    size=1080*1350*3;count=0;prev=None;motion=[];paths=[]
    while True:
        data=bytearray()
        while len(data)<size:
            chunk=p.stdout.read(size-len(data))
            if not chunk:break
            data.extend(chunk)
        if not data:break
        assert len(data)==size
        a=np.frombuffer(data,np.uint8).reshape(1350,1080,3);crop=a[350:850:3,40:1040:3].astype(np.int16)
        if prev is not None:
            diff=np.abs(crop-prev);motion.append(dict(frame=count,mae=float(diff.mean()),changed_fraction=float(np.any(diff>12,axis=2).mean())))
            assert np.any(diff),count
        prev=crop
        if count in samples:
            path=dest/f'{count/30:06.2f}s-f{count:04d}.jpg';im=Image.fromarray(a);im.save(path,quality=94);paths.append(path)
            if count==0:im.save(OUT/'media/frame-0-decoded.png');im.resize((360,450),Image.Resampling.LANCZOS).save(OUT/'media/mobile-frame-0.png')
        count+=1
    assert p.wait()==0 and count==1470,count
    for part in range(math.ceil(len(paths)/20)):sheet(paths[part*20:(part+1)*20],OUT/f'media/decoded-contact-{part+1}.jpg',5,300)
    selected=[min(paths,key=lambda p:abs(float(p.stem.split('s')[0])-t)) for t in [0,3,5,8,9,12,13,16,17,22,27,28,30,33,34,37,39,41,43,45,47,48.9667]]
    sheet(selected,OUT/'media/contact-sheet.jpg')
    save(OUT/'evidence/DECODED_FRAME_MOTION_RC3.json',motion)
    return count,motion

def main():
    audit();film=Film();source_checks(film)
    # Canonical final provenance qualifies inherited helper fields for RC3's crop.
    provenance=json.loads((OUT/'evidence/VISUAL_PROVENANCE_RC3.json').read_text())
    provenance.update(imported_point_rasterizer_sha256=sha(ROOT/'tools/streaming_r2_rc2.py'),qa_source_sha256=sha(__file__),map_focus_xy_local=[[25,25],[15,35]],point_decimation='NONE for primary source shots. Map context explicitly cropped to 18m XY radius; ordinary projection/clipping/occlusion only.',map_recent_event_window=2200,map_annotation='Thin 3px round projected-cell outlines, screen overlay, not depth-tested. White first insertion / cyan reobservation; true per-cell geometry is not enlarged.',model_payload_sha256={str(p.relative_to(BASE/'models/unet.srxmodel')):sha(p) for p in sorted((BASE/'models/unet.srxmodel').rglob('*')) if p.is_file()},copc_scope=json.loads((OUT/'evidence/SOURCE_MAP_COPC_RC3.json').read_text()))
    save(OUT/'evidence/VISUAL_PROVENANCE_RC3.json',provenance)
    cams=json.loads((OUT/'evidence/CAMERA_FRAMES_RC3.json').read_text());assert len(cams)==1470
    reports=cameras(film,cams)
    for a,b in [(28,34),(34,40)]:
        stops=[c['map_event_stop'] for c in cams[a*30:b*30]];assert all(y>x for x,y in zip(stops,stops[1:])) and stops[-1]==400654
    count,motion=decode()
    run([FFMPEG,'-v','error','-y','-i',str(VIDEO),'-vf','scale=432:540','-c:v','libx264','-crf','21','-preset','fast','-pix_fmt','yuv420p','-an','-movflags','+faststart',str(OUT/'media/preview.mp4')],'preview-encode.log')
    run([FFMPEG,'-v','error','-xerror','-i',str(OUT/'media/preview.mp4'),'-f','null','-'],'preview-decode.log')
    t=time.monotonic();run([FFMPEG,'-v','error','-xerror','-re','-i',str(VIDEO),'-f','null','-'],'realtime-playback.log')
    playback=dict(status='PASS',seconds=time.monotonic()-t,scope='Complete realtime headless decode. No GUI/audio playback claimed.')
    info=run([FFMPEG,'-hide_banner','-i',str(VIDEO),'-map','0:v:0','-c','copy','-f','null','-'],'container-info.log').stderr.decode()
    assert all(v in info for v in ['h264','1080x1350','30 fps','yuv420p','00:00:49.00'])
    boxes=json.loads((OUT/'evidence/TEXT_LAYOUT_RC3.json').read_text());assert min(b['size'] for b in boxes)>=22
    assert all(b['box'][0]>=34 and b['box'][2]<=1048 and b['box'][3]<=1326 for b in boxes)
    visual=OUT/'evidence/VISUAL_REVIEW_RC3.json';review=json.loads(visual.read_text()) if visual.exists() else dict(status='NOT_RUN')
    save(OUT/'evidence/VIDEO_QA_RC3.json',dict(status='PASS' if review.get('status')=='PASS' and review.get('video_sha256')==sha(VIDEO) else 'AWAITING_VISUAL_REVIEW',video_sha256=sha(VIDEO),width=1080,height=1350,fps=30,seconds=49,codec='H.264',pixel_format='yuv420p',decoded_frames=count,full_resolution_decode='PASS: every native-resolution frame decoded with -xerror and exact byte count',realtime_headless_playback=playback,preview_decode='PASS',preview_scope='Full 49s reduced-resolution review copy, 432x540, 30fps',camera_pitch_degrees=[min(r['pitch_range'][0] for r in reports),max(r['pitch_range'][1] for r in reports)],frames_with_30_to_45_down_pitch=1470,static_geometry_frame_pairs=0,minimum_font_px=min(b['size'] for b in boxes),headline_font_px=57,parallax='PASS: see CAMERA_AND_PARALLAX_RC3.json',source_map_copc='PASS',editorial_visual_review=review,publication='NOT_RUN'))
    print('Full decode, cameras, source parity and headless playback complete',flush=True)

if __name__=='__main__':main()
