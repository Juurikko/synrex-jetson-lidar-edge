# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independently decode and inspect every RC2 frame, its motion and source linkage."""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from streaming_r2_rc2 import BASE,OUT,ROOT,W,H,FPS,DURATION,FFMPEG,Film,basis,sha,save

VIDEO=OUT/'media/synrex-streaming-r2-rc2.mp4'

def run(cmd,log=None):
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if log:Path(log).write_bytes(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr.decode()[-2000:]
    return result


def preview():
    target=OUT/'media/preview.mp4'
    if target.exists() and target.stat().st_mtime>=VIDEO.stat().st_mtime:return
    intervals=[(0,4),(5,7),(17,19),(25,28),(34,36),(39,41),(45,49)]
    filters=[]
    for i,(a,b) in enumerate(intervals):filters.append(f'[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[v{i}]')
    filters.append(''.join(f'[v{i}]' for i in range(len(intervals)))+f'concat=n={len(intervals)}:v=1:a=0[out]')
    run([FFMPEG,'-v','error','-y','-i',str(VIDEO),'-filter_complex',';'.join(filters),'-map','[out]',
        '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-r','30','-movflags','+faststart',str(target)],OUT/'evidence/preview-encode.log')


def decode():
    decoded=OUT/'media/decoded';decoded.mkdir(exist_ok=True)
    # Independent decoder reads all 1470 frames at native resolution and checks raw byte count.
    cmd=[FFMPEG,'-v','error','-xerror','-i',str(VIDEO),'-map','0:v:0','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-']
    error=(OUT/'evidence/full-decode.log').open('wb')
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=error)
    size=W*H*3;count=0;previous=None;motions=[];outline=[];identical=[];samples=[]
    sample_frames=sorted(set([0,1,15,30,90,119,120,165,240,285,330,375,420,480,540,600,660,720,749,
       750,765,780,795,810,825,839,840,855,870,885,900,915,930,945,960,975,989,990,1005,1020,1035,1050,
       1065,1080,1095,1110,1125,1139,1140,1185,1238,1239,1260,1320,1349,1350,1410,1469]))
    while True:
        buf=bytearray()
        while len(buf)<size:
            b=p.stdout.read(size-len(buf))
            if not b:break
            buf.extend(b)
        if not buf:break
        assert len(buf)==size,'Truncated decoded frame'
        a=np.frombuffer(buf,np.uint8).reshape(H,W,3)
        # Exclude title, counters, footer and changing progress line.
        crop=a[430:1020:3,36:1044:3].astype(np.int16)
        if previous is not None:
            diff=np.abs(crop-previous)
            motions.append(dict(frame=count,mae=float(diff.mean()),changed_pixel_fraction=float(np.any(diff>12,axis=2).mean())))
            if not np.any(diff):identical.append(count)
        previous=crop
        if 750<=count<1140:
            roi=a[430:1020,36:1044].astype(np.int16)
            white=(roi[:,:,0]>185)&(roi[:,:,1]>185)&(roi[:,:,2]>185)&(np.max(roi,axis=2)-np.min(roi,axis=2)<40)
            cyan=(roi[:,:,0]>30)&(roi[:,:,0]<135)&(roi[:,:,1]>160)&(roi[:,:,2]>130)&(roi[:,:,1]-roi[:,:,0]>60)
            outline.append(dict(frame=count,white_pixels=int(white.sum()),cyan_pixels=int(cyan.sum())))
        if count in sample_frames:
            im=Image.fromarray(a);im.save(decoded/f'frame-{count:04d}.jpg',quality=94)
            samples.append((count,im.resize((270,338),Image.Resampling.LANCZOS)))
            if count==0:
                im.save(OUT/'media/frame-0-decoded.png')
                im.resize((360,450),Image.Resampling.LANCZOS).save(OUT/'media/mobile-frame-0.png')
        count+=1
    assert p.wait()==0;error.close();assert count==1470,count
    assert not identical,identical
    # Detailed sheets include decoded frames at 0.5 s spacing through the map sequence.
    for part in range(math.ceil(len(samples)/20)):
        chosen=samples[part*20:(part+1)*20];sheet=Image.new('RGB',(1350,370*4),'#07101c');d=ImageDraw.Draw(sheet)
        for i,(n,im) in enumerate(chosen):
            x=i%5*270;y=i//5*370;sheet.paste(im,(x,y+28));d.text((x+8,y+7),f'{n/FPS:.3f} s / frame {n}',fill='white')
        sheet.save(OUT/f'media/decoded-contact-{part+1}.jpg',quality=94)
    times=[0,5.5,9.5,12.5,16,20,24,25.5,27.5,28.5,30.5,32.5,33.5,35.5,37.5,39.5,42,45,47,1469/30]
    contact=Image.new('RGB',(1440,470*5),'#07101c');d=ImageDraw.Draw(contact)
    lookup=dict(samples)
    for i,t in enumerate(times):
        n=min(sample_frames,key=lambda k:abs(k/30-t));im=Image.open(decoded/f'frame-{n:04d}.jpg').resize((360,450),Image.Resampling.LANCZOS)
        x=i%4*360;y=i//4*470;contact.paste(im,(x,y+20));d.text((x+8,y+4),f'{n/30:.2f}s',fill='white')
    contact.save(OUT/'media/contact-sheet.jpg',quality=94)
    save(OUT/'evidence/DECODED_FRAME_MOTION_RC2.json',motions)
    save(OUT/'evidence/MAP_OUTLINE_PIXELS_RC2.json',outline)
    return count,motions,outline


def depth_and_camera():
    cams=json.loads((OUT/'evidence/CAMERA_FRAMES_RC2.json').read_text());assert len(cams)==1470
    film=Film();shots=[(0,4),(4,8),(8,11),(11,14),(14,25),(25,28),(28,33),(33,38),(38,45),(45,49)]
    reports=[]
    for start,end in shots:
        records=cams[int(start*30):int(end*30)]
        eyes=np.array([r['eye_local'] for r in records]);targets=np.array([r['target_local'] for r in records])
        delta=np.linalg.norm(np.diff(eyes,axis=0),axis=1)
        vec=targets-eyes;angles=np.degrees(np.arctan2(-vec[:,2],np.linalg.norm(vec[:,:2],axis=1)))
        assert delta.min()>0 and delta.max()<.5, (start,delta.min(),delta.max())
        assert np.abs(angles).max()<30,(start,angles.max())
        a,b=records[0],records[15];xyz=film.scenes[a['scene']][0]
        def projection(record):
            eye=np.array(record['eye_local']);r,u,f=basis(eye,np.array(record['target_local']));p=xyz-eye;dep=p@f
            return np.column_stack((540+850*(p@r)/dep,785-850*(p@u)/dep)),dep
        pa,da=projection(a);pb,db=projection(b)
        valid=(da>1.5)&(db>1.5)&(pa[:,0]>30)&(pa[:,0]<1050)&(pa[:,1]>430)&(pa[:,1]<1020)
        ds=da[valid];motion=(pb-pa)[valid]
        near=motion[ds<np.quantile(ds,.25)];far=motion[ds>np.quantile(ds,.75)]
        reports.append(dict(time=[start,end],translation_metres=float(np.linalg.norm(eyes[-1]-eyes[0])),
            min_step_metres=float(delta.min()),max_step_metres=float(delta.max()),
            depression_degrees_range=[float(angles.min()),float(angles.max())],
            depth_p05_p95_metres=np.quantile(ds,[.05,.95]).tolist(),
            near_motion_median_px=float(np.median(np.linalg.norm(near,axis=1))),
            far_motion_median_px=float(np.median(np.linalg.norm(far,axis=1))),
            foreground_background_motion_difference_px=float(np.linalg.norm(np.median(near,axis=0)-np.median(far,axis=0))),
            witness_scope='Actual source points visible in geometric frustum; 0.5s camera interval; occlusion not filtered'))
    assert min(r['foreground_background_motion_difference_px'] for r in reports[:5])>1
    save(OUT/'evidence/DEPTH_AND_PARALLAX_RC2.json',reports)
    return reports


def provenance_checks():
    loaded=np.load(OUT/'evidence/map-events-derived.npz');cache={k:loaded[k] for k in loaded.files};state={};keys=[];mapping={};ordinal=0
    for line in (BASE/'recording/events.jsonl').open():
        e=json.loads(line)
        if e['schema']!='synrex.semantic_delta.v1':continue
        key=tuple(e['key']);new=key not in mapping
        if new:mapping[key]=len(keys);keys.append(key)
        label=e['semantic_id'];changed=not new and state[key]!=label
        assert cache['index'][ordinal]==mapping[key]
        assert cache['labels'][ordinal]==label and cache['new'][ordinal]==new and cache['changed'][ordinal]==changed
        state[key]=label;ordinal+=1
    assert ordinal==400654 and np.array_equal(np.array(keys),cache['keys'])
    cams=json.loads((OUT/'evidence/CAMERA_FRAMES_RC2.json').read_text())
    for first,last in [(750,840),(840,990),(990,1140)]:
        stops=[r['map_event_stop'] for r in cams[first:last]]
        assert all(b>a for a,b in zip(stops,stops[1:]))
    assert cams[839]['map_event_stop']==400654
    # Every tracked pre-existing blob must still match the immutable source seal.
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','eb4cab0'],cwd=ROOT,text=True).splitlines()
    bad=[]
    for p in paths:
        expected=subprocess.check_output(['git','rev-parse',f'eb4cab0:{p}'],cwd=ROOT,text=True).strip()
        actual=subprocess.check_output(['git','hash-object',p],cwd=ROOT,text=True).strip()
        if expected!=actual:bad.append(p)
    assert not bad,bad
    preserved=json.loads((OUT/'evidence/BASELINE_PRESERVATION_RC2.json').read_text())
    for p,h in preserved.items():assert sha(ROOT/p)==h,p
    save(OUT/'evidence/PRESERVATION_CHECK_RC2.json',dict(status='PASS',baseline='eb4cab0',tracked_files_checked=len(paths),
         changed_preexisting_files=bad,rc1_archive_and_mp4=preserved,model_runtime_benchmark='All accepted bytes unchanged',
         public_actions='NOT_RUN',target_device_access='NOT_RUN'))
    return len(paths)


def ensure_playback():
    receipt=OUT/'evidence/REALTIME_PLAYBACK_RC2.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['status']=='PASS' and old['video_sha256']==sha(VIDEO):return
    start=time.monotonic()
    run([FFMPEG,'-hide_banner','-v','error','-xerror','-re','-i',str(VIDEO),'-f','null','-'],OUT/'evidence/realtime-playback.log')
    save(receipt,dict(status='PASS',elapsed_seconds=time.monotonic()-start,
        method='FFmpeg -re -xerror: complete full-rate headless playback/decode to null sink; no GUI/audio claimed',
        video_sha256=sha(VIDEO),returncode=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reuse-decoded',action='store_true');args=parser.parse_args()
    preview();ensure_playback()
    if args.reuse_decoded:
        assert json.loads((OUT/'evidence/DECODE_COMPLETE_RC2.json').read_text())['video_sha256']==sha(VIDEO)
        motions=json.loads((OUT/'evidence/DECODED_FRAME_MOTION_RC2.json').read_text())
        outlines=json.loads((OUT/'evidence/MAP_OUTLINE_PIXELS_RC2.json').read_text())
        count=len(motions)+1;assert count==1470 and len(outlines)==390
        assert json.loads((OUT/'evidence/REALTIME_PLAYBACK_RC2.json').read_text())['video_sha256']==sha(VIDEO)
    else:
        count,motions,outlines=decode()
        run([FFMPEG,'-v','error','-xerror','-i',str(OUT/'media/preview.mp4'),'-f','null','-'],OUT/'evidence/preview-decode.log')
        save(OUT/'evidence/DECODE_COMPLETE_RC2.json',dict(status='PASS',video_sha256=sha(VIDEO),frames=count,preview_decode='PASS'))
    depth=depth_and_camera();protected=provenance_checks()
    # FFmpeg identifies codec, frame rate, pixel format and duration from the actual bitstream/container.
    info=run([FFMPEG,'-hide_banner','-i',str(VIDEO),'-map','0:v:0','-c','copy','-f','null','-'],OUT/'evidence/container-info.log').stderr.decode()
    assert 'h264' in info and '1080x1350' in info and '30 fps' in info and 'yuv420p' in info
    assert '00:00:49.00' in info
    boxes=json.loads((OUT/'evidence/TEXT_LAYOUT_RC2.json').read_text())
    assert all(r['size']>=22 and r['box'][0]>=34 and r['box'][2]<=1048 and r['box'][3]<=1326 for r in boxes)
    # Local visual inspection is explicit and separate from automated measurements.
    review_path=OUT/'evidence/VISUAL_REVIEW_RC2.json'
    review=json.loads(review_path.read_text()) if review_path.exists() else dict(status='NOT_RUN')
    sections=[]
    for a,b in [(0,4),(4,14),(14,25),(25,38),(38,45),(45,49)]:
        rows=[r for r in motions if a*30<=r['frame']<b*30]
        sections.append(dict(time=[a,b],median_mae=float(np.median([r['mae'] for r in rows])),
            min_changed_fraction=min(r['changed_pixel_fraction'] for r in rows),median_changed_fraction=float(np.median([r['changed_pixel_fraction'] for r in rows]))))
    map_visible=sum(r['white_pixels']+r['cyan_pixels']>60 for r in outlines)
    assert map_visible>len(outlines)*.5,(map_visible,len(outlines))
    assert min(s['median_changed_fraction'] for s in sections)>.01
    save(OUT/'evidence/VIDEO_QA_RC2.json',dict(schema='synrex.streaming_r2.rc2.video_qa.v1',
        status='PASS' if review['status']=='PASS' else 'NOT_RUN',video_sha256=sha(VIDEO),
        width=1080,height=1350,fps=30,seconds=49,codec='H.264',pixel_format='yuv420p',decoded_frames=count,
        full_resolution_decode='PASS: all frames read, exact byte count, FFmpeg -xerror; no decoder errors',
        full_rate_headless_playback=json.loads((OUT/'evidence/REALTIME_PLAYBACK_RC2.json').read_text()),
        exact_geometry_static_holds=0,text_layout_checks=len(boxes),minimum_font_px=22,headline_font_px=58,
        mobile_review='See mobile-frame-0.png and explicit visual review',geometry_motion=sections,
        map_frames_with_visible_event_outlines=map_visible,map_frames_total=len(outlines),
        depth_camera_receipt='DEPTH_AND_PARALLAX_RC2.json',protected_existing_source_blobs=protected,
        preview_sha256=sha(OUT/'media/preview.mp4'),contact_sheet_sha256=sha(OUT/'media/contact-sheet.jpg'),
        thumbnail_sha256=sha(OUT/'media/thumbnail.png'),visual_review=review,publication='NOT_RUN'))
    print('QA completed; explicit visual-review status:',review['status'],flush=True)

if __name__=='__main__':main()
