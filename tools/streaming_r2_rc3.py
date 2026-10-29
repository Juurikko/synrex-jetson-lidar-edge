# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Media-only RC3: coupled oblique cameras and genuine point/event rendering."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import streaming_r2_rc2 as old

ROOT=old.ROOT
OUT=ROOT/'build/streaming-r2-review/streaming-r2-rc3'
BASE=OUT/'inputs/streaming-r2-rc1'
old.OUT=OUT
old.BASE=BASE
W,H,FPS,DURATION=1080,1350,30,49
sha,save,basis=old.sha,old.save,old.basis
FFMPEG=old.FFMPEG
SHOTS=[(0,5,'urban'),(5,9,'urban'),(9,13,'forest'),(13,17,'tower'),
       (17,28,'urban'),(28,34,'forest'),(34,40,'forest'),(40,45,'urban'),(45,49,'urban')]

def camera(i):
    t=i/FPS
    a,b,scene=next(s for s in SHOTS if s[0]<=t<s[1]);u=(t-a)/(b-a)
    # Translation is shared by eye and target. Small orbit preserves radius/pitch.
    if a==0: target=old.mix([23,24,9],[20,29,9],u); offset=old.mix([35,-42,42],[29,-39,39],u)
    elif a==5: target=old.mix([22,25,8],[29,28,9],u);offset=old.mix([-29,-35,35],[-25,-31,32],u)
    elif a==9: target=old.mix([23,24,20],[29,28,21],u);offset=old.mix([32,-36,35],[29,-32,33],u)
    elif a==13:
        target=old.mix([25,25,12],[26,28,12],u);angle=-1.05+u*.18;offset=np.array([49*math.cos(angle),49*math.sin(angle),39])
    elif a==17: target=old.mix([21,25,9],[27,30,9],u);offset=old.mix([32,-38,39],[25,-35,35],u)
    elif a==28: target=old.mix([23,24,18],[27,27,18],u);offset=old.mix([25,-28,30],[23,-26,28],u)
    elif a==34: target=old.mix([16,33,18],[14,36,19],u);offset=old.mix([-26,-29,30],[-24,-27,28],u)
    elif a==40: target=old.mix([23,25,9],[26,28,9],u);offset=old.mix([30,-37,37],[28,-35,35],u)
    else: target=old.mix([24,26,9],[26,28,9],u);offset=old.mix([-31,-37,38],[-28,-35,36],u)
    eye=target+offset
    pitch=math.degrees(math.atan2(offset[2],np.linalg.norm(offset[:2])))
    return dict(frame=i,time=t,shot_start=a,shot_end=b,scene=scene,eye_local=eye.tolist(),target_local=target.tolist(),pitch_down_degrees=pitch,focal_pixels=1000 if a in (28,34) else (720 if a==13 else 850),principal_point=[540,785])

class Film(old.Film):
    def __init__(self):
        super().__init__()
        # Exact timestamps and prediction IDs from the recorded emitted point batches.
        ids=[]; times=[]; stamps=[]
        for line in (BASE/'recording/events.jsonl').open():
            e=json.loads(line)
            if e['schema']=='synrex.semantic_points.v1':
                ids.extend(p['id'] for p in e['points']);times.extend(p['timestamp_ns'] for p in e['points'])
        # Schema names are checked below if recording uses a different point envelope.
        if not ids:
            for line in (BASE/'recording/input.jsonl').open():
                e=json.loads(line);ids.extend(p['id'] for p in e['points']);times.extend(p['timestamp_ns'] for p in e['points'])
        self.source_ids=np.asarray(ids,np.int64);self.source_times=np.asarray(times,np.int64)
        assert len(ids)==len(self.events['index'])==400654
        assert np.array_equal(self.source_ids,np.arange(400654))
        self.capture_start=int(self.source_times[0])

    def stage(self,im,chapter,lines,sub):
        arr=np.asarray(im).copy()
        for y in range(340):
            w=.94*max(0,1-y/340)**.65;arr[y]=(arr[y]*(1-w)+np.array([6,13,23])*w).astype(np.uint8)
        for y in range(1080,H):
            w=min(.97,(y-1080)/70);arr[y]=(arr[y]*(1-w)+np.array([6,13,23])*w).astype(np.uint8)
        im=Image.fromarray(arr)
        self.text(im,48,34,'SynRex / NATIVE LiDAR',25,old.ACCENT,True)
        self.text(im,48,78,chapter,24,old.MUTED)
        for j,line in enumerate(lines):self.text(im,45,121+j*67,line,57,old.WHITE,True)
        self.text(im,48,130+len(lines)*67,sub,27,old.MUTED)
        self.text(im,48,1272,'IGN FRACTAL · Etalab 2.0 · illustrative camera',22,old.MUTED)
        self.text(im,48,1298,'Recorded LiDAR replay · Streaming R2: local candidate',22,old.MUTED)
        return im

    def map_overlay(self,rec,stop,detail):
        # Geometry remains the actual classified SOURCE points. Small round markers
        # annotate exact delta cell centres; no cube renderer or enlarged occupancy.
        xyz,labels,_=self.scenes['forest'];eye=np.array(rec['eye_local']);target=np.array(rec['target_local'])
        r,u,f=basis(eye,target)
        center=np.array([25.,25.]) if detail==0 else np.array([15.,35.])
        crop=np.linalg.norm(xyz[:,:2]-center,axis=1)<18
        im=Image.fromarray(old.raster(xyz[crop],old.PALETTE[labels[crop]],eye,r,u,f,1000.,.045,np.zeros(crop.sum(),np.uint8)))
        if stop<self.event_pos:self.event_pos=0;self.last[:]=-1
        for j in range(self.event_pos,stop):
            k=self.events['index'][j];self.maplabels[k]=self.events['labels'][j];self.last[k]=j
        self.event_pos=stop
        active=(self.last>=max(0,stop-2200))&(np.linalg.norm(self.mapxyz[:,:2]-center,axis=1)<18)
        indices=np.flatnonzero(active);p=self.mapxyz[indices]-eye;depth=p@f
        sx=540+1000*(p@r)/depth;sy=785-1000*(p@u)/depth
        d=ImageDraw.Draw(im)
        # Thin round outlines, fixed 3px radius: annotation, never a voxel surface.
        for n,k in enumerate(indices):
            if depth[n]<1.5 or not (40<sx[n]<1040 and 340<sy[n]<1080):continue
            x,y=sx[n],sy[n];new=self.first[k]>=max(0,stop-2200)
            d.ellipse((x-3,y-3,x+3,y+3),outline=old.WHITE if new else old.ACCENT,width=1)
        return im,center,indices

    def frame(self,i,metadata=True):
        rec=camera(i);t=rec['time'];a=rec['shot_start'];scene=rec['scene'];stop=None
        im=self.draw(scene,rec['eye_local'],rec['target_local'])
        if a==13:
            xyz,labels,_=self.scenes[scene];eye=np.array(rec['eye_local']);r,u,f=basis(eye,np.array(rec['target_local']))
            im=Image.fromarray(old.raster(xyz,old.PALETTE[labels],eye,r,u,f,720.,.055,np.zeros(len(xyz),np.uint8)))
        if a==0: chapter='RTX 4090 · recorded LiDAR replay';title=['UP TO 2 MILLION','LiDAR POINTS/S'];sub='Native GPU classification + semantic mapping'
        elif a==5: chapter='01 / URBAN STRUCTURE';title=['Roofs. Walls.','Real spatial depth.'];sub='Original points · actual trained-model predictions'
        elif a==9: chapter='01 / TREE CROWNS';title=['Read the canopy.','See the ground.'];sub='Elevated perspective · original FRACTAL geometry'
        elif a==13: chapter='01 / A SECOND REAL SCENE';title=['Seven classes.','Model errors retained.'];sub='One frozen 62,103-parameter 3D U-Net'
        elif a==17:
            chapter='02 / ORIGINAL → NEURAL CLASSIFICATION';title=['Same geometry.','Same moving camera.'];sub='Original XYZ: height shade / predicted semantic IDs'
            split=.12+.76*(.5-.5*math.cos(2*math.pi*(t-17)/11));x=round(split*W)
            raw=self.draw(scene,rec['eye_local'],rec['target_local'],mode='geometry')
            im.paste(raw.crop((0,0,x,H)),(0,0));ImageDraw.Draw(im).line((x,345,x,1080),fill=old.WHITE,width=2)
            rec['reveal_x']=x;rec['synchronized_camera']=True
        elif a in (28,34):
            detail=int(a==34);u=(t-a+1/FPS)/6
            stop=min(400654,max(1,round(u*400654)))
            im,center,indices=self.map_overlay(rec,stop,detail)
            chapter=f'03 / RECORDED MAP DELTAS · CROP {detail+1}'
            title=['Point observations.','Bounded map updates.'];sub='Source predictions + actual 1 m cell-centre annotations'
            rec.update(map_event_stop=stop,map_crop_center_xy=center.tolist(),map_crop_radius_m=18,replay_repeat=detail==1,recorded_point_timestamp_ns=int(self.source_times[stop-1]),visible_recent_delta_cells=len(indices))
        elif a==40: chapter='04 / QUALIFIED RTX 4090 REPLAY';title=['Measured throughput.'];sub='Fresh neural classifications + map observation updates/s'
        else: chapter='05 / PUBLIC R4 BYOM FOUNDATION';title=['SynRex Native','LiDAR Streaming'];sub='C++ / CUDA / TensorRT'
        im=self.stage(im,chapter,title,sub)
        if a==0:
            self.text(im,48,1130,'Qualified scene A · sustained 60-second runs',31,old.WHITE,True)
            self.text(im,48,1182,'Scene-specific result · compact trained model',28,old.MUTED)
        elif a in (5,9,13):self.legend(im,1120)
        elif a==17:
            self.text(im,48,1100,'ORIGINAL XYZ',27,old.WHITE,True);self.text(im,690,1100,'PREDICTED',27,old.WHITE,True)
            self.text(im,48,1150,'48.756% held-out mIoU · seven-class FRACTAL',30,old.WHITE,True)
            self.text(im,48,1200,'Bounded engineering proof · accuracy limits retained',27,old.MUTED)
        elif a in (28,34):
            n=int(self.events['new'][:stop].sum());ts=int(self.source_times[stop-1]);elapsed=(ts-self.capture_start)/1e9
            self.text(im,48,1106,f'{n:,} new cells · {stop-n:,} reobservations',30,old.WHITE,True)
            self.text(im,48,1152,f'Replay clock +{elapsed:.6f} s · event {stop:,} / 400,654',26,old.MUTED)
            self.text(im,48,1195,'White: new / cyan: reobserved · recent-event window',26,old.MUTED)
            self.text(im,48,1233,'6 s replay · spatial crop · repeated capture' if a==34 else '4.16 s capture → 6 s · event-count retiming',25,old.MUTED)
        elif a==40:
            # Narrow lower card keeps buildings visible throughout benchmark chapter.
            d=ImageDraw.Draw(im);d.rectangle((32,860,1048,1245),fill='#091522')
            if t<42.4:
                self.text(im,48,878,'~2.0M/s',58,old.ACCENT,True)
                self.text(im,48,956,'Scene A · qualified 60-second RTX replay runs',29,old.WHITE,True)
                self.text(im,48,1020,'~1.5M/s · accepted dense / full-churn trials',32,old.WHITE,True)
                self.text(im,48,1090,'Fresh source classifications + map observations',29,old.MUTED)
                self.text(im,48,1156,'Recorded replay · scene-specific small-model result',27,old.MUTED)
            else:
                self.text(im,48,878,'600,014,848',58,old.ACCENT,True)
                self.text(im,48,955,'Freshly classified points · separate 10-minute run',29,old.WHITE,True)
                self.text(im,48,1020,'~1.0M/s · full churn',38,old.WHITE,True)
                self.text(im,48,1083,'100% point coverage · zero drops',35,old.WHITE,True)
                self.text(im,48,1156,'Separate from the ~2M/s scene-A tests',28,old.MUTED)
        else:
            self.text(im,48,1110,'github.com/Juurikko/',39,old.ACCENT,True)
            self.text(im,48,1164,'synrex-jetson-lidar-edge',39,old.ACCENT,True)
            self.text(im,48,1224,'R4 foundation public · R2 local release candidate',26,old.MUTED)
        if metadata:self.cameras.append(rec)
        return im

def encode(film,path,start,end):
    old.encode(film,path,start,end)

def provenance(film):
    old.provenance(film)
    p=json.loads((OUT/'evidence/VISUAL_PROVENANCE_RC2.json').read_text())
    p.update(schema='synrex.streaming_r2.rc3.visual_provenance.v1',renderer_sha256=sha(__file__),map_representation='Real classified source cloud with fixed-radius 3px round outlines at actual recent delta cell centres. No cubes, surfaces, occupancy enlargement or invented geometry.',map_timing=[dict(film=[28,34],events=[0,400654],timing='4.156687229 s capture stretched to 6 s, uniform event-count retiming'),dict(film=[34,40],events=[0,400654],timing='Same capture repeated with different 18 m XY crop and viewpoint')],map_detail_crop='18m XY radius, centers [25,25] and [15,35]; cumulative counters cover full capture',timestamp='Actual recorded per-source-point Unix ns, emulated replay clock; paired by verified source ID. No sensor clock or per-delta wall-clock claim.',semantic_map_scope='Bounded 1m representation, not reconstruction. Incoming source observations, frozen fresh predictions and native upsert events remain distinct.')
    save(OUT/'evidence/VISUAL_PROVENANCE_RC3.json',p)
    (OUT/'evidence/VISUAL_PROVENANCE_RC2.json').unlink()

def proof():
    n=old.verify_inputs();film=Film()
    encode(film,OUT/'media/camera-proof.mp4',0,60)
    encode(film,OUT/'media/ending-proof.mp4',45*30,49*30)
    for t in [0,1.5,5,8,9,12,13,16,17,22,27,28,30,33.9,34,37,39.9,41,43,45,47,48.9]:film.frame(round(t*30)).save(OUT/f'media/proof-{t:g}.png')
    save(OUT/'evidence/PROOF_RENDER_RC3.json',dict(status='AWAITING_VISUAL_REVIEW',verified_payloads=n,renderer_sha256=sha(__file__)))
    provenance(film)

def render():
    proof=json.loads((OUT/'evidence/PROOF_REVIEW_RC3.json').read_text());assert proof['status']=='PASS'
    assert proof['renderer_sha256']==sha(__file__)
    film=Film();encode(film,OUT/'media/synrex-streaming-r2-rc3.mp4',0,49*30)
    save(OUT/'evidence/CAMERA_FRAMES_RC3.json',film.cameras)
    save(OUT/'evidence/TEXT_LAYOUT_RC3.json',list({(r['text'],tuple(r['box'])):r for r in film.text_boxes}.values()))
    film.frame(0,False).save(OUT/'media/thumbnail.png');provenance(film)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['proof','render'],required=True);args=p.parse_args();globals()[args.stage]()
