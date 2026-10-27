# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""RC2 media-only perspective point rendering from immutable, verified RC1 payloads.

No inferred geometry, reference-label colours, neural execution or sensor pose creation.
Camera positions are illustrative local-coordinate view transforms, never sensor poses.
Run with .venv/bin/python tools/streaming_r2_rc2.py --stage proof|render|qa|package.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import laspy
import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/streaming-r2-review/streaming-r2-rc2'
BASE = OUT / 'inputs/streaming-r2-rc1'
W, H, FPS, DURATION = 1080, 1350, 30, 49
PALETTE = np.array([[138,146,151],[167,124,67],[44,155,89],[242,89,75],
                    [53,123,217],[185,110,208],[247,182,46]],dtype=np.uint8)
NAMES = ['Other','Ground','Vegetation','Building','Water','Bridge','Permanent structure']
WHITE, MUTED, ACCENT = '#f3f8fc', '#b5c6d4', '#48e0c1'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
BOLD = FONT.with_name('DejaVuSans-Bold.ttf')

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

@njit(cache=True)
def raster(xyz, colors, eye, right, up, forward, focal, point_size, flags):
    """Perspective opaque point splats, true nearest-depth buffer; no surfaces or fill."""
    arr=np.empty((H,W,3),np.uint8)
    for y in range(H):
        for x in range(W):
            arr[y,x,0]=7+int(4*y/H)
            arr[y,x,1]=15+int(7*y/H)
            arr[y,x,2]=25+int(9*y/H)
    depth=np.full((H,W),1.e20,np.float32)
    for i in range(len(xyz)):
        px=xyz[i,0]-eye[0]; py=xyz[i,1]-eye[1]; pz=xyz[i,2]-eye[2]
        z=px*forward[0]+py*forward[1]+pz*forward[2]
        if z<1.5: continue
        sx=int(W/2+focal*(px*right[0]+py*right[1]+pz*right[2])/z)
        sy=int(785-focal*(px*up[0]+py*up[1]+pz*up[2])/z)
        if sx<0 or sx>=W or sy<0 or sy>=H: continue
        rad=min(5,max(1,int(point_size*focal/z)))
        flag=flags[i]
        ring=2 if flag else 0
        for dy in range(-rad-ring,rad+ring+1):
            for dx in range(-rad-ring,rad+ring+1):
                xx=sx+dx; yy=sy+dy
                if xx<0 or xx>=W or yy<0 or yy>=H or dx*dx+dy*dy>(rad+ring)**2: continue
                if z>=depth[yy,xx]: continue
                depth[yy,xx]=z
                # White=new; cyan=reobserved. These are explicitly event outlines.
                if flag and dx*dx+dy*dy>rad*rad:
                    arr[yy,xx,0]=242 if flag==1 else 72
                    arr[yy,xx,1]=248 if flag==1 else 224
                    arr[yy,xx,2]=252 if flag==1 else 193
                else:
                    arr[yy,xx]=colors[i]
    return arr


@njit(cache=True)
def voxel_raster(xyz,colors,eye,right,up,forward,flags):
    """Rasterize literal 1m map voxels (0.84m display cubes, no spatial interpolation)."""
    arr=np.empty((H,W,3),np.uint8)
    arr[:,:,0]=9;arr[:,:,1]=19;arr[:,:,2]=30
    zbuf=np.full((H,W),1.e20,np.float32)
    corners=np.array([[-1.,-1.,-1.],[1.,-1.,-1.],[1.,1.,-1.],[-1.,1.,-1.],
                      [-1.,-1.,1.],[1.,-1.,1.],[1.,1.,1.],[-1.,1.,1.]])*.42
    faces=np.array([[0,3,7,4],[1,5,6,2],[0,4,5,1],[3,2,6,7],[0,1,2,3],[4,7,6,5]])
    for i in range(len(xyz)):
        screen=np.empty((8,3),np.float64)
        valid=True
        for k in range(8):
            p=xyz[i]+corners[k]-eye;z=p[0]*forward[0]+p[1]*forward[1]+p[2]*forward[2]
            if z<1.5:valid=False;break
            screen[k,0]=W/2+850*(p[0]*right[0]+p[1]*right[1]+p[2]*right[2])/z
            screen[k,1]=785-850*(p[0]*up[0]+p[1]*up[1]+p[2]*up[2])/z
            screen[k,2]=1/z
        if not valid:continue
        for axis in range(3):
            side=1 if eye[axis]>xyz[i,axis] else 0
            face=faces[axis*2+side]
            shade=.65+axis*.15
            q=screen[face]
            for tri in range(2):
                ids=np.array([0,1,2]) if tri==0 else np.array([0,2,3])
                a,b,c=q[ids[0]],q[ids[1]],q[ids[2]]
                den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
                if abs(den)<.001:continue
                xmin=max(0,int(min(a[0],b[0],c[0])));xmax=min(W-1,int(max(a[0],b[0],c[0]))+1)
                ymin=max(0,int(min(a[1],b[1],c[1])));ymax=min(H-1,int(max(a[1],b[1],c[1]))+1)
                for yy in range(ymin,ymax+1):
                    for xx in range(xmin,xmax+1):
                        x=xx+.5;y=yy+.5
                        u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
                        v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
                        w=1-u-v
                        if min(u,v,w)<0:continue
                        z=1/(u*a[2]+v*b[2]+w*c[2])
                        if z>=zbuf[yy,xx]:continue
                        zbuf[yy,xx]=z;edge=False
                        if flags[i]:
                            for k in range(4):
                                aa=q[k];bb=q[(k+1)%4];dx=bb[0]-aa[0];dy=bb[1]-aa[1]
                                if abs(dx*(y-aa[1])-dy*(x-aa[0]))<1.5*math.sqrt(dx*dx+dy*dy):edge=True
                        if edge:
                            arr[yy,xx,0]=242 if flags[i]==1 else 72
                            arr[yy,xx,1]=248 if flags[i]==1 else 224
                            arr[yy,xx,2]=252 if flags[i]==1 else 193
                        else:
                            for k in range(3):arr[yy,xx,k]=int(colors[i,k]*shade)
    return arr


def basis(eye,target):
    f=target-eye; f/=np.linalg.norm(f)
    r=np.cross(f,np.array([0.,0.,1.])); r/=np.linalg.norm(r)
    return r,np.cross(r,f),f


def mix(a,b,u): return np.array(a,float)*(1-u)+np.array(b,float)*u


def verify_inputs():
    """Verify complete extracted baseline, then reconstruct native map and predictions anew."""
    manifest=json.loads((BASE/'RELEASE_MANIFEST.json').read_text())
    print('manifest keys', list(manifest), flush=True)
    entries=manifest['files']
    for rel,rec in entries.items():
        p=BASE/rel
        assert p.is_file() and sha(p)==rec['sha256'],rel
    rec=BASE/'recording'; proof=json.loads((rec/'verification.json').read_text())
    for name,key in [('input.jsonl','input_sha256'),('events.jsonl','events_sha256'),
                     ('predictions.u8','predictions_sha256'),('snapshots.npz','snapshots_sha256'),
                     ('native-receipt.json','receipt_sha256')]:
        assert sha(rec/name)==proof[key], name
    source=BASE/'qgis/data/source/TEST-0944_6564-005900112.laz'
    assert sha(source)==proof['source_sha256']
    # Existing verifier independently checks each point, original ID, timestamp and upsert.
    fresh=OUT/'evidence/reverified-recording'; fresh.mkdir(exist_ok=True)
    from streaming_r2_replay import verify
    for name in ['input.jsonl','input-provenance.json','native-receipt.json','events.jsonl']:
        if not (fresh/name).exists(): shutil.copy2(rec/name,fresh/name)
    for name in ['verification.json','predictions.u8','snapshots.npz']:
        if (fresh/name).exists(): (fresh/name).unlink()
    verify(source,fresh,BASE/'models/unet.srxmodel')
    assert sha(fresh/'predictions.u8')==proof['predictions_sha256']
    a=np.load(fresh/'snapshots.npz'); b=np.load(rec/'snapshots.npz')
    assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    return len(entries)


class Film:
    def __init__(self):
        self.fonts={}; self.text_boxes=[]; self.cameras=[]
        self.scenes={}; self.bindings=[]
        sys.path.insert(0,str(ROOT/'src'))
        from verify_semantics import verify_semantics
        for name,stem in [('urban','TEST-0438_6393-002595911'),('tower','TEST-0741_6244-006630086')]:
            q=BASE/'qgis-r4-foundation'; source=q/'data/source'/f'{stem}.laz'
            side=q/'data/semantics'/f'{stem}-semantics.json'
            output=q/'data/classified'/f'{stem}-classified.laz'
            binding=q/'data/semantics'/f'{stem}-binding.json'
            verification=verify_semantics(source,output,side,q/'model/unet.srxmodel',binding,ROOT/'examples/unet/semantic_classes.json')
            cloud=laspy.read(source); xyz=np.column_stack((cloud.x,cloud.y,cloud.z))
            origin=xyz.min(0); xyz-=origin
            labels=np.array(json.loads(side.read_text())['semantic_ids'],np.uint8)
            self.scenes[name]=(xyz,labels,origin)
            self.bindings.append(dict(scene=name,source=str(source.relative_to(OUT)),source_sha256=sha(source),
                sidecar_sha256=sha(side),classified_sha256=sha(output),binding_sha256=sha(binding),
                local_origin=origin.tolist(),points=len(xyz),verification=verification))
        source=BASE/'qgis/data/source/TEST-0944_6564-005900112.laz'
        c=laspy.read(source); xyz=np.column_stack((c.x,c.y,c.z)); origin=xyz.min(0); xyz-=origin
        labels=np.fromfile(BASE/'recording/predictions.u8',np.uint8)
        self.scenes['forest']=(xyz,labels,origin)
        self.build_events()

    def build_events(self):
        cache=OUT/'evidence/map-events-derived.npz'
        if not cache.exists():
            indices=[]; ids=[]; isnew=[]; changes=[]; keys=[]; keyindex={}; state={}
            for line in (BASE/'recording/events.jsonl').open():
                row=json.loads(line)
                if row['schema']!='synrex.semantic_delta.v1': continue
                assert row['evicted'] is None
                key=tuple(row['key']); label=row['semantic_id']; new=key not in keyindex
                if new: keyindex[key]=len(keys); keys.append(key)
                indices.append(keyindex[key]); ids.append(label); isnew.append(new)
                changes.append(not new and state[key]!=label); state[key]=label
            np.savez_compressed(cache,keys=np.array(keys),index=np.array(indices,np.int32),
                labels=np.array(ids,np.uint8),new=np.array(isnew),changed=np.array(changes))
        e=np.load(cache); self.events={k:e[k] for k in e.files}
        self.mapxyz=self.events['keys']+.5-self.scenes['forest'][2]
        self.maplabels=np.zeros(len(self.mapxyz),np.uint8)
        self.last=np.full(len(self.mapxyz),-1,np.int32)
        self.first=np.full(len(self.mapxyz),-1,np.int32)
        for j,k in enumerate(self.events['index']):
            if self.first[k]<0: self.first[k]=j
        self.event_pos=0
        # Focus chosen by actual event activity within a compact XY region, not class.
        centers=[]
        for lo,hi in [(0,200000),(200000,len(self.events['index']))]:
            xy=self.mapxyz[self.events['index'][lo:hi],:2]
            hist,xe,ye=np.histogram2d(xy[:,0],xy[:,1],bins=5,range=((0,50),(0,50)))
            a,b=np.unravel_index(hist.argmax(),hist.shape); centers.append([int(5+10*a),int(5+10*b)])
        self.focus=centers
        self.active_ordinals=[];self.detail_targets=[]
        for detail,(lo,hi) in enumerate([(0,230000),(170000,len(self.events['index']))]):
            cx,cy=self.focus[detail]
            valid=np.linalg.norm(self.mapxyz[:,:2]-np.array([cx,cy]),axis=1)<14
            selected=np.flatnonzero(valid[self.events['index'][lo:hi]])+lo
            self.active_ordinals.append(selected)
            self.detail_targets.append(np.array([cx,cy,float(np.median(self.mapxyz[valid,2]))]))

    def text(self,im,x,y,value,size=32,color=WHITE,bold=False):
        k=(size,bold)
        if k not in self.fonts:self.fonts[k]=ImageFont.truetype(str(BOLD if bold else FONT),size)
        d=ImageDraw.Draw(im); box=d.textbbox((x,y),value,font=self.fonts[k])
        assert 34<=box[0] and box[2]<=W-32 and box[3]<=H-24,(value,box)
        d.text((x,y),value,font=self.fonts[k],fill=color)
        self.text_boxes.append(dict(text=value,size=size,box=box))

    def draw(self,scene,eye,target,mode='prediction',mapstate=None,mapcrop=None):
        xyz,labels,_=self.scenes[scene]
        if mapstate is not None:
            # Exact event order; visible state only from accepted events to this ordinal.
            stop=mapstate
            if stop<self.event_pos:
                self.event_pos=0;self.last[:]=-1
            for j in range(self.event_pos,stop):
                k=self.events['index'][j];self.maplabels[k]=self.events['labels'][j];self.last[k]=j
            self.event_pos=stop
            valid=self.last>=0
            if mapcrop is not None: valid &= np.linalg.norm(self.mapxyz[:,:2]-mapcrop[:2],axis=1)<14
            xyz=self.mapxyz[valid]; labels=self.maplabels[valid]
            latest=self.last[valid];first=self.first[valid]
            recent=latest>=max(0,stop-9000)
            flags=np.where(recent,np.where(first>=max(0,stop-9000),1,2),0).astype(np.uint8)
            colors=PALETTE[labels]; size=.16
        else:
            flags=np.zeros(len(xyz),np.uint8);size=.055
            if mode=='geometry':
                h=xyz[:,2]/max(1,xyz[:,2].max());colors=np.clip(np.array([105,147,171])+h[:,None]*65,0,255).astype(np.uint8)
            else: colors=PALETTE[labels]
        eye=np.array(eye,float);target=np.array(target,float);r,u,f=basis(eye,target)
        return Image.fromarray(voxel_raster(xyz,colors,eye,r,u,f,flags) if mapstate is not None else raster(xyz,colors,eye,r,u,f,850.,size,flags))

    def legend(self,im,y=1115):
        d=ImageDraw.Draw(im)
        for i,name in enumerate(NAMES):
            x=48+(i%3)*330; yy=y+(i//3)*37
            d.ellipse((x,yy+10,x+12,yy+22),fill=tuple(PALETTE[i]))
            self.text(im,x+22,yy,name,24)

    def chrome(self,im,chapter,title,sub):
        # Soft, reproducible UI scrims leave a continuous full-frame 3D stage.
        a=np.array(im).copy()
        for y in range(0,430):
            weight=.88*(1-y/430)**.6
            a[y]=(a[y]*(1-weight)+np.array([6,13,23])*weight).astype(np.uint8)
        for y in range(1030,H):
            weight=min(.94,(y-1030)/90*.94)
            a[y]=(a[y]*(1-weight)+np.array([6,13,23])*weight).astype(np.uint8)
        im=Image.fromarray(a);d=ImageDraw.Draw(im)
        self.text(im,48,35,'SynRex / NATIVE LiDAR',27,ACCENT,True)
        self.text(im,48,85,chapter,24,MUTED)
        for n,line in enumerate(title):self.text(im,46,133+n*68,line,58,WHITE,True)
        self.text(im,48,133+len(title)*68+10,sub,28,MUTED)
        self.text(im,48,1260,'IGN FRACTAL • Etalab 2.0 • source update 2025-04-05',22,MUTED)
        self.text(im,48,1298,'Recorded data • illustrative camera • no sensor trajectory',22,MUTED)
        return im

    def frame(self,i,metadata=True,camera_only=False):
        t=i/FPS; mapstop=None; mapcrop=None; mode='prediction'; split=None
        if t<4:
            scene='urban';u=t/4;eye=mix([56,-12,19],[39,-4,15],u);target=mix([19,31,11],[18,34,12],u)
            chapter='RTX 4090 • recorded LiDAR replay'
            title=['UP TO 2 MILLION','LiDAR points / second']
            sub='Native GPU classification + semantic mapping'
        elif t<8:
            scene='urban';u=(t-4)/4;eye=mix([7,-1,13],[27,6,15],u);target=mix([14,34,13],[20,39,14],u)
            chapter='01 / CLOSE TO THE GEOMETRY';title=['Real points.','Actual neural predictions.'];sub='Low oblique tracking • frozen trained U-Net'
        elif t<11:
            scene='forest';u=(t-8)/3;eye=mix([-4,-8,19],[14,-3,21],u);target=mix([22,27,18],[27,29,20],u)
            chapter='01 / VEGETATION PASS';title=['Depth you can see.'];sub='Original point geometry • no reconstructed surfaces'
        elif t<14:
            scene='tower';u=(t-11)/3;angle=-1.05+u*.55;eye=np.array([25+44*math.cos(angle),25+44*math.sin(angle),19]);target=np.array([25,29,20.])
            chapter='01 / CONTROLLED ORBIT';title=['One frozen model.','Different real scenes.'];sub='Seven semantic channels • errors retained'
        elif t<25:
            scene='urban';u=(t-14)/11;eye=mix([45,-9,18],[9,1,14],u);target=mix([18,33,12],[18,34,13],u)
            chapter='02 / ALIGNED INPUT → PREDICTION';title=['Same points.','Same moving camera.'];sub='Input: height shade • output: trained U-Net classes'
            # Two full-size renders share exactly one camera. Reveal, then reverse.
            split=.18+.64*(.5-.5*math.cos(2*math.pi*(t-14)/11))
        elif t<38:
            scene='forest';v=t-25
            if v<3:
                u=(v+1/FPS)/3;mapstop=max(1,min(len(self.events['index']),round(u*len(self.events['index']))))
                eye=mix([-13,-16,29],[8,-13,27],u);target=np.array([25,27,19.])
                chapter='03 / GENUINE NATIVE MAP DELTAS';title=['A map that updates.'];sub='Full capture: 4.16 s → 3 s • event-count retiming'
            else:
                detail=0 if v<8 else 1;u=(v-3+1/FPS)/5 if detail==0 else (v-8+1/FPS)/5
                active=self.active_ordinals[detail]
                mapstop=int(active[min(len(active)-1,int(u*len(active)))])+1
                target=self.detail_targets[detail];mapcrop=target.copy()
                angle=-1.2+u*.5;eye=target+np.array([27*math.cos(angle),27*math.sin(angle),10-2*u])
                chapter=f'03 / DETAIL REPLAY {detail+1} OF 2 • SAME CAPTURE'
                title=['New cells.','Real reobservations.'];sub='Active-crop retiming • emulated replay clock'
        elif t<45:
            scene='urban';u=(t-38)/7;eye=mix([54,6,21],[30,-4,17],u);target=np.array([18,32,12.])
            chapter='04 / QUALIFIED RTX 4090 REPLAY';title=['Measured.','With a defined scope.'];sub='Fresh neural classifications + map observation updates'
        else:
            scene='urban';u=(t-45)/4;eye=mix([2,-9,14],[18,-2,14],u);target=np.array([18,34,13.])
            chapter='05 / MIT BYOM FOUNDATION';title=['Native C++ / CUDA','TensorRT'];sub='Public R4 foundation • streaming R2 not yet public'
        if camera_only: return dict(frame=i,scene=scene,eye=eye.tolist(),target=target.tolist(),mapstop=mapstop)
        im=self.draw(scene,eye,target,mapstate=mapstop,mapcrop=mapcrop)
        if split is not None:
            raw=self.draw(scene,eye,target,mode='geometry');x=int(split*W)
            im.paste(raw.crop((0,0,x,H)),(0,0));d=ImageDraw.Draw(im)
            d.line((x,420,x,1065),fill=WHITE,width=3)
            d.rounded_rectangle((x-17,732,x+17,790),radius=13,fill=WHITE)
        im=self.chrome(im,chapter,title,sub)
        if t<4:
            self.text(im,48,1100,'Qualified scene A • 60-second replay tests',31,WHITE,True)
            self.text(im,48,1147,'Fresh neural classifications + map observation updates/s',27,MUTED)
            self.text(im,48,1192,'Scene-specific small-model result',27,MUTED)
        elif t<14:
            self.legend(im)
            self.text(im,48,1067,{'urban':'Urban scene • 83,641 source points','forest':'Vegetation scene • 400,654 source points','tower':'Second urban scene • 85,222 source points'}[scene],29,WHITE,True)
        elif t<25:
            self.text(im,48,1035,'INPUT GEOMETRY',26,WHITE,True)
            self.text(im,720,1035,'PREDICTIONS',26,WHITE,True)
            self.legend(im,1083)
            self.text(im,48,1204,'Compact FRACTAL U-Net • held-out mIoU 48.756%',28,WHITE,True)
        elif t<38:
            nnew=int(self.events['new'][:mapstop].sum()); nup=mapstop-nnew
            self.text(im,48,1058,f'{nnew:,} new cells   /   {nup:,} reobservations',32,WHITE,True)
            self.text(im,48,1107,'○ WHITE: new    ○ CYAN: updated existing cell',29,MUTED)
            self.text(im,48,1150,'Semantic voxel fill • outlines show recent event type',27,MUTED)
            self.text(im,48,1193,'Reobserved ≠ changed class • inactive intervals skipped',27,MUTED)
        elif t<45:
            # Readable sequential metric cards over continuously moving geometry.
            overlay=Image.new('RGBA',(W,H));d=ImageDraw.Draw(overlay)
            d.rounded_rectangle((40,425,1040,1010),radius=18,fill=(7,18,30,211))
            im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
            if t<41.25:
                self.text(im,65,453,'UP TO 2.0M/s',77,ACCENT,True)
                self.text(im,65,561,'Scene A • two qualified 60 s replay tests',32,WHITE,True)
                self.text(im,65,629,'1.5M/s',65,WHITE,True)
                self.text(im,65,718,'Qualified dense + full-churn tests',32,MUTED)
                self.text(im,65,799,'Fresh neural classifications and',33)
                self.text(im,65,846,'semantic-map observation updates',33)
                self.text(im,65,927,'Small model • scene-specific • recorded replay',27,MUTED)
            elif t<45:
                self.text(im,65,453,'600,014,848',74,ACCENT,True)
                self.text(im,65,557,'freshly classified points in 10 minutes',33,WHITE,True)
                self.text(im,65,643,'100% coverage',54,WHITE,True)
                self.text(im,65,722,'Zero drops',54,WHITE,True)
                self.text(im,65,829,'Approximately 1.0M/s in this ten-minute run',30,MUTED)
                self.text(im,65,922,'Separate from the 2.0M/s scene-A tests',28,MUTED)
            else:
                self.text(im,65,453,'48.756% mIoU',64,WHITE,True)
                self.text(im,65,555,'Held-out seven-class FRACTAL evaluation',30,MUTED)
                self.text(im,65,647,'Compact trained U-Net',41,WHITE,True)
                self.text(im,65,734,'Useful output. Visible model errors.',32,MUTED)
                self.text(im,65,827,'No state-of-the-art quality claim',31,MUTED)
            self.text(im,48,1093,'RTX 4090 • recorded LiDAR replay',30,WHITE,True)
            self.text(im,48,1144,'No live sensor / Jetson AGX / Thor qualification',28,MUTED)
            self.text(im,48,1194,'Accepted benchmark receipts retained in review archive',26,MUTED)
        else:
            overlay=Image.new('RGBA',(W,H));ImageDraw.Draw(overlay).rectangle((32,922,1048,1074),fill=(6,13,23,220))
            im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
            self.text(im,48,940,'github.com/Juurikko/',43,ACCENT,True)
            self.text(im,48,1002,'synrex-jetson-lidar-edge',43,ACCENT,True)
            self.text(im,48,1100,'MIT native source • model/data licences separate',29,WHITE,True)
            self.text(im,48,1150,'Tuomas Pietilä • with OpenAI Codex assistance',27,MUTED)
            self.text(im,48,1195,'Streaming R2 RC2 • local owner review',28,MUTED)
        ImageDraw.Draw(im).line((48,1240,48+984*t/DURATION,1240),fill=ACCENT,width=3)
        if metadata:self.cameras.append(dict(frame=i,time=t,scene=scene,eye_local=eye.tolist(),target_local=target.tolist(),
            focal_pixels=850.,principal_point=[540,785],map_event_stop=mapstop,map_crop_center=None if mapcrop is None else mapcrop.tolist(),map_crop_radius=14 if mapcrop is not None else None,reveal_x=None if split is None else int(split*W)))
        return im


def encode(film,path,start,end):
    command=[FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
        '-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    with path.with_suffix('.encode.log').open('w') as log:
        p=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for i in range(start,end):
                p.stdin.write(film.frame(i).tobytes())
                if i%60==0:print(f'{path.name}: {i/FPS:.1f}s',flush=True)
        finally:p.stdin.close()
        assert p.wait()==0


def proof():
    n=verify_inputs(); film=Film();media=OUT/'media';media.mkdir(exist_ok=True)
    encode(film,media/'opening-proof.mp4',0,4*FPS)
    for t in [0,1,3,5,9,12,16,20,24,25.5,27.5,30,32,35,37,39,42,44,47]:
        film.frame(round(t*FPS)).save(media/f'proof-{t:g}.png')
    save(OUT/'evidence/OPENING_PROOF_RC2.json',dict(status='RENDERED_AWAITING_VISUAL_REVIEW',baseline_payloads_verified=n,
        proof_sha256=sha(media/'opening-proof.mp4'),frames=120,fps=30,seconds=4))
    provenance(film)


def provenance(film):
    p=json.loads((BASE/'recording/verification.json').read_text())
    save(OUT/'evidence/VISUAL_PROVENANCE_RC2.json',dict(schema='synrex.streaming_r2.rc2.visual_provenance.v1',status='PASS',
        baseline_source_seal='eb4cab0',candidate_source='e33679c',renderer='Deterministic CPU perspective point splats, Numba z buffer; no meshes or inpainting',
        renderer_sha256=sha(__file__),font_sha256=sha(FONT),bold_font_sha256=sha(BOLD),palette=PALETTE.tolist(),classes=NAMES,
        scene_bindings=film.bindings,forest_recording=p,derived_map_event_cache_sha256=sha(OUT/'evidence/map-events-derived.npz'),
        native_events=len(film.events['index']),new_cells=int(film.events['new'].sum()),reobservations=int((~film.events['new']).sum()),
        semantic_changes=int(film.events['changed'].sum()),
        map_representation='Actual 1m cell centres displayed as 0.84m cubes with gaps; flat face shading of the same semantic palette. No invented occupancy. Event-outline annotation.',
        map_timing=[dict(film=[25,28],events=[0,400654],timing='4.156687229s native replay compressed to 3s; uniform event-count retiming, 1.386x total duration'),
                    dict(film=[28,33],events=[0,230000],timing='Same capture detail replay; uniform active-crop event count, outside-crop gaps skipped'),
                    dict(film=[33,38],events=[170000,400654],timing='Same capture detail replay; uniform active-crop event count, outside-crop gaps skipped')],
        map_focus_xy_local=film.focus,map_detail_crop='14m XY radius around actual activity-density peak; event-count retiming skips outside-crop intervals; controlled illustrative orbit',point_decimation='NONE: all source points rendered; pixel occlusion only',
        geometry='Original XYZ minus recorded local origin, no deformation; cell centres floor(XYZ)+0.5',
        camera='Illustrative view only. Never sensor motion or acquisition trajectory.',
        prediction_source='Frozen actual trained U-Net outputs only. Reference labels never rendered.',
        historical_diagnostic_latency=dict(status='FAIL',p99_ms=274.477963,scope='Historical JSON recording; excluded from accepted benchmark claims'),
        publication='NOT_RUN'))


def render():
    receipt=json.loads((OUT/'evidence/OPENING_PROOF_RC2.json').read_text())
    assert receipt['status']=='PASS','Review opening before full render'
    film=Film();encode(film,OUT/'media/synrex-streaming-r2-rc2.mp4',0,DURATION*FPS)
    save(OUT/'evidence/CAMERA_FRAMES_RC2.json',film.cameras)
    save(OUT/'evidence/TEXT_LAYOUT_RC2.json',list({(r['text'],tuple(r['box'])):r for r in film.text_boxes}.values()))
    film.frame(0,False).save(OUT/'media/thumbnail.png')
    provenance(film)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['proof','render'],required=True)
    a=p.parse_args();globals()[a.stage]()

if __name__=='__main__':main()
