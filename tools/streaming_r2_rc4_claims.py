# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Read-only numeric/source checks and draft receipts for the RC4 media edit."""
import csv
import json
import re
from pathlib import Path

from streaming_r2_rc4 import ROOT, OUT, RC3, sha, save


def main():
    rows=list(csv.DictReader((ROOT/'evidence/streaming_r1_transport_trials.csv').open()))
    accepted=[r for r in rows if r['campaign']=='streaming_r1_transport_sustained.json'
              and r['status']=='PASS' and r['requested_seconds']=='60'
              and r['binary_sha256']=='85d09ceccffdceee7ce780746b3b495bbfa613a8e9e3b555cdcb779d50f204f2']
    scene_a=[r for r in accepted if r['requested_points_s']=='2000000' and r['churn_mm']=='0']
    dense=[r for r in accepted if r['requested_points_s']=='1500000' and r['source_sha256'].startswith('50106aa5')]
    churn=[r for r in accepted if r['requested_points_s']=='1500000' and r['churn_mm']=='64000']
    assert len(scene_a)==2 and len(dense)==2 and len(churn)==4
    assert {r['map_capacity'] for r in churn}=={'4096','65536'}
    for r in scene_a+dense+churn:
        assert r['async_export']=='False' and r['complete_coverage']=='1.0'
        assert r['classification_drops']=='0' and r['failed_point_deliveries']=='0'
        assert r['freshly_classified_points_s']==r['map_updates_s']
    long=json.loads((ROOT/'evidence/streaming_r1_long_stability.json').read_text())['trials'][0]
    recv=long['native']['receive'];sub=long['native']['subscribe']
    assert long['status']=='PASS' and long['seconds']==600 and long['churn_mm']==64000
    assert recv['freshly_classified_measured_points']==recv['measured_map_updates']==600014848
    assert recv['measured_seconds']==600.026055098
    assert abs(600014848/recv['measured_seconds']-999981.3223144148)<1e-7
    assert sub['coverage']==1 and recv['service']['dropped_points']==0
    assert sub['source_to_subscriber']['p99_seconds_upper_bound']*1000==39.845888
    params=json.loads((ROOT/'evidence/unet_r2_model_freeze.json').read_text())
    assert params['model_parameters']==62103
    assert params['weights_sha256']=='788b65c42bb5d50d9509a56d38368f51890c4a93300d53578e290d193456c7c4'
    # Native test mIoU is recomputed from its independent confusion matrix.
    import numpy as np
    metrics=json.loads((ROOT/'evidence/unet_r2_independent_metrics.json').read_text())
    confusion=np.array(metrics['metrics']['native_fp16']['confusion'],dtype=np.float64)
    miou=float(np.mean(np.diag(confusion)/(confusion.sum(0)+confusion.sum(1)-np.diag(confusion))))
    assert abs(miou-.48755974311318834)<1e-14
    original=json.loads((RC3/'evidence/BENCHMARK_WORDING_RC3.json').read_text())
    assert original['diagnostic_json_p99']['status']=='FAIL' and original['diagnostic_json_p99']['ms']==274.477963
    assert original['diagnostic_json_p99']['accepted_latency_evidence'] is False
    files=['LINKEDIN_POST_MAIN_RC4.md','LINKEDIN_POST_TEASER_RC4.md','LINKEDIN_FIRST_COMMENT_RC4.md',
           'LINKEDIN_CLAIM_AUDIT_RC4.md','THUMBNAIL_CAPTION_RC4.md']
    hashes={name:sha(ROOT/'linkedin'/name) for name in files}
    text={name:(ROOT/'linkedin'/name).read_text() for name in files}
    body=re.sub(r'<!--.*?-->','',text[files[0]],flags=re.S).strip()
    assert 130<=len(body.split())<=220
    assert 3<=len(re.findall(r'(?<!\w)#[A-Za-z]+',body))<=6
    assert '[ADD VERIFIED STREAMING R2 RELEASE URL AFTER OWNER PUBLICATION]' in body
    assert 'local candidate' in text[files[1]] and 'R4 BYOM foundation is public' in text[files[1]]
    for name in files[:2]:
        for claim in ['600,014,848','39.85','48.756%','62,103','~1.5M','Jetson AGX Thor','Jetson AGX Orin','OpenAI Codex','MIT']:
            assert claim in text[name], (name,claim)
    # Check every repository evidence path named by the audit actually exists.
    audit=text['LINKEDIN_CLAIM_AUDIT_RC4.md']
    referenced=sorted(set(re.findall(r'`((?:evidence|native|tools|docs|examples)/[^`]+)`',audit)))
    for path in referenced:
        assert (ROOT/path).exists(),path
    evidence=['evidence/streaming_r1_transport_trials.csv','evidence/streaming_r1_transport_summary.json',
              'evidence/streaming_r1_long_stability.json','evidence/unet_r2_model_freeze.json',
              'evidence/unet_r2_independent_metrics.json','evidence/streaming_r2_release_candidate.json',
              'evidence/streaming_r2_rc3_benchmark_wording.json','evidence/streaming_r2_public_foundation.json']
    save(OUT/'evidence/CLAIM_AUDIT_RC4.json',dict(status='PASS',method='Independent CSV contexts, raw endurance JSON, test-confusion arithmetic and manual claim/rights review',
         evidence_sha256={p:sha(ROOT/p) for p in evidence},draft_sha256=hashes,
         scene_A_qualified_rows=scene_a,dense_qualified_rows=dense,churn_qualified_rows=churn,
         endurance=dict(fresh_points=600014848,seconds=600.026055098,points_per_second=999981.3223144148,
                        coverage=1.0,drops=0,p99_ms=39.845888,source_json_pointer='/trials/0/native'),
         quality=dict(parameters=62103,independent_test_miou=miou),
         historical_diagnostic=dict(status='FAIL',p99_ms=274.477963,accepted_latency_citation=False),
         hardware='No streaming throughput or flight qualification for Jetson AGX Thor / Jetson Thor or Jetson AGX Orin',
         public_release_status='R4 public; R2 local; main draft only for use after owner publication with verified placeholder replaced',
         official_context_read_date='2026-10-27',context_links=['https://developer.nvidia.com/embedded/jetson-modules/',
         'https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/','https://github.com/Juurikko/synrex-jetson-lidar-edge'],
         publication='NOT_RUN'))
    save(OUT/'evidence/LINKEDIN_DRAFTS_RC4.json',dict(status='PASS',draft_sha256=hashes,main_words=len(body.split()),
        main_hashtags=re.findall(r'(?<!\w)#[A-Za-z]+',body),manual_voice_and_scope_review='PASS',
        publication='NOT_RUN',release_url='[ADD VERIFIED STREAMING R2 RELEASE URL AFTER OWNER PUBLICATION]'))
    print(f'Claim contexts, independent mIoU and five drafts PASS; main {len(body.split())} words')


if __name__=='__main__':
    main()
