# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independent RC4 archive check using only the standard library and archive bytes.

Does not import the packager or trust its in-memory payload table. Checks the embedded
manifest and checksum file, every compressed member, gzip EOF/CRC and the external seal.
"""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    args=parser.parse_args();archive=args.archive.resolve()
    prefix='streaming-r2-rc4/'
    with tarfile.open(archive,'r:gz') as tar:
        checksum_bytes=tar.extractfile(prefix+'SHA256SUMS').read()
        manifest_bytes=tar.extractfile(prefix+'MANIFEST_RC4.json').read()
        report=json.load(tar.extractfile(prefix+'evidence/FINAL_REPORT_RC4.json'))
    expected={}
    for line in checksum_bytes.decode().splitlines():
        value,path=line.split('  ',1)
        assert path not in expected and len(value)==64
        assert not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts
        expected[path]=value
    assert expected['MANIFEST_RC4.json']==hashlib.sha256(manifest_bytes).hexdigest()
    manifest=json.loads(manifest_bytes)
    assert set(expected)==set(manifest['files'])|{'MANIFEST_RC4.json'}
    assert all(expected[p]==row['sha256'] for p,row in manifest['files'].items())
    expected['SHA256SUMS']=hashlib.sha256(checksum_bytes).hexdigest()
    checked=set();total=0
    with tarfile.open(archive,'r:gz') as tar:
        for entry in tar:
            assert entry.name.startswith(prefix) and entry.isfile()
            path=entry.name[len(prefix):]
            assert path in expected and path not in checked,path
            actual=hashlib.file_digest(tar.extractfile(entry),'sha256').hexdigest()
            assert actual==expected[path],path
            if path in manifest['files']:
                assert entry.size==manifest['files'][path]['bytes'],path
            checked.add(path);total+=entry.size
    assert checked==set(expected)
    uncompressed=0
    with gzip.open(archive,'rb') as stream:
        while block:=stream.read(8*1024*1024):
            uncompressed+=len(block)
    assert uncompressed>total
    actual_archive=digest(archive)
    external=json.loads((archive.parent/'ARCHIVE_INTEGRITY_RC4.json').read_text())
    assert external['archive_sha256']==actual_archive
    assert external['video_sha256']==expected['media/synrex-streaming-r2-rc4.mp4']==report['video_sha256']
    assert external['source_commit']==manifest['source_commit']==report['source_commit']
    assert all(v=='PASS' for k,v in report['gates'].items() if k!='PUBLICATION')
    assert report['gates']['PUBLICATION']=='NOT_RUN'
    receipt=dict(status='PASS',archive_sha256=actual_archive,video_sha256=report['video_sha256'],
        source_commit=report['source_commit'],payloads_checked=len(checked),payload_bytes=total,
        gzip_bytes_read_to_eof=uncompressed,gzip_crc_and_trailer='PASS',
        method='Separate standard-library verifier; embedded checksums/manifest parsed from archive; every member read and SHA256 compared; full gzip EOF/CRC; external seal cross-check',
        verifier_sha256=digest(Path(__file__)),publication='NOT_RUN')
    (archive.parent/'INDEPENDENT_READBACK_RC4.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
