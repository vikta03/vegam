"""S3 transfer with checksums and explicit keys; no listing-dependent recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from .transport import atomic_json, digest


class Store:
    def __init__(self, bucket, endpoint, region, prefix='nextchess', client=None, control=False, keep_checkpoints=0):
        if not endpoint.startswith('https://') or '..' in prefix.split('/'):
            raise ValueError('HTTPS endpoint and safe prefix required')
        self.bucket, self.prefix = bucket, prefix.strip('/')
        if not isinstance(keep_checkpoints, int) or keep_checkpoints < 0 or keep_checkpoints == 1:
            raise ValueError('retain at least two checkpoints, or disable retention')
        self.keep_checkpoints = keep_checkpoints
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client('s3', endpoint_url=endpoint, region_name=region,
                                  config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
                                                request_checksum_calculation='when_required',
                                                response_checksum_validation='when_required',
                                                connect_timeout=3 if control else 10, read_timeout=5 if control else 60,
                                                retries={'total_max_attempts': 1 if control else 4, 'mode': 'standard'}))
        self.client = client

    @classmethod
    def from_config(cls, path, **overrides):
        config = json.loads(Path(path).read_text())
        config.update(overrides)
        return cls(**config)

    def key(self, suffix):
        if suffix.startswith('/') or '..' in suffix.split('/'):
            raise ValueError('unsafe object key')
        return '/'.join(filter(None, [self.prefix, suffix]))

    def receipt_key(self, key):
        return self.key('.upload-receipts/' + hashlib.sha256(key.encode()).hexdigest() + '.json')

    @staticmethod
    def missing(error):
        code = str(getattr(error, 'response', {}).get('Error', {}).get('Code', ''))
        return code in {'404', 'NoSuchKey', 'NotFound'}

    def matches(self, suffix, sha, size):
        """Resume hint, not full remote SHA verification; get() hashes bytes.

        RunPod EU-RO-1 currently drops user metadata. For that case use a small
        receipt bound to the object's opaque ETag. ETags are identity tokens,
        never interpreted as SHA256 or as a multipart content checksum.
        """
        key = self.key(suffix)
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            if self.missing(error): return False
            raise
        if head['ContentLength'] != size: return False
        remote_sha = head.get('Metadata', {}).get('sha256')
        if remote_sha is not None: return remote_sha == sha
        etag = head.get('ETag')
        if not etag: return False
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self.receipt_key(key))['Body']
        except Exception as error:
            if self.missing(error): return False
            raise
        try:
            raw = body.read(16385)
        finally: body.close()
        if len(raw) > 16384: return False
        try: receipt = json.loads(raw)
        except (ValueError, UnicodeError): return False
        return (isinstance(receipt, dict) and receipt.get('schema') == 'next-upload-object-v1'
                and receipt.get('key') == key and receipt.get('sha256') == sha
                and receipt.get('bytes') == size and receipt.get('etag') == etag)

    def put(self, path, suffix, *, verify_readback=False, progress=None):
        path = Path(path)
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        def transfer():
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != identity:
                raise ValueError('source changed before upload retry')
            return self._put(path, suffix, verify_readback=verify_readback, progress=progress)
        return self.retry_transfer(transfer)

    @staticmethod
    def retry_transfer(operation):
        # RunPod can return nonstandard HTTP 520 for UploadPart, which the
        # SDK's normal retry policy does not cover. Retry the same operation,
        # never publish an unverified pointer or swallow integrity failures.
        for attempt in range(3):
            try:
                return operation()
            except Exception as error:
                response = getattr(error, 'response', {})
                code = str(response.get('ResponseMetadata', {}).get('HTTPStatusCode', ''))
                transient = (code == '429' or (code.isdigit() and 500 <= int(code) < 600)
                    or (type(error).__name__ == 'S3UploadFailedError'
                        and re.search(r'\((?:429|5\d\d)\)', str(error)) is not None)
                    or type(error).__name__ in ('EndpointConnectionError', 'ConnectionClosedError',
                                                'ReadTimeoutError', 'ConnectTimeoutError'))
                if not transient or attempt == 2:
                    raise
                time.sleep(2**(attempt+1))

    def _put(self, path, suffix, *, verify_readback=False, progress=None):
        path = Path(path)
        before = path.stat()
        sha = digest(path)
        key = self.key(suffix)
        # Bound concurrent multipart uploads; avoid ten simultaneous requests
        # against the volume endpoint for a single checkpoint.
        from boto3.s3.transfer import TransferConfig
        kwargs = {'ExtraArgs': {'Metadata': {'sha256': sha}},
                  'Config': TransferConfig(max_concurrency=2)}
        if progress is not None: kwargs['Callback'] = progress
        self.client.upload_file(str(path), self.bucket, key, **kwargs)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('source changed during upload')
        head = self.client.head_object(Bucket=self.bucket, Key=key)
        remote_sha = head.get('Metadata', {}).get('sha256')
        if head['ContentLength'] != before.st_size or (remote_sha is not None and remote_sha != sha):
            raise ValueError('uploaded object metadata mismatch')
        if verify_readback:
            body = self.client.get_object(Bucket=self.bucket, Key=key)['Body']
            h = hashlib.sha256()
            try:
                for chunk in iter(lambda: body.read(8*1024*1024), b''): h.update(chunk)
            finally: body.close()
            if h.hexdigest() != sha: raise ValueError('remote checkpoint checksum mismatch')
        result = dict(key=key, sha256=sha, bytes=before.st_size, readback_verified=verify_readback,
                      metadata_sha256_available=remote_sha is not None)
        if remote_sha is None and head.get('ETag'):
            receipt = dict(result, schema='next-upload-object-v1', etag=head['ETag'])
            self.client.put_object(Bucket=self.bucket, Key=self.receipt_key(key),
                                   Body=(json.dumps(receipt)+'\n').encode(), ContentType='application/json')
        return result

    def get(self, suffix, path, sha):
        path = Path(path)
        if path.exists() and digest(path) == sha: return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix+'.download')
        self.client.download_file(self.bucket, self.key(suffix), str(tmp))
        if digest(tmp) != sha:
            raise ValueError(f'download checksum mismatch: {path.name}')
        os.replace(tmp, path)

    def checkpoint(self, path, status):
        suffix = 'checkpoints/'+status['sha256']+'.pt'
        obj = self.put(path, suffix, verify_readback=True)
        manifest = dict(status, remote_checkpoint=obj)
        # Publish the pointer only AFTER a complete checkpoint readback agrees.
        self.retry_transfer(lambda: self.client.put_object(Bucket=self.bucket, Key=self.key('latest.json'),
                               Body=(json.dumps(manifest, indent=2)+'\n').encode(), ContentType='application/json'))
        if self.keep_checkpoints:
            self._retain_checkpoints(path, manifest)
        return manifest

    def _retain_checkpoints(self, path, manifest):
        """Prune only this run's recorded checkpoints after verified publication.

        Keep complete-pass checkpoints indefinitely and a rolling recovery set.
        No bucket listing, old-run cleanup, dataset deletion or unverified
        replacement is permitted. A failed cleanup leaves extra copies.
        """
        ledger_key = self.key('checkpoint-retention.json')
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=ledger_key)['Body']
            try: ledger = json.loads(body.read(1024*1024))
            finally: body.close()
            if ledger.get('schema') != 'next-checkpoint-retention-v1' or ledger.get('prefix') != self.prefix:
                raise ValueError('retention ledger identity differs')
            entries = ledger['entries']
        except Exception as error:
            if not self.missing(error):
                raise
            entries = []
        current = manifest['sha256']
        partition = manifest['partition_positions']
        complete_pass = manifest['positions'] > 0 and manifest['positions'] % partition == 0
        entry = dict(sha256=current, file=Path(path).name, positions=manifest['positions'],
                     milestone=complete_pass)
        # Duplicate final/periodic saves can have the same serialized content.
        entries = [e for e in entries if e['sha256'] != current] + [entry]
        for e in entries:
            if (not re.fullmatch(r'[0-9a-f]{64}', e['sha256'])
                    or not re.fullmatch(r'step-\d{9}\.pt', e['file'])):
                raise ValueError('unsafe retention entry')
        recent = {e['sha256'] for e in entries[-self.keep_checkpoints:]}
        kept = [e for e in entries if e['sha256'] in recent or e['milestone']]
        dropped = [e for e in entries if e not in kept]
        if complete_pass:
            self.retry_transfer(lambda: self.client.put_object(Bucket=self.bucket,
                Key=self.key(f"epochs/{manifest['positions']//partition}.json"),
                Body=(json.dumps(manifest)+'\n').encode(), ContentType='application/json'))
        # Record deletion intentions first; retrying after a crash is idempotent.
        ledger = dict(schema='next-checkpoint-retention-v1', prefix=self.prefix, entries=entries)
        self.retry_transfer(lambda: self.client.put_object(Bucket=self.bucket, Key=ledger_key,
            Body=(json.dumps(ledger)+'\n').encode(), ContentType='application/json'))
        for e in dropped:
            self.retry_transfer(lambda e=e: self.client.delete_object(Bucket=self.bucket,
                Key=self.key('checkpoints/'+e['sha256']+'.pt')))
            local = Path(path).parent/e['file']
            # A resumed checkpoint may have another directory or the same step
            # filename with different bytes. Never unlink such a file.
            if local != Path(path) and local.exists() and digest(local) == e['sha256']:
                local.unlink()
        ledger['entries'] = kept
        self.retry_transfer(lambda: self.client.put_object(Bucket=self.bucket, Key=ledger_key,
            Body=(json.dumps(ledger)+'\n').encode(), ContentType='application/json'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--direction', choices=['upload', 'download'], required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--receipt', required=True)
    a = p.parse_args(); store = Store.from_config(a.config)
    items = json.loads(Path(a.inventory).read_text())['files']
    finished = []
    for item in items:
        relative = Path(item['name'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('unsafe inventory path')
        path = Path(item['source']) if a.direction == 'upload' and 'source' in item else Path(a.root)/relative
        key = 'corpus/'+item['name']
        if a.direction == 'upload':
            if digest(path) != item['sha256']: raise ValueError('source differs from inventory')
            exists = store.matches(key, item['sha256'], item['bytes'])
            if not exists: store.put(path, key)
        else: store.get(key, path, item['sha256'])
        finished.append(item['name'])
        atomic_json(a.receipt, dict(direction=a.direction, complete=len(finished)==len(items), verified_files=finished))
        print(json.dumps(dict(completed=len(finished), total=len(items), file=item['name'])), flush=True)


if __name__ == '__main__': main()
