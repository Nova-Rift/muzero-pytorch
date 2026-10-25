"""Replication-specific bindings around the frozen compressed-state verifier.

Integrity of all four full-state milestones is required; this does not establish
numerical split-resume parity, playing strength, or admission permission.
"""
import argparse
import json
from pathlib import Path
import replication_contract as rc
import replication_collection as collection
from verify_compressed_states import verify as verify_chain


def verify(root, source_path, contract, source_receipt, side, initialization, bound,
           collection_sha, encoding_sha):
    manifest=collection.validate_manifest(root,contract,source_receipt,side,initialization)
    receipt=collection.pinned(root/collection.RECEIPT,collection_sha)
    rc.require(type(receipt.get('schema')) is int and receipt['schema']==1 and receipt.get('kind')=='replication-collection-v1' and receipt.get('mode')=='train' and receipt.get('complete') is True and
        type(receipt.get('optimizer_updates')) is int and receipt['optimizer_updates']==0 and
        collection.is_digest(receipt.get('own_reference_sha256')) and
        (collection.is_digest(receipt.get('paired_reference_sha256')) if side=='candidate' else receipt.get('paired_reference_sha256') is None) and
        receipt.get('side')==side and receipt.get('variant')==contract['variant'] and
        rc.exact(receipt.get('bindings'),bound) and
        receipt.get('guard_sha256')==rc.digest(Path(collection.__file__)) and
        receipt.get('checked_before_first_collect') is True and receipt.get('checked_before_first_update') is True and
        receipt.get('comparisons',{}).get('own',{}).get('full_128_trajectory_parity') is True and
        receipt.get('comparisons',{}).get('own',{}).get('non_timing_aggregates_equal') is True,
        'Missing successful own first-block guard')
    rc.require(collection.init.timestamp(bound['initialization_review_completed_at'])<=collection.init.timestamp(receipt['started_at'])<=collection.init.timestamp(receipt['completed_at'])<=collection.init.timestamp(collection.init.now()), 'Recovery collection chronology differs')
    rc.require(receipt.get('paired_parity')==collection.parity_semantics(side,contract),'Collection policy semantics differ')
    keys={'own','paired'} if side=='candidate' and contract['variant']=='MIR' else {'own'}
    rc.require(set(receipt.get('comparisons',{}))==keys,'Recovery parity set differs')
    if side=='candidate' and contract['variant']=='MIR':
        rc.require(receipt.get('comparisons',{}).get('paired',{}).get('full_128_trajectory_parity') is True and
            receipt['comparisons']['paired'].get('non_timing_aggregates_equal') is True,'MIR paired parity missing')
    if side=='candidate' and contract['variant']=='EGO':
        rc.require(receipt.get('paired_parity')=='inapplicable-relative-aim-changes-policy' and
            'paired' not in receipt.get('comparisons',{}),'False EGO paired parity claim')
    rc.require(set(receipt.get('files',{}))==set(collection.FILES),'Incomplete collection artifact set')
    for name,sha in receipt['files'].items():rc.require(rc.digest(root/name)==sha,'First-block artifact changed')
    encoded=collection.pinned(root/'compressed/manifest.json',encoding_sha)
    rc.require(type(encoded.get('encoded_bytes')) is int and 0<encoded['encoded_bytes']<=600000000,'Encoded recovery allowance exceeded')
    report=json.loads((root/'report.json').read_text())
    rc.require(report.get('checkpoint_sha256')==rc.digest(root/'weights.npz'),'Endpoint weight binding differs')
    rc.require(rc.exact(json.loads(source_path.read_text()),source_receipt),'Recovery source receipt differs')
    result=verify_chain(root,source_path,'arena/train_loss.py','replication-'+side+'-'+contract['variant'],
        expected=(512,1024,1536,2048),expected_loss_contract='baseline',expected_updates_per_game=1)
    result.update(kind='replication-recovery-v1',training_seed=rc.TRAINING_SEED,side=side,
        variant=contract['variant'],bindings=bound,collection_receipt_sha256=collection_sha,
        encoded_manifest_sha256=encoding_sha,recovery_adapter_sha256=rc.digest(Path(__file__)),
        verifier_sha256=rc.digest(Path(__file__).with_name('verify_compressed_states.py')))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','source-receipt','initialization-dir','initialization-review','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('source-receipt','contract','initialization','initialization-review','collection','encoding'):p.add_argument('--'+key+'-sha256',required=True)
    p.add_argument('--side',choices=('candidate','paired'),required=True);a=p.parse_args()
    source=collection.pinned(a.source_receipt,a.source_receipt_sha256)
    contract,initialization,bound=collection.bindings(a.initialization_dir,a.contract_sha256,a.initialization_sha256,
        a.initialization_review,a.initialization_review_sha256,source)
    rc.require(not a.output.exists(),'Fresh recovery receipt required')
    result=verify(a.root,a.source_receipt,contract,source,a.side,initialization,bound,a.collection_sha256,a.encoding_sha256)
    collection.write(a.output,result)

if __name__=='__main__':main()
