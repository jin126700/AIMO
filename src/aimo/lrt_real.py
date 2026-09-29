"""실제 FP32 Page의 label-free LRT 실행과 frozen gate."""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import LRTConfig
from .flow_representation import digest, split_ids
from .lrt import LRTModel, MacroNormStats, build_pair_tensors, fit_macro_norm_stats, landmark_mask, query_folds
from .lrt_experiment import (
    Rank4LinearTransport, TrainMeanTransport, ZeroTransport, audit_macro_page,
    evaluate_transport, load_lrt_checkpoint, original_balanced_mean, save_lrt_checkpoint,
    train_lrt, transport_loss, transport_term,
)
from .macro_page import to_macro_page
from .page import load_pages
from .runtime import atomic_save, atomic_write_json as write


def ids_hash(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest()


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'frozen artifact mismatch: {path}')
    else:
        write(path, value)


def source_manifest(source):
    """허용된 세 metadata 파일만 읽고 Page는 아직 열지 않습니다."""
    source = Path(source).resolve()
    names = ['collection/groups.json', 'pages/index.json', 'protocol/policy.json']
    groups, index, policy = [json.loads((source / n).read_text()) for n in names]
    expected = policy['observation_provenance']
    if expected['forward_dtype'] != 'float32' or expected['tf32'] or expected['autocast']:
        raise ValueError('FP32 source provenance required')
    originals = {}
    for g in groups:
        oid, split = g['original_id'], g['split']
        if oid in originals:
            raise ValueError('duplicate original')
        entries = []
        for prompt in g['prompts']:
            pid = prompt['prompt_id']
            item = index[pid]
            path = Path(item['path']).resolve()
            if not path.is_relative_to(source / 'pages' / 'artifacts'):
                raise ValueError('Page path outside allowlist')
            if item['original_id'] != oid or item['split'] != split:
                raise ValueError('source ID/split mismatch')
            entries.append({'prompt_id': pid, 'path': str(path), 'sha256': item['sha256']})
        if sum(e['prompt_id'] == oid for e in entries) != 1:
            raise ValueError('exactly one original Page required')
        originals[oid] = {'split': split, 'pages': entries}
    return {'source': str(source), 'metadata_hashes': {n: digest(source / n) for n in names},
            'expected_provenance': expected, 'originals': originals,
            'behavior_files_read': False}


def load_fp32_page_source(manifest, ids=None):
    """명시된 original ID만 load; known-test는 gate 이후 호출합니다."""
    if not isinstance(manifest, dict):
        manifest = source_manifest(manifest)
    if ids is None:
        ids = [i for i, r in manifest['originals'].items() if r['split'] == 'train']
    pairs = []
    for oid in ids:
        row = manifest['originals'][oid]
        pages = {}
        for item in row['pages']:
            if digest(item['path']) != item['sha256']:
                raise ValueError('source checksum mismatch')
            with np.load(item['path'], allow_pickle=False) as raw:
                if raw['state_0'].dtype != np.float32 or raw['updates_0'].dtype != np.float32:
                    raise ValueError('stored Page dtype must be FP32')
            loaded = load_pages(item['path'])
            if len(loaded) != 1:
                raise ValueError('expected single Page artifact')
            page = loaded[0]
            if page.original_id != oid or page.variant_id != item['prompt_id']:
                raise ValueError('Page original/variant ID mismatch')
            for key, value in manifest['expected_provenance'].items():
                if page.provenance.get(key) != value:
                    raise ValueError(f'Page provenance mismatch: {key}')
            if page.state.dtype != torch.float32 or page.updates.dtype != torch.float32:
                raise ValueError('non-FP32 Page')
            page.validate(tol=1e-3)  # 기존 Qwen FP32 extractor와 동일한 tolerance
            pages[item['prompt_id']] = page
        pairs.extend((pages[oid], p) for pid, p in pages.items() if pid != oid)
    if not pairs:
        raise ValueError('empty selected source')
    return pairs


def macro_pairs(pairs, fine=False):
    cache = {}
    def convert(p):
        key = (p.original_id, p.variant_id)
        if key not in cache:
            cache[key] = to_macro_page(p, p.n_blocks if fine else 8)
        return cache[key]
    return [(convert(o), convert(v)) for o, v in pairs]


def unique_originals(pairs):
    return list({o.original_id: o for o, _ in pairs}.values())


def freeze_floor(root, pairs, stats, ids):
    values, zero = [], 0
    for o, v in pairs:
        payload = build_pair_tensors(o, v, stats)
        valid = landmark_mask(o, v)
        for fold in query_folds(8):
            target = payload['delta_norm'][0, list(fold)]
            term = transport_term(target, torch.zeros_like(target), valid, 1e-12)
            if v.is_identity or term.mse_zero == 0:
                zero += 1
            else:
                values.append(term.mse_zero)
    if not values:
        raise ValueError('no positive train relation energy')
    qs = np.quantile(values, [.01, .05, .5, .95])
    result = {'rule': 'max(1e-12, train-positive per-scalar zero MSE q05); identity excluded',
              'train_ids_hash': ids_hash(ids), 'number_positive': len(values), 'number_zero': zero,
              'q01': float(qs[0]), 'q05': float(qs[1]), 'median': float(qs[2]),
              'q95': float(qs[3]), 'selected_tau': max(1e-12, float(qs[1]))}
    freeze(root / 'floor_manifest.json', result)
    return result['selected_tau']


def evaluate_fine(pairs, stats, baseline, floor, mode='all_common'):
    rows, per = [], {}
    for o, v in pairs:
        payload = build_pair_tensors(o, v, stats)
        mask = landmark_mask(o, v, mode)
        landmarks = torch.where(mask)[0]
        if not len(landmarks):
            raise ValueError('missing landmarks')
        bounds = [(g * o.n_macro) // 8 for g in range(9)]
        for fold in query_folds(8):
            layers = tuple(range(bounds[fold[0]], bounds[fold[-1] + 1]))
            target = payload['delta_norm'][0, list(layers)][:, landmarks]
            prediction = baseline.predict(payload, layers, landmarks)[0]
            t = transport_term(target, prediction, mask[landmarks], floor)
            per.setdefault(o.original_id, []).append(t.ratio)
            rows.append({'original_id': o.original_id, 'variant_id': v.variant_id,
                         'fold': list(fold), 'query_layers': list(layers), 'mse_pred': t.mse_pred,
                         'mse_zero': t.mse_zero, 'ratio': t.ratio, 'below_floor': t.below_floor,
                         'n_scalar': t.n_scalar, 'n_cells': t.n_cells})
    return {'transport_ratio': original_balanced_mean(per), 'records': rows,
            'per_original': {k: float(np.mean(v)) for k, v in per.items()},
            'landmark_mode': mode, 'n_originals': len(per)}


def bridge_decision(fine, macro):
    return 'MACROPAGE_COMPRESSION_LIMIT' if fine < .98 and macro >= 1 else 'PROCEED_MACRO_LRT'


def prepare(source, root):
    root.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest(source)
    freeze(root / 'source_manifest.json', manifest)
    splits = {name: sorted(i for i, r in manifest['originals'].items() if r['split'] == label)
              for name, label in [('train', 'train'), ('validation', 'validation'),
                                  ('known_test', 'known_original_test')]}
    splits['lrt_train'], splits['lrt_dev'] = split_ids(splits['train'])
    splits['ids_hashes'] = {k: ids_hash(v) for k, v in splits.items()}
    freeze(root / 'split_manifest.json', splits)
    spec = {'schema': 'lrt-real-v1-mse', 'source': str(Path(source).resolve()),
            'train_seeds': [0, 1, 2], 'eval_seed': 0, 'bootstrap_seed': 0, 'bootstrap_samples': 2000,
            'architecture': dataclasses.asdict(LRTConfig()), 'max_epochs': 200, 'patience': 40,
            'lr': 3e-4, 'weight_decay': 1e-3, 'grad_clip': 1.,
            'bridge_split': 'lrt_dev', 'bridge_gate': 'fine < .98 and macro >= 1 => stop',
            'validation_gate': 'mean R < 1; >=2 seeds R<1; mean swap_gap>0; >=2 seeds gap>0',
            'normalization_rank4_trainmean': 'lrt_train originals only',
            'fine_normalization': 'fine train original per-layer/stream RMS',
            'fine_floor': 'same frozen macro tau; report below-floor per fold',
            'checkpoint_selection': 'lrt_dev transport_ratio only',
            'known_test_access': 'only after primary validation gate B',
            'primary_mode': 'all_common', 'diagnostic_mode': 'final_token'}
    freeze(root / 'spec.json', spec)
    if (root / 'prepared.json').exists():
        return
    print('prepare: train Pages / normalization / floor', flush=True)
    train_fine = load_fp32_page_source(manifest, splits['lrt_train'])
    train = macro_pairs(train_fine)
    stats = fit_macro_norm_stats(unique_originals(train))
    tau = freeze_floor(root, train, stats, splits['lrt_train'])
    atomic_save(stats.state_dict(), root / 'normalization.pt')
    write(root / 'normalization.json', {'hash': stats.hash(), 'train_ids_hash': ids_hash(splits['lrt_train']),
                                      'n_originals': stats.n_originals})
    print('prepare: train-only rank4 Macro basis', flush=True)
    rank = Rank4LinearTransport.fit(train, stats)
    mean = TrainMeanTransport.fit(train, stats)
    fine_train = macro_pairs(train_fine, fine=True)
    fine_stats = fit_macro_norm_stats(unique_originals(fine_train))
    print('prepare: train-only rank4 Fine basis', flush=True)
    fine_rank = Rank4LinearTransport.fit(fine_train, fine_stats)
    atomic_save({'rank4': rank.basis, 'mean': mean.mean, 'fine_rank4': fine_rank.basis,
                 'fine_stats': fine_stats.state_dict()}, root / 'baselines' / 'fitted.pt')
    del fine_train, train_fine
    print('prepare: dev4 bridge audit (external validation/test unopened)', flush=True)
    dev_fine = load_fp32_page_source(manifest, splits['lrt_dev'])
    dev = macro_pairs(dev_fine)
    audit = audit_macro_page(dev_fine)
    c = np.concatenate([np.asarray(r.pop('cancellation_values')).reshape(8, -1, 2)
                        for r in audit['pairs']], axis=1)
    quant = lambda x: dict(zip(['p10', 'p50', 'p90'], np.quantile(x, [.1, .5, .9]).tolist()))
    audit['cancellation'] = {'overall': quant(c), 'median': float(np.median(c)),
                             'by_macro': {str(g): quant(c[g]) for g in range(8)},
                             'by_stream': {str(k): quant(c[:, :, k]) for k in range(2)}}
    audit['audit_split'] = 'lrt_dev4; normalization and basis train28 only'
    macro_result = evaluate_transport(None, dev, stats, floor=tau, baseline=rank)
    fine_result = evaluate_fine(macro_pairs(dev_fine, fine=True), fine_stats, fine_rank, tau)
    write(root / 'baselines' / 'bridge_macro.json', macro_result)
    write(root / 'baselines' / 'bridge_fine.json', fine_result)
    audit['rank4_macro_transport_ratio'] = macro_result['transport_ratio']
    audit['rank4_fine_transport_ratio'] = fine_result['transport_ratio']
    audit['bridge_decision'] = bridge_decision(fine_result['transport_ratio'], macro_result['transport_ratio'])
    write(root / 'macro_audit.json', audit)
    write(root / 'prepared.json', {'prepared': True, 'time': time.time(), 'bridge': audit['bridge_decision']})
    print(json.dumps({'bridge': audit['bridge_decision'], 'fine_R': fine_result['transport_ratio'],
                      'macro_R': macro_result['transport_ratio'], 'tau': tau}), flush=True)


def precision():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def cuda_smoke(root, pairs, stats, cfg):
    precision()
    torch.cuda.reset_peak_memory_stats()
    model = LRTModel(hidden_size=pairs[0][0].hidden_size, n_macro=8,
                     n_landmarks=pairs[0][0].n_landmarks).cuda()
    tensors = build_pair_tensors(*pairs[0], stats.to(torch.device('cuda')))
    if not all(v.device.type == 'cuda' and v.dtype == torch.float32 for v in tensors.values()):
        raise ValueError('CUDA FP32 tensor path')
    mask = landmark_mask(*pairs[0]); landmarks = torch.where(mask)[0].cuda()
    fold = (0, 1); support = list(range(2, 8))
    begin = time.monotonic()
    out = model(tensors['delta_norm'], tensors['original_state_norm'], tensors['original_update_norm'],
                support, fold, landmarks, tensors['relative_positions'])
    target = tensors['delta_norm'][0, :2][:, landmarks]
    loss = transport_loss(target, out.delta_hat[0], mask[landmarks.cpu()].cuda(), cfg.denominator_floor)
    loss.backward(); torch.cuda.synchronize()
    if not torch.isfinite(loss) or not all(p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError('nonfinite or non-FP32 model')
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(g.dtype == torch.float32 and bool(torch.isfinite(g).all()) for g in grads):
        raise ValueError('gradient dtype/finite check')
    save_lrt_checkpoint(root / 'training' / 'smoke.pt', {'model': model, 'best': {}, 'history': []}, cfg, stats)
    reloaded, _, _ = load_lrt_checkpoint(root / 'training' / 'smoke.pt')
    if not all(torch.equal(v.cpu(), reloaded.state_dict()[k]) for k, v in model.state_dict().items()):
        raise ValueError('checkpoint roundtrip mismatch')
    result = {'passed': True, 'forward_backward_seconds': time.monotonic()-begin,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'loss': float(loss),
              'model_input_gradient_dtype': 'float32', 'autocast': False, 'tf32': False,
              'checkpoint_roundtrip': True}
    write(root / 'training' / 'cuda_smoke.json', result)
    (root / 'training' / 'smoke.pt').unlink()
    report = model.param_report().as_dict()
    o = pairs[0][0]
    report.update(fine_sequence_logical_sites=o.n_blocks*o.n_landmarks*2,
                  macro_relation_logical_sites=8*o.n_landmarks*2,
                  support_attention_sites=6*o.n_landmarks*2,
                  note='sequence depth reduction differs from adapter parameter reduction; shared IO adapters unchanged')
    write(root / 'parameter_report.json', report)
    print('CUDA smoke: '+json.dumps(result), flush=True)
    del model, reloaded, tensors, loss, out, grads
    torch.cuda.empty_cache()


def decision(results, rank):
    ratios = [r['transport_ratio'] for r in results]
    gaps = [r['swap_gap'] for r in results]
    passed = np.mean(ratios) < 1 and sum(x < 1 for x in ratios) >= 2 and np.mean(gaps) > 0 and sum(x > 0 for x in gaps) >= 2
    label = 'RELATIONAL_SIGNAL_NOT_ESTABLISHED'
    if passed:
        label = 'LRT_ADDS_OVER_LINEAR' if np.mean(ratios) < rank else 'RELATION_SIGNAL_LINEAR_BASELINE_SUFFICIENT'
    return {'decision': label, 'validation_gate_B': bool(passed), 'mean_transport_ratio': float(np.mean(ratios)),
            'mean_swap_gap': float(np.mean(gaps)), 'seeds_R_below_one': sum(x < 1 for x in ratios),
            'seeds_positive_swap_gap': sum(x > 0 for x in gaps),
            'signal_status': 'RELATIONAL_SIGNAL_ESTABLISHED_PRELIMINARY' if passed else label}


def bootstrap(left, right, seed=0):
    ids = sorted(left)
    if set(ids) != set(right):
        raise ValueError('paired bootstrap original mismatch')
    delta = np.array([left[i]-right[i] for i in ids])
    rng = np.random.default_rng(seed)
    sample = delta[rng.integers(0, len(ids), (2000, len(ids)))].mean(1)
    return {'difference': 'left minus right; negative favors left', 'mean': float(delta.mean()),
            'ci95': np.quantile(sample, [.025, .975]).tolist(), 'n_originals': len(ids),
            'resamples': 2000, 'unit': 'original', 'seed': seed}


def evaluate_split(root, name, manifest, ids, stats, cfg, fitted):
    cached = root / 'evaluation' / name / 'results.json'
    if cached.exists():
        return json.loads(cached.read_text()), json.loads((cached.parent/'bootstrap.json').read_text())
    pairs = macro_pairs(load_fp32_page_source(manifest, ids))
    result, boot = {}, {}
    for mode in ['all_common', 'final_token']:
        controls = {'zero': ZeroTransport(), 'train_mean': TrainMeanTransport(fitted['mean']),
                    'rank4_macro': Rank4LinearTransport(fitted['rank4'])}
        rows = {k: evaluate_transport(None, pairs, stats, floor=cfg.denominator_floor,
                                     landmark_mode=mode, baseline=b) for k, b in controls.items()}
        for seed in [0, 1, 2]:
            model, _, _ = load_lrt_checkpoint(root / 'training' / f'seed{seed}' / 'best.pt')
            rows[f'seed{seed}'] = evaluate_transport(model.cuda(), pairs, stats,
                            floor=cfg.denominator_floor, landmark_mode=mode, support_swap=True, eval_seed=0)
            row = rows[f'seed{seed}']
            boot[f'{mode}/seed{seed}'] = {
                'lrt_minus_zero': bootstrap(row['per_original'], rows['zero']['per_original']),
                'lrt_minus_rank4': bootstrap(row['per_original'], rows['rank4_macro']['per_original']),
                'correct_minus_swap': bootstrap(row['per_original'], row['swap_per_original'])}
            del model
        result[mode] = rows
    write(root / 'evaluation' / name / 'results.json', result)
    write(root / 'evaluation' / name / 'bootstrap.json', boot)
    return result, boot


def verify_source(manifest, read_ids):
    source = Path(manifest['source'])
    for name, checksum in manifest['metadata_hashes'].items():
        if digest(source/name) != checksum:
            raise ValueError('source metadata changed')
    for oid in read_ids:
        for entry in manifest['originals'][oid]['pages']:
            if digest(entry['path']) != entry['sha256']:
                raise ValueError('source Page changed')


def report(root):
    get = lambda n: json.loads((root/n).read_text())
    dec, audit, floor, splits = [get(n) for n in ['decision.json', 'macro_audit.json', 'floor_manifest.json', 'split_manifest.json']]
    lines = [f'# {root.name}', '', '목적: FP32 Fine→Macro relation 보존과 learned z_rel 추가 설명력 검증.',
             'Stage1 label-free; correctness/drop/recipe 미사용. 새 generation/extraction 없음.',
             f"판정: {dec['decision']}", '',
             f"Fine rank4 R={audit['rank4_fine_transport_ratio']:.8g}; Macro rank4 R={audit['rank4_macro_transport_ratio']:.8g} (dev4 bridge)",
             f"Macro audit: {json.dumps({k:v for k,v in audit.items() if k not in ['pairs','notes']}, ensure_ascii=False)}",
             f"tau: {json.dumps(floor, ensure_ascii=False)}",
             f"Original IDs hashes: {json.dumps(splits['ids_hashes'])}",
             f"Source manifest hash: {digest(root/'source_manifest.json')}",
             f"Code: {json.dumps(get('code_provenance.json'))}"]
    for name in ['validation', 'known_test']:
        path = root / 'evaluation' / name / 'results.json'
        if not path.exists():
            continue
        lines += ['', f'## {name}', 'known_test는 secondary held-out confirmation; untouched final이 아님.' if name=='known_test' else 'validation8 primary architecture evaluation.',
                  '|mode|model|R|swap R|swap gap|below floor|', '|---|---|---:|---:|---:|---:|']
        for mode, rows in json.loads(path.read_text()).items():
            for model, r in rows.items():
                lines.append(f"|{mode}|{model}|{r['transport_ratio']:.8g}|{r.get('swap_transport_ratio','')}|{r.get('swap_gap','')}|{r.get('n_below_denominator_floor','')}|")
        lines += ['', 'Original-level paired bootstrap 2000:', json.dumps(get(f'evaluation/{name}/bootstrap.json'), ensure_ascii=False)]
    lines += ['', '## 판단과 한계', json.dumps(dec, ensure_ascii=False),
              'n=8 uncertainty; ordinal alignment과 macro cancellation은 diagnostic이며 인과/robustness 결론이 아님.',
              'Historical E-FLOW-1: Zero flow_total=0.742116, learned=0.744428. Transport ratio와 다른 metric.',
              'Fine control은 U4-like low-rank transport control이며 historical U4 재현이 아님.',
              f"Runtime: {json.dumps(get('completed.json'))}", f'Artifacts: {root}',
              f'Reproduce: python -m aimo lrt-experiment --source {get("source_manifest.json")["source"]} --run-dir {root} --execute-gpu']
    if (root/'parameter_report.json').exists():
        lines += ['Parameters / logical sites: '+json.dumps(get('parameter_report.json'))]
    for seed in [0,1,2]:
        p=root/'training'/f'seed{seed}'/'summary.json'
        if p.exists(): lines += [f'Seed{seed} training: '+p.read_text()]
    content = '\n'.join(lines)+'\n'
    (root/'report.md').write_text(content)
    log = Path('/data1/HKM/AIMO/Loop_result.md')
    marker = f'## {root.name}'
    existing = log.read_text() if log.exists() else ''
    if marker not in existing:
        with log.open('a') as f: f.write('\n'+marker+'\n\n'+content)


def execute(root):
    start = time.monotonic()
    get = lambda n: json.loads((root/n).read_text())
    if (root/'completed.json').exists():
        report(root)
        return
    manifest, splits, audit = [get(n) for n in ['source_manifest.json', 'split_manifest.json', 'macro_audit.json']]
    if audit['bridge_decision'] == 'MACROPAGE_COMPRESSION_LIMIT':
        write(root/'decision.json', {'decision': 'MACROPAGE_COMPRESSION_LIMIT', 'validation_gate_B': False})
        write(root/'completed.json', {'status': 'SCIENTIFIC_GATE_STOP', 'seconds': time.monotonic()-start})
        report(root)
        return
    precision()
    cfg = dataclasses.replace(LRTConfig(), denominator_floor=get('floor_manifest.json')['selected_tau'])
    stats = MacroNormStats.from_state_dict(torch.load(root/'normalization.pt', weights_only=False))
    train = macro_pairs(load_fp32_page_source(manifest, splits['lrt_train']))
    dev = macro_pairs(load_fp32_page_source(manifest, splits['lrt_dev']))
    if not (root/'training'/'cuda_smoke.json').exists():
        cuda_smoke(root, train, stats, cfg)
    for seed in [0,1,2]:
        folder = root/'training'/f'seed{seed}'
        if (folder/'summary.json').exists():
            continue
        write(root/'status.json', {'stage': 'training', 'seed': seed, 'time': time.time()})
        begin = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        result = train_lrt(train, dev, stats, cfg, floor=cfg.denominator_floor,
                           seed=seed, device=torch.device('cuda'), checkpoint_dir=folder)
        save_lrt_checkpoint(folder/'best.pt', result, cfg, stats)
        write(folder/'summary.json', {'best': result['best'], 'epochs_run': result['epochs_run'],
                                     'seconds': time.monotonic()-begin,
                                     'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                     'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        del result
        torch.cuda.empty_cache()
    fitted = torch.load(root/'baselines'/'fitted.pt', weights_only=False)
    write(root/'status.json', {'stage': 'validation', 'time': time.time()})
    val, boot = evaluate_split(root, 'validation', manifest, splits['validation'], stats, cfg, fitted)
    primary = val['all_common']
    dec = decision([primary[f'seed{s}'] for s in [0,1,2]], primary['rank4_macro']['transport_ratio'])
    freeze(root/'decision.json', dec)
    all_boot = {'validation': boot}; read_ids = splits['train']+splits['validation']
    if dec['validation_gate_B']:
        _, secondary = evaluate_split(root, 'known_test', manifest, splits['known_test'], stats, cfg, fitted)
        all_boot['known_test'] = secondary
        read_ids += splits['known_test']
    write(root/'bootstrap.json', all_boot)
    verify_source(manifest, read_ids)
    write(root/'completed.json', {'status': 'COMPLETED', 'seconds': time.monotonic()-start,
                                 'wall_seconds_since_execution_start': time.time()-get('execution_started.json')['unix_time'],
                                 'source_immutable_verified': True, 'known_test_opened': dec['validation_gate_B']})
    report(root)
    write(root/'status.json', {'stage': 'completed', 'decision': dec['decision'], 'time': time.time()})
    print(json.dumps(dec), flush=True)


def command(args):
    root = Path(args.run_dir); root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (root/'completed.json').exists():
            head = (root/'head.txt').read_text().strip() if (root/'head.txt').exists() else 'NOT_AVAILABLE'
            patch = (root/'implementation.patch').read_bytes() if (root/'implementation.patch').exists() else b''
            files = ['src/aimo/lrt.py','src/aimo/lrt_experiment.py','src/aimo/lrt_real.py',
                     'src/aimo/macro_page.py','src/aimo/cli.py','tests/test_lrt_real.py']
            write(root/'code_provenance.json', {'head': head,
                  'dirty_state': (root/'dirty_state.txt').read_text() if (root/'dirty_state.txt').exists() else 'NOT_AVAILABLE',
                  'tracked_patch_sha256': hashlib.sha256(patch).hexdigest(),
                  'files_sha256': {f:digest(f) for f in files}})
            (root/'implementation.patch').write_bytes(patch)
        if getattr(args, 'source', None):
            prepare(Path(args.source), root)
        if not (root/'prepared.json').exists():
            raise ValueError('prepare --source first')
        if getattr(args, 'execute_gpu', False):
            freeze(root/'execution_started.json', json.loads((root/'execution_started.json').read_text()) if (root/'execution_started.json').exists() else {'unix_time': time.time()})
            try:
                execute(root)
            except Exception as exc:
                write(root/'failed.json', {'type': type(exc).__name__, 'message': str(exc), 'time': time.time()})
                raise
    return 0
