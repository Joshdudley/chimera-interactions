#!/usr/bin/env python3
"""Compare a ChimeraX export with PDBe Arpeggio JSON by residue pair.

    python3 compare_interactions.py 2_interactions.csv arpeggio.json \
        --output interaction_comparison.csv

Use the same structure and author chain/residue numbering for both programs.
A difference is a candidate for review, not proof that either tool is wrong.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

# There is no direct Arpeggio equivalent for our bad_hydrophobic flag.
COMPARABLE = {'hbond', 'salt_bridge', 'hydrophobic', 'polar', 'pi'}
COLUMNS = ('status', 'kind', 'chain_a', 'resnum_a', 'icode_a',
           'chain_b', 'resnum_b', 'icode_b', 'chimera_atom_pairs',
           'arpeggio_labels', 'arpeggio_atom_records')


def _residue(chain, number, icode=''):
    num = str(number).strip()
    code = str(icode or '').strip()
    if code in ('.', '?'):
        code = ''
    match = re.fullmatch(r'(-?\d+)([A-Za-z]?)', num)
    if not match:
        raise ValueError('unsupported residue number ' + repr(number))
    if not code:
        code = match[2]
    return (str(chain).strip(), int(match[1]), code)


def _pair(a, b):
    return tuple(sorted((a, b)))


def _arpeggio_kinds(record):
    labels = record.get('contact', [])
    if isinstance(labels, str):
        labels = [labels]
    tags = {str(t).upper() for t in labels}
    kinds = set()
    if 'HBOND' in tags:
        kinds.add('hbond')
    if 'IONIC' in tags:
        kinds.add('salt_bridge')
    if 'HYDROPHOBIC' in tags:
        kinds.add('hydrophobic')
    if 'POLAR' in tags:
        kinds.add('polar')
    if 'AROMATIC' in tags or 'CATIONPI' in tags or 'METSULPHURPI' in tags:
        kinds.add('pi')
    if record.get('type', '').lower() == 'plane-plane':
        kinds.add('pi')
    return kinds, tags


def _records_from_json(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('interactions', 'contacts', 'results'):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError('Expected a PDBe Arpeggio JSON list of contact objects.')


def compare(chimera_path, arpeggio_path, output_path):
    scopes = set()
    ours = defaultdict(list)
    incomplete = 0
    with open(chimera_path, newline='', encoding='utf-8-sig') as infile:
        for row in csv.DictReader(infile):
            kind = row.get('kind', '')
            if kind not in COMPARABLE:
                continue
            chains = frozenset((row['chain_a'], row['chain_b']))
            if row.get('record_type') == 'group':
                scopes.add((chains, kind))
            elif row.get('record_type') == 'contact':
                try:
                    a = _residue(row['chain_a'], row['resnum_a'], row['icode_a'])
                    b = _residue(row['chain_b'], row['resnum_b'], row['icode_b'])
                except ValueError:
                    incomplete += 1
                    continue
                key = (kind, _pair(a, b))
                scopes.add((chains, kind))
                ours[key].append(f"{row['atom_a']}:{row['atom_b']}")
    if not scopes:
        raise ValueError('No comparable chain-pair groups in the ChimeraX CSV.')

    arpeggio = defaultdict(lambda: {'tags': set(), 'records': 0})
    skipped = Counter()
    for record in _records_from_json(arpeggio_path):
        if not isinstance(record, dict):
            skipped['invalid_record'] += 1
            continue
        a, b = record.get('bgn'), record.get('end')
        if not isinstance(a, dict) or not isinstance(b, dict):
            skipped['missing_endpoints'] += 1
            continue
        try:
            ra = _residue(a['auth_asym_id'], a['auth_seq_id'], a.get('pdbx_PDB_ins_code', ''))
            rb = _residue(b['auth_asym_id'], b['auth_seq_id'], b.get('pdbx_PDB_ins_code', ''))
        except (KeyError, ValueError, TypeError):
            skipped['unmapped_residue'] += 1
            continue
        if ra[0] == rb[0]:
            skipped['same_chain'] += 1
            continue
        kinds, tags = _arpeggio_kinds(record)
        if not kinds:
            skipped['unmapped_category'] += 1
        for kind in kinds:
            if (frozenset((ra[0], rb[0])), kind) not in scopes:
                skipped['outside_analyzed_group'] += 1
                continue
            entry = arpeggio[(kind, _pair(ra, rb))]
            entry['tags'].update(tags)
            entry['records'] += 1

    result = []
    for kind, (a, b) in sorted(set(ours) | set(arpeggio)):
        key = (kind, (a, b))
        our_hits = ours.get(key, [])
        their_hit = arpeggio.get(key)
        status = ('both' if our_hits and their_hit else
                  'chimera_only' if our_hits else 'arpeggio_only')
        result.append(dict(status=status, kind=kind,
                           chain_a=a[0], resnum_a=a[1], icode_a=a[2],
                           chain_b=b[0], resnum_b=b[1], icode_b=b[2],
                           chimera_atom_pairs=';'.join(sorted(set(our_hits))),
                           arpeggio_labels=';'.join(sorted(their_hit['tags'])) if their_hit else '',
                           arpeggio_atom_records=their_hit['records'] if their_hit else 0))

    output = Path(output_path)
    with output.open('w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(result)
    totals = Counter((r['kind'], r['status']) for r in result)
    summary = output.with_name(output.stem + '_summary.txt')
    lines = ['Comparison by chain and residue pair (one result per interaction type)',
             'A difference is a review candidate; the programs use different rules.',
             f'ChimeraX group scopes compared: {len(scopes)}',
             'bad_hydrophobic excluded: Arpeggio has no equivalent category.', '']
    for kind in sorted(COMPARABLE):
        lines.append(f"{kind}: both={totals[kind, 'both']}, "
                     f"ChimeraX only={totals[kind, 'chimera_only']}, "
                     f"Arpeggio only={totals[kind, 'arpeggio_only']}")
    lines.extend(['', f'ChimeraX contact rows with invalid residue IDs: {incomplete}',
                  'Arpeggio records skipped: ' +
                  (', '.join(f'{k}={v}' for k, v in sorted(skipped.items())) or 'none'),
                  'Only chain pairs and categories present as group rows in the CSV are compared.',
                  'Chain IDs, residue numbers, and insertion codes must match the mmCIF file.'])
    summary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {output} ({len(result)} residue-pair/category results)')
    print(f'Wrote {summary}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chimera_csv', help='CSV from export_interactions in ChimeraX')
    parser.add_argument('arpeggio_json', help='PDBe Arpeggio interaction JSON')
    parser.add_argument('--output', default='interaction_comparison.csv')
    options = parser.parse_args()
    try:
        compare(options.chimera_csv, options.arpeggio_json, options.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, 'Cannot compare interactions: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
