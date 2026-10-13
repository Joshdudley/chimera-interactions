"""Export existing add_* ChimeraX groups for comparison with PDBe Arpeggio.

Load: runscript /path/to/export_interactions.py
Run:  export_interactions #2 #2_interactions.csv

The CSV includes a group row for every analyzed chain pair, even when it has
zero contacts. It exports existing pseudobonds, including those currently hidden.
"""
import csv
import math
import re
from chimerax.core.commands import CmdDesc, StringArg, register

NAME = re.compile(
    r'^Model_(?P<source>\d+(?:_\d+)*)_'
    r'(?P<kind>bad_hydrophobic|salt_bridge|hydrophobic|hbond|polar|pi)_'
    r'Chain_(?P<a>[^_]+)_to_Chain_(?P<b>[^_]+)$'
)
COLUMNS = ('record_type', 'kind', 'chain_a', 'resnum_a', 'icode_a', 'resname_a',
           'atom_a', 'chain_b', 'resnum_b', 'icode_b', 'resname_b', 'atom_b',
           'group_id', 'distance')


def all_models(session):
    seen = set()
    for root in session.models:
        try:
            descendants = root.all_models()
        except Exception:
            descendants = (root,)
        for model in descendants:
            if id(model) not in seen:
                seen.add(id(model))
                yield model


def model_spec(model):
    return '#' + '.'.join(map(str, model.id))


def _coord(atom):
    value = getattr(atom, 'scene_coord', None)
    return value if value is not None else atom.coord


def _distance(a, b):
    try:
        return math.sqrt(sum((float(x) - float(y)) ** 2
                             for x, y in zip(_coord(a), _coord(b))))
    except (AttributeError, TypeError, ValueError):
        return None


def _source_residue(atom, source, chains):
    residue = getattr(atom, 'residue', None)
    if residue is not None and getattr(residue, 'structure', None) is source:
        return residue
    marker_set = getattr(atom, 'structure', None)
    residue = getattr(marker_set, '_pi_source_residue_map', {}).get(id(atom))
    if residue is not None and getattr(residue, 'structure', None) is source:
        return residue
    # Runtime-only centroid mappings can be lost when reopening a .cxs file.
    # Use geometry only when a marker is clearly closest to one source residue.
    distances = []
    for candidate in source.residues:
        if candidate.chain_id not in chains:
            continue
        heavy = [_distance(atom, a) for a in candidate.atoms
                 if getattr(getattr(a, 'element', None), 'name', '') != 'H']
        heavy = [d for d in heavy if d is not None]
        if heavy:
            distances.append((min(heavy), candidate))
    distances.sort(key=lambda item: item[0])
    if (distances and distances[0][0] <= 3.0 and
            (len(distances) == 1 or distances[1][0] - distances[0][0] >= 0.15)):
        return distances[0][1]
    return None


def _icode(residue):
    return str(getattr(residue, 'insertion_code', '') or '').strip()


def _group_details(group, source):
    if not hasattr(group, 'pseudobonds') or hasattr(group, 'atoms'):
        return None
    match = NAME.fullmatch(str(getattr(group, 'name', '')))
    if match and match['source'] == '_'.join(map(str, source.id)):
        return match['kind'], match['a'], match['b']
    if str(getattr(group, '_interaction_suite_source_id', '')) == '_'.join(map(str, source.id)):
        vals = [getattr(group, '_interaction_suite_' + key, None)
                for key in ('kind', 'chain1', 'chain2')]
        if all(vals):
            return tuple(map(str, vals))
    return None


def export_interactions(session, source_model, output_file=None):
    """Write one row per group and one row per existing pseudobond."""
    spec = str(source_model).strip()
    models = list(all_models(session))
    source = next((m for m in models if model_spec(m) == spec and hasattr(m, 'atoms')), None)
    if source is None:
        session.logger.error('[export_interactions] Coordinate model not found: ' + spec)
        return
    groups = [(m, d) for m in models if (d := _group_details(m, source)) is not None]
    if not groups:
        session.logger.error('[export_interactions] No existing interaction groups for ' + spec)
        return
    path = str(output_file).strip() if output_file else spec.lstrip('#').replace('.', '_') + '_interactions.csv'
    rows = []
    unresolved = 0
    for group, (kind, first, second) in groups:
        gid = model_spec(group)
        rows.append(dict(record_type='group', kind=kind, chain_a=first,
                         chain_b=second, group_id=gid))
        for pb in group.pseudobonds:
            try:
                atoms = tuple(pb.atoms)
                if len(atoms) != 2:
                    raise ValueError('expected two endpoints')
                endpoints = [(atom, _source_residue(atom, source, {first, second}))
                             for atom in atoms]
                if any(r is None for _, r in endpoints):
                    raise ValueError('could not map centroid to a unique protein residue')
                by_chain = {r.chain_id: (atom, r) for atom, r in endpoints}
                if set(by_chain) != {first, second}:
                    raise ValueError('endpoints do not match the named chain pair')
                a, ra = by_chain[first]
                b, rb = by_chain[second]
                distance = _distance(a, b)
                rows.append(dict(record_type='contact', kind=kind,
                                 chain_a=first, resnum_a=ra.number,
                                 icode_a=_icode(ra), resname_a=ra.name,
                                 atom_a=a.name, chain_b=second,
                                 resnum_b=rb.number, icode_b=_icode(rb),
                                 resname_b=rb.name, atom_b=b.name,
                                 group_id=gid,
                                 distance=round(distance, 3) if distance is not None else ''))
            except (AttributeError, TypeError, ValueError) as error:
                unresolved += 1
                session.logger.warning(f'[export_interactions] Skipped contact in {gid}: {error}')
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    from os.path import abspath
    session.logger.info(f'[export_interactions] Saved {len(groups)} groups and '
                        f'{len(rows) - len(groups)} contacts to {abspath(path)}; '
                        f'{unresolved} unresolved endpoint(s).')


register('export_interactions', CmdDesc(required=[('source_model', StringArg)],
         optional=[('output_file', StringArg)],
         synopsis='Export existing protein interaction pairs as CSV'),
         export_interactions, logger=session.logger)
print('[export_interactions] Command initialized: export_interactions')
