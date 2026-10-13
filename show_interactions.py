"""Show interaction groups made by the six add_* ChimeraX commands.

    runscript /path/to/show_interactions.py
    show_interactions #2
    show_interactions #2 A-D A-E keep #2/D:8,44,70 #2/E:8,44,70 display_Dist false
    show_interactions #2 A-B hide_other_chains false
    hide_interactions #2

Chain pairs such as A-D select interfaces. ``keep`` shows extra residues.
Omitting display_Dist preserves the distance-label visibility of selected groups.
For specific pairs, hide_other_chains defaults to true; false preserves the
current cartoon and atom display of chains outside the requested interfaces.
Without chain pairs, chains with displayed interactions are shown even if their
cartoons were previously hidden.
"""
import re
from chimerax.core.commands import CmdDesc, RestOfLine, register, run

NAME = re.compile(r'^Model_(?P<model>\d+(?:_\d+)*)_(?P<kind>bad_hydrophobic|salt_bridge|hydrophobic|hbond|polar|pi)_Chain_(?P<a>[^_]+)_to_Chain_(?P<b>[^_]+)$')
KEEP = re.compile(r'^(#\d+(?:\.\d+)*)/([^/:]+):([\d,]+)$')
PAIR = re.compile(r'^([^/#:\s-]+)-([^/#:\s-]+)$')


def models(session):
    seen = set()
    for root in session.models:
        try:
            family = root.all_models()
        except Exception:
            family = (root,)
        for model in family:
            if id(model) not in seen:
                seen.add(id(model))
                yield model


def spec(model):
    return '#' + '.'.join(map(str, model.id))


def parse(arguments):
    words = str(arguments).split()
    if not words or not re.fullmatch(r'#\d+(?:\.\d+)*', words[0]):
        raise ValueError('Specify a coordinate model, such as #2.')
    source_spec = words.pop(0)
    pairs, keeps, labels, in_keep, hide_other_chains = set(), [], None, False, True
    i = 0
    while i < len(words):
        word = words[i]
        if word.lower() == 'keep':
            in_keep = True
        elif word.lower() in ('display_dist', 'displaydist'):
            i += 1
            if i >= len(words) or words[i].lower() not in ('true', 'false', 'yes', 'no'):
                raise ValueError('display_Dist requires true or false.')
            labels = words[i].lower() in ('true', 'yes')
            in_keep = False
        elif word.lower() in ('hide_other_chains', 'hideotherchains'):
            i += 1
            if i >= len(words) or words[i].lower() not in ('true', 'false', 'yes', 'no'):
                raise ValueError('hide_other_chains requires true or false.')
            hide_other_chains = words[i].lower() in ('true', 'yes')
            in_keep = False
        elif in_keep:
            match = KEEP.fullmatch(word)
            if not match:
                raise ValueError('Invalid keep residue specification: ' + word)
            if match[1] != source_spec:
                raise ValueError('Extra residues must reference the same coordinate model.')
            keeps.append((match[2], {int(n) for n in match[3].split(',') if n}))
        else:
            match = PAIR.fullmatch(word)
            if not match:
                raise ValueError('Invalid chain pair ' + word)
            pairs.add(frozenset((match[1], match[2])))
        i += 1
    return source_spec, pairs, keeps, labels, hide_other_chains


def details(group, source):
    if not hasattr(group, 'pseudobonds') or hasattr(group, 'atoms'):
        return None
    match = NAME.fullmatch(str(getattr(group, 'name', '')))
    if match and match['model'] == '_'.join(map(str, source.id)):
        return match['kind'], match['a'], match['b']
    if str(getattr(group, '_interaction_suite_source_id', '')) == '_'.join(map(str, source.id)):
        values = [getattr(group, '_interaction_suite_' + key, None)
                  for key in ('kind', 'chain1', 'chain2')]
        if all(values):
            return tuple(map(str, values))
    return None


def label_models(group):
    try:
        children = list(group.all_models())[1:]
    except Exception:
        children = list(group.child_models()) if hasattr(group, 'child_models') else []
    return [child for child in children if str(getattr(child, 'name', '')).lower()
            in ('distance', 'distances', 'label', 'labels')]


def protein_residue(atom, source, chains):
    residue = getattr(atom, 'residue', None)
    if residue is not None and getattr(residue, 'structure', None) is source:
        return residue
    marker = getattr(atom, 'structure', None)
    residue = getattr(marker, '_pi_source_residue_map', {}).get(id(atom))
    if residue is not None and getattr(residue, 'structure', None) is source:
        return residue
    # Arbitrary marker attributes are not guaranteed to survive a saved session.
    # Accept a reloaded pi centroid only if its nearest source residue is unique.
    try:
        import numpy as np
        pos = getattr(atom, 'scene_coord', None)
        if pos is None:
            pos = atom.coord
        candidates = []
        for r in source.residues:
            if r.chain_id not in chains:
                continue
            ds = []
            for a in r.atoms:
                if getattr(a.element, 'name', '') != 'H':
                    coord = getattr(a, 'scene_coord', None)
                    ds.append(float(np.linalg.norm(pos - (coord if coord is not None else a.coord))))
            if ds:
                candidates.append((min(ds), r))
        candidates.sort(key=lambda item: item[0])
        if candidates and candidates[0][0] <= 3.0 and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= 0.15):
            return candidates[0][1]
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def show_protein(session, source, residues, endpoints):
    by_chain = {}
    for r in residues:
        by_chain.setdefault(r.chain_id, set()).add(r.number)
    chosen = [f'{spec(source)}/{c}:{",".join(map(str, sorted(nums)))} & sidechain & ~H'
              for c, nums in sorted(by_chain.items())]
    extras = set()
    for atom in endpoints:
        r = getattr(atom, 'residue', None)
        if r is None or getattr(r, 'structure', None) is not source:
            continue
        if getattr(getattr(atom, 'element', None), 'name', '') == 'H' or getattr(atom, 'is_backbone', False):
            extras.add(atom)
            if getattr(atom, 'is_backbone', False):
                for bond in getattr(atom, 'bonds', ()):
                    extras.update(a for a in bond.atoms if getattr(a, 'is_backbone', False))
    for a in sorted(extras, key=lambda atom: (atom.residue.chain_id, atom.residue.number, atom.name)):
        chosen.append(f'{spec(source)}/{a.residue.chain_id}:{a.residue.number}@{a.name}')
    run(session, 'select clear')
    if chosen:
        run(session, 'select ' + chosen[0])
        for expression in chosen[1:]:
            run(session, 'select add ' + expression)
        run(session, 'show sel atoms')
        run(session, 'style sel stick')
        run(session, 'color sel byhetero')
    run(session, 'select clear')


def show_interactions(session, arguments):
    try:
        source_spec, pairs, keeps, dist, hide_others = parse(arguments)
        all_models = list(models(session))
        source = next((m for m in all_models if spec(m) == source_spec and hasattr(m, 'atoms')), None)
        if source is None:
            raise ValueError('Could not locate coordinate model ' + source_spec)
        groups = [(m, d) for m in all_models if (d := details(m, source)) is not None]
        if not groups:
            raise ValueError('No interaction groups found for ' + source_spec)
        chosen = [(m, d) for m, d in groups if not pairs or frozenset(d[1:]) in pairs]
        if not chosen:
            raise ValueError('No interaction groups match those chain pairs.')
        old_labels = {id(m): [(child, child.display) for child in label_models(m)]
                      for m, _ in chosen}
        for group, _ in groups:
            group.display = False
            for pb in group.pseudobonds:
                pb.display = False
            for child in label_models(group):
                child.display = False
        if pairs and not hide_others:
            selected_chains = sorted(set().union(*pairs))
            for chain in selected_chains:
                run(session, f'hide {source_spec}/{chain} atoms')
        else:
            run(session, f'hide {source_spec} atoms')
        residues, endpoints, count = set(), set(), 0
        for group, (kind, first, second) in chosen:
            gid = tuple(group.id)
            for parent in all_models:
                pid = tuple(getattr(parent, 'id', ()))
                if pid and len(pid) < len(gid) and gid[:len(pid)] == pid:
                    parent.display = True
            group.display = True
            for pb in group.pseudobonds:
                for atom in pb.atoms:
                    r = protein_residue(atom, source, {first, second})
                    if r is not None:
                        residues.add(r)
                    endpoints.add(atom)
                    atom.display = True
                    if hasattr(atom, 'hide'):
                        atom.hide = 0
                    marker = getattr(atom, 'structure', None)
                    if marker is not None and marker is not source:
                        marker.display = True
                        if kind == 'pi':
                            atom.radius = 0.001
                try:
                    pb.shown_when_atoms_hidden = True
                except (AttributeError, TypeError):
                    pass
                pb.display = True
                count += 1
            for child, previous in old_labels[id(group)]:
                child.display = previous if dist is None else dist
        for chain, numbers in keeps:
            residues.update(r for r in source.residues if r.chain_id == chain and r.number in numbers)
        available_chains = {r.chain_id for r in source.residues}
        if pairs:
            shown_chains = set().union(*pairs) & available_chains
            for chain in sorted(available_chains if hide_others else shown_chains):
                action = 'show' if chain in shown_chains else 'hide'
                run(session, f'{action} {source_spec}/{chain} cartoons')
        else:
            shown_chains = {chain for group, (_, *chains) in chosen
                            if group.pseudobonds for chain in chains}
            shown_chains.update(r.chain_id for r in residues)
            for chain in sorted(shown_chains & available_chains):
                run(session, f'show {source_spec}/{chain} cartoons')
        show_protein(session, source, residues, endpoints)
        session.logger.info(f'[show_interactions] {source_spec}: {count} interaction pseudobonds and {len(residues)} unique residues')
    except ValueError as error:
        session.logger.error(f'[show_interactions] Error: {error}')


def hide_interactions(session, arguments):
    try:
        source_spec, pairs, _, _, _ = parse(arguments)
        all_models = list(models(session))
        source = next((m for m in all_models if spec(m) == source_spec and hasattr(m, 'atoms')), None)
        if source is None:
            raise ValueError('Could not locate coordinate model ' + source_spec)
        hidden = 0
        for group in all_models:
            d = details(group, source)
            if d is None or (pairs and frozenset(d[1:]) not in pairs):
                continue
            group.display = False
            for pb in group.pseudobonds:
                pb.display = False
                hidden += 1
            for child in label_models(group):
                child.display = False
        run(session, f'hide {source_spec} atoms')
        session.logger.info(f'[hide_interactions] {source_spec}: hid {hidden} interaction pseudobonds')
    except ValueError as error:
        session.logger.error(f'[hide_interactions] Error: {error}')


register('show_interactions', CmdDesc(required=[('arguments', RestOfLine)],
         synopsis='Show existing interaction groups and protein residues'),
         show_interactions, logger=session.logger)
register('hide_interactions', CmdDesc(required=[('arguments', RestOfLine)],
         synopsis='Hide existing interaction groups and protein atoms'),
         hide_interactions, logger=session.logger)
print('[show_interactions] Commands initialized: show_interactions, hide_interactions')
