# add_polar.py
# ChimeraX native command.
# Usage:
#   add_polar #4 A B-F printFile yes #4.3 [showLabels yes|no]
#
# Residual polar-contact detector using the closest eligible O/N/S heavy-atom
# pair for each cross-interface residue pair. Exclude pairs assigned to an
# H-bond or conventional salt bridge to avoid overlapping interaction lines.

import os
import numpy as np
from chimerax.core.commands import run, register, CmdDesc, StringArg


def _parse_show_labels(value):
    v = str(value).strip().lower()
    if v in ("yes", "true", "on", "1"):
        return True
    if v in ("no", "false", "off", "0"):
        return False
    raise ValueError("showLabels must be yes/no, true/false, on/off, or 1/0")

STANDARD_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
POLAR_RES = {"ASN", "GLN", "SER", "THR", "ASP", "GLU", "ARG", "LYS", "HIS", "CYS", "TYR", "TRP", "MET"}
POLAR_ELEMENTS = {"O", "N", "S"}
POLAR_CUTOFF = 4.0
HBOND_CUTOFF = 3.5

# Keep these atom definitions in step with add_hbond.py. That command uses a
# heavy-atom distance proxy even for structures without explicit hydrogens.
HBOND_DONORS = {
    "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"}, "LYS": {"NZ"}, "SER": {"OG"},
    "THR": {"OG1"}, "TRP": {"NE1"}, "TYR": {"OH"}, "CYS": {"SG"},
}
HBOND_ACCEPTORS = {
    "ASN": {"OD1"}, "ASP": {"OD1", "OD2"}, "GLN": {"OE1"},
    "GLU": {"OE1", "OE2"}, "HIS": {"ND1", "NE2"}, "SER": {"OG"},
    "THR": {"OG1"}, "TYR": {"OH"}, "CYS": {"SG"}, "MET": {"SD"},
}

# Conventional side-chain salt-bridge atom definitions.
# These are used only to exclude salt-bridge contacts from the residual
# polar-contact category; they do not exclude other charged-residue
# polar contacts such as ARG-ARG N...N.
SALT_BRIDGE_ATOMS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "LYS": {"NZ"},
    "ARG": {"NE", "NH1", "NH2"},
}


def _parse_chains(chain_spec):
    if "-" in chain_spec:
        start, end = chain_spec.split("-", 1)
        if len(start) != 1 or len(end) != 1:
            raise ValueError("Chain range must look like B-F")
        return [chr(c) for c in range(ord(start), ord(end) + 1)]
    return [chain_spec]


def _find_structure(session, model_input):
    spec = str(model_input).strip()
    if not spec.startswith("#") or spec.startswith("##"):
        session.logger.error("[add_polar] Error: source model must include '#', for example #4.")
        return None
    model_id = spec[1:]
    if not model_id or any(p == "" or not p.isdigit() for p in model_id.split(".")):
        session.logger.error("[add_polar] Error: source model must be numeric, such as #4 or #4.2.")
        return None
    for m in session.models:
        if hasattr(m, "atoms") and ".".join(map(str, m.id)) == model_id:
            return m
    session.logger.error(f"[add_polar] Error: Could not locate molecular coordinate model matching {spec}")
    return None


def _eligible_polar_atom(atom):
    residue = atom.residue
    return (atom.element.name in POLAR_ELEMENTS
            and (residue.name in POLAR_RES
                 or (residue.name in STANDARD_RES and atom.name in {"N", "O", "OXT"})))


def _is_hbond_donor_heavy_atom(atom):
    return ((atom.name == "N" and atom.residue.name != "PRO")
            or atom.name in HBOND_DONORS.get(atom.residue.name, set()))


def _is_hbond_acceptor(atom):
    return (atom.name == "O"
            or atom.name in HBOND_ACCEPTORS.get(atom.residue.name, set()))


def _is_hbond_like_pair(a1, a2):
    # Match add_hbond's donor/acceptor classification in either orientation.
    return (
        (_is_hbond_donor_heavy_atom(a1) and _is_hbond_acceptor(a2))
        or (_is_hbond_donor_heavy_atom(a2) and _is_hbond_acceptor(a1))
    )


def _native_add_polar(session, model_input, chain1_id, chain2_id,
                            print_file_keyword, print_file_value, interaction_model_id, showLabels="yes"):
    struct_model = _find_structure(session, model_input)
    if struct_model is None:
        return

    if str(print_file_keyword).strip().lower() not in ("printf_file", "print_file", "printfile"):
        session.logger.error("Usage: add_polar #MODEL CHAIN1 CHAIN2 printFile yes #INTERACTION_MODEL_ID")
        return
    print_file_mode = str(print_file_value).strip().lower()
    if print_file_mode not in ("yes", "no", "true", "false", "on", "off", "1", "0"):
        session.logger.error("Error: printFile must be yes/no, true/false, on/off, or 1/0.")
        return
    write_file = print_file_mode in ("yes", "true", "on", "1")

    interaction_input = str(interaction_model_id).strip()
    if not interaction_input.startswith("#") or interaction_input.startswith("##"):
        session.logger.error("[add_polar] Error: destination model must include '#', for example #4.3.")
        return
    interaction_id_str = interaction_input[1:]
    if not interaction_id_str or any(p == "" or not p.isdigit() for p in interaction_id_str.split(".")):
        session.logger.error("[add_polar] Error: destination model must be numeric, such as #4.3.")
        return

    try:
        show_labels = _parse_show_labels(showLabels)
    except ValueError as e:
        session.logger.error(f"[add_polar] Error: {e}")
        return

    try:
        target_chains = _parse_chains(chain2_id)
    except ValueError as e:
        session.logger.error(f"[add_polar] Error: {e}")
        return

    model_spec = f"#{'.'.join(map(str, struct_model.id))}"
    source_id = model_spec[1:].replace('.', '_')
    dest_id = interaction_id_str.replace('.', '_')
    # ChimeraX tracks pseudobond groups by name. Use the final, unique
    # chain-pair name when creating the group, including empty groups.
    final_group_name = f"Model_{source_id}_polar_Chain_{chain1_id}_to_Chain_{chain2_id}"
    temp_group_name = final_group_name

    for m in session.models:
        if (hasattr(m, "pseudobonds") and not hasattr(m, "atoms")
                and getattr(m, "name", "") == final_group_name):
            session.logger.error(f"[add_polar] Error: interaction group '{final_group_name}' already exists.")
            return

    atoms_c1 = [a for a in struct_model.atoms
                if a.residue.chain_id == chain1_id and _eligible_polar_atom(a)]
    atoms_c2 = [a for a in struct_model.atoms
                if a.residue.chain_id in target_chains and _eligible_polar_atom(a)]

    donor_heavy_c1 = sum(
        1 for a in struct_model.atoms
        if a.residue.chain_id == chain1_id
        and _eligible_polar_atom(a)
        and _is_hbond_donor_heavy_atom(a)
    )
    donor_heavy_c2 = sum(
        1 for a in struct_model.atoms
        if a.residue.chain_id in target_chains
        and _eligible_polar_atom(a)
        and _is_hbond_donor_heavy_atom(a)
    )

    session.logger.info(
        f"[add_polar] Found {len(atoms_c1)} eligible polar heavy atoms on Chain {chain1_id} "
        f"and {len(atoms_c2)} on Chains {chain2_id}; residue pairs with a qualifying "
        "hydrogen bond are assigned to add_hbond instead."
    )

    def is_salt_bridge_pair(a1, a2, distance):
        # Only exclude conventional opposite-charge side-chain pairs.
        # Keep same-charge contacts (e.g. ARG-ARG) available to add_polar.
        if distance > POLAR_CUTOFF:
            return False
        r1 = a1.residue.name
        r2 = a2.residue.name
        if r1 in {"ASP", "GLU"} and r2 in {"LYS", "ARG"}:
            return a1.name in SALT_BRIDGE_ATOMS[r1] and a2.name in SALT_BRIDGE_ATOMS[r2]
        if r1 in {"LYS", "ARG"} and r2 in {"ASP", "GLU"}:
            return a1.name in SALT_BRIDGE_ATOMS[r1] and a2.name in SALT_BRIDGE_ATOMS[r2]
        return False

    # Deliberately permissive: one closest O/N/S pair per residue pair,
    # unless add_hbond would classify that residue pair as a hydrogen bond.
    residue_pairs = {}
    excluded_salt_pairs = set()
    excluded_hbond_pairs = set()
    for a1 in atoms_c1:
        res1 = a1.residue
        for a2 in atoms_c2:
            res2 = a2.residue
            if res1 is res2:
                continue
            d = float(np.linalg.norm(a1.coord - a2.coord))
            if d <= POLAR_CUTOFF:
                key = (res1, res2)
                if is_salt_bridge_pair(a1, a2, d):
                    excluded_salt_pairs.add(key)
                    continue
                if d <= HBOND_CUTOFF and _is_hbond_like_pair(a1, a2):
                    # add_hbond keeps the closest such atom pair. Exclude the
                    # whole residue pair here so another O/N/S contact between
                    # those residues cannot overlap its H-bond line.
                    excluded_hbond_pairs.add(key)
                    continue
                if key not in residue_pairs or d < residue_pairs[key]["dist"]:
                    residue_pairs[key] = {"atom1": a1, "atom2": a2, "dist": d}

    # A residue pair is omitted entirely if its closest qualifying polar
    # contact is itself a conventional salt-bridge atom pair. This prevents
    # the same residue pair from appearing in both categories.
    for key in excluded_hbond_pairs:
        residue_pairs.pop(key, None)
    for key in excluded_salt_pairs:
        if key in residue_pairs and is_salt_bridge_pair(
                residue_pairs[key]["atom1"], residue_pairs[key]["atom2"], residue_pairs[key]["dist"]):
            del residue_pairs[key]

    affected_residues = set()
    chain_counts = {chain1_id: 0, **{c: 0 for c in target_chains}}
    lines = []

    for data in residue_pairs.values():
        a1, a2 = data["atom1"], data["atom2"]
        affected_residues.update((a1.residue, a2.residue))
        chain_counts[a1.residue.chain_id] += 1
        chain_counts[a2.residue.chain_id] += 1

        spec1 = f"{model_spec}/{a1.residue.chain_id}:{a1.residue.number}@{a1.name}"
        spec2 = f"{model_spec}/{a2.residue.chain_id}:{a2.residue.number}@{a2.name}"
        cmd = (
            f'pbond {spec1} {spec2} color purple radius 0.2 dashes 8 '
            f'showDist true name "{temp_group_name}" global'
        )
        run(session, cmd)
        lines.append(
            f"pbond {spec1} {spec2} color purple radius 0.2 dashes 8 showDist true name 'polar' global"
        )

    detected = None
    for m in session.models:
        if (hasattr(m, "pseudobonds") and not hasattr(m, "atoms")
                and getattr(m, "name", "") == temp_group_name):
            detected = f"#{'.'.join(map(str, m.id))}"
            break

    if detected is None:
        # A valid interface may contain no contacts. Keep its empty subgroup.
        try:
            from chimerax.atomic import PseudobondManager
            empty_group = PseudobondManager(session).get_group(temp_group_name, create=True)
            if empty_group is None:
                raise RuntimeError("PseudobondManager.get_group() returned None")
            if not any(m is empty_group for m in _suite_all_models(session)):
                session.models.add([empty_group])
            detected = _suite_model_spec(empty_group)
            session.logger.info(f"[add_polar] No contacts found for {interaction_input}; created an empty subgroup.")
        except Exception as e:
            session.logger.error(f"[add_polar] ERROR: Could not create empty subgroup for {interaction_input}: {e}")
            return

    run(session, f"rename {detected} id {interaction_id_str}")
    assigned = None
    for m in session.models:
        if (f"#{'.'.join(map(str, m.id))}" == interaction_input
                and hasattr(m, "pseudobonds") and not hasattr(m, "atoms")):
            assigned = m
            break
    if assigned is None:
        session.logger.error(f"[add_polar] ERROR: Could not assign generated group to {interaction_input}.")
        return

    session.logger.info(
        f"[add_polar] Successfully assigned interaction model {interaction_input} "
        f"with group name '{final_group_name}'"
    )

    # Always create the ChimeraX distance-label child model via showDist true.
    # showLabels only controls its initial visibility. When false, keep the child
    # object in the hierarchy but hide that child so it can be re-enabled later.
    if not show_labels:
        # IMPORTANT: showDist remains TRUE so ChimeraX creates the distance
        # label object. Hide only that child after creation. Use the parent
        # model's full hierarchy because session.models is not guaranteed to
        # enumerate nested child models consistently across ChimeraX versions.
        try:
            children = list(assigned.all_models())[1:]
        except Exception:
            children = []
        for child in children:
            child_name = str(getattr(child, "name", "")).strip().lower()
            if child_name in {"distance", "distances", "labels"}:
                try:
                    child.display = False
                except Exception:
                    try:
                        run(session, f"hide #{'.'.join(map(str, child.id))} models")
                    except Exception:
                        pass

    if write_file:
        try:
            with open("polar.txt", "w") as f:
                f.write("\n".join(lines) + "\n\n")
                f.write("=" * 50 + "\n")
                f.write("      INTERFACE INTERACTION METRICS SUMMARY      \n")
                f.write("=" * 50 + "\n")
                f.write(f"Total number of polar interactions found: {len(residue_pairs)}\n")
                f.write("-" * 50 + "\n")
                for chain_name, count in sorted(chain_counts.items()):
                    f.write(f"  • Chain {chain_name} : {count} polar interaction contacts found.\n")
                f.write("=" * 50 + "\n")
            session.logger.info(f"[add_polar] Exported polar.txt to {os.path.abspath('polar.txt')}")
        except Exception as e:
            session.logger.error(f"[add_polar] Error writing polar.txt: {e}")

    # Keep protein atoms in their existing representation. Do not promote
    # atoms to ball-and-stick; the polar interaction itself is the dashed line.
    run(session, "select clear")

    print("\n" + "=" * 50)
    print("      INTERFACE INTERACTION METRICS PER CHAIN      ")
    print("=" * 50)
    for chain_name, count in sorted(chain_counts.items()):
        print(f"  • Chain {chain_name} : {count} polar interaction contacts found.")
    print("=" * 50)
    print(
        f"Successfully generated exactly {len(residue_pairs)} polar interactions "
        f"inside group '{final_group_name}' at model '{interaction_input}'!\n"
    )




# ---------------------------------------------------------------------------
# Chain-pair hierarchy wrapper added without changing the native detector body.
# The original detector remains the chemistry authority and is run once per
# target chain. Each successful result is moved under a family parent.
# ---------------------------------------------------------------------------
from chimerax.core.models import Model


def _suite_all_models(session):
    seen=set(); out=[]
    def walk(items):
        for m in items:
            if id(m) in seen:
                continue
            seen.add(id(m)); out.append(m)
            try:
                children=list(m.child_models())
            except Exception:
                children=[]
            walk(children)
    walk(list(session.models))
    return out


def _suite_model_spec(model):
    return '#' + '.'.join(map(str, model.id))


def _suite_find_model(session, spec):
    s=str(spec).strip()
    if not s.startswith('#'):
        return None
    try:
        wanted=tuple(int(x) for x in s[1:].split('.'))
    except Exception:
        return None
    for m in _suite_all_models(session):
        if tuple(getattr(m,'id',())) == wanted:
            return m
    return None


def _suite_chain_list(spec):
    s=str(spec).strip()
    if '-' not in s:
        return [s] if s else []
    parts=s.split('-')
    if len(parts)!=2 or len(parts[0])!=1 or len(parts[1])!=1:
        raise ValueError(f'Chain range must be like B-F, got {s}')
    a,b=parts
    step=1 if ord(b)>=ord(a) else -1
    return [chr(c) for c in range(ord(a),ord(b)+step,step)]


def _suite_hide_group(group):
    try: group.display=False
    except Exception: pass
    for pb in getattr(group,'pseudobonds',()):
        try: pb.display=False
        except Exception: pass
    try: children=list(group.all_models())[1:]
    except Exception: children=[]
    for child in children:
        name=str(getattr(child,'name','')).lower()
        if 'distance' in name or 'label' in name:
            try: child.display=False
            except Exception: pass


def _suite_snapshot(source):
    atoms=[(a,getattr(a,'display',False),getattr(a,'style',None),getattr(a,'radius',None)) for a in source.atoms]
    bonds=[]; seen=set()
    for a in source.atoms:
        for b in getattr(a,'bonds',()):
            if id(b) in seen: continue
            seen.add(id(b)); bonds.append((b,getattr(b,'display',False),getattr(b,'radius',None)))
    return atoms,bonds


def _suite_restore(snap):
    atoms,bonds=snap
    for a,d,style,radius in atoms:
        try: a.display=d
        except Exception: pass
        try: a.style=style
        except Exception: pass
        try: a.radius=radius
        except Exception: pass
    for b,d,radius in bonds:
        try: b.display=d
        except Exception: pass
        try: b.radius=radius
        except Exception: pass


def add_polar(session, model_input, chain1_id, chain2_id, print_file_keyword, print_file_value, interaction_model_id, showLabels='yes'):
    chain2_list=_suite_chain_list(chain2_id)
    if not chain2_list:
        session.logger.error('[add_polar] No target chains were supplied.')
        return
    source=_suite_find_model(session,model_input)
    if source is None or not hasattr(source,'atoms'):
        session.logger.error(f'[add_polar] Could not locate coordinate model {model_input}.')
        return
    parent_spec=str(interaction_model_id).strip()
    if not parent_spec.startswith('#'):
        session.logger.error('[add_polar] Parent interaction model must look like #2.1.')
        return
    if _suite_find_model(session,parent_spec) is not None:
        session.logger.error(f'[add_polar] Parent model {parent_spec} already exists. Remove it before rebuilding this interaction family.')
        return

    source_id='_'.join(map(str,source.id))
    parent=Model(f'Model_{source_id}_polar',session)
    session.models.add([parent])
    run(session, f'rename {_suite_model_spec(parent)} name "Model_{source_id}_polar" id {parent_spec}')
    parent=_suite_find_model(session,parent_spec)
    if parent is None:
        session.logger.error(f'[add_polar] Could not create parent interaction model {parent_spec}.')
        return

    counts={}; created=[]
    for idx,c2 in enumerate(chain2_list,start=1):
        child_spec=parent_spec+'.'+str(idx)
        snap=_suite_snapshot(source)
        before_ids={id(m) for m in _suite_all_models(session)}
        try:
            _native_add_polar(session,_suite_model_spec(source),str(chain1_id),str(c2),'printFile','no',child_spec,showLabels)
        finally:
            _suite_restore(snap)

        child=_suite_find_model(session,child_spec)
        if child is None or not hasattr(child,'pseudobonds'):
            counts[c2]=0
            continue

        # The native detector created this subgroup with its final chain-pair name.
        try:
            child._interaction_suite_kind='polar'
            child._interaction_suite_source_id=source_id
            child._interaction_suite_chain1=str(chain1_id)
            child._interaction_suite_chain2=str(c2)
        except Exception:
            pass
        _suite_hide_group(child)
        created.append(child)
        counts[c2]=len(list(getattr(child,'pseudobonds',())))

        if 'polar'=='pi':
            for m in _suite_all_models(session):
                if str(getattr(m,'name','')).startswith('pi_centroids_') and id(m) not in before_ids:
                    try: m.display=False
                    except Exception: pass

    mode=str(print_file_value).strip().lower()
    total=sum(counts.values())
    print('\n'+'='*50)
    print('      INTERFACE INTERACTION METRICS PER CHAIN      ')
    print('='*50)
    for c2 in chain2_list:
        print(f'  - Chain {c2} : {counts.get(c2,0)} polar interaction contacts found.')
    print('='*50)
    print(f'Created parent {parent_spec} with {len(created)} chain-pair subgroup(s); total contacts: {total}')

    if mode in ('yes','true','on','1'):
        filename='polar.txt'
        with open(filename,'w') as f:
            f.write('='*50+'\n')
            f.write('      INTERFACE INTERACTION METRICS SUMMARY      \n')
            f.write('='*50+'\n')
            f.write('Total number of polar interactions found: '+str(total)+'\n')
            f.write('-'*50+'\n')
            for c2 in chain2_list:
                f.write(f'  - Chain {c2} : {counts.get(c2,0)} polar interaction contacts found.\n')
            f.write('='*50+'\n')


desc = CmdDesc(
    required=[("model_input", StringArg), ("chain1_id", StringArg), ("chain2_id", StringArg), ("print_file_keyword", StringArg), ("print_file_value", StringArg), ("interaction_model_id", StringArg)],
    keyword=[("showLabels", StringArg)],
    synopsis="Generate polar interaction subgroups by chain pair"
)
register("add_polar", desc, add_polar, logger=session.logger)
print("[add_polar] Command initialized successfully with chain-pair subgroups.")
